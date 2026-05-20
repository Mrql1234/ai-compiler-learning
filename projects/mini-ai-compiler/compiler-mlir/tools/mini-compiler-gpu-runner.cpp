#include "MiniCompiler/MiniDialect.h"
#include "MiniCompiler/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#ifdef MINI_ENABLE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

#ifndef MINI_CUDA_RUNTIME_WRAPPERS_PATH
#define MINI_CUDA_RUNTIME_WRAPPERS_PATH ""
#endif

#ifndef MINI_MLIR_RUNNER_UTILS_PATH
#define MINI_MLIR_RUNNER_UTILS_PATH ""
#endif

#ifndef MINI_MLIR_C_RUNNER_UTILS_PATH
#define MINI_MLIR_C_RUNNER_UTILS_PATH ""
#endif

using namespace mlir;

namespace {

class NvtxRange {
public:
  explicit NvtxRange(const char *name) {
#ifdef MINI_ENABLE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~NvtxRange() {
#ifdef MINI_ENABLE_NVTX
    nvtxRangePop();
#endif
  }
};

class PerfRuntimeHooks {
public:
  struct KernelTimingSequence {
    int64_t count = 0;
    double totalMs = 0.0;
  };

  explicit PerfRuntimeHooks(const char *runtimePath) {
#ifndef _WIN32
    if (!runtimePath || runtimePath[0] == '\0')
      return;
    handle = dlopen(runtimePath, RTLD_LAZY | RTLD_LOCAL);
    if (!handle)
      return;
    resetFn = reinterpret_cast<ResetFn>(dlsym(handle, "miniPerfResetKernelTimings"));
    countFn = reinterpret_cast<CountFn>(dlsym(handle, "miniPerfGetKernelTimingCount"));
    getFn = reinterpret_cast<GetFn>(dlsym(handle, "miniPerfGetKernelTimingMs"));
#else
    (void)runtimePath;
#endif
  }

  bool available() const { return resetFn && countFn && getFn; }

  void reset() const {
    if (available())
      resetFn();
  }

  KernelTimingSequence collectSequence() const {
    KernelTimingSequence sequence;
    if (!available())
      return sequence;
    sequence.count = countFn();
    for (int64_t index = 0; index < sequence.count; ++index)
      sequence.totalMs += getFn(index);
    return sequence;
  }

private:
  using ResetFn = void (*)();
  using CountFn = int64_t (*)();
  using GetFn = double (*)(int64_t);

#ifndef _WIN32
  void *handle = nullptr;
#endif
  ResetFn resetFn = nullptr;
  CountFn countFn = nullptr;
  GetFn getFn = nullptr;
};

struct RunnerOptions {
  std::string inputFilename;
  std::string entryFunction = "run";
  std::string resultType = "f32";
  std::string kernelBackend = "generated_nvvm";
  bool quantized = false;
  std::string gpuChip = "sm_86";
  std::string cubinFormat = "fatbin";
  std::string ptxasCmdOptions;
  std::string jsonOutput;
  std::string dumpLowered;
  int optLevel = 3;
  int warmup = 0;
  int repeat = 1;
};

static void printUsage(llvm::raw_ostream &os, llvm::StringRef programName) {
  os << "Usage: " << programName
     << " <input.mlir> [-e function] [--entry-point-result=f32|void]"
        " [--quantized]"
        " [--kernel-backend=generated_nvvm|mlir_nvvm|cuda_hand|cublas|cutlass]"
        " [--gpu-chip=sm_86] [--cubin-format=fatbin|isa]"
        " [--ptxas-cmd-options=...] [--opt-level=0..3]"
        " [--warmup=N] [--repeat=N] [--json-output=path]"
        " [--dump-lowered=path]\n";
}

static bool parseNonNegativeInt(llvm::StringRef value,
                                llvm::StringRef optionName, int &result) {
  int parsed = 0;
  if (value.getAsInteger(10, parsed) || parsed < 0) {
    llvm::errs() << "Expected " << optionName
                 << " to be a non-negative integer\n";
    return false;
  }
  result = parsed;
  return true;
}

static bool parseOptions(int argc, char **argv, RunnerOptions &options,
                         bool &printedHelp) {
  for (int index = 1; index < argc; ++index) {
    llvm::StringRef arg(argv[index]);
    if (arg == "-h" || arg == "--help") {
      printUsage(llvm::outs(), argv[0]);
      printedHelp = true;
      return false;
    }
    if (arg == "-e") {
      if (index + 1 >= argc) {
        llvm::errs() << "Missing function name after -e\n";
        return false;
      }
      options.entryFunction = argv[++index];
      continue;
    }
    if (arg.consume_front("--entry-point-result=")) {
      options.resultType = arg.str();
      continue;
    }
    if (arg.consume_front("--kernel-backend=")) {
      options.kernelBackend = arg.str();
      continue;
    }
    if (arg == "--quantized") {
      options.quantized = true;
      continue;
    }
    if (arg.consume_front("--gpu-chip=")) {
      options.gpuChip = arg.str();
      continue;
    }
    if (arg.consume_front("--cubin-format=")) {
      options.cubinFormat = arg.str();
      continue;
    }
    if (arg.consume_front("--ptxas-cmd-options=")) {
      options.ptxasCmdOptions = arg.str();
      continue;
    }
    if (arg.consume_front("--opt-level=")) {
      int level = 0;
      if (arg.getAsInteger(10, level) || level < 0 || level > 3) {
        llvm::errs() << "Expected --opt-level to be an integer in [0, 3]\n";
        return false;
      }
      options.optLevel = level;
      continue;
    }
    if (arg.consume_front("--warmup=")) {
      if (!parseNonNegativeInt(arg, "--warmup", options.warmup))
        return false;
      continue;
    }
    if (arg.consume_front("--repeat=")) {
      if (!parseNonNegativeInt(arg, "--repeat", options.repeat))
        return false;
      if (options.repeat == 0) {
        llvm::errs() << "Expected --repeat to be greater than zero\n";
        return false;
      }
      continue;
    }
    if (arg.consume_front("--json-output=")) {
      options.jsonOutput = arg.str();
      continue;
    }
    if (arg.consume_front("--dump-lowered=")) {
      options.dumpLowered = arg.str();
      continue;
    }
    if (arg.starts_with("-")) {
      llvm::errs() << "Unknown option: " << arg << "\n";
      return false;
    }
    if (!options.inputFilename.empty()) {
      llvm::errs() << "Only one input file is supported\n";
      return false;
    }
    options.inputFilename = arg.str();
  }

  if (options.inputFilename.empty()) {
    llvm::errs() << "Missing input file\n";
    printUsage(llvm::errs(), argv[0]);
    return false;
  }
  if (options.kernelBackend != "generated_nvvm" &&
      options.kernelBackend != "mlir_nvvm" &&
      options.kernelBackend != "cuda_hand" &&
      options.kernelBackend != "cublas" &&
      options.kernelBackend != "cutlass") {
    llvm::errs() << "Unsupported --kernel-backend: " << options.kernelBackend
                 << "\n";
    return false;
  }
  return true;
}

static bool isGeneratedNvvmBackend(llvm::StringRef backend) {
  return backend == "generated_nvvm" || backend == "mlir_nvvm";
}

static void printUnsupportedIntegratedBackend(llvm::StringRef backend) {
  llvm::errs()
      << "kernel backend '" << backend
      << "' is recognized but not implemented in the compiler-integrated "
         "runner yet. Use generated_nvvm/mlir_nvvm for the automatic MLIR GPU "
         "route, or use mini-compiler-kernel-bench through perf_run.py for the "
         "current external CUDA/cuBLAS baseline.\n";
}

static OwningOpRef<ModuleOp> loadModule(MLIRContext &context,
                                        llvm::StringRef inputFilename) {
  std::string errorMessage;
  auto inputFile = openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return {};
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

static LogicalResult runMiniGpuLowering(ModuleOp module,
                                        const RunnerOptions &options) {
  PassManager pm(module.getContext());
  if (failed(parsePassPipeline(
          options.quantized ? "mini-quantized-gpu-lowering" : "mini-gpu-lowering",
          pm)))
    return failure();

  std::string pipeline =
      "gpu-lower-to-nvvm-pipeline{cubin-chip=" + options.gpuChip +
      " cubin-format=" + options.cubinFormat + " opt-level=" +
      std::to_string(options.optLevel);
  if (!options.ptxasCmdOptions.empty())
    pipeline += " ptxas-cmd-options=" + options.ptxasCmdOptions;
  pipeline += "}";
  if (failed(parsePassPipeline(pipeline, pm)))
    return failure();
  return pm.run(module);
}

static LogicalResult dumpModule(ModuleOp module, llvm::StringRef path) {
  if (path.empty())
    return success();

  std::error_code error;
  llvm::raw_fd_ostream os(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "Failed to open lowered MLIR dump path '" << path
                 << "': " << error.message() << "\n";
    return failure();
  }
  module.print(os);
  os << "\n";
  return success();
}

static std::unique_ptr<llvm::Module>
convertMLIRModule(Operation *operation, llvm::LLVMContext &llvmContext) {
  auto module = dyn_cast<ModuleOp>(operation);
  if (!module)
    return operation->emitError("expected builtin.module"), nullptr;
  return translateModuleToLLVMIR(module, llvmContext);
}

template <typename ResultT>
static int invokeWithResult(ExecutionEngine &engine, llvm::StringRef functionName,
                            ResultT &result) {
  if (auto error =
          engine.invoke(functionName, ExecutionEngine::result(result))) {
    llvm::errs() << "Failed to invoke entry function: "
                 << llvm::toString(std::move(error)) << "\n";
    return 1;
  }
  return 0;
}

static int invokeVoid(ExecutionEngine &engine, llvm::StringRef functionName) {
  if (auto error = engine.invoke(functionName)) {
    llvm::errs() << "Failed to invoke entry function: "
                 << llvm::toString(std::move(error)) << "\n";
    return 1;
  }
  return 0;
}

static void writeJsonString(llvm::raw_ostream &os, llvm::StringRef value) {
  os << "\"";
  for (char c : value) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << "\"";
}

static double average(llvm::ArrayRef<double> values) {
  if (values.empty())
    return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

static double median(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  size_t middle = values.size() / 2;
  if (values.size() % 2 == 1)
    return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

static double minValue(llvm::ArrayRef<double> values) {
  return values.empty() ? 0.0
                        : *std::min_element(values.begin(), values.end());
}

static double maxValue(llvm::ArrayRef<double> values) {
  return values.empty() ? 0.0
                        : *std::max_element(values.begin(), values.end());
}

static void writeMetricObject(llvm::raw_ostream &os, llvm::StringRef source,
                              llvm::ArrayRef<double> timingsMs,
                              bool includeTimings = true) {
  std::vector<double> timingCopy(timingsMs.begin(), timingsMs.end());
  os << "{\n";
  os << "      \"source\": ";
  writeJsonString(os, source);
  os << ",\n";
  os << "      \"min\": " << minValue(timingsMs) << ",\n";
  os << "      \"mean\": " << average(timingsMs) << ",\n";
  os << "      \"median\": " << median(std::move(timingCopy)) << ",\n";
  os << "      \"max\": " << maxValue(timingsMs);
  if (includeTimings) {
    os << ",\n";
    os << "      \"timings_ms\": [";
    for (size_t i = 0, e = timingsMs.size(); i < e; ++i) {
      if (i != 0)
        os << ", ";
      os << timingsMs[i];
    }
    os << "]\n";
  } else {
    os << "\n";
  }
  os << "    }";
}

static LogicalResult writeJsonReport(const RunnerOptions &options,
                                     llvm::ArrayRef<double> timingsMs,
                                     llvm::ArrayRef<double> kernelTimingsMs,
                                     bool hasResult, float result,
                                     double compileMs,
                                     double engineCreateMs,
                                     double endToEndMs) {
  if (options.jsonOutput.empty())
    return success();

  std::error_code error;
  llvm::raw_fd_ostream os(options.jsonOutput, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "Failed to open JSON output path '" << options.jsonOutput
                 << "': " << error.message() << "\n";
    return failure();
  }

  std::vector<double> timingCopy(timingsMs.begin(), timingsMs.end());

  os << "{\n";
  os << "  \"backend\": \"mlir_nvvm\",\n";
  os << "  \"input\": ";
  writeJsonString(os, options.inputFilename);
  os << ",\n";
  os << "  \"entry_function\": ";
  writeJsonString(os, options.entryFunction);
  os << ",\n";
  os << "  \"kernel_backend\": ";
  writeJsonString(os, options.kernelBackend);
  os << ",\n";
  os << "  \"result_type\": ";
  writeJsonString(os, options.resultType);
  os << ",\n";
  os << "  \"quantized\": " << (options.quantized ? "true" : "false") << ",\n";
  os << "  \"gpu_chip\": ";
  writeJsonString(os, options.gpuChip);
  os << ",\n";
  os << "  \"cubin_format\": ";
  writeJsonString(os, options.cubinFormat);
  os << ",\n";
  os << "  \"opt_level\": " << options.optLevel << ",\n";
  os << "  \"ptxas_cmd_options\": ";
  writeJsonString(os, options.ptxasCmdOptions);
  os << ",\n";
  os << "  \"warmup\": " << options.warmup << ",\n";
  os << "  \"repeat\": " << options.repeat << ",\n";
  os << "  \"result\": ";
  if (hasResult)
    os << result;
  else
    os << "null";
  os << ",\n";
  os << "  \"latency_ms\": {\n";
  os << "    \"min\": " << minValue(timingsMs) << ",\n";
  os << "    \"mean\": " << average(timingsMs) << ",\n";
  os << "    \"median\": " << median(std::move(timingCopy)) << ",\n";
  os << "    \"max\": " << maxValue(timingsMs) << "\n";
  os << "  },\n";
  os << "  \"timings_ms\": [";
  for (size_t i = 0, e = timingsMs.size(); i < e; ++i) {
    if (i != 0)
      os << ", ";
    os << timingsMs[i];
  }
  os << "],\n";
  os << "  \"metrics\": {\n";
  if (!kernelTimingsMs.empty()) {
    os << "    \"kernel_ms\": ";
    writeMetricObject(os, "cuda_driver_event_kernel_sequence", kernelTimingsMs);
    os << ",\n";
  }
  os << "    \"invoke_ms\": ";
  writeMetricObject(os, "host_steady_clock_engine_invoke", timingsMs);
  os << ",\n";
  os << "    \"compile_ms\": " << compileMs << ",\n";
  os << "    \"engine_create_ms\": " << engineCreateMs << ",\n";
  os << "    \"end_to_end_ms\": " << endToEndMs << "\n";
  os << "  },\n";
  os << "  \"measurement_contract\": {\n";
  os << "    \"default_compare_metric\": \"kernel_ms\",\n";
  os << "    \"kernel_ms_available\": "
     << (!kernelTimingsMs.empty() ? "true" : "false") << ",\n";
  os << "    \"notes\": ";
  if (!kernelTimingsMs.empty())
    writeJsonString(os,
                    "kernel_ms is the sum of CUDA driver event timings recorded "
                    "inside mgpuLaunchKernel for one engine.invoke iteration");
  else
    writeJsonString(os,
                    "kernel_ms is unavailable because CUDA runtime perf hooks "
                    "were not loaded or no kernels were launched");
  os << "\n";
  os << "  },\n";
  os << "  \"artifacts\": {\n";
  os << "    \"lowered_mlir\": ";
  if (!options.dumpLowered.empty())
    writeJsonString(os, options.dumpLowered);
  else
    os << "null";
  os << "\n";
  os << "  }\n";
  os << "}\n";
  return success();
}

} // namespace

int main(int argc, char **argv) {
  auto endToEndStart = std::chrono::steady_clock::now();
  llvm::InitLLVM initLLVM(argc, argv);
  RunnerOptions runnerOptions;
  bool printedHelp = false;
  if (!parseOptions(argc, argv, runnerOptions, printedHelp))
    return printedHelp ? 0 : 1;

  if (!isGeneratedNvvmBackend(runnerOptions.kernelBackend)) {
    printUnsupportedIntegratedBackend(runnerOptions.kernelBackend);
    return 2;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  registry.insert<mini::MiniDialect>();
  mini::registerMiniPasses();
  mini::registerMiniPassPipelines();

  MLIRContext context(registry);
  context.getOrLoadDialect<mini::MiniDialect>();
  OwningOpRef<ModuleOp> module;
  double compileMs = 0.0;

  {
    NvtxRange range("compile");
    auto start = std::chrono::steady_clock::now();
    module = loadModule(context, runnerOptions.inputFilename);
    if (!module) {
      llvm::errs() << "Failed to parse input module\n";
      return 1;
    }

    if (failed(runMiniGpuLowering(*module, runnerOptions))) {
      llvm::errs() << "Failed to run GPU lowering pipeline\n";
      return 1;
    }
    if (failed(dumpModule(*module, runnerOptions.dumpLowered)))
      return 1;
    auto end = std::chrono::steady_clock::now();
    compileMs = std::chrono::duration<double, std::milli>(end - start).count();
  }

  ExecutionEngineOptions engineOptions;
  engineOptions.llvmModuleBuilder = convertMLIRModule;
  engineOptions.sharedLibPaths = {
      MINI_CUDA_RUNTIME_WRAPPERS_PATH,
      MINI_MLIR_RUNNER_UTILS_PATH,
      MINI_MLIR_C_RUNNER_UTILS_PATH,
  };

  std::unique_ptr<ExecutionEngine> engine;
  double engineCreateMs = 0.0;
  {
    NvtxRange range("engine_create");
    auto start = std::chrono::steady_clock::now();
    auto maybeEngine =
        ExecutionEngine::create(module->getOperation(), engineOptions);
    if (!maybeEngine) {
      llvm::errs() << "Failed to create execution engine: "
                   << llvm::toString(maybeEngine.takeError()) << "\n";
      return 1;
    }
    engine = std::move(*maybeEngine);
    auto end = std::chrono::steady_clock::now();
    engineCreateMs =
        std::chrono::duration<double, std::milli>(end - start).count();
  }

  float result = 0.0f;
  bool hasResult = runnerOptions.resultType == "f32";
  PerfRuntimeHooks perfHooks(MINI_CUDA_RUNTIME_WRAPPERS_PATH);
  auto runOnce = [&]() -> int {
    if (runnerOptions.resultType == "void")
      return invokeVoid(*engine, runnerOptions.entryFunction);
    if (runnerOptions.resultType == "f32")
      return invokeWithResult(*engine, runnerOptions.entryFunction, result);
    llvm::errs() << "Unsupported result kind: " << runnerOptions.resultType
                 << "\n";
    return 1;
  };

  {
    NvtxRange range("warmup");
    for (int iteration = 0; iteration < runnerOptions.warmup; ++iteration) {
      perfHooks.reset();
      if (runOnce() != 0)
        return 1;
    }
  }

  std::vector<double> timingsMs;
  std::vector<double> kernelTimingsMs;
  timingsMs.reserve(static_cast<size_t>(runnerOptions.repeat));
  kernelTimingsMs.reserve(static_cast<size_t>(runnerOptions.repeat));
  {
    NvtxRange range("benchmark");
    for (int iteration = 0; iteration < runnerOptions.repeat; ++iteration) {
      perfHooks.reset();
      auto start = std::chrono::steady_clock::now();
      if (runOnce() != 0)
        return 1;
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      timingsMs.push_back(elapsed.count());
      auto kernelTiming = perfHooks.collectSequence();
      if (kernelTiming.count > 0)
        kernelTimingsMs.push_back(kernelTiming.totalMs);
    }
  }

  {
    NvtxRange range("verify");
    auto endToEndEnd = std::chrono::steady_clock::now();
    double endToEndMs =
        std::chrono::duration<double, std::milli>(endToEndEnd - endToEndStart)
            .count();
    if (failed(writeJsonReport(runnerOptions, timingsMs, kernelTimingsMs,
                               hasResult, result, compileMs, engineCreateMs,
                               endToEndMs)))
      return 1;
  }

  if (hasResult)
    llvm::outs() << result << "\n";
  return 0;
}
