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

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace mlir;

namespace {

struct RunnerOptions {
  std::string inputFilename;
  std::string entryFunction = "run";
  std::string resultType = "f32";
  std::string gpuChip = "sm_86";
  std::string cubinFormat = "fatbin";
  int optLevel = 3;
};

static void printUsage(llvm::raw_ostream &os, llvm::StringRef programName) {
  os << "Usage: " << programName
     << " <input.mlir> [-e function] [--entry-point-result=f32|void]"
        " [--gpu-chip=sm_86] [--cubin-format=fatbin|isa] [--opt-level=0..3]\n";
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
    if (arg.consume_front("--gpu-chip=")) {
      options.gpuChip = arg.str();
      continue;
    }
    if (arg.consume_front("--cubin-format=")) {
      options.cubinFormat = arg.str();
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
  return true;
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
  if (failed(parsePassPipeline("mini-gpu-lowering", pm)))
    return failure();

  std::string pipeline =
      "gpu-lower-to-nvvm-pipeline{cubin-chip=" + options.gpuChip +
      " cubin-format=" + options.cubinFormat + " opt-level=" +
      std::to_string(options.optLevel) + "}";
  if (failed(parsePassPipeline(pipeline, pm)))
    return failure();
  return pm.run(module);
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
  llvm::outs() << result << "\n";
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

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  RunnerOptions runnerOptions;
  bool printedHelp = false;
  if (!parseOptions(argc, argv, runnerOptions, printedHelp))
    return printedHelp ? 0 : 1;

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
  auto module = loadModule(context, runnerOptions.inputFilename);
  if (!module) {
    llvm::errs() << "Failed to parse input module\n";
    return 1;
  }

  if (failed(runMiniGpuLowering(*module, runnerOptions))) {
    llvm::errs() << "Failed to run GPU lowering pipeline\n";
    return 1;
  }

  ExecutionEngineOptions engineOptions;
  engineOptions.llvmModuleBuilder = convertMLIRModule;
  engineOptions.sharedLibPaths = {
      MINI_CUDA_RUNTIME_WRAPPERS_PATH,
      MINI_MLIR_RUNNER_UTILS_PATH,
      MINI_MLIR_C_RUNNER_UTILS_PATH,
  };

  auto maybeEngine = ExecutionEngine::create(module->getOperation(), engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Failed to create execution engine: "
                 << llvm::toString(maybeEngine.takeError()) << "\n";
    return 1;
  }

  auto engine = std::move(*maybeEngine);
  if (runnerOptions.resultType == "void")
    return invokeVoid(*engine, runnerOptions.entryFunction);
  if (runnerOptions.resultType == "f32") {
    float result = 0.0f;
    return invokeWithResult(*engine, runnerOptions.entryFunction, result);
  }

  llvm::errs() << "Unsupported result kind: " << runnerOptions.resultType << "\n";
  return 1;
}
