#include "MiniCompiler/MiniDialect.h"
#include "MiniCompiler/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
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
  bool quantized = false;
  std::vector<std::string> sharedLibs;
};

static void printUsage(llvm::raw_ostream &os, llvm::StringRef programName) {
  os << "Usage: " << programName
     << " <input.mlir> [-e function] [--entry-point-result=f32|void]"
        " [--quantized] [--shared-libs=lib1,lib2]\n";
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
    if (arg == "--quantized") {
      options.quantized = true;
      continue;
    }
    if (arg.consume_front("--shared-libs=")) {
      SmallVector<llvm::StringRef> paths;
      arg.split(paths, ',', -1, false);
      for (llvm::StringRef path : paths)
        options.sharedLibs.push_back(path.str());
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

static LogicalResult runMiniCpuLowering(ModuleOp module, bool quantized) {
  PassManager pm(module.getContext());
  if (failed(parsePassPipeline(
          quantized ? "mini-quantized-cpu-lowering" : "mini-cpu-lowering", pm)))
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

  if (failed(runMiniCpuLowering(*module, runnerOptions.quantized))) {
    llvm::errs() << "Failed to run selected CPU lowering pipeline\n";
    return 1;
  }

  ExecutionEngineOptions engineOptions;
  engineOptions.llvmModuleBuilder = convertMLIRModule;
  SmallVector<llvm::StringRef> sharedLibPaths;
  for (const std::string &path : runnerOptions.sharedLibs)
    sharedLibPaths.push_back(path);
  engineOptions.sharedLibPaths = sharedLibPaths;

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
