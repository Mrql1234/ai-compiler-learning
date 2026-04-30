#include "Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace {

struct BuiltinCanonicalizeFoldPass
    : public PassWrapper<BuiltinCanonicalizeFoldPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuiltinCanonicalizeFoldPass)

  StringRef getArgument() const override { return "constant-fold-builtin"; }

  StringRef getDescription() const override {
    return "Fold arithmetic constants using MLIR built-in canonicalization and folding";
  }

  void runOnOperation() override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    pipeline.addPass(createCanonicalizerPass());

    if (failed(runPipeline(pipeline, getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createBuiltinCanonicalizeFoldPass() {
  return std::make_unique<BuiltinCanonicalizeFoldPass>();
}

void registerBuiltinCanonicalizeFoldPass() {
  PassRegistration<BuiltinCanonicalizeFoldPass>();
}

} // namespace mlir
