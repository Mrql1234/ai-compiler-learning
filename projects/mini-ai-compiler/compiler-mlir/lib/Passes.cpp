#include "MiniCompiler/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mini {
namespace {

struct MiniCanonicalizePass
    : public PassWrapper<MiniCanonicalizePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "mini-canonicalize"; }
  StringRef getDescription() const final {
    return "Mini dialect canonicalization pass skeleton";
  }
};

struct MiniConstantFoldPass
    : public PassWrapper<MiniConstantFoldPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "mini-const-fold"; }
  StringRef getDescription() const final {
    return "Mini dialect constant fold pass skeleton";
  }
};

struct MiniDCEPass : public PassWrapper<MiniDCEPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "mini-dce"; }
  StringRef getDescription() const final {
    return "Mini dialect dead code elimination pass skeleton";
  }
};

struct MiniFusionPass
    : public PassWrapper<MiniFusionPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "mini-fusion"; }
  StringRef getDescription() const final {
    return "Mini dialect fusion pass skeleton";
  }
};

} // namespace

std::unique_ptr<Pass> createMiniCanonicalizePass() {
  return std::make_unique<MiniCanonicalizePass>();
}

std::unique_ptr<Pass> createMiniConstantFoldPass() {
  return std::make_unique<MiniConstantFoldPass>();
}

std::unique_ptr<Pass> createMiniDCEPass() {
  return std::make_unique<MiniDCEPass>();
}

std::unique_ptr<Pass> createMiniFusionPass() {
  return std::make_unique<MiniFusionPass>();
}

void registerMiniPasses() {
  PassRegistration<MiniCanonicalizePass>();
  PassRegistration<MiniConstantFoldPass>();
  PassRegistration<MiniDCEPass>();
  PassRegistration<MiniFusionPass>();
}

} // namespace mini
