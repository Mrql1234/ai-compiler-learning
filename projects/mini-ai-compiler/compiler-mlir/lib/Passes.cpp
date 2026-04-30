#include "MiniCompiler/Passes.h"
#include "MiniCompiler/MiniDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

using namespace mlir;

namespace mini {
namespace {

struct FoldReluConstantPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto constant = op.getInput().getDefiningOp<ConstantOp>();
    if (!constant)
      return failure();

    auto denseAttr = constant.getValue().dyn_cast_or_null<DenseFPElementsAttr>();
    auto resultType = op.getOutput().getType().dyn_cast<RankedTensorType>();
    if (!denseAttr || !resultType || !resultType.getElementType().isF32())
      return failure();

    SmallVector<float> folded;
    folded.reserve(denseAttr.getNumElements());
    for (const APFloat &element : denseAttr.getValues<APFloat>())
      folded.push_back(std::max(0.0f, element.convertToFloat()));

    auto newAttr = DenseElementsAttr::get(resultType, ArrayRef<float>(folded));
    auto newConstant =
        rewriter.create<ConstantOp>(op.getLoc(), newAttr, op.getOutput().getType());
    rewriter.replaceOp(op, newConstant.getOutput());
    if (constant->use_empty())
      rewriter.eraseOp(constant);
    return success();
  }
};

struct FuseLinearReluPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto linear = op.getInput().getDefiningOp<LinearOp>();
    if (!linear || !linear->hasOneUse())
      return failure();

    auto fused = rewriter.create<FusedLinearReluOp>(
        op.getLoc(), op.getOutput().getType(), linear.getInput(),
        linear.getWeight(), linear.getBias());
    rewriter.replaceOp(op, fused.getOutput());
    rewriter.eraseOp(linear);
    return success();
  }
};

struct MiniCanonicalizePass
    : public PassWrapper<MiniCanonicalizePass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldReluConstantPattern, FuseLinearReluPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "mini-canonicalize"; }
  StringRef getDescription() const final {
    return "Run canonicalization-style rewrites for the mini dialect";
  }
};

struct MiniConstantFoldPass
    : public PassWrapper<MiniConstantFoldPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldReluConstantPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "mini-const-fold"; }
  StringRef getDescription() const final {
    return "Fold relu(constant) into mini.constant";
  }
};

struct MiniDCEPass : public PassWrapper<MiniDCEPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "mini-dce"; }
  StringRef getDescription() const final {
    return "Mini dialect dead code elimination pass skeleton";
  }
};

struct MiniFusionPass
    : public PassWrapper<MiniFusionPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseLinearReluPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "mini-fusion"; }
  StringRef getDescription() const final {
    return "Fuse mini.linear followed by mini.relu";
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
