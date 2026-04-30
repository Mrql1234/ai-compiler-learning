#include "Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

struct ConstantFoldAddI : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lhs || !rhs)
      return failure();

    auto lhsAttr = lhs.getValue();
    auto rhsAttr = rhs.getValue();

    Attribute resultAttr;
    if (auto lhsInt = lhsAttr.dyn_cast<IntegerAttr>()) {
      if (auto rhsInt = rhsAttr.dyn_cast<IntegerAttr>()) {
        auto result = lhsInt.getValue() + rhsInt.getValue();
        resultAttr = IntegerAttr::get(lhsInt.getType(), result);
      }
    }

    if (!resultAttr)
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultAttr);
    return success();
  }
};

struct ConstantFoldMulI : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lhs || !rhs)
      return failure();

    auto lhsAttr = lhs.getValue();
    auto rhsAttr = rhs.getValue();

    Attribute resultAttr;
    if (auto lhsInt = lhsAttr.dyn_cast<IntegerAttr>()) {
      if (auto rhsInt = rhsAttr.dyn_cast<IntegerAttr>()) {
        auto result = lhsInt.getValue() * rhsInt.getValue();
        resultAttr = IntegerAttr::get(lhsInt.getType(), result);
      }
    }

    if (!resultAttr)
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultAttr);
    return success();
  }
};

struct ConstantFoldAddF : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lhs || !rhs)
      return failure();

    auto lhsAttr = lhs.getValue();
    auto rhsAttr = rhs.getValue();

    Attribute resultAttr;
    if (auto lhsFloat = lhsAttr.dyn_cast<FloatAttr>()) {
      if (auto rhsFloat = rhsAttr.dyn_cast<FloatAttr>()) {
        auto result = lhsFloat.getValue() + rhsFloat.getValue();
        resultAttr = FloatAttr::get(lhsFloat.getType(), result);
      }
    }

    if (!resultAttr)
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultAttr);
    return success();
  }
};

struct ConstantFoldPass
    : public PassWrapper<ConstantFoldPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldPass)

  StringRef getArgument() const override { return "constant-fold"; }

  StringRef getDescription() const override {
    return "Fold addi, muli, and addf constant expressions with explicit patterns";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantFoldAddI, ConstantFoldMulI, ConstantFoldAddF>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConstantFoldPass() {
  return std::make_unique<ConstantFoldPass>();
}

void registerConstantFoldPass() { PassRegistration<ConstantFoldPass>(); }

} // namespace mlir
