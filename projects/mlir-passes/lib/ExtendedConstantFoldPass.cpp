#include "Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

template <typename AttrT>
static FailureOr<std::pair<AttrT, AttrT>>
getConstantOperands(Value lhs, Value rhs) {
  auto lhsConst = lhs.getDefiningOp<arith::ConstantOp>();
  auto rhsConst = rhs.getDefiningOp<arith::ConstantOp>();
  if (!lhsConst || !rhsConst)
    return failure();

  auto lhsAttr = lhsConst.getValue().dyn_cast<AttrT>();
  auto rhsAttr = rhsConst.getValue().dyn_cast<AttrT>();
  if (!lhsAttr || !rhsAttr)
    return failure();

  return std::make_pair(lhsAttr, rhsAttr);
}

template <typename OpT>
struct ManualFoldAddI : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<IntegerAttr>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();
    auto [lhs, rhs] = *operands;
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, IntegerAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue()));
    return success();
  }
};

template <typename OpT>
struct ManualFoldSubI : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<IntegerAttr>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();
    auto [lhs, rhs] = *operands;
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, IntegerAttr::get(lhs.getType(), lhs.getValue() - rhs.getValue()));
    return success();
  }
};

template <typename OpT>
struct ManualFoldMulI : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<IntegerAttr>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();
    auto [lhs, rhs] = *operands;
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, IntegerAttr::get(lhs.getType(), lhs.getValue() * rhs.getValue()));
    return success();
  }
};

template <typename OpT>
struct ManualFoldAddF : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<FloatAttr>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();
    auto [lhs, rhs] = *operands;
    llvm::APFloat result = lhs.getValue();
    result.add(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                   FloatAttr::get(lhs.getType(), result));
    return success();
  }
};

template <typename OpT>
struct ManualFoldSubF : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<FloatAttr>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();
    auto [lhs, rhs] = *operands;
    llvm::APFloat result = lhs.getValue();
    result.subtract(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                   FloatAttr::get(lhs.getType(), result));
    return success();
  }
};

template <typename OpT>
struct ManualFoldMulF : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<FloatAttr>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();
    auto [lhs, rhs] = *operands;
    llvm::APFloat result = lhs.getValue();
    result.multiply(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                   FloatAttr::get(lhs.getType(), result));
    return success();
  }
};

struct ExtendedConstantFoldPass
    : public PassWrapper<ExtendedConstantFoldPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtendedConstantFoldPass)

  StringRef getArgument() const override { return "constant-fold-extended"; }
  StringRef getDescription() const override {
    return "Extended handwritten constant folding for add/sub/mul on ints and floats";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ManualFoldAddI<arith::AddIOp>, ManualFoldSubI<arith::SubIOp>,
                 ManualFoldMulI<arith::MulIOp>, ManualFoldAddF<arith::AddFOp>,
                 ManualFoldSubF<arith::SubFOp>, ManualFoldMulF<arith::MulFOp>>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createExtendedConstantFoldPass() {
  return std::make_unique<ExtendedConstantFoldPass>();
}

void registerExtendedConstantFoldPass() {
  PassRegistration<ExtendedConstantFoldPass>();
}

} // namespace mlir
