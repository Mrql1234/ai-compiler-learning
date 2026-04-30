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

struct AddIntegerFolder {
  static Attribute fold(IntegerAttr lhs, IntegerAttr rhs) {
    return IntegerAttr::get(lhs.getType(), lhs.getValue() + rhs.getValue());
  }
};

struct SubIntegerFolder {
  static Attribute fold(IntegerAttr lhs, IntegerAttr rhs) {
    return IntegerAttr::get(lhs.getType(), lhs.getValue() - rhs.getValue());
  }
};

struct MulIntegerFolder {
  static Attribute fold(IntegerAttr lhs, IntegerAttr rhs) {
    return IntegerAttr::get(lhs.getType(), lhs.getValue() * rhs.getValue());
  }
};

struct AddFloatFolder {
  static Attribute fold(FloatAttr lhs, FloatAttr rhs) {
    llvm::APFloat result = lhs.getValue();
    result.add(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
    return FloatAttr::get(lhs.getType(), result);
  }
};

struct SubFloatFolder {
  static Attribute fold(FloatAttr lhs, FloatAttr rhs) {
    llvm::APFloat result = lhs.getValue();
    result.subtract(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
    return FloatAttr::get(lhs.getType(), result);
  }
};

struct MulFloatFolder {
  static Attribute fold(FloatAttr lhs, FloatAttr rhs) {
    llvm::APFloat result = lhs.getValue();
    result.multiply(rhs.getValue(), llvm::APFloat::rmNearestTiesToEven);
    return FloatAttr::get(lhs.getType(), result);
  }
};

template <typename OpT, typename AttrT, typename FolderT>
struct BinaryConstantFoldPattern : OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    auto operands = getConstantOperands<AttrT>(op.getLhs(), op.getRhs());
    if (failed(operands))
      return failure();

    auto [lhs, rhs] = *operands;
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op,
                                                   FolderT::fold(lhs, rhs));
    return success();
  }
};

struct TemplateConstantFoldPass
    : public PassWrapper<TemplateConstantFoldPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TemplateConstantFoldPass)

  StringRef getArgument() const override { return "constant-fold-template"; }

  StringRef getDescription() const override {
    return "Fold arithmetic constants using template-based reusable rewrite patterns";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<
        BinaryConstantFoldPattern<arith::AddIOp, IntegerAttr, AddIntegerFolder>,
        BinaryConstantFoldPattern<arith::SubIOp, IntegerAttr, SubIntegerFolder>,
        BinaryConstantFoldPattern<arith::MulIOp, IntegerAttr, MulIntegerFolder>,
        BinaryConstantFoldPattern<arith::AddFOp, FloatAttr, AddFloatFolder>,
        BinaryConstantFoldPattern<arith::SubFOp, FloatAttr, SubFloatFolder>,
        BinaryConstantFoldPattern<arith::MulFOp, FloatAttr, MulFloatFolder>>(
        &getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTemplateConstantFoldPass() {
  return std::make_unique<TemplateConstantFoldPass>();
}

void registerTemplateConstantFoldPass() {
  PassRegistration<TemplateConstantFoldPass>();
}

} // namespace mlir
