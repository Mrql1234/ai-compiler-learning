/**
 * 常量折叠 Pass
 * 
 * 功能：在编译时计算常量表达式
 * 
 * 优化示例：
 *   %0 = arith.constant 42 : i32
 *   %1 = arith.constant 0 : i32
 *   %2 = arith.addi %0, %1 : i32
 * 
 * 优化后：
 *   %0 = arith.constant 42 : i32
 */

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

/**
 * 常量折叠：arith.addi
 * 
 * 匹配：arith.addi(constant, constant)
 * 替换：constant
 */
struct ConstantFoldAddI : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter& rewriter) const override {
    // 检查两个操作数是否都是常量
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    
    if (!lhs || !rhs) {
      return failure();
    }

    // 获取常数值
    auto lhsAttr = lhs.getValue();
    auto rhsAttr = rhs.getValue();

    // 计算结果（仅支持整数）
    Attribute resultAttr;
    if (auto lhsInt = lhsAttr.dyn_cast<IntegerAttr>()) {
      if (auto rhsInt = rhsAttr.dyn_cast<IntegerAttr>()) {
        auto result = lhsInt.getValue() + rhsInt.getValue();
        resultAttr = IntegerAttr::get(lhsInt.getType(), result);
      }
    }

    if (!resultAttr) {
      return failure();
    }

    // 用常量替换原操作
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultAttr);
    return success();
  }
};

/**
 * 常量折叠：arith.muli
 */
struct ConstantFoldMulI : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    
    if (!lhs || !rhs) {
      return failure();
    }

    auto lhsAttr = lhs.getValue();
    auto rhsAttr = rhs.getValue();

    Attribute resultAttr;
    if (auto lhsInt = lhsAttr.dyn_cast<IntegerAttr>()) {
      if (auto rhsInt = rhsAttr.dyn_cast<IntegerAttr>()) {
        auto result = lhsInt.getValue() * rhsInt.getValue();
        resultAttr = IntegerAttr::get(lhsInt.getType(), result);
      }
    }

    if (!resultAttr) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultAttr);
    return success();
  }
};

/**
 * 常量折叠：arith.addf (浮点)
 */
struct ConstantFoldAddF : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    
    if (!lhs || !rhs) {
      return failure();
    }

    auto lhsAttr = lhs.getValue();
    auto rhsAttr = rhs.getValue();

    Attribute resultAttr;
    if (auto lhsFloat = lhsAttr.dyn_cast<FloatAttr>()) {
      if (auto rhsFloat = rhsAttr.dyn_cast<FloatAttr>()) {
        auto result = lhsFloat.getValue() + rhsFloat.getValue();
        resultAttr = FloatAttr::get(lhsFloat.getType(), result);
      }
    }

    if (!resultAttr) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultAttr);
    return success();
  }
};

/**
 * 常量折叠 Pass 定义
 */
struct ConstantFoldPass : public PassWrapper<ConstantFoldPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantFoldPass)

  StringRef getArgument() const override {
    return "constant-fold";
  }

  StringRef getDescription() const override {
    return "Fold constant expressions at compile time";
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    
    // 创建重写模式
    RewritePatternSet patterns(context);
    patterns.add<ConstantFoldAddI, ConstantFoldMulI, ConstantFoldAddF>(context);
    
    // 应用贪婪重写
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

/**
 * 创建 Pass 的工厂函数
 */
std::unique_ptr<Pass> createConstantFoldPass() {
  return std::make_unique<ConstantFoldPass>();
}

} // namespace mlir

/**
 * 注册 Pass 到 MLIR
 */
namespace mlir {
void registerConstantFoldPass() {
  PassRegistration<ConstantFoldPass>();
}
} // namespace mlir
