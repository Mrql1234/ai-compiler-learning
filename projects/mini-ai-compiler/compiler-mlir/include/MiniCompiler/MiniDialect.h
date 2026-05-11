#ifndef MINI_COMPILER_MINI_DIALECT_H
#define MINI_COMPILER_MINI_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mini {

class MiniDialect : public mlir::Dialect {
public:
  explicit MiniDialect(mlir::MLIRContext *context);

  static llvm::StringRef getDialectNamespace() { return "mini"; }
};

class ConstantOp : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
                                   mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.constant"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Attribute value, mlir::Type resultType);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();

  mlir::Attribute getValue() { return (*this)->getAttr("value"); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }

  mlir::LogicalResult verify();
};

class LinearOp : public mlir::Op<LinearOp, mlir::OpTrait::NOperands<3>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.linear"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value input,
                    mlir::Value weight, mlir::Value bias);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
  static llvm::StringRef getWeightScaleAttrName() { return "weight_scale"; }

  mlir::Value getInput() { return getOperand(0); }
  mlir::Value getWeight() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
  mlir::FloatAttr getWeightScaleAttr() {
    return (*this)->getAttrOfType<mlir::FloatAttr>(getWeightScaleAttrName());
  }

  mlir::LogicalResult verify();
};

class QLinearOp : public mlir::Op<QLinearOp, mlir::OpTrait::NOperands<3>::Impl,
                                  mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.qlinear"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value input,
                    mlir::Value qweight, mlir::Value bias,
                    mlir::FloatAttr weightScale);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
  static llvm::StringRef getWeightScaleAttrName() { return "weight_scale"; }

  mlir::Value getInput() { return getOperand(0); }
  mlir::Value getWeight() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
  mlir::FloatAttr getWeightScaleAttr() {
    return (*this)->template getAttrOfType<mlir::FloatAttr>(
        getWeightScaleAttrName());
  }

  mlir::LogicalResult verify();
};

class MatmulOp : public mlir::Op<MatmulOp, mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.matmul"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();

  mlir::Value getLhs() { return getOperand(0); }
  mlir::Value getRhs() { return getOperand(1); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
};

class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.add"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();

  mlir::Value getLhs() { return getOperand(0); }
  mlir::Value getRhs() { return getOperand(1); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
};

class ReluOp : public mlir::Op<ReluOp, mlir::OpTrait::OneOperand,
                               mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.relu"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value input);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();

  mlir::Value getInput() { return getOperand(); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
};

class FusedLinearReluOp
    : public mlir::Op<FusedLinearReluOp, mlir::OpTrait::NOperands<3>::Impl,
                      mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.fused_linear_relu"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value input,
                    mlir::Value weight, mlir::Value bias);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
  static llvm::StringRef getWeightScaleAttrName() { return "weight_scale"; }

  mlir::Value getInput() { return getOperand(0); }
  mlir::Value getWeight() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
  mlir::FloatAttr getWeightScaleAttr() {
    return (*this)->getAttrOfType<mlir::FloatAttr>(getWeightScaleAttrName());
  }

  mlir::LogicalResult verify();
};

class QLinearReluOp
    : public mlir::Op<QLinearReluOp, mlir::OpTrait::NOperands<3>::Impl,
                      mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mini.qlinear_relu"; }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value input,
                    mlir::Value qweight, mlir::Value bias,
                    mlir::FloatAttr weightScale);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
  static llvm::StringRef getWeightScaleAttrName() { return "weight_scale"; }

  mlir::Value getInput() { return getOperand(0); }
  mlir::Value getWeight() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
  mlir::FloatAttr getWeightScaleAttr() {
    return (*this)->template getAttrOfType<mlir::FloatAttr>(
        getWeightScaleAttrName());
  }

  mlir::LogicalResult verify();
};

class FusedMatmulAddReluOp
    : public mlir::Op<FusedMatmulAddReluOp, mlir::OpTrait::NOperands<3>::Impl,
                      mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() {
    return "mini.fused_matmul_add_relu";
  }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value lhs, mlir::Value rhs,
                    mlir::Value bias);

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames();

  mlir::Value getLhs() { return getOperand(0); }
  mlir::Value getRhs() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
};

} // namespace mini

#endif // MINI_COMPILER_MINI_DIALECT_H
