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

  mlir::Value getInput() { return getOperand(0); }
  mlir::Value getWeight() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
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

  mlir::Value getInput() { return getOperand(0); }
  mlir::Value getWeight() { return getOperand(1); }
  mlir::Value getBias() { return getOperand(2); }
  mlir::Value getOutput() { return getOperation()->getResult(0); }
};

} // namespace mini

#endif // MINI_COMPILER_MINI_DIALECT_H
