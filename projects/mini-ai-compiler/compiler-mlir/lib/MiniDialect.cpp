#include "MiniCompiler/MiniDialect.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace mini {

MiniDialect::MiniDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniDialect>()) {
  addOperations<ConstantOp, LinearOp, ReluOp, FusedLinearReluOp>();
}

ArrayRef<StringRef> ConstantOp::getAttributeNames() {
  static const StringRef names[] = {"value"};
  return names;
}

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       Attribute value, Type resultType) {
  (void)builder;
  state.addAttribute("value", value);
  state.addTypes(resultType);
}

LogicalResult ConstantOp::verify() {
  Attribute valueAttr = getValue();
  if (!valueAttr)
    return emitOpError("requires a typed 'value' attribute");
  if (valueAttr.getType() != getOutput().getType())
    return emitOpError("requires 'value' attribute type to match result type");
  return success();
}

ArrayRef<StringRef> LinearOp::getAttributeNames() { return ArrayRef<StringRef>(); }

void LinearOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                     Value input, Value weight, Value bias) {
  (void)builder;
  state.addOperands({input, weight, bias});
  state.addTypes(resultType);
}

ArrayRef<StringRef> ReluOp::getAttributeNames() { return ArrayRef<StringRef>(); }

void ReluOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                   Value input) {
  (void)builder;
  state.addOperands(input);
  state.addTypes(resultType);
}

ArrayRef<StringRef> FusedLinearReluOp::getAttributeNames() { return ArrayRef<StringRef>(); }

void FusedLinearReluOp::build(OpBuilder &builder, OperationState &state,
                              Type resultType, Value input, Value weight,
                              Value bias) {
  (void)builder;
  state.addOperands({input, weight, bias});
  state.addTypes(resultType);
}

} // namespace mini
