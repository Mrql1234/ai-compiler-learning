#include "MiniCompiler/MiniDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace mini {

MiniDialect::MiniDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniDialect>()) {
  addOperations<ConstantOp, LinearOp, MatmulOp, AddOp, ReluOp,
                FusedLinearReluOp, FusedMatmulAddReluOp>();
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
  auto elementsAttr = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!elementsAttr)
    return emitOpError("requires a dense typed 'value' attribute");
  if (elementsAttr.getType() != getOutput().getType())
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

ArrayRef<StringRef> MatmulOp::getAttributeNames() {
  return ArrayRef<StringRef>();
}

void MatmulOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                     Value lhs, Value rhs) {
  (void)builder;
  state.addOperands({lhs, rhs});
  state.addTypes(resultType);
}

ArrayRef<StringRef> AddOp::getAttributeNames() { return ArrayRef<StringRef>(); }

void AddOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                  Value lhs, Value rhs) {
  (void)builder;
  state.addOperands({lhs, rhs});
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

ArrayRef<StringRef> FusedMatmulAddReluOp::getAttributeNames() {
  return ArrayRef<StringRef>();
}

void FusedMatmulAddReluOp::build(OpBuilder &builder, OperationState &state,
                                 Type resultType, Value lhs, Value rhs,
                                 Value bias) {
  (void)builder;
  state.addOperands({lhs, rhs, bias});
  state.addTypes(resultType);
}

} // namespace mini
