#include "MiniCompiler/MiniDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace mini {

MiniDialect::MiniDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniDialect>()) {
  addOperations<ConstantOp, LinearOp, QLinearOp, MatmulOp, AddOp, ReluOp,
                FusedLinearReluOp, QLinearReluOp, FusedMatmulAddReluOp>();
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

LogicalResult LinearOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto resultType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !weightType || !biasType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.getElementType().isF32() || !biasType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return emitOpError("currently expects f32 input, bias, and result tensors");
  auto weightElementType = weightType.getElementType();
  if (weightElementType.isF32()) {
    if (getWeightScaleAttr())
      return emitOpError("must not set 'weight_scale' when weight is already f32");
    return success();
  }
  if (!weightElementType.isSignlessInteger(8))
    return emitOpError("currently supports only f32 or i8 weights");
  if (!getWeightScaleAttr())
    return emitOpError("requires 'weight_scale' when weight tensor element type is i8");
  return success();
}

ArrayRef<StringRef> QLinearOp::getAttributeNames() {
  return ArrayRef<StringRef>();
}

void QLinearOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                      Value input, Value qweight, Value bias,
                      FloatAttr weightScale) {
  (void)builder;
  state.addOperands({input, qweight, bias});
  state.addTypes(resultType);
  state.addAttribute(getWeightScaleAttrName(), weightScale);
}

LogicalResult QLinearOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto resultType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !weightType || !biasType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.getElementType().isF32() || !biasType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return emitOpError("currently expects f32 input, bias, and result tensors");
  if (!weightType.getElementType().isSignlessInteger(8))
    return emitOpError("requires i8 weight tensor");
  if (!getWeightScaleAttr())
    return emitOpError("requires 'weight_scale' attribute");
  return success();
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

LogicalResult FusedLinearReluOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto resultType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !weightType || !biasType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.getElementType().isF32() || !biasType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return emitOpError("currently expects f32 input, bias, and result tensors");
  auto weightElementType = weightType.getElementType();
  if (weightElementType.isF32()) {
    if (getWeightScaleAttr())
      return emitOpError("must not set 'weight_scale' when weight is already f32");
    return success();
  }
  if (!weightElementType.isSignlessInteger(8))
    return emitOpError("currently supports only f32 or i8 weights");
  if (!getWeightScaleAttr())
    return emitOpError("requires 'weight_scale' when weight tensor element type is i8");
  return success();
}

ArrayRef<StringRef> QLinearReluOp::getAttributeNames() {
  return ArrayRef<StringRef>();
}

void QLinearReluOp::build(OpBuilder &builder, OperationState &state,
                          Type resultType, Value input, Value qweight,
                          Value bias, FloatAttr weightScale) {
  (void)builder;
  state.addOperands({input, qweight, bias});
  state.addTypes(resultType);
  state.addAttribute(getWeightScaleAttrName(), weightScale);
}

LogicalResult QLinearReluOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto resultType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !weightType || !biasType || !resultType)
    return emitOpError("requires ranked tensor operands and result");
  if (!inputType.getElementType().isF32() || !biasType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return emitOpError("currently expects f32 input, bias, and result tensors");
  if (!weightType.getElementType().isSignlessInteger(8))
    return emitOpError("requires i8 weight tensor");
  if (!getWeightScaleAttr())
    return emitOpError("requires 'weight_scale' attribute");
  return success();
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
