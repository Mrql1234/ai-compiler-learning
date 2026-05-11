#include "MiniCompiler/MiniDialect.h"
#include "MiniCompiler/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <algorithm>
#include <cmath>

using namespace mlir;

namespace mini {
namespace {

static FailureOr<RankedTensorType> getRankedTensorType(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return failure();
  return tensorType;
}

static SmallVector<Value> collectDynamicDims(Location loc, Value source,
                                             RankedTensorType resultType,
                                             OpBuilder &builder) {
  SmallVector<Value> dynamicDims;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (ShapedType::isDynamic(dim))
      dynamicDims.push_back(builder.create<tensor::DimOp>(loc, source, index));
  }
  return dynamicDims;
}

static FailureOr<Value> createEmptyTensorFor(Location loc, RankedTensorType type,
                                             Value shapeSource,
                                             OpBuilder &builder) {
  if (!type)
    return failure();
  auto dynamicDims = collectDynamicDims(loc, shapeSource, type, builder);
  return builder.create<tensor::EmptyOp>(loc, type, dynamicDims).getResult();
}

static FailureOr<Value> lowerReluValue(Location loc, Value input,
                                       RankedTensorType resultType,
                                       OpBuilder &builder) {
  auto outputInit = createEmptyTensorFor(loc, resultType, input, builder);
  if (failed(outputInit))
    return failure();

  auto identityMap =
      AffineMap::getMultiDimIdentityMap(resultType.getRank(), builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                 utils::IteratorType::parallel);

  auto reluOp = builder.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, ValueRange{input}, ValueRange{*outputInit},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        auto zero = nestedBuilder.create<arith::ConstantOp>(
            nestedLoc, nestedBuilder.getFloatAttr(args[0].getType(), 0.0));
        auto max =
            nestedBuilder.create<arith::MaximumFOp>(nestedLoc, args[0], zero);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, max.getResult());
      });
  return reluOp->getResult(0);
}

static FailureOr<Value> lowerQuantizedLinearValue(Location loc, Value input,
                                                  Value qweight, Value bias,
                                                  Value scale,
                                                  RankedTensorType resultType,
                                                  bool addRelu,
                                                  OpBuilder &builder) {
  auto inputType = getRankedTensorType(input);
  auto weightType = getRankedTensorType(qweight);
  auto biasType = getRankedTensorType(bias);
  if (failed(inputType) || failed(weightType) || failed(biasType))
    return failure();
  if (inputType->getRank() != 2 || weightType->getRank() != 2 ||
      biasType->getRank() != 1 || resultType.getRank() != 2)
    return failure();
  if (!inputType->getElementType().isF32() || !biasType->getElementType().isF32() ||
      !resultType.getElementType().isF32() ||
      !weightType->getElementType().isSignlessInteger(8))
    return failure();

  auto biasInit = createEmptyTensorFor(loc, resultType, input, builder);
  if (failed(biasInit))
    return failure();

  auto d0 = builder.getAffineDimExpr(0);
  auto d1 = builder.getAffineDimExpr(1);
  auto d2 = builder.getAffineDimExpr(2);

  SmallVector<AffineMap> biasIndexingMaps{
      AffineMap::get(2, 0, {d1}, builder.getContext()),
      AffineMap::get(2, 0, {d0, d1}, builder.getContext())};
  SmallVector<utils::IteratorType> biasIteratorTypes{
      utils::IteratorType::parallel, utils::IteratorType::parallel};

  auto initialized = builder.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, ValueRange{bias}, ValueRange{*biasInit},
      biasIndexingMaps, biasIteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  SmallVector<AffineMap> indexingMaps{
      AffineMap::get(3, 0, {d0, d2}, builder.getContext()),
      AffineMap::get(3, 0, {d1, d2}, builder.getContext()),
      AffineMap::get(3, 0, {d0, d1}, builder.getContext())};
  SmallVector<utils::IteratorType> iteratorTypes{
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};

  auto qlinear = builder.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, ValueRange{input, qweight},
      ValueRange{initialized->getResult(0)}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        auto weightAsFloat = nestedBuilder.create<arith::SIToFPOp>(
            nestedLoc, nestedBuilder.getF32Type(), args[1]);
        auto scaledWeight =
            nestedBuilder.create<arith::MulFOp>(nestedLoc, weightAsFloat, scale);
        auto product =
            nestedBuilder.create<arith::MulFOp>(nestedLoc, args[0], scaledWeight);
        auto sum =
            nestedBuilder.create<arith::AddFOp>(nestedLoc, args[2], product);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum.getResult());
      });

  if (!addRelu)
    return qlinear->getResult(0);
  return lowerReluValue(loc, qlinear->getResult(0), resultType, builder);
}

static bool shouldQuantizeWeightValue(Value weight) {
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  return weightType && weightType.getElementType().isF32();
}

struct QuantizedWeightData {
  RankedTensorType type;
  DenseElementsAttr attr;
  float scale;
};

static FailureOr<QuantizedWeightData>
quantizeWeightConstant(Value weight) {
  if (!shouldQuantizeWeightValue(weight))
    return failure();

  auto weightConstant = weight.getDefiningOp<ConstantOp>();
  if (!weightConstant)
    return failure();

  auto denseAttr = dyn_cast_or_null<DenseFPElementsAttr>(weightConstant.getValue());
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  if (!denseAttr || !weightType)
    return failure();

  SmallVector<int8_t> quantizedValues;
  quantizedValues.reserve(denseAttr.getNumElements());
  float absMax = 0.0f;
  for (const APFloat &element : denseAttr.template getValues<APFloat>())
    absMax = std::max(absMax, std::abs(element.convertToFloat()));
  float scale = std::max(absMax / 127.0f, 1.0e-8f);

  for (const APFloat &element : denseAttr.template getValues<APFloat>()) {
    float value = element.convertToFloat();
    float scaled = std::round(value / scale);
    scaled = std::clamp(scaled, -127.0f, 127.0f);
    quantizedValues.push_back(static_cast<int8_t>(scaled));
  }

  Builder builder(weight.getContext());
  auto quantizedType =
      RankedTensorType::get(weightType.getShape(), builder.getIntegerType(8));
  auto quantizedAttr =
      DenseElementsAttr::get<int8_t>(quantizedType, ArrayRef<int8_t>(quantizedValues));
  return QuantizedWeightData{quantizedType, quantizedAttr, scale};
}

static LogicalResult quantizeLinearLikeWeight(LinearOp op) {
  if (op.getWeightScaleAttr() || !shouldQuantizeWeightValue(op.getWeight()))
    return success();

  auto quantizedWeight = quantizeWeightConstant(op.getWeight());
  if (failed(quantizedWeight))
    return success();

  auto weightConstant = op.getWeight().getDefiningOp<ConstantOp>();
  auto [quantizedType, quantizedAttr, scale] = *quantizedWeight;
  OpBuilder opBuilder(op);
  Builder attrBuilder(op.getContext());

  auto newWeight =
      opBuilder.create<ConstantOp>(weightConstant.getLoc(), quantizedAttr, quantizedType);
  auto qlinear = opBuilder.create<QLinearOp>(
      op.getLoc(), op.getOutput().getType(), op.getInput(), newWeight.getOutput(),
      op.getBias(), attrBuilder.getF32FloatAttr(scale));
  op.replaceAllUsesWith(qlinear.getOutput());
  op.erase();
  if (weightConstant->use_empty())
    weightConstant.erase();
  return success();
}

static LogicalResult quantizeLinearLikeWeight(FusedLinearReluOp op) {
  if (op.getWeightScaleAttr() || !shouldQuantizeWeightValue(op.getWeight()))
    return success();

  auto quantizedWeight = quantizeWeightConstant(op.getWeight());
  if (failed(quantizedWeight))
    return success();

  auto weightConstant = op.getWeight().getDefiningOp<ConstantOp>();
  auto [quantizedType, quantizedAttr, scale] = *quantizedWeight;
  OpBuilder opBuilder(op);
  Builder attrBuilder(op.getContext());

  auto newWeight =
      opBuilder.create<ConstantOp>(weightConstant.getLoc(), quantizedAttr, quantizedType);
  auto qlinear = opBuilder.create<QLinearReluOp>(
      op.getLoc(), op.getOutput().getType(), op.getInput(), newWeight.getOutput(),
      op.getBias(), attrBuilder.getF32FloatAttr(scale));
  op.replaceAllUsesWith(qlinear.getOutput());
  op.erase();
  if (weightConstant->use_empty())
    weightConstant.erase();
  return success();
}

struct LowerQLinearPattern : public OpRewritePattern<QLinearOp> {
  using OpRewritePattern<QLinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(QLinearOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType)
      return failure();
    auto scale = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(op.getWeightScaleAttr().getValueAsDouble()));
    auto lowered = lowerQuantizedLinearValue(op.getLoc(), op.getInput(), op.getWeight(),
                                             op.getBias(), scale, resultType,
                                             /*addRelu=*/false, rewriter);
    if (failed(lowered))
      return failure();
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

struct LowerQLinearReluPattern : public OpRewritePattern<QLinearReluOp> {
  using OpRewritePattern<QLinearReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(QLinearReluOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType)
      return failure();
    auto scale = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getF32FloatAttr(op.getWeightScaleAttr().getValueAsDouble()));
    auto lowered = lowerQuantizedLinearValue(op.getLoc(), op.getInput(), op.getWeight(),
                                             op.getBias(), scale, resultType,
                                             /*addRelu=*/true, rewriter);
    if (failed(lowered))
      return failure();
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

struct MiniQuantizeWeightsPass
    : public PassWrapper<MiniQuantizeWeightsPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MiniDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> worklist;
    getOperation()->walk([&](Operation *operation) {
      if (isa<LinearOp, FusedLinearReluOp>(operation))
        worklist.push_back(operation);
    });
    for (Operation *operation : worklist) {
      if (!operation->getBlock())
        continue;
      LogicalResult status = success();
      if (auto linear = dyn_cast<LinearOp>(operation))
        status = quantizeLinearLikeWeight(linear);
      else if (auto fused = dyn_cast<FusedLinearReluOp>(operation))
        status = quantizeLinearLikeWeight(fused);
      if (failed(status)) {
        signalPassFailure();
        return;
      }
    }
  }
  StringRef getArgument() const final { return "mini-quantize-weights"; }
  StringRef getDescription() const final {
    return "Convert constant linear weights to symmetric int8 plus scale";
  }
};

struct MiniLowerQLinearToLinalgPass
    : public PassWrapper<MiniLowerQLinearToLinalgPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, MiniDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerQLinearPattern, LowerQLinearReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "mini-lower-qlinear-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower mini.qlinear ops into dequant-plus-linalg form";
  }
};

struct MiniLowerQuantToCpuRuntimePass
    : public PassWrapper<MiniLowerQuantToCpuRuntimePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    MiniDialect, tensor::TensorDialect>();
  }

  func::FuncOp createRuntimeHelper(ModuleOp module, Location loc, StringRef baseName,
                                   FunctionType type, bool addRelu) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    std::string name = (baseName + std::to_string(nextHelperId++)).str();
    auto helper = builder.create<func::FuncOp>(loc, name, type);
    helper.setSymVisibilityAttr(builder.getStringAttr("private"));
    Block *entry = helper.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);

    auto resultType = dyn_cast<RankedTensorType>(type.getResult(0));
    if (!resultType)
      return {};

    auto lowered = lowerQuantizedLinearValue(loc, entry->getArgument(0),
                                             entry->getArgument(1),
                                             entry->getArgument(2),
                                             entry->getArgument(3), resultType,
                                             addRelu, bodyBuilder);
    if (failed(lowered))
      return {};
    bodyBuilder.create<func::ReturnOp>(loc, ValueRange{*lowered});
    return helper;
  }

  LogicalResult rewriteQuantizedLinear(ModuleOp module, Operation *operation,
                                       bool addRelu) {
    auto resultType =
        dyn_cast<RankedTensorType>(operation->getResult(0).getType());
    if (!resultType)
      return failure();

    Value input = operation->getOperand(0);
    Value weight = operation->getOperand(1);
    Value bias = operation->getOperand(2);
    auto weightType = dyn_cast<RankedTensorType>(weight.getType());
    if (!weightType || !weightType.getElementType().isSignlessInteger(8))
      return success();

    auto weightScaleAttr =
        operation->getAttrOfType<FloatAttr>(LinearOp::getWeightScaleAttrName());
    if (!weightScaleAttr)
      return failure();

    OpBuilder builder(operation);
    auto scaleValue = builder.create<arith::ConstantOp>(
        operation->getLoc(),
        builder.getF32FloatAttr(weightScaleAttr.getValueAsDouble()));

    auto functionType = builder.getFunctionType(
        TypeRange{input.getType(), weight.getType(), bias.getType(), scaleValue.getType()},
        TypeRange{resultType});
    auto helper = createRuntimeHelper(
        module, operation->getLoc(),
        addRelu ? "__mini_cpu_qlinear_relu_runtime_"
                : "__mini_cpu_qlinear_runtime_",
        functionType, addRelu);
    if (!helper)
      return failure();

    auto call = builder.create<func::CallOp>(
        operation->getLoc(), helper.getName(), TypeRange{resultType},
        ValueRange{input, weight, bias, scaleValue});
    operation->getResult(0).replaceAllUsesWith(call.getResult(0));
    operation->erase();
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *operation) {
      if (isa<QLinearOp, QLinearReluOp>(operation))
        worklist.push_back(operation);
    });

    for (Operation *operation : worklist) {
      if (!operation->getBlock())
        continue;
      LogicalResult status = success();
      if (isa<QLinearOp>(operation))
        status = rewriteQuantizedLinear(module, operation, /*addRelu=*/false);
      else if (isa<QLinearReluOp>(operation))
        status = rewriteQuantizedLinear(module, operation, /*addRelu=*/true);
      if (failed(status)) {
        signalPassFailure();
        return;
      }
    }
  }

  StringRef getArgument() const final { return "mini-lower-quant-to-cpu-runtime"; }
  StringRef getDescription() const final {
    return "Lower quantized mini linear ops into runtime helper calls";
  }

private:
  int nextHelperId = 0;
};

} // namespace

std::unique_ptr<Pass> createMiniQuantizeWeightsPass() {
  return std::make_unique<MiniQuantizeWeightsPass>();
}

std::unique_ptr<Pass> createMiniLowerQuantToCpuRuntimePass() {
  return std::make_unique<MiniLowerQuantToCpuRuntimePass>();
}

std::unique_ptr<Pass> createMiniLowerQLinearToLinalgPass() {
  return std::make_unique<MiniLowerQLinearToLinalgPass>();
}

void registerMiniQuantPasses() {
  PassRegistration<MiniQuantizeWeightsPass>();
  PassRegistration<MiniLowerQuantToCpuRuntimePass>();
  PassRegistration<MiniLowerQLinearToLinalgPass>();
}

void registerMiniQuantPassPipelines() {
  PassPipelineRegistration<>(
      "mini-quantized-cpu-lowering",
      "Quantize supported constant linear weights and lower them to CPU runtime "
      "helper calls before the existing CPU pipeline",
      [](OpPassManager &pm) {
        const char *pipelineText =
            "func.func(mini-canonicalize,mini-fusion,mini-quantize-weights),"
            "mini-lower-quant-to-cpu-runtime,"
            "func.func(mini-lower-to-linalg),"
            "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
            "drop-equivalent-buffer-results,"
            "buffer-results-to-out-params,"
            "convert-bufferization-to-memref,"
            "convert-linalg-to-loops,"
            "convert-scf-to-cf,"
            "convert-cf-to-llvm,"
            "convert-arith-to-llvm,"
            "convert-index-to-llvm,"
            "expand-realloc,"
            "finalize-memref-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts";
        if (failed(parsePassPipeline(pipelineText, pm)))
          llvm::report_fatal_error(
              "failed to parse mini-quantized-cpu-lowering pipeline");
      });

  PassPipelineRegistration<>(
      "mini-quantized-gpu-prep",
      "Quantize supported constant linear weights, keep mini.qlinear late, then "
      "lower into GPU-kernel dequant-plus-linalg form",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createMiniCanonicalizePass());
        pm.addNestedPass<func::FuncOp>(createMiniFusionPass());
        pm.addNestedPass<func::FuncOp>(createMiniQuantizeWeightsPass());
        pm.addNestedPass<func::FuncOp>(createMiniLowerToLinalgPass());
        pm.addNestedPass<func::FuncOp>(createMiniLowerQLinearToLinalgPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });

  PassPipelineRegistration<>(
      "mini-quantized-gpu-lowering",
      "Lower quantized mini linear ops into GPU kernel IR with dequantization "
      "kept inside the generated kernel path",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createMiniCanonicalizePass());
        pm.addNestedPass<func::FuncOp>(createMiniFusionPass());
        pm.addNestedPass<func::FuncOp>(createMiniQuantizeWeightsPass());
        pm.addNestedPass<func::FuncOp>(createMiniLowerToLinalgPass());
        pm.addNestedPass<func::FuncOp>(createMiniLowerQLinearToLinalgPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        if (failed(parsePassPipeline(
                "one-shot-bufferize{bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map},"
                "drop-equivalent-buffer-results,"
                "buffer-results-to-out-params,"
                "convert-bufferization-to-memref,"
                "canonicalize,cse",
                pm)))
          llvm::report_fatal_error(
              "failed to parse shared quantized GPU bufferization pipeline");
        pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
        pm.addNestedPass<func::FuncOp>(createMiniGpuTilePass());
        pm.addNestedPass<func::FuncOp>(createMiniGpuMapPass());
        pm.addNestedPass<func::FuncOp>(createConvertParallelLoopToGpuPass());
        pm.addPass(createGpuKernelOutliningPass());
        pm.addNestedPass<func::FuncOp>(createMiniGpuHostSharedPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
}

} // namespace mini
