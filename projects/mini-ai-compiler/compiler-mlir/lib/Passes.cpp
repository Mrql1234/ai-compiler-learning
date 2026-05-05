#include "MiniCompiler/Passes.h"
#include "MiniCompiler/MiniDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>
#include <cctype>

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
                                             PatternRewriter &rewriter) {
  SmallVector<Value> dynamicDims;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (ShapedType::isDynamic(dim))
      dynamicDims.push_back(rewriter.create<tensor::DimOp>(loc, source, index));
  }
  return dynamicDims;
}

static FailureOr<Value> createEmptyTensorFor(Location loc, RankedTensorType type,
                                             Value shapeSource,
                                             PatternRewriter &rewriter) {
  if (!type)
    return failure();
  auto dynamicDims = collectDynamicDims(loc, shapeSource, type, rewriter);
  return rewriter.create<tensor::EmptyOp>(loc, type, dynamicDims).getResult();
}

static FailureOr<Value> lowerReluValue(Location loc, Value input,
                                       RankedTensorType resultType,
                                       PatternRewriter &rewriter) {
  auto outputInit = createEmptyTensorFor(loc, resultType, input, rewriter);
  if (failed(outputInit))
    return failure();

  auto identityMap =
      AffineMap::getMultiDimIdentityMap(resultType.getRank(), rewriter.getContext());
  SmallVector<AffineMap> indexingMaps{
      identityMap, identityMap};
  SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                 utils::IteratorType::parallel);

  auto reluOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, ValueRange{input}, ValueRange{*outputInit},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        auto zero = builder.create<arith::ConstantOp>(
            nestedLoc, builder.getFloatAttr(args[0].getType(), 0.0));
        auto max = builder.create<arith::MaximumFOp>(nestedLoc, args[0], zero);
        builder.create<linalg::YieldOp>(nestedLoc, max.getResult());
      });
  return reluOp->getResult(0);
}

static FailureOr<Value> lowerLinearValue(Location loc, Value input, Value weight,
                                         Value bias, RankedTensorType resultType,
                                         PatternRewriter &rewriter) {
  auto inputType = getRankedTensorType(input);
  auto weightType = getRankedTensorType(weight);
  auto biasType = getRankedTensorType(bias);
  if (failed(inputType) || failed(weightType) || failed(biasType))
    return failure();
  if (inputType->getRank() != 2 || weightType->getRank() != 2 ||
      biasType->getRank() != 1 || resultType.getRank() != 2)
    return failure();
  if (!resultType.getElementType().isF32())
    return failure();

  auto zeroTensor = createEmptyTensorFor(loc, resultType, input, rewriter);
  if (failed(zeroTensor))
    return failure();

  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(resultType.getElementType(), 0.0));
  auto fill = rewriter.create<linalg::FillOp>(
      loc, TypeRange{resultType}, ValueRange{zero}, ValueRange{*zeroTensor});

  auto matmul = rewriter.create<linalg::MatmulTransposeBOp>(
      loc, TypeRange{resultType}, ValueRange{input, weight},
      ValueRange{fill->getResult(0)});

  auto biasInit = createEmptyTensorFor(loc, resultType, input, rewriter);
  if (failed(biasInit))
    return failure();

  auto d0 = rewriter.getAffineDimExpr(0);
  auto d1 = rewriter.getAffineDimExpr(1);
  SmallVector<AffineMap> indexingMaps{
      AffineMap::get(2, 0, {d0, d1}, rewriter.getContext()),
      AffineMap::get(2, 0, {d1}, rewriter.getContext()),
      AffineMap::get(2, 0, {d0, d1}, rewriter.getContext())};
  SmallVector<utils::IteratorType> iteratorTypes{
      utils::IteratorType::parallel, utils::IteratorType::parallel};

  auto addBias = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, ValueRange{matmul->getResult(0), bias},
      ValueRange{*biasInit}, indexingMaps, iteratorTypes,
      [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
        auto sum = builder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        builder.create<linalg::YieldOp>(nestedLoc, sum.getResult());
      });
  return addBias->getResult(0);
}

struct FoldReluConstantPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto constant = op.getInput().getDefiningOp<ConstantOp>();
    if (!constant)
      return failure();

    auto denseAttr = dyn_cast_or_null<DenseFPElementsAttr>(constant.getValue());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!denseAttr || !resultType || !resultType.getElementType().isF32())
      return failure();

    SmallVector<float> folded;
    folded.reserve(denseAttr.getNumElements());
    for (const APFloat &element : denseAttr.getValues<APFloat>())
      folded.push_back(std::max(0.0f, element.convertToFloat()));

    auto newAttr = DenseElementsAttr::get(resultType, ArrayRef<float>(folded));
    auto newConstant =
        rewriter.create<ConstantOp>(op.getLoc(), newAttr, op.getOutput().getType());
    rewriter.replaceOp(op, newConstant.getOutput());
    if (constant->use_empty())
      rewriter.eraseOp(constant);
    return success();
  }
};

struct FuseLinearReluPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto linear = op.getInput().getDefiningOp<LinearOp>();
    if (!linear || !linear->hasOneUse())
      return failure();

    auto fused = rewriter.create<FusedLinearReluOp>(
        op.getLoc(), op.getOutput().getType(), linear.getInput(),
        linear.getWeight(), linear.getBias());
    rewriter.replaceOp(op, fused.getOutput());
    rewriter.eraseOp(linear);
    return success();
  }
};

struct LowerConstantPattern : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto valueAttr = dyn_cast_or_null<TypedAttr>(op.getValue());
    if (!valueAttr)
      return failure();
    auto lowered = rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                      valueAttr);
    rewriter.replaceOp(op, lowered.getResult());
    return success();
  }
};

struct LowerReluPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType)
      return failure();
    auto lowered = lowerReluValue(op.getLoc(), op.getInput(), resultType, rewriter);
    if (failed(lowered))
      return failure();
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

struct LowerLinearPattern : public OpRewritePattern<LinearOp> {
  using OpRewritePattern<LinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType)
      return failure();
    auto lowered = lowerLinearValue(op.getLoc(), op.getInput(), op.getWeight(),
                                    op.getBias(), resultType, rewriter);
    if (failed(lowered))
      return failure();
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

struct LowerFusedLinearReluPattern : public OpRewritePattern<FusedLinearReluOp> {
  using OpRewritePattern<FusedLinearReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedLinearReluOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType)
      return failure();
    auto linear = lowerLinearValue(op.getLoc(), op.getInput(), op.getWeight(),
                                   op.getBias(), resultType, rewriter);
    if (failed(linear))
      return failure();
    auto relu = lowerReluValue(op.getLoc(), *linear, resultType, rewriter);
    if (failed(relu))
      return failure();
    rewriter.replaceOp(op, *relu);
    return success();
  }
};

struct MiniCanonicalizePass
    : public PassWrapper<MiniCanonicalizePass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MiniDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldReluConstantPattern, FuseLinearReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "mini-canonicalize"; }
  StringRef getDescription() const final {
    return "Run canonicalization-style rewrites for the mini dialect";
  }
};

struct MiniConstantFoldPass
    : public PassWrapper<MiniConstantFoldPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MiniDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldReluConstantPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "mini-const-fold"; }
  StringRef getDescription() const final {
    return "Fold relu(constant) into mini.constant";
  }
};

struct MiniDCEPass : public PassWrapper<MiniDCEPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "mini-dce"; }
  StringRef getDescription() const final {
    return "Mini dialect dead code elimination pass skeleton";
  }
};

struct MiniFusionPass
    : public PassWrapper<MiniFusionPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<MiniDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseLinearReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  StringRef getArgument() const final { return "mini-fusion"; }
  StringRef getDescription() const final {
    return "Fuse mini.linear followed by mini.relu";
  }
};

struct MiniLowerToLinalgPass
    : public PassWrapper<MiniLowerToLinalgPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, MiniDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerConstantPattern, LowerReluPattern, LowerLinearPattern,
                 LowerFusedLinearReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    bool hasIllegalMiniOps = false;
    getOperation()->walk([&](Operation *operation) {
      if (isa<ConstantOp, LinearOp, ReluOp, FusedLinearReluOp>(operation))
        hasIllegalMiniOps = true;
    });
    if (hasIllegalMiniOps) {
      getOperation().emitError(
          "mini-lower-to-linalg left unsupported mini ops behind");
      signalPassFailure();
    }
  }
  StringRef getArgument() const final { return "mini-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower mini dialect ops to linalg/arith/tensor";
  }
};

} // namespace

std::unique_ptr<Pass> createMiniCanonicalizePass() {
  return std::make_unique<MiniCanonicalizePass>();
}

std::unique_ptr<Pass> createMiniConstantFoldPass() {
  return std::make_unique<MiniConstantFoldPass>();
}

std::unique_ptr<Pass> createMiniDCEPass() {
  return std::make_unique<MiniDCEPass>();
}

std::unique_ptr<Pass> createMiniFusionPass() {
  return std::make_unique<MiniFusionPass>();
}

std::unique_ptr<Pass> createMiniLowerToLinalgPass() {
  return std::make_unique<MiniLowerToLinalgPass>();
}

void registerMiniPasses() {
  PassRegistration<MiniCanonicalizePass>();
  PassRegistration<MiniConstantFoldPass>();
  PassRegistration<MiniDCEPass>();
  PassRegistration<MiniFusionPass>();
  PassRegistration<MiniLowerToLinalgPass>();
}

void registerMiniPassPipelines() {
  PassPipelineRegistration<>(
      "mini-cpu-lowering",
      "Run the project CPU lowering pipeline from mini dialect to LLVM dialect",
      [](OpPassManager &pm) {
        const char *pipelineText =
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
          llvm::report_fatal_error("failed to parse mini-cpu-lowering pipeline");
      });

  PassPipelineRegistration<>(
      "mini-gpu-prep",
      "Prepare mini dialect programs for later GPU/Triton mapping by lowering "
      "to standard tensor/linalg dialects and cleaning up the IR",
      [](OpPassManager &pm) {
        const char *pipelineText =
            "func.func(mini-canonicalize,mini-fusion,mini-lower-to-linalg),"
            "canonicalize,"
            "cse";
        if (failed(parsePassPipeline(pipelineText, pm)))
          llvm::report_fatal_error("failed to parse mini-gpu-prep pipeline");
      });
}

} // namespace mini
