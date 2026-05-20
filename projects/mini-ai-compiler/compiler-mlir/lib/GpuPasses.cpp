#include "MiniCompiler/Passes.h"
#include "MiniCompiler/MiniDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir;

namespace mini {
namespace {

static constexpr int64_t kMiniGpuTileSizes[] = {8, 8, 1};

static FailureOr<RankedTensorType> getRankedF32Tensor(Value value,
                                                      unsigned rank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != static_cast<int64_t>(rank) ||
      !type.getElementType().isF32())
    return failure();
  if (!type.hasStaticShape())
    return failure();
  return type;
}

static MemRefType getIdentityMemRefFor(RankedTensorType tensorType) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
}

static Value createToBuffer(Location loc, Value tensor, MemRefType memrefType,
                            OpBuilder &builder, bool readOnly) {
  OperationState state(loc, bufferization::ToBufferOp::getOperationName());
  state.addOperands(tensor);
  state.addTypes(memrefType);
  if (readOnly)
    state.addAttribute("read_only", builder.getUnitAttr());
  return builder.create(state)->getResult(0);
}

static Value createToTensor(Location loc, Value buffer,
                            RankedTensorType tensorType, OpBuilder &builder) {
  OperationState state(loc, bufferization::ToTensorOp::getOperationName());
  state.addOperands(buffer);
  state.addTypes(tensorType);
  state.addAttribute("restrict", builder.getUnitAttr());
  state.addAttribute("writable", builder.getUnitAttr());
  return builder.create(state)->getResult(0);
}

static StringRef getLinearReluRuntimeCallee(StringRef backend) {
  if (backend == "cublas")
    return "mini_cublas_linear_relu_f32_memref";
  return "mini_cuda_linear_relu_f32_memref";
}

static LogicalResult getOrCreateRuntimeDecl(ModuleOp module, Location loc,
                                            StringRef callee,
                                            FunctionType functionType) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(callee)) {
    if (existing.getFunctionType() != functionType)
      return existing.emitError("runtime declaration type mismatch for ")
             << callee;
    return success();
  }

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto func = builder.create<func::FuncOp>(loc, callee, functionType);
  func.setPrivate();
  return success();
}

static FailureOr<SmallVector<int64_t>> parseTileSizes(StringRef spec) {
  SmallVector<int64_t> tileSizes;
  SmallVector<StringRef> pieces;
  spec.split(pieces, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (pieces.empty())
    return failure();
  for (StringRef piece : pieces) {
    int64_t tileSize = 0;
    if (piece.trim().getAsInteger(10, tileSize) || tileSize <= 0)
      return failure();
    tileSizes.push_back(tileSize);
  }
  return tileSizes;
}

enum class MiniGpuMappingLevel { Block, Thread, Sequential };

struct MiniGpuTilePipelineOptions
    : public PassPipelineOptions<MiniGpuTilePipelineOptions> {
  PassOptions::Option<std::string> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Comma-separated tile sizes for outer GPU blocks"),
      llvm::cl::init("")};
};

struct MiniGpuRuntimeCallPipelineOptions
    : public PassPipelineOptions<MiniGpuRuntimeCallPipelineOptions> {
  PassOptions::Option<std::string> backend{
      *this, "backend",
      llvm::cl::desc("Runtime backend for mini.fused_linear_relu: cuda_hand or cublas"),
      llvm::cl::init("cuda_hand")};
};

static gpu::Processor getProcessorFor(MiniGpuMappingLevel level, int dimension) {
  if (dimension > 2 || level == MiniGpuMappingLevel::Sequential)
    return gpu::Processor::Sequential;

  switch (level) {
  case MiniGpuMappingLevel::Block:
    switch (dimension) {
    case 0:
      return gpu::Processor::BlockX;
    case 1:
      return gpu::Processor::BlockY;
    case 2:
      return gpu::Processor::BlockZ;
    }
    break;
  case MiniGpuMappingLevel::Thread:
    switch (dimension) {
    case 0:
      return gpu::Processor::ThreadX;
    case 1:
      return gpu::Processor::ThreadY;
    case 2:
      return gpu::Processor::ThreadZ;
    }
    break;
  case MiniGpuMappingLevel::Sequential:
    break;
  }
  return gpu::Processor::Sequential;
}

static LogicalResult applyMapping(scf::ParallelOp parallelOp,
                                  MiniGpuMappingLevel level) {
  if (parallelOp->getAttr(gpu::getMappingAttrName()))
    return success();

  Builder builder(parallelOp.getContext());
  const int numLoops = static_cast<int>(parallelOp.getNumLoops());
  const int loopsToMap = std::min(numLoops, 3);
  SmallVector<gpu::ParallelLoopDimMappingAttr> mapping;
  mapping.reserve(numLoops);

  for (int index = 0; index < numLoops; ++index) {
    int hardwareDimension = index < loopsToMap ? (loopsToMap - 1 - index) : 3;
    mapping.push_back(builder.getAttr<gpu::ParallelLoopDimMappingAttr>(
        getProcessorFor(level, hardwareDimension), builder.getDimIdentityMap(),
        builder.getDimIdentityMap()));
  }

  return gpu::setMappingAttr(parallelOp, mapping);
}

static void mapNestedParallelLoops(scf::ParallelOp parallelOp,
                                   MiniGpuMappingLevel level) {
  if (failed(applyMapping(parallelOp, level)))
    return;

  MiniGpuMappingLevel nestedLevel = MiniGpuMappingLevel::Sequential;
  if (level == MiniGpuMappingLevel::Block)
    nestedLevel = MiniGpuMappingLevel::Thread;

  for (Operation &operation : *parallelOp.getBody()) {
    if (auto nestedParallel = dyn_cast<scf::ParallelOp>(operation))
      mapNestedParallelLoops(nestedParallel, nestedLevel);
  }
}

struct MiniGpuMapPass
    : public PassWrapper<MiniGpuMapPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    for (Block &block : func.getBlocks()) {
      for (Operation &operation : block) {
        auto parallelOp = dyn_cast<scf::ParallelOp>(operation);
        if (!parallelOp)
          continue;
        mapNestedParallelLoops(parallelOp, MiniGpuMappingLevel::Block);
      }
    }
  }

  StringRef getArgument() const final { return "mini-gpu-map"; }
  StringRef getDescription() const final {
    return "Apply a project-defined GPU mapping strategy to scf.parallel loops";
  }
};

struct MiniGpuTilePass
    : public PassWrapper<MiniGpuTilePass, OperationPass<func::FuncOp>> {
  MiniGpuTilePass()
      : tileSizes(std::begin(kMiniGpuTileSizes), std::end(kMiniGpuTileSizes)) {}
  explicit MiniGpuTilePass(ArrayRef<int64_t> configuredTileSizes)
      : tileSizes(configuredTileSizes.begin(), configuredTileSizes.end()) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<scf::ParallelOp> innermostParallelLoops;
    func.walk([&](scf::ParallelOp parallelOp) {
      bool hasNestedParallel = false;
      parallelOp.getBody()->walk([&](scf::ParallelOp nestedParallel) {
        if (nestedParallel != parallelOp) {
          hasNestedParallel = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!hasNestedParallel)
        innermostParallelLoops.push_back(parallelOp);
    });

    for (scf::ParallelOp parallelOp : innermostParallelLoops) {
      if (!parallelOp->getBlock())
        continue;
      if (parallelOp.getNumReductions() != 0)
        continue;
      if (parallelOp->getParentOfType<scf::ParallelOp>())
        continue;
      scf::tileParallelLoop(parallelOp, tileSizes,
                            /*noMinMaxBounds=*/true);
    }
  }

  StringRef getArgument() const final { return "mini-gpu-tile"; }
  StringRef getDescription() const final {
    return "Tile top-level scf.parallel loops for the project GPU mapping strategy";
  }

private:
  SmallVector<int64_t> tileSizes;
};

static SmallVector<Value> collectDynamicMemRefDims(Location loc, Value source,
                                                   MemRefType type,
                                                   OpBuilder &builder) {
  SmallVector<Value> dynamicDims;
  for (auto [index, dim] : llvm::enumerate(type.getShape())) {
    if (ShapedType::isDynamic(dim))
      dynamicDims.push_back(builder.create<memref::DimOp>(loc, source, index));
  }
  return dynamicDims;
}

struct MiniGpuHostSharedPass
    : public PassWrapper<MiniGpuHostSharedPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, memref::MemRefDialect>();
  }

  StringRef getArgument() const final { return "mini-gpu-host-shared"; }
  StringRef getDescription() const final {
    return "Materialize GPU-visible host_shared buffers for gpu.launch_func "
           "and conservatively copy mutable operands back to the host";
  }

  static bool isReadOnlySource(Value source) {
    if (auto getGlobal = source.getDefiningOp<memref::GetGlobalOp>()) {
      if (auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
              getGlobal, getGlobal.getNameAttr()))
        return global.getConstant();
      return true;
    }
    return false;
  }

  enum class LaunchOperandPlan {
    AlreadyDeviceVisible,
    CopyInOnly,
    CopyInAndCopyBack
  };

  static bool isKernelOperandWritten(gpu::LaunchFuncOp launch,
                                     unsigned kernelOperandIndex) {
    auto kernel = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(
        launch, launch.getKernelAttr());
    if (!kernel || kernel.getBody().empty())
      return true;
    Block &entryBlock = kernel.getBody().front();
    if (kernelOperandIndex >= entryBlock.getNumArguments())
      return true;

    Value kernelArgument = entryBlock.getArgument(kernelOperandIndex);
    bool written = false;
    kernel.walk([&](memref::StoreOp store) {
      if (store.getMemref() == kernelArgument) {
        written = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return written;
  }

  static LaunchOperandPlan classifyLaunchOperand(Value source,
                                                 gpu::LaunchFuncOp launch,
                                                 unsigned kernelOperandIndex) {
    if (source.getDefiningOp<gpu::AllocOp>())
      return LaunchOperandPlan::AlreadyDeviceVisible;
    if (isReadOnlySource(source))
      return LaunchOperandPlan::CopyInOnly;
    if (!isKernelOperandWritten(launch, kernelOperandIndex))
      return LaunchOperandPlan::CopyInOnly;
    return LaunchOperandPlan::CopyInAndCopyBack;
  }

  Value materializeGpuVisibleCopy(Value source, gpu::LaunchFuncOp launch,
                                  DenseMap<Value, Value> &cache) {
    auto memrefType = dyn_cast<MemRefType>(source.getType());
    if (!memrefType)
      return {};
    if (auto it = cache.find(source); it != cache.end())
      return it->second;

    OpBuilder builder(launch);
    auto dynamicDims =
        collectDynamicMemRefDims(launch.getLoc(), source, memrefType, builder);
    auto sharedAlloc = builder.create<gpu::AllocOp>(
        launch.getLoc(), memrefType,
        /*asyncDependencies=*/ValueRange{},
        /*dynamicSizes=*/dynamicDims,
        /*symbolOperands=*/ValueRange{}, builder.getUnitAttr());
    builder.create<memref::CopyOp>(launch.getLoc(), source,
                                   sharedAlloc.getMemref());
    cache[source] = sharedAlloc.getMemref();
    return sharedAlloc.getMemref();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    SmallVector<memref::AllocOp> allocs;
    func.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
    for (memref::AllocOp alloc : allocs) {
      bool needsGpuShared = llvm::any_of(alloc.getResult().getUsers(),
                                         [](Operation *user) {
                                           return isa<gpu::LaunchFuncOp>(user);
                                         });
      if (!needsGpuShared)
        continue;

      OpBuilder builder(alloc);
      auto sharedAlloc = builder.create<gpu::AllocOp>(
          alloc.getLoc(), alloc.getType(),
          /*asyncDependencies=*/ValueRange{},
          /*dynamicSizes=*/alloc.getDynamicSizes(),
          /*symbolOperands=*/alloc.getSymbolOperands(), builder.getUnitAttr());
      alloc.getResult().replaceAllUsesWith(sharedAlloc.getMemref());
      alloc.erase();
    }

    SmallVector<gpu::LaunchFuncOp> launches;
    func.walk([&](gpu::LaunchFuncOp launch) { launches.push_back(launch); });
    for (gpu::LaunchFuncOp launch : launches) {
      DenseMap<Value, Value> copiedGpuOperands;
      SmallVector<std::pair<Value, Value>> copyBacks;
      DenseSet<Value> seenCopyBackSources;
      MutableOperandRange kernelOperands = launch.getKernelOperandsMutable();
      for (auto [kernelOperandIndex, operand] :
           llvm::enumerate(kernelOperands)) {
        auto memrefType = dyn_cast<MemRefType>(operand.get().getType());
        if (!memrefType)
          continue;
        Value originalValue = operand.get();
        LaunchOperandPlan plan = classifyLaunchOperand(
            originalValue, launch, static_cast<unsigned>(kernelOperandIndex));
        if (plan == LaunchOperandPlan::AlreadyDeviceVisible)
          continue;
        Value sharedValue =
            materializeGpuVisibleCopy(originalValue, launch, copiedGpuOperands);
        if (!sharedValue) {
          launch.emitError("failed to materialize host_shared GPU operand");
          signalPassFailure();
          return;
        }
        operand.set(sharedValue);
        if (plan == LaunchOperandPlan::CopyInAndCopyBack &&
            seenCopyBackSources.insert(originalValue).second) {
          copyBacks.emplace_back(sharedValue, originalValue);
        }
      }

      if (!copyBacks.empty()) {
        OpBuilder builder(launch);
        builder.setInsertionPointAfter(launch);
        for (auto [sharedValue, originalValue] : copyBacks) {
          builder.create<memref::CopyOp>(launch.getLoc(), sharedValue,
                                         originalValue);
        }
      }
    }
  }
};

struct MiniGpuRuntimeCallLoweringPass
    : public PassWrapper<MiniGpuRuntimeCallLoweringPass,
                         OperationPass<ModuleOp>> {
  MiniGpuRuntimeCallLoweringPass() = default;
  MiniGpuRuntimeCallLoweringPass(const MiniGpuRuntimeCallLoweringPass &pass)
      : PassWrapper(pass) {}

  Option<std::string> backend{
      *this, "backend",
      llvm::cl::desc("Runtime backend for mini.fused_linear_relu: cuda_hand or cublas"),
      llvm::cl::init("cuda_hand")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    func::FuncDialect, memref::MemRefDialect, MiniDialect>();
  }

  StringRef getArgument() const final {
    return "mini-gpu-runtime-call-lowering";
  }
  StringRef getDescription() const final {
    return "Lower mini.fused_linear_relu to an explicit GPU runtime func.call";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (backend != "cuda_hand" && backend != "cublas") {
      module.emitError("mini-gpu-runtime-call-lowering supports backend=cuda_hand "
                       "or backend=cublas");
      signalPassFailure();
      return;
    }

    SmallVector<FusedLinearReluOp> worklist;
    module.walk([&](FusedLinearReluOp op) { worklist.push_back(op); });

    for (FusedLinearReluOp op : worklist) {
      auto inputType = getRankedF32Tensor(op.getInput(), 2);
      auto weightType = getRankedF32Tensor(op.getWeight(), 2);
      auto biasType = getRankedF32Tensor(op.getBias(), 1);
      auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
      if (failed(inputType) || failed(weightType) || failed(biasType) ||
          !resultType || resultType.getRank() != 2 ||
          !resultType.getElementType().isF32() || !resultType.hasStaticShape()) {
        op.emitError("runtime call lowering currently supports only static "
                     "ranked f32 linear_relu tensors");
        signalPassFailure();
        return;
      }

      int64_t m = inputType->getShape()[0];
      int64_t k = inputType->getShape()[1];
      int64_t n = weightType->getShape()[0];
      if (weightType->getShape()[1] != k || biasType->getShape()[0] != n ||
          resultType.getShape()[0] != m || resultType.getShape()[1] != n) {
        op.emitError("linear_relu runtime call shape contract mismatch");
        signalPassFailure();
        return;
      }

      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto inputMemRefType = getIdentityMemRefFor(*inputType);
      auto weightMemRefType = getIdentityMemRefFor(*weightType);
      auto biasMemRefType = getIdentityMemRefFor(*biasType);
      auto outputMemRefType = getIdentityMemRefFor(resultType);

      Value inputBuffer =
          createToBuffer(loc, op.getInput(), inputMemRefType, builder,
                         /*readOnly=*/true);
      Value weightBuffer =
          createToBuffer(loc, op.getWeight(), weightMemRefType, builder,
                         /*readOnly=*/true);
      Value biasBuffer = createToBuffer(loc, op.getBias(), biasMemRefType,
                                        builder, /*readOnly=*/true);
      auto outputBuffer =
          builder.create<memref::AllocOp>(loc, outputMemRefType).getResult();
      Value mValue = builder.create<arith::ConstantIndexOp>(loc, m);
      Value nValue = builder.create<arith::ConstantIndexOp>(loc, n);
      Value kValue = builder.create<arith::ConstantIndexOp>(loc, k);

      StringRef callee = getLinearReluRuntimeCallee(backend);
      auto functionType = builder.getFunctionType(
          TypeRange{inputMemRefType, weightMemRefType, biasMemRefType,
                    outputMemRefType, builder.getIndexType(),
                    builder.getIndexType(), builder.getIndexType()},
          TypeRange{});
      if (failed(getOrCreateRuntimeDecl(module, loc, callee, functionType))) {
        signalPassFailure();
        return;
      }

      builder.create<func::CallOp>(
          loc, callee, TypeRange{},
          ValueRange{inputBuffer, weightBuffer, biasBuffer, outputBuffer, mValue,
                     nValue, kValue});
      Value resultTensor = createToTensor(loc, outputBuffer, resultType, builder);
      op.getOutput().replaceAllUsesWith(resultTensor);
      op.erase();
    }

    SmallVector<ConstantOp> constants;
    module.walk([&](ConstantOp op) { constants.push_back(op); });
    for (ConstantOp op : constants) {
      auto valueAttr = dyn_cast_or_null<TypedAttr>(op.getValue());
      if (!valueAttr) {
        op.emitError("expected typed mini.constant value");
        signalPassFailure();
        return;
      }
      OpBuilder builder(op);
      auto lowered = builder.create<arith::ConstantOp>(op.getLoc(), valueAttr);
      op.getOutput().replaceAllUsesWith(lowered.getResult());
      op.erase();
    }
  }
};

static FailureOr<SmallVector<int64_t>>
getConfiguredTileSizes(StringRef spec) {
  if (spec.empty())
    return SmallVector<int64_t>(std::begin(kMiniGpuTileSizes),
                                std::end(kMiniGpuTileSizes));
  return parseTileSizes(spec);
}

static void buildMiniGpuTilePipeline(OpPassManager &pm,
                                     const MiniGpuTilePipelineOptions &options) {
  auto tileSizes = getConfiguredTileSizes(options.tileSizes);
  if (failed(tileSizes))
    llvm::report_fatal_error(
        "mini-gpu-tile-pipeline expects tile-sizes to be a comma-separated "
        "list of positive integers");
  pm.addNestedPass<func::FuncOp>(createMiniGpuTilePass(*tileSizes));
}

static void buildMiniGpuRuntimeCallPipeline(
    OpPassManager &pm, const MiniGpuRuntimeCallPipelineOptions &options) {
  std::string pipelineText =
      "func.func(mini-canonicalize,mini-fusion),"
      "mini-gpu-runtime-call-lowering{backend=" +
      options.backend +
      "},"
      "one-shot-bufferize{bufferize-function-boundaries "
      "function-boundary-type-conversion=identity-layout-map},"
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
        "failed to parse mini-gpu-runtime-call-lowering-pipeline");
}

} // namespace

std::unique_ptr<Pass> createMiniGpuTilePass() {
  return std::make_unique<MiniGpuTilePass>();
}

std::unique_ptr<Pass> createMiniGpuTilePass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<MiniGpuTilePass>(tileSizes);
}

std::unique_ptr<Pass> createMiniGpuMapPass() {
  return std::make_unique<MiniGpuMapPass>();
}

std::unique_ptr<Pass> createMiniGpuHostSharedPass() {
  return std::make_unique<MiniGpuHostSharedPass>();
}

std::unique_ptr<Pass> createMiniGpuRuntimeCallLoweringPass() {
  return std::make_unique<MiniGpuRuntimeCallLoweringPass>();
}

void registerMiniGpuPasses() {
  PassRegistration<MiniGpuTilePass>();
  PassRegistration<MiniGpuMapPass>();
  PassRegistration<MiniGpuHostSharedPass>();
  PassRegistration<MiniGpuRuntimeCallLoweringPass>();
}

void registerMiniGpuPassPipelines() {
  PassPipelineRegistration<MiniGpuTilePipelineOptions>(
      "mini-gpu-tile-pipeline",
      "Run the project GPU tiling pass with configurable tile sizes",
      buildMiniGpuTilePipeline);
  PassPipelineRegistration<MiniGpuRuntimeCallPipelineOptions>(
      "mini-gpu-runtime-call-lowering-pipeline",
      "Lower static mini.fused_linear_relu to an executable CUDA runtime call path",
      buildMiniGpuRuntimeCallPipeline);
}

} // namespace mini
