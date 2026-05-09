#include "MiniCompiler/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

  static LaunchOperandPlan classifyLaunchOperand(Value source) {
    if (source.getDefiningOp<gpu::AllocOp>())
      return LaunchOperandPlan::AlreadyDeviceVisible;
    if (isReadOnlySource(source))
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
      for (OpOperand &operand : launch->getOpOperands()) {
        auto memrefType = dyn_cast<MemRefType>(operand.get().getType());
        if (!memrefType)
          continue;
        Value originalValue = operand.get();
        LaunchOperandPlan plan = classifyLaunchOperand(originalValue);
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

void registerMiniGpuPasses() {
  PassRegistration<MiniGpuTilePass>();
  PassRegistration<MiniGpuMapPass>();
  PassRegistration<MiniGpuHostSharedPass>();
}

void registerMiniGpuPassPipelines() {
  PassPipelineRegistration<MiniGpuTilePipelineOptions>(
      "mini-gpu-tile-pipeline",
      "Run the project GPU tiling pass with configurable tile sizes",
      buildMiniGpuTilePipeline);
}

} // namespace mini
