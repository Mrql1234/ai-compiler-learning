#ifndef MINI_COMPILER_PASSES_H
#define MINI_COMPILER_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mini {

std::unique_ptr<mlir::Pass> createMiniCanonicalizePass();
std::unique_ptr<mlir::Pass> createMiniConstantFoldPass();
std::unique_ptr<mlir::Pass> createMiniDCEPass();
std::unique_ptr<mlir::Pass> createMiniFusionPass();
std::unique_ptr<mlir::Pass> createMiniLowerToLinalgPass();
std::unique_ptr<mlir::Pass> createMiniGpuTilePass();
std::unique_ptr<mlir::Pass>
createMiniGpuTilePass(llvm::ArrayRef<int64_t> tileSizes);
std::unique_ptr<mlir::Pass> createMiniGpuMapPass();
std::unique_ptr<mlir::Pass> createMiniGpuHostSharedPass();
void registerMiniGpuPasses();
void registerMiniGpuPassPipelines();

void registerMiniPasses();
void registerMiniPassPipelines();

} // namespace mini

#endif // MINI_COMPILER_PASSES_H
