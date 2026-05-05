#ifndef MINI_COMPILER_PASSES_H
#define MINI_COMPILER_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mini {

std::unique_ptr<mlir::Pass> createMiniCanonicalizePass();
std::unique_ptr<mlir::Pass> createMiniConstantFoldPass();
std::unique_ptr<mlir::Pass> createMiniDCEPass();
std::unique_ptr<mlir::Pass> createMiniFusionPass();
std::unique_ptr<mlir::Pass> createMiniLowerToLinalgPass();

void registerMiniPasses();
void registerMiniPassPipelines();

} // namespace mini

#endif // MINI_COMPILER_PASSES_H
