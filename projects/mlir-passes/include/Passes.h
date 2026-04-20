#ifndef MLIR_PASSES_PASSES_H
#define MLIR_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

/// 创建常量折叠 Pass
std::unique_ptr<Pass> createConstantFoldPass();

/// 创建死代码消除 Pass
std::unique_ptr<Pass> createDeadCodeElimPass();

/// 创建算子融合 Pass
std::unique_ptr<Pass> createOperatorFusionPass();

/// 注册所有 Pass
void registerAllCustomPasses();

} // namespace mlir

#endif // MLIR_PASSES_PASSES_H
