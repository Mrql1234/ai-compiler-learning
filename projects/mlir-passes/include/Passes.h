#ifndef MLIR_PASSES_PASSES_H
#define MLIR_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

/// 创建常量折叠 Pass
std::unique_ptr<Pass> createConstantFoldPass();
void registerConstantFoldPass();

/// 创建模板化常量折叠 Pass
std::unique_ptr<Pass> createTemplateConstantFoldPass();
void registerTemplateConstantFoldPass();

/// 创建增强版手写常量折叠 Pass
std::unique_ptr<Pass> createExtendedConstantFoldPass();
void registerExtendedConstantFoldPass();

/// 创建基于 MLIR 内建 folding/canonicalization 的常量折叠 Pass
std::unique_ptr<Pass> createBuiltinCanonicalizeFoldPass();
void registerBuiltinCanonicalizeFoldPass();

/// 创建死代码消除 Pass
std::unique_ptr<Pass> createDeadCodeElimPass();

/// 创建算子融合 Pass
std::unique_ptr<Pass> createOperatorFusionPass();

/// 注册所有 Pass
void registerAllCustomPasses();

} // namespace mlir

#endif // MLIR_PASSES_PASSES_H
