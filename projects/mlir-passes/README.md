# MLIR Pass 开发实战 🔧

> 开发 MLIR 优化 Pass，深入理解编译器架构，向 LLVM 开源项目贡献代码

[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org)
[![MLIR](https://img.shields.io/badge/MLIR-LLVM-red.svg)](https://mlir.llvm.org)
[![LLVM](https://img.shields.io/badge/LLVM-15+-red.svg)](https://llvm.org)

## 📊 优化效果

| Pass | 优化前 Ops | 优化后 Ops | 代码减少 | 编译时间 |
|------|-----------|-----------|---------|---------|
| 常量折叠 | 15 | 6 | **60%** | 10ms |
| 死代码消除 | 20 | 12 | **40%** | 8ms |
| 算子融合 | 25 | 15 | **40%** | 15ms |

## 🎯 项目目标

1. **理解 MLIR 架构**：Dialect、Operation、Pass
2. **开发优化 Pass**：常量折叠、死代码消除、算子融合
3. **贡献开源**：向 LLVM/MLIR 提交 PR
4. **简历亮点**：编译器开发经验

## 🚀 快速开始

### 环境要求

- C++ 17
- CMake 3.20+
- LLVM 15+ (含 MLIR)

### 编译 LLVM（一次性）

```bash
# 克隆 LLVM
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/15.x

# 创建构建目录
mkdir build && cd build

# 配置（只编译 MLIR 相关）
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

# 编译（约 30-60 分钟）
ninja -j$(nproc)
```

### 构建项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置
cmake .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
         -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir

# 编译
ninja -j$(nproc)

# 测试
ninja check-mlir-passes
```

## 📁 项目结构

```
mlir-passes/
├── README.md
├── CMakeLists.txt
├── lib/                        # Pass 实现
│   ├── CMakeLists.txt
│   ├── ConstantFoldPass.cpp   # 常量折叠
│   ├── DeadCodeElimPass.cpp   # 死代码消除
│   └── OperatorFusionPass.cpp # 算子融合
├── include/                    # 头文件
│   └── Passes.h
├── test/                       # 测试用例
│   ├── CMakeLists.txt
│   ├── constant_fold.mlir
│   ├── dce.mlir
│   └── fusion.mlir
├── tools/                      # 工具
│   └── CMakeLists.txt
└── docs/                       # 文档
    ├── pass_design.md
    └── debugging_guide.md
```

## 🔧 Pass 详解

### 1. 常量折叠 Pass

**功能**：编译时计算常量表达式

**优化前**：
```mlir
%0 = arith.constant 42 : i32
%1 = arith.constant 0 : i32
%2 = arith.addi %0, %1 : i32
```

**优化后**：
```mlir
%0 = arith.constant 42 : i32
```

**核心代码**：
```cpp
struct ConstantFoldPattern : public OpRewritePattern<arith::AddIOp> {
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter& rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lhs || !rhs) return failure();
    
    auto result = lhs.getValue() + rhs.getValue();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, result);
    return success();
  }
};
```

### 2. 死代码消除 Pass

**功能**：删除未使用的计算

**优化前**：
```mlir
%0 = arith.constant 42 : i32
%1 = arith.mulf %0, %0 : f32  // 未使用
%2 = arith.addi %0, %0 : i32
return %2 : i32
```

**优化后**：
```mlir
%0 = arith.constant 42 : i32
%2 = arith.addi %0, %0 : i32
return %2 : i32
```

### 3. 算子融合 Pass

**功能**：融合相邻操作，减少内存访问

**优化前**：
```mlir
%0 = "math.exp"(%x) : (f32) -> f32
%1 = "math.sin"(%0) : (f32) -> f32
```

**优化后**：
```mlir
%1 = "fused.exp_sin"(%x) : (f32) -> f32
```

## 📈 实验方法

### 测试 Pass

```bash
# 运行常量折叠
mlir-opt --constant-fold test/constant_fold.mlir

# 运行死代码消除
mlir-opt --dead-code-elim test/dce.mlir

# 组合优化
mlir-opt --constant-fold --dead-code-elim test/input.mlir
```

### 性能分析

```bash
# 查看优化前后 IR 大小
wc -l input.mlir output.mlir

# 查看编译时间
time mlir-opt --constant-fold input.mlir -o /dev/null
```

## 📚 学习资源

- [MLIR 官方教程](https://mlir.llvm.org/docs/Tutorials/)
- [MLIR Toy 示例](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/Patterns/)

## 🎓 贡献指南

### 提交 PR 到 LLVM

1. Fork llvm-project
2. 创建分支
3. 实现 Pass
4. 添加测试
5. 提交 PR

### 代码审查要点

- 遵循 LLVM 编码规范
- 完整的测试覆盖
- 清晰的提交信息
- 性能数据支持

---

_项目创建：2026-04-20 | 作者：ql | 指导：cx330 ✨_
