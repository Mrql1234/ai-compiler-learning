# Design: Mini AI Compiler

## Overview
`Mini AI Compiler` 现在采用双轨架构：

- **Python 轨**
  - 保留现有原型实现
  - 负责 `FX / ONNX` 导入、图规范化、bridge 导出、reference 执行、验证与 benchmark
- **MLIR 轨**
  - 新增 `compiler-mlir/` out-of-tree C++ 子工程
  - 作为正式编译器主线
  - 负责 dialect、pass、lowering 与后端路线

主链路改为：

`PyTorch FX / ONNX -> Python bridge -> MLIR module/dialect -> MLIR passes -> CPU(LLVM) + Triton/GPU lowering -> execution + validation`

## Design Principles
- MLIR 是正式主 IR，不再只是后续升级项。
- Python 原型继续保留，但角色变为桥接与 reference harness。
- 文档、目录结构、测试与工具链都要体现双轨架构，而不是把 MLIR 挂在原型后面。
- CPU 和 Triton/GPU 在设计上是同级主后端，只是实现节奏可以不同。

## System Architecture

### 1. Frontend Bridge Layer（Python）
职责：

- 导入外部模型图
- 做图规范化
- 生成 bridge 输出
- 驱动验证和 benchmark

当前包含：

- `frontend/fx_importer.py`
- `frontend/onnx_importer.py`
- `ir/`
- `passes/`
- `backend/cpu/`
- `tools/`

推荐角色划分：

- `frontend/`
  - 模型读取与图规范化
- `ir/`
  - 教学原型 IR
- `tools/export_mlir.py`
  - 继续保留，作为 bridge / dump 工具
- 新增 bridge 导出工具
  - 面向 `compiler-mlir/` 的正式输入

### 2. MLIR Core Compiler（C++ out-of-tree）
职责：

- 注册自定义 dialect
- 注册与运行 MLIR pass
- 把高层表示 lower 到标准或目标相关 dialect
- 生成 CPU / GPU 路线可继续消费的中间产物

新目录：

```text
compiler-mlir/
  CMakeLists.txt
  README.md
  include/
  lib/
  tools/
  test/
```

工程约束：

- 使用官方 LLVM/MLIR CMake 体系
- 不直接改造 `projects/mlir-passes/`
- 可以复用 `projects/mlir-passes/` 的经验与 pass 结构设计

### 3. Dialect Strategy
第一版采用 `Mini` 自定义 dialect 作为高层入口。

它至少要能表达：

- `matmul`
- `add`
- `mul`
- `relu`
- `reshape`
- `transpose`

后续逐步扩展到：

- `gelu`
- `layernorm`
- `softmax`
- fused op

设计目标不是一开始就把所有 op 都 fully custom，而是建立一条清晰路径：

- high-level mini dialect
- -> `arith/tensor/linalg/scf`
- -> `LLVM` 或 `GPU/Triton` 相关路径

### 4. Pass Pipeline Strategy
正式 pass 主线落在 MLIR 工程中。

第一批 pass 设计为：

- `canonicalize`
- `constant fold`
- `DCE`
- `fusion`

Python 侧现有 pass 的角色：

- 行为原型
- 教学参考
- 基线对照

MLIR 侧 pass 的角色：

- 正式优化链路
- 面向真实 lowering 的实现

### 5. Backend Strategy

#### CPU Backend
正式路线：

- `MLIR -> LLVM IR -> CPU execution`

当前 Python CPU backend 继续保留：

- 用于 reference 执行
- 用于语义对照

#### Triton/GPU Backend
正式路线：

- `MLIR -> Triton/GPU lowering`

第一阶段允许先实现：

- lowering plan
- 受限可执行路径
- 热点算子 MVP

第二阶段再逐步扩展：

- fused op
- 更多算子
- 更真实的执行路径

### 6. Validation Harness
Python 继续承担统一验证层角色。

职责：

- 调用 eager baseline
- 调用 Python reference backend
- 调用 `compiler-mlir` toolchain
- 统一做：
  - correctness 对照
  - benchmark
  - artifact dump

这意味着后续 `tools/` 应逐步具备：

- 导出 bridge 输入
- 调用 MLIR tool
- 收集结果
- 对照 eager/reference 输出

## Bridge Format Strategy
第一版桥接优先采用**文本桥接**。

推荐顺序：

1. Python 输出 MLIR 文本或 bridge 文本
2. `compiler-mlir` 解析 bridge 输入
3. 构建 MLIR module

这样做的原因：

- 实现快
- 可读性强
- 易测试
- 易 debug

后续允许升级为：

- JSON/Proto bridge
- in-memory API

但这些不作为当前第一批实现的前提。

## Proposed Directory Structure

```text
mini-ai-compiler/
  README.md
  requirements.md
  design.md
  tasks.md
  frontend/                # Python bridge / importer / normalization
  ir/                      # Python prototype IR
  passes/                  # Python prototype pass reference
  backend/cpu/             # Python reference backend
  tools/                   # bridge export / validation / benchmark / dump
  tests/
  benchmarks/
  compiler-mlir/           # out-of-tree MLIR C++ project
    CMakeLists.txt
    README.md
    include/
    lib/
    tools/
    test/
```

## Staged Implementation Plan

### Phase A: 文档与架构重置
- 重写 `requirements.md`
- 重写 `design.md`
- 重写 `tasks.md`
- 更新 `README.md`

### Phase B: MLIR 工程骨架
- 新增 `compiler-mlir/`
- 接通 CMake
- 提供 dialect / pass / tool skeleton
- 提供 smoke test

### Phase C: 前端桥接
- Python FX/ONNX -> bridge text
- bridge text -> MLIR module

### Phase D: MLIR Pass
- canonicalize
- constant fold
- DCE
- fusion

### Phase E: CPU 路线
- lower 到 `LLVM IR`
- 跑通 `MLP`

### Phase F: Triton/GPU 路线
- lowering 到 Triton/GPU 相关路径
- 支持核心算子与 fused op

### Phase G: 统一验证
- Python harness 调用 MLIR 主工程
- 正确性验证
- benchmark

## Notes on Current Repo
- `projects/mlir-passes/` 是学习与实验资产，不直接等于新主工程。
- 当前 `mini-ai-compiler` Python 原型已经具备：
  - 自定义 IR
  - Python pass
  - CPU reference backend
  - MLIR 风格文本导出
- 新设计不是删除这些内容，而是重新定义它们在整体架构中的位置。
