# AGENTS.md

本文件用于说明当前仓库的开发背景、目录用途、已落地约定与后续开发建议。作用范围为**仓库根目录及其所有子目录**；若更深层目录存在新的 `AGENTS.md`，则以更深层文件为准。

---

## 1. 仓库概览

当前仓库是一个以 **AI 编译器 / MLIR / Triton / LLVM 学习与实验** 为主题的工作区，主要包含以下内容：

- `projects/mlir-passes/`
  - MLIR 外部 pass 学习项目
  - 重点是 C++ / MLIR pass 开发、IR 变换、`mlir-opt` 验证
- `projects/mini-ai-compiler/`
  - Python 实现的端到端教学型 AI 编译器项目
  - 当前是本仓库里最完整的“前端 -> IR -> pass -> backend -> 验证”闭环原型
- `other/`
  - 杂项资料目录
  - **该目录下已有单独的 `other/AGENTS.md`，进入该目录树后需要优先遵守它**

根目录下还有一些环境记录文件，例如：

- `README.md`
- `WSL_MLIR_SETUP_LOG.md`

---

## 2. 当前重点项目：`projects/mini-ai-compiler/`

### 2.1 项目定位

`mini-ai-compiler` 是一个**双轨架构**项目，目标是实现一个面向小模型子集的端到端 AI 编译器。

项目题目可以概括为：

> Mini AI Compiler: From ONNX / PyTorch FX to MLIR IR, Optimized CPU/Triton Execution

当前分为两条轨道：

- Python 轨：教学原型、bridge、reference backend、验证与 benchmark
- MLIR 轨：正式编译器主线，位于 `projects/mini-ai-compiler/compiler-mlir/`

### 2.2 当前范围

当前项目**不是完整大模型编译器**，而是面向一个可控的小子集：

- 推荐模型范围：
  - `MLP / FFN`
  - `single attention block`
- 当前真正跑通的示例：
  - `TinyMLP`

### 2.3 当前实现状态

当前项目已经具备以下能力：

- `PyTorch FX` 前端导入
- 自定义 IR
- `Constant Folding`
- `DCE`
- `linear + relu` 的 `FusionPass`
- CPU reference backend
- IR 文本 dump
- benchmark 脚本
- ONNX importer MVP 代码框架（运行依赖 `onnx` 包）
- MLIR 风格文本导出
- MLIR 风格 rewrite 原型
- `compiler-mlir/` out-of-tree MLIR 子工程骨架

当前还**没有**真正进入：

- 正式 MLIR pass 实装
- `MLIR -> LLVM` CPU 闭环
- `MLIR -> Triton/GPU` 正式闭环

---

## 3. `mini-ai-compiler` 目录结构说明

当前 `projects/mini-ai-compiler/` 主要结构如下：

- `requirements.md`
  - 需求规格
- `design.md`
  - 架构与设计说明
- `tasks.md`
  - 分阶段任务清单
- `README.md`
  - 项目说明与使用方式
- `frontend/`
  - 前端导入器
  - 当前包括：
    - `fx_importer.py`
    - `onnx_importer.py`
- `ir/`
  - 自定义 IR 定义
  - 包括：
    - `graph.py`
    - `node.py`
    - `value.py`
    - `types.py`
    - `printer.py`
- `passes/`
  - 图优化 pass
  - 当前包括：
    - `constant_fold.py`
    - `dce.py`
    - `fusion.py`
    - `manager.py`
- `backend/cpu/`
  - CPU reference backend
- `compiler-mlir/`
  - 正式 MLIR C++ 子工程
  - 使用 out-of-tree LLVM/MLIR CMake 体系
- `examples/`
  - 示例模型
  - 当前主要是 `mlp.py`
- `tools/`
  - 运行与调试工具
  - 当前包括：
    - `run_mlp_example.py`
    - `dump_ir.py`
- `benchmarks/`
  - benchmark 脚本
  - 当前主要是 `bench_mlp.py`
- `tests/`
  - 基础单测

---

## 4. `mini-ai-compiler` 的核心设计约定

### 4.1 IR 设计约定

当前项目采用双层 IR 视角：

- Python 原型 IR：
  - 采用“**Node 表示操作，Value 表示数据边**”的思路
- MLIR 主线 IR：
  - 采用 `mini` dialect 与 MLIR module/pass/lowering 体系

Python 原型 IR 具体约定如下：

- `Node`
  - 表示一个操作 / 算子
  - 例如：`constant`、`add`、`mul`、`linear`、`relu`
- `Value`
  - 表示某个操作产生的结果
  - 或图输入值
- `Graph`
  - 维护：
    - `inputs`
    - `outputs`
    - `nodes`

这是一个典型的图 IR：

- 点 = `Node`
- 边 = `Value`

### 4.2 当前 CPU backend 的语义约定

当前 CPU backend 基于 `numpy` 做 reference 执行。

需要注意：

- `matmul`
  - 使用 `args[0] @ args[1]`
- `linear`
  - 使用 `args[0] @ args[1].T`
  - 原因是 PyTorch `Linear.weight` 存储为 `[out_features, in_features]`
- `fused_linear_relu`
  - 先做线性层，再做 `relu`

### 4.3 当前 fusion 约定

当前只实现了一个最小 fusion 规则：

- `linear + relu`
  -> `fused_linear_relu`

这是一个**教学型融合示例**，目的主要是：

- 展示 pass 如何改图
- 展示优化前后 IR 差异
- 给后续 Triton / fused op 路线打基础

MLIR 轨当前则已新增：

- dialect skeleton
- pass registration skeleton
- compiler driver skeleton
- smoke test skeleton

### 4.4 当前 DCE 方法

当前 DCE 使用的是：

- **从图输出反向做可达性 / 活跃性传播**

即：

- 从 `graph.outputs` 出发
- 沿着 `Value -> producer Node -> input Values`
- 反向追踪所有活节点
- 未被追踪到的节点删除

这是针对当前“无复杂控制流、以数据流为主”的神经网络图非常合适的方法。

---

## 5. 当前阶段状态

### Phase 1

当前 Phase 1 已完成：

- 项目骨架
- 自定义 IR
- FX importer MVP
- CPU backend MVP
- `constant_fold`
- `dce`
- MLP 示例
- 基础测试

### Phase 2

当前 Phase 2 已完成第一批：

- `FusionPass`
- IR dump 增强
- benchmark 脚本
- ONNX importer MVP 代码

### Phase 3 / 4

当前 Phase 3 / 4 已完成“架构升级的第一批”：

- MLIR 风格文本导出
- MLIR 风格 rewrite 原型
- Triton lowering / executor 骨架
- `compiler-mlir/` 子工程骨架

但还没有完成正式主链路：

- MLIR-native pass 实装
- `MLIR -> LLVM` CPU 路线
- `MLIR -> Triton/GPU` 正式路线

### Phase 3

当前尚未开始真正实现：

- Triton kernel MVP
- Triton executor
- fused op lowering
- CPU / Triton 对照 benchmark

### Phase 4

当前尚未开始真正实现：

- MLIR 风格 IR 输出
- IR 到 MLIR 概念映射
- MLIR-based pass 迁移

---

## 6. 开发建议

### 6.1 修改 `mini-ai-compiler` 时的原则

- 优先保持“**教学性 + 可读性 + 最小闭环**”
- 先修主链路，不要过早引入复杂抽象
- 保持代码小而清晰，避免为了“像工业框架”而过度设计
- 任何新增能力最好配：
  - 示例
  - 测试
  - 或 dump/benchmark 入口

### 6.2 后续推荐顺序

如果继续推进 `mini-ai-compiler`，建议顺序是：

1. 稳定 Python bridge 输出
2. 在 `compiler-mlir/` 中实现正式 dialect / pass
3. 打通 `MLIR -> LLVM` CPU 闭环
4. 再打通 Triton/GPU 路线
5. 最后统一 Python harness 验证与 benchmark

### 6.3 不建议的方向

当前阶段不建议：

- 直接大改成复杂框架式架构
- 一开始就追求完整 Transformer / 完整 Qwen 支持
- 直接切到完整 MLIR-native 实现而放弃现有 Python 原型

---

## 7. 运行与验证习惯

在 `projects/mini-ai-compiler/` 下，常用命令包括：

- 运行 MLP 示例：
  - `python3 -m tools.run_mlp_example`
- 输出 IR：
  - `python3 -m tools.dump_ir`
- 跑 benchmark：
  - `python3 -m benchmarks.bench_mlp`
- 跑测试：
  - `python3 -m unittest discover -s tests`

依赖方面：

- 当前至少需要：
  - `numpy`
  - `torch`
- 若要验证 ONNX importer，还需要：
  - `onnx`

---

## 8. 给后续协作者的提醒

- 如果只是想学习“最小闭环原型”，优先看 `mini-ai-compiler` 的 Python 轨
- 如果想学习“更真实的编译器主线”，优先看 `mini-ai-compiler/compiler-mlir/`
- 如果是研究 MLIR C++ pass，请看 `projects/mlir-passes/`
- 如果是研究 Triton kernel，可把 `mini-ai-compiler` 视作前端 / IR / pass 原型，再与 Triton 实验结合

后续修改时，建议优先同步更新：

- `requirements.md`
- `design.md`
- `tasks.md`
- `README.md`

避免代码和文档长期脱节。
