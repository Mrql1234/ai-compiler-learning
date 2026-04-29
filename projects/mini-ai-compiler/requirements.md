# Requirements: Mini AI Compiler

## Summary
本项目题目为：**Mini AI Compiler: From ONNX / PyTorch FX to MLIR IR, Optimized CPU/Triton Execution**。

项目目标是实现一个面向**小型神经网络子集**的端到端 AI 编译器，覆盖从前端模型导入，到中间表示构建、图优化、后端执行、正确性验证与 benchmark 的完整闭环。项目强调“全流程可运行”和“技术深度可继续升级”两点：第一版先以可控的小模型子集和自定义 IR 跑通闭环，后续再逐步引入 Triton backend 和 MLIR 升级路线。

## Product Goal
系统需要支持以下完整流程：
- 前端导入：`PyTorch FX` 和 `ONNX`
- 中间表示：自定义简化 IR，并预留 `MLIR IR` 表达或升级路径
- 优化 Pass：`Constant Folding`、`Dead Code Elimination`、`Operator Fusion`、`Layout / Memory Planning`
- 后端执行：`CPU reference backend` 与 `Triton backend`
- 验证：与 `PyTorch eager` 或 `ONNXRuntime` 对齐、IR dump、性能 benchmark

## Scope Boundary
本项目**不以支持完整模型生态为目标**，而是限制在一个小而完整的子集内，以保证项目能稳定落地并形成完整编译链路。

### Supported Operator Subset
第一版推荐优先支持以下算子：
- `MatMul`
- `Add`
- `Mul`
- `Relu`
- `Gelu`
- `LayerNorm`
- `Softmax`
- `Transpose`
- `Reshape`

### Supported Model Subset
第一版推荐优先支持以下网络：
- `MLP / FFN`
- `单层 Attention Block 的简化版`

## Goals
- 构建一个可以从模型导入到执行验证的完整 AI 编译器闭环。
- 使用小模型子集控制复杂度，保证工程可完成。
- 在第一版中优先完成自定义 IR 路线。
- 在后续阶段增加 ONNX、fusion、benchmark、Triton 和 MLIR 升级。
- 让项目同时具备学习价值、展示价值和扩展价值。

## Non-Goals
- 不追求一开始支持完整 PyTorch / ONNX 模型生态。
- 不在第一阶段实现完整 MLIR-native 编译管线。
- 不在第一阶段支持复杂训练图、自动求导或动态图。
- 不在第一阶段完成复杂调度器、量化系统或工业级内存优化。
- 不要求 Triton backend 在一开始覆盖全部算子。

## High-Level Architecture Requirements

### R1. Frontend Import
系统必须支持从外部模型表示导入图结构。

**Acceptance criteria:**
- Phase 1 必须支持 `PyTorch FX` 导入。
- Phase 2 必须支持 `ONNX` 导入。
- 导入结果必须映射为统一内部 IR，而不是为每个前端单独设计执行路径。
- 当输入模型包含未支持算子时，系统必须显式报错并指出不支持的节点。

### R2. Intermediate Representation
系统必须具有统一的中间表示层，用于承载优化和后端 lowering。

**Acceptance criteria:**
- 第一版必须实现自定义简化 IR。
- IR 至少应包含 `Graph`、`Node`、`TensorType`、`Attribute` 等核心概念。
- 每个 node 至少应包含 `op_type`、`inputs`、`outputs`、`shape`、`dtype`、`attrs` 信息。
- IR 必须可打印、可遍历、可改写。

### R3. MLIR Path
系统必须为 `MLIR IR` 预留路径。

**Acceptance criteria:**
- 第一版可以先采用自定义 IR。
- 后续版本必须支持“输出 MLIR 风格 IR 文本”或“迁移到 MLIR-based IR / pass”。
- IR 设计不得阻断向 MLIR 升级。

### R4. Optimization Passes
系统必须支持至少四类图优化能力。

**Acceptance criteria:**
- 必须支持 `Constant Folding`。
- 必须支持 `Dead Code Elimination`。
- 必须支持 `Operator Fusion`。
- 必须支持轻量级 `Layout / Memory Planning`，至少能表达 buffer 分配顺序、简单复用或 layout 元信息。

### R5. CPU Backend
系统必须提供一个 CPU reference backend。

**Acceptance criteria:**
- CPU backend 必须能执行受支持子集的 IR。
- CPU backend 可以基于 `NumPy`、`PyTorch eager` 或自定义解释器实现。
- CPU backend 的主要职责是正确性验证和与 Triton backend 的对照。

### R6. Triton Backend
系统必须提供一个逐步扩展的 Triton backend。

**Acceptance criteria:**
- Triton backend 第一批至少支持 `matmul`、`add`、`relu`。
- `layernorm` 可以作为后续增强项。
- Triton backend 后续必须支持部分 fused op，如 `fused linear + relu`、`fused linear + gelu`。

### R7. Correctness Validation
系统必须具备正确性验证能力。

**Acceptance criteria:**
- 对 FX 路线，输出必须可与 `PyTorch eager` 对齐。
- 对 ONNX 路线，输出必须可与 `ONNXRuntime` 对齐。
- 系统必须能比较误差并判断是否在允许范围内。

### R8. IR Dump
系统必须支持 IR 可观测性。

**Acceptance criteria:**
- 必须能够输出原始 IR。
- 必须能够输出优化后 IR。
- 必须能够输出 backend lowered IR 或 execution plan。

### R9. Benchmark
系统必须支持性能评测。

**Acceptance criteria:**
- 至少支持 `eager baseline`、`compiler CPU backend`、`Triton backend` 三类路径的对比。
- benchmark 至少输出 `latency`。
- 可选输出 `throughput` 和 `memory usage`。

## Recommended Project Structure Requirements
项目结构应至少覆盖以下模块：
- `docs/`
- `examples/`
- `frontend/`
- `ir/`
- `passes/`
- `backend/cpu/`
- `backend/triton/`
- `runtime/`
- `tests/`
- `benchmarks/`

## Staged Delivery Plan

### Phase 1: 最小闭环
**目标：**
- 从 `PyTorch FX` 导入
- 建立自定义 IR
- 实现 CPU backend
- 做 `constant fold + DCE`
- 跑通 `MLP`

**交付：**
- 能从模型到 IR 到执行
- 有正确性验证

### Phase 2: 优化与可视化
**目标：**
- 加 `fusion`
- 加 `IR dump`
- 加 `benchmark`
- 支持 `ONNX importer`

**交付：**
- 优化前后 IR 对比
- latency 对比

### Phase 3: Triton backend
**目标：**
- 为核心算子生成 Triton kernel
- 支持 fused op
- 和 CPU backend 对照

**交付：**
- 有真实 GPU backend
- 有 benchmark 提升

### Phase 4: MLIR 升级版（可选）
**目标：**
- 输出 `MLIR 风格 IR`
- 或把 IR pass 迁移到 `MLIR-based` 实现

**交付：**
- 项目具备从自定义 IR 迈向 MLIR 的清晰升级路径

## Why This Project Is Recommended
该项目之所以作为最终推荐题目，是因为它同时满足以下要求：
- **真正全流程**：覆盖 import、IR、optimization、lowering、execution、validation
- **难度可控**：通过限制算子和模型子集保证可完成性
- **有扩展空间**：后续可接 MLIR、量化、更多后端、更多模型结构
- **技术表达强**：能清晰体现编译器结构、图优化、IR 设计、kernel/backend 和性能验证能力

## Project Name and One-Line Introduction
**项目名：** `Mini AI Compiler`

**一句话介绍：**
> A small end-to-end AI compiler that imports ONNX / PyTorch FX graphs, lowers them into an intermediate representation, applies graph-level optimizations, and executes optimized kernels on CPU and Triton backends.

## Recommended Engineering Order
项目实施顺序必须遵循以下优先级：
1. `PyTorch FX importer`
2. `自定义 IR`
3. `CPU interpreter backend`
4. `constant fold / DCE`
5. `fusion`
6. `benchmark`
7. `Triton backend`
8. `MLIR 升级`

## Open Questions
- CPU backend 最终选择 `NumPy`、`PyTorch eager` 还是纯自定义解释执行？
- `Layout / Memory Planning` 第一版只做元信息记录，还是实际引入 buffer reuse？
- `MLIR IR` 路线是先做文本输出，还是直接在第二版引入更接近 dialect 的结构？
