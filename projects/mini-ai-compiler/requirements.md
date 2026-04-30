# Requirements: Mini AI Compiler

## Summary
`Mini AI Compiler` 现在采用“双轨并存，但 MLIR 是真实工程主链路”的定位。

- Python 轨继续保留，负责前端导入、图规范化、样例生成、reference 执行、正确性验证与 benchmark 驱动。
- MLIR 轨作为正式编译器主工程，采用 out-of-tree C++ MLIR 项目，负责 dialect、pass、lowering 与真实后端路线。

项目新的目标不再是“先做自定义 IR，再把 MLIR 当升级项”，而是：

`PyTorch FX / ONNX -> Python bridge -> MLIR dialect/module -> MLIR passes -> CPU(LLVM) + Triton/GPU lowering -> execution + validation`

## Product Goal
系统必须同时具备以下两种能力：

- **教学原型能力**
  - Python 侧保留自定义 IR 与可运行原型
  - 便于学习、讲解、快速实验与 baseline 对照
- **真实编译器能力**
  - 新增 `compiler-mlir/` 子工程
  - 使用官方 LLVM/MLIR CMake 体系
  - 在 MLIR 中实现核心优化 pass 与 lowering
  - 支持 `CPU via LLVM` 与 `Triton/GPU` 两条主后端路线

## Scope Boundary
本项目仍然面向“小模型子集”，不追求一开始覆盖完整大模型生态。

### Supported Operator Subset
第一批聚焦以下算子或其直接等价表达：

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
第一批聚焦：

- `MLP / FFN`
- `single attention block`

## Goals
- 构建一条以 MLIR 为中心的 AI 编译器主链路。
- 保留 Python 原型作为桥接层和 reference harness。
- 把图优化 pass 的正式实现落在 MLIR 中，而不是只停留在自定义 IR。
- 明确支持两条后端方向：
  - `MLIR -> LLVM IR -> CPU`
  - `MLIR -> Triton/GPU lowering`
- 提供统一的正确性验证、IR dump 和 benchmark 驱动。

## Non-Goals
- 不追求一开始支持完整 Transformer / Qwen 全模型推理。
- 不要求第一版就完成工业级 GPU 调度、复杂内存规划或量化体系。
- 不要求 Python 原型被完全删除。
- 不要求第一版 bridge 就实现复杂 in-memory API 集成。

## High-Level Architecture Requirements

### R1. Frontend Import
系统必须支持从外部模型表示导入图结构。

**Acceptance criteria:**
- 必须支持 `PyTorch FX` 导入。
- 必须支持 `ONNX` 导入，允许以 MVP 子集方式落地。
- 前端输出必须能进入统一 bridge 层，而不是直接绑定某个单一 backend。
- 当遇到未支持算子时，必须显式报错。

### R2. Python Bridge Layer
Python 侧必须作为前端桥接层和验证层，而不是未来唯一编译核心。

**Acceptance criteria:**
- 保留现有 `frontend/`、`ir/`、`passes/`、`backend/cpu/` 原型。
- Python 侧必须能输出一种稳定 bridge 格式给 MLIR 工程消费。
- 第一版 bridge 默认采用 **文本桥接**，优先选择 MLIR 文本或结构化桥接文本。
- 后续允许升级为 in-memory 或结构化 bridge，但不作为第一阶段硬要求。

### R3. MLIR-Native Compiler Core
系统必须新增一个正式的 out-of-tree C++ MLIR 子工程作为主编译器实现。

**Acceptance criteria:**
- 子工程目录为 `compiler-mlir/`。
- 必须基于官方 LLVM/MLIR CMake 体系构建。
- 必须具备：
  - 自定义 dialect 注册能力
  - pass 注册能力
  - 编译器 driver/tool 入口
- 必须能够消费 bridge 层导出的输入并形成 MLIR module。

### R4. MLIR Dialect and IR Strategy
系统必须明确以 MLIR dialect/module 作为正式中间表示主线。

**Acceptance criteria:**
- 自定义 dialect 至少能表达第一批目标算子。
- dialect 设计必须能与 `func`、`arith`、`tensor`、`linalg`、`scf`、`LLVM` 等后续 lowering 路线兼容。
- 当前 Python 自定义 IR 允许继续存在，但其角色是原型与桥接，不再是唯一主 IR。

### R5. MLIR Pass Pipeline
核心优化 pass 必须作为 MLIR 主链路的硬要求。

**Acceptance criteria:**
- 必须在 MLIR 中规划或实现：
  - `canonicalize`
  - `constant fold`
  - `DCE`
  - `fusion`
- 这些 pass 不再被定义为“可选升级项”。
- Python 原型中的同名 pass 可以继续保留，作为参考实现和行为对照。

### R6. CPU Backend
系统必须支持 `MLIR -> LLVM IR -> CPU` 的正式后端路线。

**Acceptance criteria:**
- 文档中必须把 `CPU via LLVM` 写成主后端之一。
- Python CPU backend 继续保留，但定位为 reference backend。
- 新主链路的 CPU 路线必须以 MLIR lowering 为目标，而不是只停留在 Python 解释执行。

### R7. Triton/GPU Backend
系统必须支持 `MLIR -> Triton/GPU lowering` 的主后端路线。

**Acceptance criteria:**
- 文档中必须把 Triton/GPU 作为与 CPU 同级的正式后端方向。
- 第一版允许先做到 lowering plan、受限执行或热点算子 MVP。
- fused op 必须被纳入 Triton/GPU 路线设计范围。

### R8. Validation Harness
系统必须有统一验证与评测层。

**Acceptance criteria:**
- Python harness 必须负责：
  - eager/reference 对照
  - ONNXRuntime 对照（当 ONNX 路线启用时）
  - benchmark
  - IR dump
- 验证层必须能同时服务 Python 原型和 MLIR 主链路。

### R9. IR Dump and Artifact Output
系统必须支持分层 IR 可观测性。

**Acceptance criteria:**
- 必须能够输出：
  - Python 原型 IR
  - bridge 输出
  - MLIR module 文本
  - lowering plan 或后端中间产物
- 优化前后 IR 必须可对照。

## Recommended Project Structure Requirements
项目结构应采用双轨布局：

- `frontend/`
- `ir/`
- `passes/`
- `backend/cpu/`
- `tools/`
- `tests/`
- `benchmarks/`
- `compiler-mlir/`

其中：

- Python 目录负责桥接与验证
- `compiler-mlir/` 负责正式 MLIR 编译器实现

## Staged Delivery Plan

### Phase A: 文档与架构重置
**目标：**
- 重写 `requirements/design/tasks`
- 把 MLIR 提升为正式主链路
- 明确目录结构、bridge 协议、后端路线

**交付：**
- 新版 SDD 文档
- 新版项目 README

### Phase B: MLIR 工程骨架
**目标：**
- 新增 `compiler-mlir/`
- 接通官方 LLVM/MLIR CMake 体系
- 提供自定义 dialect、pass 注册、driver tool 骨架

**交付：**
- 可构建的 out-of-tree MLIR skeleton
- 最小 smoke test

### Phase C: 前端桥接
**目标：**
- Python FX/ONNX -> bridge format
- bridge format -> MLIR module

**交付：**
- bridge 导出工具
- 样例模型桥接产物

### Phase D: MLIR Pass
**目标：**
- 在 MLIR 轨中实现或规划：
  - `constant fold`
  - `canonicalize`
  - `DCE`
  - `fusion`

**交付：**
- MLIR pass pipeline
- `mlir-opt` 级测试

### Phase E: CPU 路线
**目标：**
- `MLIR -> LLVM IR`
- 跑通 `MLP`

**交付：**
- CPU 正式后端闭环

### Phase F: Triton/GPU 路线
**目标：**
- 对核心算子与 fused op 做 Triton/GPU lowering

**交付：**
- Triton/GPU 路线 MVP

### Phase G: 统一验证
**目标：**
- Python harness 统一驱动
- 正确性对照与 benchmark

**交付：**
- 双轨统一验证入口

## Open Questions
- bridge format 第一版最终选择纯 MLIR 文本，还是保留一个结构化中间层？
- Triton/GPU 路线第一批是直接对接 Triton dialect，还是先做可解释的 lowering plan？
