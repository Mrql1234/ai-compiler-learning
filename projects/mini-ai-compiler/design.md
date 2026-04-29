# Design: Mini AI Compiler

## Overview
本设计严格对应项目题目：**Mini AI Compiler: From ONNX / PyTorch FX to MLIR IR, Optimized CPU/Triton Execution**。

实现策略采用“先闭环、后增强、再升级”的路线：
- 第一阶段以 `PyTorch FX + 自定义 IR + CPU backend + constant fold / DCE` 跑通最小闭环
- 第二阶段加入 `fusion + IR dump + benchmark + ONNX importer`
- 第三阶段引入 `Triton backend`
- 第四阶段再升级到 `MLIR 风格 IR` 或 `MLIR-based pass`

这样做的原因是：自定义 IR 更适合第一版快速掌控全流程，而 MLIR 更适合作为第二层技术深度升级。

## Design Principles
- 先保证完整编译链路能运行，再逐步提升技术深度。
- 通过限制算子和模型子集控制复杂度。
- 所有前端共享统一 IR。
- 所有优化基于统一 IR，而不是分散在 importer 或 backend 内。
- CPU backend 作为正确性基线。
- Triton backend 作为性能与 GPU lowering 路线。
- MLIR 作为后续升级方向，而不是第一阶段阻塞项。

## Supported Scope

### Operator Subset
初版围绕以下算子构建：
- `MatMul`
- `Add`
- `Mul`
- `Relu`
- `Gelu`
- `LayerNorm`
- `Softmax`
- `Transpose`
- `Reshape`

### Model Subset
初版围绕以下模型形态构建：
- `MLP / FFN`
- `单层 Attention Block 的简化版`

## System Architecture

### 1. Frontend
#### A. PyTorch FX Importer
输入：PyTorch 小模型。

流程：
1. 使用 `torch.fx.symbolic_trace`
2. 得到 FX graph
3. 转换成内部 IR 节点

定位：
- Phase 1 主入口
- 调试简单
- 最适合作为 MVP 前端

#### B. ONNX Importer
输入：导出的 ONNX 模型。

定位：
- Phase 2 引入
- 用于展示“通用前端”能力
- 与 FX importer 共同映射到同一内部 IR

### 2. Intermediate Representation

#### Option Chosen for Version 1
第一版采用**简化自定义 IR**。

建议对象：
- `Graph`
- `Node`
- `TensorType`
- `Attribute`

建议的节点结构至少包含：

```text
op_type = "matmul"
inputs = [...]
outputs = [...]
shape = ...
dtype = ...
attrs = ...
```

这样设计的原因：
- 更容易完全掌控
- 更适合第一版做全流程
- 开发速度快于直接上完整 MLIR 基础设施

#### MLIR Upgrade Path
第二版及以后有两个方向：
- 输出 `MLIR 风格 IR` 文本
- 或逐步迁移到 `MLIR-based IR / pass`

设计约束：
- 当前 IR 必须具备向 MLIR 概念映射的清晰路径
- 不允许把内部结构设计成只能服务单一 backend 的临时脚本格式

### 3. Optimization Layer
系统至少需要四个 pass。

#### Pass 1: Constant Folding
示例：
- `Add(Const, Const) -> Const`
- `Mul(Const, Const) -> Const`

职责：
- 在图优化阶段提前计算常量表达式
- 减少运行时算子数量

#### Pass 2: Dead Code Elimination
职责：
- 删除无用户的中间结果
- 删除不影响最终输出的节点

#### Pass 3: Operator Fusion
典型融合模式：
- `MatMul + Add + Relu`
- `MatMul + Add + Gelu`

融合结果示例：
- `fused_linear_relu`
- `fused_linear_gelu`

#### Pass 4: Layout / Memory Planning
第一版至少做一个轻量版本，支持以下之一或组合：
- 决定 tensor buffer 的分配顺序
- 做简单 inplace / buffer reuse
- 记录 tensor layout，如 `row-major`

### 4. Backend Layer

#### CPU Backend
第一版先实现 reference backend。

可选实现方式：
- `NumPy`
- `PyTorch eager`
- 自定义解释器

核心作用：
- 验证正确性
- 作为 Triton backend 的对照基线

#### Triton Backend
按阶段渐进支持：
- 第一批：`matmul`、`add`、`relu`
- 后续增强：`layernorm`
- 再后续：
  - `fused linear + relu`
  - `fused linear + gelu`

### 5. Validation Layer

#### Correctness Validation
必须与以下路径做对照：
- `PyTorch eager`
- `ONNXRuntime`（当 ONNX 路线启用时）

验证内容：
- 输出误差
- 优化前后语义一致性
- CPU / Triton 结果一致性

#### IR Dump
必须输出：
- 原始 IR
- 优化后 IR
- backend lowered IR 或 execution plan

#### Benchmark
至少比较：
- `eager baseline`
- `compiler CPU backend`
- `Triton backend`

指标：
- `latency`
- `throughput`（可选）
- `memory usage`（可选）

## Proposed Directory Structure

```text
mini-ai-compiler/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── ir-design.md
│   ├── pass-design.md
│   └── benchmark.md
├── examples/
│   ├── mlp.py
│   ├── attention_block.py
│   └── export_onnx.py
├── frontend/
│   ├── fx_importer.py
│   └── onnx_importer.py
├── ir/
│   ├── graph.py
│   ├── node.py
│   ├── types.py
│   └── printer.py
├── passes/
│   ├── constant_fold.py
│   ├── dce.py
│   ├── fusion.py
│   └── memory_plan.py
├── backend/
│   ├── cpu/
│   │   └── executor.py
│   └── triton/
│       ├── kernels.py
│       └── executor.py
├── runtime/
│   ├── tensor.py
│   └── memory.py
├── tests/
│   ├── test_frontend.py
│   ├── test_passes.py
│   ├── test_cpu_backend.py
│   └── test_triton_backend.py
└── benchmarks/
    └── bench_mlp.py
```

## Staged Implementation Plan

### Phase 1: 最小闭环
目标：
- 从 `PyTorch FX` 导入
- 建立自定义 IR
- 实现 CPU backend
- 做 `constant fold + DCE`
- 跑通 `MLP`

交付：
- 能从模型到 IR 到执行
- 有正确性验证

设计说明：
- 这一阶段不先做 MLIR、不先做 Triton、不先做量化
- 重点是“真正把闭环跑起来”

### Phase 2: 优化与可视化
目标：
- 加 `fusion`
- 加 `IR dump`
- 加 `benchmark`
- 支持 `ONNX importer`

交付：
- 优化前后 IR 对比
- latency 对比

设计说明：
- 这一阶段开始体现“编译器展示力”
- 让项目不只是能跑，还能讲清楚优化做了什么

### Phase 3: Triton backend
目标：
- 为核心算子生成 Triton kernel
- 支持 fused op
- 和 CPU backend 对照

交付：
- 有真实 GPU backend
- 有 benchmark 提升

设计说明：
- 这一步把项目从“图优化器”升级为“真正有后端 lowering 的 AI 编译器”

### Phase 4: MLIR 升级版（可选）
目标：
- 输出 `MLIR 风格 IR`
- 或把 IR pass 迁移到 `MLIR-based` 实现

设计说明：
- 这个阶段不是 MVP 必须项
- 但它能显著提高项目技术深度和 MLIR 相关性
- 当前推荐的过渡方案不是立刻接入完整 MLIR C++ API，而是先在 Python IR 上引入
  `Pattern + Rewriter + Greedy Driver` 这类 MLIR 风格重写结构，让“pass 如何从命令式遍历
  迁移到声明式重写”这件事先变得可见、可测试、可讲解

## Why This Design
这个设计被选中，是因为它同时满足：

### 1. 真正全流程
覆盖：
- import
- IR
- optimization
- lowering
- execution
- validation

### 2. 难度可控
- 通过限制算子和模型子集控制范围
- 通过阶段化设计减少一次性复杂度

### 3. 有扩展空间
后续可以继续加入：
- MLIR
- quantization
- 更多 backend
- 更复杂的 attention / transformer

### 4. 简历和表达强
这个项目能够同时体现：
- 编译器结构理解
- 图优化能力
- IR 设计能力
- backend / kernel 理解
- benchmark 与验证意识

## Execution Order
工程执行顺序必须保持为：
1. `PyTorch FX importer`
2. `自定义 IR`
3. `CPU interpreter backend`
4. `constant fold / DCE`
5. `fusion`
6. `benchmark`
7. `Triton backend`
8. `MLIR 升级`

## Next Step
下一份产物建议是 `tasks.md`，把 Phase 1 拆成可直接动手的文件级任务，例如：
- `fx_importer.py` 要先支持哪些 node
- `graph.py / node.py / types.py` 要先定义哪些字段
- `executor.py` 第一版如何执行 `matmul/add/relu`
- `constant_fold.py` 与 `dce.py` 第一版覆盖哪些规则
