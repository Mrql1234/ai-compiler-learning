# AI Compiler Learning Notes 🚀

> AI 编译器学习笔记 - Java 背景入门  
> 学习路线参考：[ai-compiler-study-guide.md](../ai-compiler-study-guide.md)

## 📚 学习进度

- [x] 阶段 1：AI 编译器入门（TVM、MLIR、PyTorch 2.x）— 笔记 01-04 完成
- [ ] 阶段 2：CUDA 算子优化（GEMM、Triton、FlashAttention）— 笔记 05-07 进行中
- [ ] 阶段 3：LLM 推理引擎（vLLM、PagedAttention、Continuous Batching）— 计划中
- [ ] 阶段 4：实战项目 — 计划中

## 📝 笔记目录

### 阶段 1：AI 编译器入门 ✅

| 编号 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [01](./notes/01-ai-compiler-intro.md) | AI 编译器入门：给 Java 开发者的第一堂课 | ✅ 完成 | 核心概念、计算图、IR、算子融合 |
| [02](./notes/02-mlir-deep-dive.md) | 深入 MLIR：编译器的事实标准 | ✅ 完成 | Dialect、Operation、Pass 详解 |
| [03](./notes/03-compute-graph-optimization.md) | 计算图优化实战：从理论到性能提升 | ✅ 完成 | torch.compile、TVM 编译、算子融合实践 |
| [04](./notes/04-tvm-scheduling-optimization.md) | TVM 调度优化深入：从自动调优到手写优化 | ✅ 完成 | 分块、并行、向量化、AutoTVM |

### 阶段 2：CUDA 算子优化 🚧

| 编号 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [05](./notes/05-cuda-programming-basics.md) | CUDA 编程基础：GPU 并行模型入门 | 🚧 进行中 | Thread/Block/Grid、内存层次、第一个 CUDA 程序 |
| [06] | GEMM 优化实战：从朴素实现到接近 cuBLAS | 📅 计划中 | Tiling、Shared Memory、Register 重用 |
| [07] | Triton 编程入门：用 Python 写 GPU 算子 | 📅 计划中 | Triton 语言基础、实现 LayerNorm/Attention |

### 阶段 3：LLM 推理引擎 📅 计划中

| 编号 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [08] | LLM 推理优化：KV Cache 和 Continuous Batching | 📅 计划中 | Transformer 架构、KV Cache 原理 |
| [09] | FlashAttention 原理与 Triton 实现 | 📅 计划中 | IO 感知 Attention、分块计算 |
| [10] | vLLM 源码解读：PagedAttention 实战 | 📅 计划中 | 显存管理、调度器实现 |

### 阶段 4：实战项目 📅 计划中

- [ ] 项目 1：TVM 模型部署（ResNet/MobileNet 编译到 CPU/GPU）
- [ ] 项目 2：Triton 算子库（LayerNorm、GELU、Attention）
- [ ] 项目 3：简易 LLM 推理引擎（KV Cache + Continuous Batching）
- [ ] 项目 4：MLIR Pass 开发（贡献到开源项目）

---

## 📂 目录结构

```
ai-compiler-learning/
├── README.md                 # 本文件
├── notes/                    # 学习笔记
│   ├── 01-ai-compiler-intro.md
│   ├── 02-mlir-deep-dive.md
│   ├── 03-compute-graph-optimization.md
│   ├── 04-tvm-scheduling-optimization.md
│   └── 05-cuda-programming-basics.md
├── code/                     # 实践代码（按笔记编号组织）
│   ├── 01/
│   ├── 02/
│   ├── 03/
│   ├── 04/
│   └── 05/
├── papers/                   # 论文阅读笔记
└── resources/                # 学习资源
    └── ai-compiler-study-guide.md
```

## 🎯 目标

- 理解 AI 编译器基本工作流程
- 能手写简单的 CUDA/Triton 算子
- 部署和优化 LLM 推理引擎
- 完成 1-2 个实战项目

---

_Started: 2026-03-17_  
_Last Updated: 2026-03-22_
