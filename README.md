# AI Compiler Learning Notes 🚀

> AI 编译器学习笔记 - Java 背景入门  
> 学习路线参考：[ai-compiler-study-guide.md](../ai-compiler-study-guide.md)

## 📚 学习进度

- [x] 阶段 1：AI 编译器入门（TVM、MLIR、PyTorch 2.x）— 笔记 01-04 完成 ✅
- [x] 阶段 2：CUDA 算子优化（GEMM、Triton、算子融合、卷积、量化）— 笔记 05-10 完成 ✅
- [x] 阶段 3：LLM 推理引擎（Transformer、KV Cache、推测解码、MoE、推理引擎对比）— 笔记 11-14 完成 ✅
- [ ] 阶段 4：实战项目 — 计划中

## 📝 笔记目录

### 阶段 1：AI 编译器入门 ✅

| 编号 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [01](./notes/01-ai-compiler-intro.md) | AI 编译器入门：给 Java 开发者的第一堂课 | ✅ 完成 | 核心概念、计算图、IR、算子融合 |
| [02](./notes/02-mlir-deep-dive.md) | 深入 MLIR：编译器的事实标准 | ✅ 完成 | Dialect、Operation、Pass 详解 |
| [03](./notes/03-compute-graph-optimization.md) | 计算图优化实战：从理论到性能提升 | ✅ 完成 | torch.compile、TVM 编译、算子融合实践 |
| [04](./notes/04-tvm-scheduling-optimization.md) | TVM 调度优化深入：从自动调优到手写优化 | ✅ 完成 | 分块、并行、向量化、AutoTVM |

### 阶段 2：CUDA 算子优化 ✅

| 编号 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [05](./notes/05-cuda-programming-basics.md) | CUDA 编程基础：GPU 并行模型入门 | ✅ 完成 | Thread/Block/Grid、内存层次、第一个 CUDA 程序 |
| [06](./notes/06-gemm-optimization.md) | GEMM 优化实战：从朴素实现到接近 cuBLAS | ✅ 完成 | Tiling、Shared Memory、Register 重用 |
| [07](./notes/07-triton-programming.md) | Triton 编程入门：用 Python 写 GPU 算子 | ✅ 完成 | Triton 语言基础、实现 LayerNorm/Attention |
| [08](./notes/08-operator-fusion.md) | 算子融合与内存优化：减少 Global Memory 访问 | ✅ 完成 | 融合 Kernel、内存优化、TVM/MLIR 融合策略 |
| [09](./notes/09-convolution-optimization.md) | 卷积优化实战：从 im2col 到 Winograd | ✅ 完成 | Conv 算法、im2col、Winograd、FFT |
| [10](./notes/10-quantization-sparsity.md) | 量化与稀疏化优化：模型压缩与加速 | ✅ 完成 | INT8/FP8 量化、稀疏化、Tensor Core |

### 阶段 3：LLM 推理引擎 ✅

| 编号 | 主题 | 状态 | 说明 |
|------|------|------|------|
| [11](./notes/11-transformer-architecture-deep-dive.md) | Transformer 架构深入：从 Self-Attention 到 RoPE | ✅ 完成 | Attention 变体、位置编码、预归一化 |
| [12](./notes/12-llm-inference-kv-cache.md) | LLM 推理优化：KV Cache 与 Continuous Batching | ✅ 完成 | 显存优化、PagedAttention、调度策略 |
| [13](./notes/13-speculative-decoding-moe.md) | 推测解码与 MoE 架构：加速 LLM 推理 | ✅ 完成 | 投机采样、专家混合、负载均衡 |
| [14](./notes/14-inference-engines-comparison.md) | 推理引擎对比：vLLM、TGI、SGLang、TensorRT-LLM | ✅ 完成 | 架构对比、性能基准、选型指南 |

### 阶段 4：实战项目 📅 计划中

- [ ] 项目 1：TVM 模型部署（ResNet/MobileNet 编译到 CPU/GPU）
- [ ] 项目 2：Triton 算子库（LayerNorm、GELU、Attention）
- [ ] 项目 3：简易 LLM 推理引擎（KV Cache + Continuous Batching）
- [ ] 项目 4：MLIR Pass 开发（贡献到开源项目）
- [x] 项目 5：手写 CUDA 算子实验室（基础算子 + Nsight 性能分析）

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
│   ├── 05-cuda-programming-basics.md
│   ├── 06-gemm-optimization.md
│   ├── 07-triton-programming.md
│   ├── 08-operator-fusion.md
│   ├── 09-convolution-optimization.md
│   ├── 10-quantization-sparsity.md
│   ├── 11-transformer-architecture-deep-dive.md
│   ├── 12-llm-inference-kv-cache.md
│   ├── 13-speculative-decoding-moe.md
│   └── 14-inference-engines-comparison.md
├── code/                     # 实践代码（按笔记编号组织）
│   ├── 01/
│   ├── 02/
│   ├── 03/
│   ├── 04/
│   ├── 05/
│   ├── 06/
│   ├── 07/
│   ├── 08/
│   ├── 09/
│   ├── 10/
│   ├── 11/
│   ├── 12/
│   ├── 13/
│   └── 14/
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
_Last Updated: 2026-04-15_
