# AI 编译器学习攻略 & 技术名词清单

> 为 Java 应用开发背景的同学定制  
> 最后更新：2026-03-06  
> 整理人：cx330 ✨

---

## 一、招聘岗位 JD 高频技术名词解释

### 📌 核心框架类

| 名词 | 解释 | 重要程度 |
|------|------|----------|
| **TVM** | Apache 开源的深度学习编译器栈，端到端优化模型部署 | ⭐⭐⭐⭐⭐ |
| **MLIR** | LLVM 的多级中间表示框架，AI 编译器的事实标准 | ⭐⭐⭐⭐⭐ |
| **LLVM** | 经典的编译器基础设施，MLIR 的底层基础 | ⭐⭐⭐⭐ |
| **Triton** | OpenAI 开发的 GPU 算子编程语言，基于 MLIR | ⭐⭐⭐⭐ |
| **TensorRT** | NVIDIA 的推理优化引擎，支持图优化和算子融合 | ⭐⭐⭐⭐ |
| **XLA** | Google 的加速器线性代数编译器，用于 TensorFlow/JAX | ⭐⭐⭐ |
| **ONE** | 三星开源的神经网络编译器（原 TVM 分支） | ⭐⭐ |
| **MNN/NCNN** | 阿里/腾讯的移动端推理引擎 | ⭐⭐⭐ |

### 📌 模型/图优化类

| 名词 | 解释 | 重要程度 |
|------|------|----------|
| **计算图 (Compute Graph)** | 用图结构表示神经网络的计算流程 | ⭐⭐⭐⭐⭐ |
| **算子融合 (Operator Fusion)** | 合并多个算子减少内存访问，提升性能 | ⭐⭐⭐⭐⭐ |
| **常量折叠 (Constant Folding)** | 编译期计算常量表达式，减少运行时开销 | ⭐⭐⭐⭐ |
| **死代码消除 (DCE)** | 删除不被使用的计算节点 | ⭐⭐⭐⭐ |
| **内存复用 (Memory Reuse)** | 优化张量内存分配，减少峰值内存 | ⭐⭐⭐⭐ |
| **量化 (Quantization)** | FP32→INT8/FP16，减少模型大小和推理延迟 | ⭐⭐⭐⭐⭐ |
| **剪枝 (Pruning)** | 删除不重要的权重或通道，压缩模型 | ⭐⭐⭐ |
| **蒸馏 (Distillation)** | 大模型→小模型，保持性能的同时压缩 | ⭐⭐ |
| **动态 Shape** | 支持可变输入尺寸的模型推理 | ⭐⭐⭐⭐ |
| **子图替换 (Subgraph Replacement)** | 将部分计算图替换为优化后的实现 | ⭐⭐⭐⭐ |

### 📌 GPU/CUDA 相关

| 名词 | 解释 | 重要程度 |
|------|------|----------|
| **CUDA** | NVIDIA 的 GPU 并行计算平台和编程模型 | ⭐⭐⭐⭐⭐ |
| **GEMM** | 通用矩阵乘法，深度学习最核心的算子 | ⭐⭐⭐⭐⭐ |
| **Tensor Core** | NVIDIA GPU 专用矩阵加速单元 | ⭐⭐⭐⭐ |
| **Shared Memory** | GPU 块内共享的高速缓存 | ⭐⭐⭐⭐ |
| **Register** | GPU 线程私有的高速存储 | ⭐⭐⭐⭐ |
| **Warp** | GPU 调度的基本单位（32 线程） | ⭐⭐⭐⭐ |
| **Occupancy** | GPU 资源利用率指标 | ⭐⭐⭐ |
| **Coalesced Access** | 合并内存访问，提升带宽利用率 | ⭐⭐⭐⭐ |
| **Stream** | CUDA 异步执行队列 | ⭐⭐⭐ |
| **Graph Capture** | CUDA 图捕获，减少 kernel 启动开销 | ⭐⭐⭐ |

### 📌 LLM 专用优化

| 名词 | 解释 | 重要程度 |
|------|------|----------|
| **FlashAttention** | 高效的 Attention 实现，减少内存访问 | ⭐⭐⭐⭐⭐ |
| **PagedAttention** | vLLM 的显存管理技术，类似操作系统分页 | ⭐⭐⭐⭐ |
| **KV Cache** | 缓存 Attention 的 K/V 状态，加速自回归生成 | ⭐⭐⭐⭐⭐ |
| **Continuous Batching** | 动态批处理，提升 LLM 推理吞吐 | ⭐⭐⭐⭐ |
| **Speculative Decoding** | 推测解码，用小模型加速大模型生成 | ⭐⭐⭐ |
| **MoE (Mixture of Experts)** | 混合专家模型，稀疏激活 | ⭐⭐⭐⭐ |
| **Awq/SmoothQuant** | LLM 量化方案，保持精度的同时压缩 | ⭐⭐⭐⭐ |

### 📌 其他基础设施

| 名词 | 解释 | 重要程度 |
|------|------|----------|
| **ONNX** | 开放的神经网络交换格式 | ⭐⭐⭐⭐⭐ |
| **PyTorch 2.x** | 支持 torch.compile，基于 TorchDynamo+Inductor | ⭐⭐⭐⭐⭐ |
| **TorchDynamo** | PyTorch 2.x 的图捕获组件 | ⭐⭐⭐⭐ |
| **TorchInductor** | PyTorch 2.x 的编译器后端 | ⭐⭐⭐⭐ |
| **IREE** | Google 的机器学习编译器，侧重跨平台部署 | ⭐⭐⭐ |
| **OpenXLA** | OpenXLA 项目（StableHLO + XLA） | ⭐⭐⭐ |

---

## 二、AI 编译器主要方向

```
┌─────────────────────────────────────────────────────────┐
│                    AI 编译器技术栈                        │
├─────────────────────────────────────────────────────────┤
│  应用层：LLM 推理引擎 (vLLM, TGI, SGLang)                │
│     ↓                                                    │
│  图优化层：计算图优化、算子融合、内存规划                 │
│     ↓                                                    │
│  算子层：CUDA/Triton 算子开发、性能调优                    │
│     ↓                                                    │
│  编译层：MLIR Dialect 开发、Pass 优化、Codegen            │
│     ↓                                                    │
│  硬件层：GPU/NPU/TPU 指令集、内存层次                      │
└─────────────────────────────────────────────────────────┘
```

### 方向 1：LLM 模型层优化 🎯
- **工作内容**：KV Cache 管理、Continuous Batching、 speculative decoding
- **代表项目**：vLLM, SGLang, TensorRT-LLM
- **技能要求**：PyTorch、CUDA 基础、理解 Transformer 架构
- **适合 Java 背景**：✅ 偏应用层，有很多工程优化空间

### 方向 2：MLIR 编译器开发 🔧
- **工作内容**：Dialect 设计、Pass 开发、优化策略
- **代表项目**：MLIR, IREE, TorchMLIR
- **技能要求**：C++、LLVM/MLIR、编译器原理
- **适合 Java 背景**：⚠️ 需要补 C++ 和编译器基础

### 方向 3：CUDA 算子优化 ⚡
- **工作内容**：手写高性能算子、GEMM 优化、FlashAttention 实现
- **代表项目**：Cutlass, Triton, cuBLAS
- **技能要求**：CUDA C、GPU 架构、性能分析
- **适合 Java 背景**：⚠️ 需要学习 GPU 编程模型

### 方向 4：推理引擎工程 🚀
- **工作内容**：模型加载、服务化、性能监控、多卡部署
- **代表项目**：Triton Inference Server, vLLM
- **技能要求**：C++/Python、分布式系统、性能调优
- **适合 Java 背景**：✅✅ 工程经验可迁移，最容易切入

---

## 三、学习路线（Java 背景定制版）

### 📍 阶段 0：前置知识补全（2-4 周）

**目标**：补齐编译器和 GPU 编程的基础认知

| 主题 | 学习资源 | 优先级 |
|------|----------|--------|
| 编译器原理 | 《编译原理》(龙书) 前 8 章 / 南京大学《编译原理》慕课 | ⭐⭐⭐ |
| C++ 基础 | 《C++ Primer》前 10 章 / C++ 速成课 | ⭐⭐⭐⭐ |
| Python 基础 | 廖雪峰 Python 教程（你有 Java 基础，1 周搞定） | ⭐⭐⭐⭐ |
| GPU 架构基础 | 《CUDA C 编程权威指南》前 3 章 | ⭐⭐⭐⭐ |

> 💡 **cx330 的建议**：别在龙书上死磕！看慕课 + 实践为主。C++ 能读懂代码就行，不用成为专家。

---

### 📍 阶段 1：入门 AI 编译器（4-6 周）

**目标**：理解 AI 编译器的基本工作流程

#### 1.1 TVM 入门（2 周）
- 官方教程：https://tvm.apache.org/docs/tutorial/
- 重点章节：
  - `tvmc` 命令行工具使用
  - Relay IR 基础
  - AutoTVM 自动调优
- 实践：用 TVM 编译一个 ResNet 模型到 CPU/GPU

#### 1.2 MLIR 入门（2-3 周）
- 官方教程：https://mlir.llvm.org/docs/Tutorials/
- 重点概念：
  - Dialect（方言）是什么
  - Operation、Type、Attribute
  - Pass 机制
- 实践：写一个简单的 MLIR Pass

#### 1.3 PyTorch 2.x 编译栈（1 周）
- 文档：https://pytorch.org/docs/stable/torch.compiler.html
- 理解：TorchDynamo + TorchInductor 的工作流程
- 实践：用 `torch.compile` 优化自己的模型

> 💡 **Java 优势**：JVM 也有 JIT 编译，很多优化思想是相通的（如内联、逃逸分析）

---

### 📍 阶段 2：深入 CUDA 算子（6-8 周）

**目标**：能手写和优化 GPU 算子

#### 2.1 CUDA 编程基础（3 周）
- 书籍：《CUDA C 编程权威指南》
- 核心概念：
  - Thread/Block/Grid 层次
  - Shared Memory、Register
  - Warp、Occupancy
- 实践：实现矩阵乘法、向量加法

#### 2.2 GEMM 优化（2 周）
- 资源：NVIDIA GTC 演讲 "Optimizing Matrix Multiplication"
- 优化技巧：
  - Tiling（分块）
  - Shared Memory 缓存
  - Register 重用
  - Warp-level Matrix Multiply
- 实践：手写一个接近 cuBLAS 性能的 GEMM

#### 2.3 Triton 编程（2-3 周）
- 官方教程：https://triton-lang.org/main/getting-started/index.html
- 优势：用 Python 写 GPU 代码，比 CUDA C 简单
- 实践：用 Triton 实现 FlashAttention

> 💡 **Java 思维转换**：从"单线程思维"转到"万线程并行思维"是关键

---

### 📍 阶段 3：LLM 推理优化（4-6 周）

**目标**：理解并实践 LLM 推理引擎的核心技术

#### 3.1 Transformer 架构深入（1 周）
- 论文：Attention Is All You Need
- 重点：Self-Attention 的计算流程、KV Cache 原理

#### 3.2 FlashAttention（2 周）
- 论文：FlashAttention: Fast and Memory-Efficient Exact Attention
- 核心思想：IO 感知、分块计算、避免 HBM 访问
- 实践：用 Triton 实现简化版 FlashAttention

#### 3.3 vLLM 源码阅读（2-3 周）
- 仓库：https://github.com/vllm-project/vllm
- 重点模块：
  - PagedAttention 实现
  - Continuous Batching 调度
  - KV Cache 管理
- 实践：部署 vLLM，跑通 Qwen/Llama 模型

---

### 📍 阶段 4：实战项目（持续）

**目标**：有可以展示的作品

#### 项目建议（选 1-2 个）：

1. **TVM 模型部署**
   - 选一个 HuggingFace 模型
   - 用 TVM 编译到目标硬件
   - 对比原生 PyTorch 的性能提升

2. **Triton 算子库**
   - 实现 3-5 个常用算子（LayerNorm, GELU, Attention）
   - 与 PyTorch 原生实现对比性能

3. **简易推理引擎**
   - 实现 KV Cache 管理
   - 支持 Continuous Batching
   - 部署一个 LLM 模型

4. **MLIR Pass 开发**
   - 为某个 Dialect 写优化 Pass
   - 贡献到开源项目

---

## 四、学习资源汇总

### 📚 书籍
- 《CUDA C 编程权威指南》- GPU 编程入门
- 《深度学习编译器》- 系统讲解 AI 编译器（中文）
- 《编译原理》- 经典教材（选读）

### 🎥 课程
- 南京大学《编译原理》- 中国大学 MOOC
- 斯坦福 CS149: Parallel Computing - YouTube
- B 站"AI 编译器入门"系列视频

### 📄 论文（按优先级）
1. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)
2. MLIR: Scaling Compiler Infrastructure for Domain Specific Computation (CGO 2020)
3. FlashAttention: Fast and Memory-Efficient Exact Attention (NeurIPS 2022)
4. PagedAttention: vLLM 相关论文

### 🔗 关键链接
- TVM 官方：https://tvm.apache.org/
- MLIR 官方：https://mlir.llvm.org/
- Triton 官方：https://triton-lang.org/
- PyTorch 2.x 编译：https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- vLLM 项目：https://github.com/vllm-project/vllm

### 💬 社区
- TVM Discourse: https://discuss.tvm.apache.org/
- LLVM Discourse: https://discourse.llvm.org/
- 知乎"AI 编译器"话题
- OpenClaw 社区（你已经在用了😉）

---

## 五、求职建议

### 🎯 目标公司（国内）

| 公司 | 团队/产品 | 技术栈 |
|------|----------|--------|
| 字节 | 火山引擎、AML | TVM, MLIR, 自研 |
| 阿里 | 平头哥、PAI | MNN, Blade, XLA |
| 腾讯 | 优图、混元 | NCNN, 自研 |
| 百度 | 昆仑、飞桨 | Paddle Inference |
| 华为 | 昇腾 | CANN, MindSpore |
| 寒武纪 | 智能芯片 | Cambricon |
| 壁仞/摩尔线程 | GPU 芯片 | 自研编译器 |
| 月之暗面/MiniMax | LLM 推理 | vLLM, 自研优化 |

### 📝 简历准备

**项目经验 > 理论知识**
- 至少准备 1-2 个完整的实战项目
- GitHub 仓库要有 README 和性能对比数据
- 有开源贡献是巨大加分项

**面试常考**
- 手写矩阵乘法优化
- 解释算子融合原理
- KV Cache 如何管理
- FlashAttention 的核心思想
- MLIR 的 Dialect 设计思路

### 💰 薪资参考（2025-2026）
- 初级（1-3 年）：30-50w
- 中级（3-5 年）：50-80w
- 高级（5 年+）：80-150w+
- 专家/架构师：150w+

> ⚠️ 注意：AI 编译器是**高门槛高回报**方向，前期投入大，但越老越吃香

---

## 六、给 Java 背景的特别建议

### ✅ 你的优势
1. **工程能力强** - Java 项目通常更复杂，工程经验可迁移
2. **系统思维** - JVM、GC、并发模型的理解有助于理解编译器优化
3. **性能调优经验** - Java 性能优化经验（JIT、GC 调优）与编译器优化有共通之处

### ⚠️ 需要补的短板
1. **C++** - 编译器领域的主流语言，至少要能读懂
2. **底层知识** - 内存层次、缓存、指令集等
3. **数学基础** - 线性代数（矩阵运算）、数值计算

### 🚀 最佳切入路径
```
Java 应用开发
    ↓
LLM 推理引擎工程（vLLM 部署、优化、服务化）
    ↓
算子优化（Triton 写算子、性能分析）
    ↓
编译器开发（MLIR Pass、Dialect）
```

> 💡 **cx330 的真心话**：别一上来就死磕 MLIR 源码！从应用层（推理引擎）切入，边用边学，更容易坚持。你有 Java 工程经验，这是很多科班出身的人没有的优势。

---

## 七、30 天快速启动计划

| 周次 | 主题 | 具体任务 |
|------|------|----------|
| 第 1 周 | 认知建立 | 看 3 个 AI 编译器入门视频，读 TVM 官方教程前 3 章 |
| 第 2 周 | TVM 实践 | 用 TVM 编译一个模型到 CPU，跑通端到端流程 |
| 第 3 周 | CUDA 入门 | 学完《CUDA C 编程权威指南》前 5 章，实现矩阵乘法 |
| 第 4 周 | PyTorch 2.x | 用 torch.compile 优化模型，理解背后原理 |

**每天投入**：2-3 小时  
**预期成果**：能理解 AI 编译器的基本工作流程，能跑通简单示例

---

## 结语

AI 编译器是个**深坑**，但也是个**金矿**。

- 门槛高 → 竞争相对少
- 技术深 → 越老越吃香
- 需求旺 → LLM 爆发带来大量机会

你有 Java 背景，工程能力是优势。别被"编译器"三个字吓到——从应用层切入，边做边学，6 个月后你会惊讶于自己的成长。

有问题随时问我，cx330 陪你一起卷！✨

---

_最后更新：2026-03-06_  
_整理：cx330 ✨_
