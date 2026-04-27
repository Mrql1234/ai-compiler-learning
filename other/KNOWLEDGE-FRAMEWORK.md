# AI 编译器知识体系框架

> 基于现有 14 篇笔记 + 课程大纲整理  
> 最后更新：2026-04-27  
> 目标：构建从理论到 NPU 部署的完整知识地图

---

## 📚 知识体系总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层：模型部署                           │
│         LLM 聊天机器人 / YOLOv8 检测 / Stream 并行化               │
├─────────────────────────────────────────────────────────────────┤
│                     优化层：量化与性能调优                         │
│         INT8 量化 / 校准算法 / Profiling / 混合精度                 │
├─────────────────────────────────────────────────────────────────┤
│                    编译层：MLIR 工具链实践                         │
│    Dialect 设计 / Pass 开发 / Conversion / Lowering / LayerGroup  │
├─────────────────────────────────────────────────────────────────┤
│                    基础层：编译器与硬件认知                         │
│      C++ 基础 / 计算图理论 / NPU 架构 / 存储层次 / DMA 协同           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 一、基础认知层（1-2 个月）

### 1.1 C++ for AI 编译器 ⭐⭐⭐⭐
> 课程第 1 讲：AI 编译器中的 C/C++

```
├── 1.1.1 核心数据结构
│   ├── SmallVector - 小向量优化
│   ├── StringRef / ArrayRef - 轻量级引用
│   └── DenseMap / StringMap - 高效哈希表
│
├── 1.1.2 内存管理
│   ├── BumpPtrAllocator - 内存池技术
│   ├── Arena Allocation - 编译器内存分配模式
│   └── 生命周期与所有权
│
├── 1.1.3 类型系统与多态
│   ├── RTTI：isa<> / cast<> / dyn_cast<>
│   ├── CRTP 静态多态模式
│   └── Visitor 模式在 IR 遍历中的应用
│
└── 1.1.4 元编程
    ├── TableGen 基础语法
    ├── 自动生成 C++ 代码
    └── Dialect/Op 定义实践
```

**📝 对应笔记：** 待补充（现有笔记中缺少 C++ 基础）  
**🔧 Lab：** 手写一个简单的 TableGen 定义

---

### 1.2 AI 编译器概论 ⭐⭐⭐⭐⭐
> 课程第 2 讲：Graph vs Kernel

```
├── 1.2.1 计算图表示
│   ├── 节点（算子）与边（数据依赖）
│   ├── 静态图 vs 动态图
│   └── 控制流与数据流
│
├── 1.2.2 算子融合理论
│   ├── 垂直融合（Vertical Fusion）
│   ├── 水平融合（Horizontal Fusion）
│   ├── 混合融合（Mixed Fusion）
│   └── 融合条件与收益分析
│
├── 1.2.3 编译模式
│   ├── AOT（Ahead-Of-Time）编译
│   ├── JIT（Just-In-Time）编译
│   └── 混合模式（如 TorchInductor）
│
└── 1.2.4 优化基础
    ├── 常量折叠（Constant Folding）
    ├── 死代码消除（DCE）
    └── 公共子表达式消除（CSE）
```

**📝 对应笔记：** [notes/01-ai-compiler-intro.md](../notes/01-ai-compiler-intro.md) ✅  
**📝 对应笔记：** [notes/03-compute-graph-optimization.md](../notes/03-compute-graph-optimization.md) ✅

---

### 1.3 NPU 硬件架构 ⭐⭐⭐⭐
> 课程硬件与环境章节

```
├── 1.3.1 NPU 核心结构
│   ├── 计算单元（CU/TPU Core）
│   ├── 矩阵加速单元（Tensor Core）
│   └── 向量处理单元
│
├── 1.3.2 存储层次
│   ├── Global Memory（DDR）
│   ├── Shared Memory / SRAM
│   ├── Local Memory / Register
│   └── 缓存一致性
│
├── 1.3.3 数据搬运
│   ├── DMA 引擎与计算单元协同
│   ├── 双缓冲（Double Buffering）
│   └── 预取策略
│
└── 1.3.4 指令集架构
    ├── 标量指令 vs 向量指令
    ├── 矩阵指令（Tensor Instruction）
    └── 并行指令调度
```

**📝 对应笔记：** 待补充（现有笔记偏重 GPU，缺少 NPU 架构）  
**🔧 Lab2：** NPU 基础推理 Demo 运行（ResNet50 CPU/NPU 对比）

---

## 二、MLIR 核心层（2-3 个月）

### 2.1 MLIR 基础概念 ⭐⭐⭐⭐⭐
> 课程第 3 讲：Dialect / Op / Type

```
├── 2.1.1 MLIR 设计哲学
│   ├── 多级中间表示（Multi-Level IR）
│   ├── 可扩展性与领域特定
│   └── 与 LLVM IR 的关系
│
├── 2.1.2 核心概念
│   ├── Dialect（方言）- 命名空间与语义边界
│   ├── Operation（操作）- 计算的基本单元
│   ├── Type（类型）- 张量/向量/自定义类型
│   └── Attribute（属性）- 编译时常量元数据
│
├── 2.1.3 MLIR 语法
│   ├── 文本格式（`.mlir` 文件）
│   ├── SSA 形式与值定义
│   └── Region / Block / CFG
│
└── 2.1.4 常用 Builtin Dialect
    ├── `arith` - 算术运算
    ├── `tensor` - 张量操作
    ├── `vector` - 向量操作
    ├── `scf` - 结构化控制流
    ├── `func` - 函数定义
    └── `memref` - 内存引用
```

**📝 对应笔记：** [notes/02-mlir-deep-dive.md](../notes/02-mlir-deep-dive.md) ✅  
**🔧 Lab1：** 手写简单 MLIR 优化 Pass（冗余算子消除）

---

### 2.2 MLIR Pass 机制 ⭐⭐⭐⭐⭐
> 课程第 4 讲：Pass 管理与模式匹配

```
├── 2.2.1 PassManager 工作原理
│   ├── Pass 注册与发现
│   ├── Pass 依赖分析
│   └── Pass 执行顺序
│
├── 2.2.2 Pattern Rewrite
│   ├── RewritePattern 定义
│   ├── GreedyPatternRewriteDriver
│   ├── 模式匹配与收益计算
│   └── 迭代重写直到定点
│
├── 2.2.3 Conversion Framework
│   ├── ConversionTarget
│   ├── Legalization（合法化）
│   └── Materialization（物化）
│
└── 2.2.4 调试与分析
    ├── Pass 打印 IR
    ├── 性能分析
    └── 验证器（Verifier）
```

**📝 对应笔记：** 待补充（现有笔记缺少 Pass 开发细节）  
**🔧 Lab1：** 基于 MLIR 框架编写冗余算子消除 Pass

---

### 2.3 编译器工具链架构 ⭐⭐⭐⭐
> 课程 MLIR 实战剖析章节

```
├── 2.3.1 模型转换全流程
│   ├── ONNX / PyTorch / TFLite → MLIR
│   ├── 前端 Dialect（TopDialect）
│   ├── 中端优化 Dialect
│   └── 后端硬件 Dialect（TpuDialect）
│
├── 2.3.2 Frontend：TopDialect
│   ├── TopDialect 定义
│   ├── 从 ONNX/Torch 转换到 Top
│   └── Top 相关 Pass 功能
│
├── 2.3.3 Backend：TpuDialect
│   ├── TpuDialect 设计
│   ├── TopToTpu Conversion
│   ├── TopToTosa Conversion
│   └── tpu-mlir Pass 详解
│
└── 2.3.4 LayerGroup 机制
    ├── LayerGroup 概念
    ├── 算子调度策略
    ├── 内存编排优化
    └── 片上存储管理
```

**📝 对应笔记：** 待补充（现有笔记缺少 tpu-mlir 实践）  
**🔧 Lab3：** 模型转换流程全跟踪（ONNX → MLIR → BModel）

---

## 三、量化与优化层（1-2 个月）

### 3.1 量化数学原理 ⭐⭐⭐⭐⭐
> 课程量化与性能优化章节

```
├── 3.1.1 量化基础
│   ├── 对称量化 vs 非对称量化
│   ├── Per-tensor vs Per-channel
│   ├── 缩放因子（Scale）与零点（Zero Point）
│   └── 量化误差分析
│
├── 3.1.2 INT8 量化
│   ├── 权重量化（Weight Quantization）
│   ├── 激活值量化（Activation Quantization）
│   └── W8A8 / W4A16 方案
│
├── 3.1.3 校准算法
│   ├── MinMax 校准
│   ├── KL 散度校准
│   ├── Percentile 校准
│   └── 自动寻找精度敏感层
│
└── 3.1.4 混合精度策略
    ├── 精度敏感层识别
    ├── 敏感层保留 FP16/FP32
    └── 精度回退分析与修复
```

**📝 对应笔记：** [notes/10-quantization-sparsity.md](../notes/10-quantization-sparsity.md) ✅  
**🔧 Lab4：** 量化精度对比与精度回退分析

---

### 3.2 性能分析与调优 ⭐⭐⭐⭐
> 课程 Profiling 部分

```
├── 3.2.1 Profiling 工具
│   ├── NPU Profiler 使用
│   ├── 算子耗时定位
│   └── 带宽利用率分析
│
├── 3.2.2 性能瓶颈识别
│   ├── Compute-bound vs Memory-bound
│   ├── 流水线气泡
│   └── 资源冲突
│
└── 3.2.3 优化策略
    ├── 算子融合减少内存访问
    ├── 数据布局优化（NCHW → NHWC）
    └── 并行指令调度
```

**📝 对应笔记：** 待补充（现有笔记缺少 Profiling 实践）

---

## 四、模型部署层（2-3 个月）

### 4.1 CV 模型部署 ⭐⭐⭐⭐
> 课程模型部署实战（CV）章节

```
├── 4.1.1 YOLOv8 模型分析
│   ├── Backbone / Neck / Head 结构
│   ├── 输入预处理（Letterbox）
│   └── 输出后处理（NMS）
│
├── 4.1.2 NPU 部署流程
│   ├── 模型转换（ONNX → BModel）
│   ├── 量化校准
│   └── 推理引擎集成
│
└── 4.1.3 Stream 并行化
    ├── 生产者 - 消费者模型
    ├── 预处理 / 推理 / 后处理并行
    └── 流水线优化
```

**📝 对应笔记：** 待补充（现有笔记缺少 CV 模型部署）  
**🔧 Lab5：** YOLOv8 Stream 实践

---

### 4.2 LLM 模型部署 ⭐⭐⭐⭐⭐
> 课程 LLM 与 Transformer 章节

```
├── 4.2.1 Transformer 架构深入
│   ├── Multi-Head Attention 原理
│   ├── Softmax 优化
│   └── Matmul 在 NPU 上的并行指令
│
├── 4.2.2 KV Cache 管理
│   ├── KV Cache 原理与显存瓶颈
│   ├── NPU 上的 KV 缓存复用
│   ├── PagedAttention 思想
│   └── Prefix Caching
│
├── 4.2.3 Qwen 模型量化部署
│   ├── W8A8 量化方案
│   ├── W4A16 量化方案
│   ├── 量化精度验证
│   └── NPU 平台实现
│
└── 4.2.4 低延迟推理应用
    ├── 流式输出
    ├── 并发请求处理
    └── 端到端延迟优化
```

**📝 对应笔记：**
- [notes/11-transformer-architecture-deep-dive.md](../notes/11-transformer-architecture-deep-dive.md) ✅
- [notes/12-llm-inference-kv-cache.md](../notes/12-llm-inference-kv-cache.md) ✅
- [notes/13-speculative-decoding-moe.md](../notes/13-speculative-decoding-moe.md) ✅
- [notes/14-inference-engines-comparison.md](../notes/14-inference-engines-comparison.md) ✅

**🔧 Lab6：** 部署一个 NPU 加速聊天机器人（Qwen3）

---

## 五、现有笔记覆盖情况

| 框架章节 | 对应笔记 | 状态 | 备注 |
|---------|---------|------|------|
| 1.2 AI 编译器概论 | notes/01 | ✅ | |
| 1.2 计算图优化 | notes/03 | ✅ | |
| 2.1 MLIR 基础 | notes/02 | ✅ | |
| 2.2 TVM 调度 | notes/04 | ✅ | 偏 TVM，需补充 MLIR Pass |
| 3.1 量化与稀疏 | notes/10 | ✅ | |
| 4.2 Transformer | notes/11 | ✅ | |
| 4.2 KV Cache | notes/12 | ✅ | |
| 4.2 推测解码/MoE | notes/13 | ✅ | |
| 4.2 推理引擎 | notes/14 | ✅ | |
| 1.1 C++ 基础 | - | ❌ | 待补充 |
| 1.3 NPU 架构 | - | ❌ | 待补充（现有偏 GPU） |
| 2.2 MLIR Pass 开发 | - | ❌ | 待补充 |
| 2.3 tpu-mlir 工具链 | - | ❌ | 待补充 |
| 3.2 Profiling | - | ❌ | 待补充 |
| 4.1 CV 模型部署 | - | ❌ | 待补充 |

---

## 六、学习路径建议

### 🎯 路径 A：MLIR 编译器开发（偏底层）
```
C++ 基础 → MLIR 基础 → Pass 开发 → tpu-mlir 工具链 → 贡献开源
  ↓          ↓           ↓            ↓              ↓
1.1       2.1        2.2          2.3           Lab1/3
```

### 🎯 路径 B：模型部署工程（偏应用）⭐ 推荐 Java 背景
```
计算图优化 → 量化原理 → CV 模型部署 → LLM 部署 → Stream 并行化
    ↓           ↓           ↓           ↓           ↓
  1.2        3.1        4.1        4.2        Lab5/6
```

### 🎯 路径 C：性能优化专家（偏调优）
```
NPU 架构 → CUDA/Triton → GEMM 优化 → Profiling → 混合精度
   ↓          ↓            ↓           ↓          ↓
 1.3       现有笔记     现有笔记      3.2        3.1
```

---

## 七、下一步行动

### 待补充笔记（优先级排序）

1. **C++ for AI 编译器** - 补充课程第 1 讲内容
2. **NPU 硬件架构** - 对比 GPU 架构差异
3. **MLIR Pass 开发实战** - 基于 Lab1 编写教程
4. **tpu-mlir 工具链剖析** - 基于课程第 3 部分
5. **Profiling 性能分析** - 工具使用与案例分析
6. **YOLOv8 NPU 部署** - CV 模型部署实践

### 待完成 Lab

- [ ] Lab1：手写 MLIR Pass
- [ ] Lab2：NPU 基础推理 Demo
- [ ] Lab3：模型转换全流程
- [ ] Lab4：量化精度对比
- [ ] Lab5：YOLOv8 Stream
- [ ] Lab6：Qwen3 NPU 聊天机器人

---

_此框架基于 14 篇现有笔记 + AI 编译器/NPU/MLIR 课程大纲整理_  
_持续更新中..._
