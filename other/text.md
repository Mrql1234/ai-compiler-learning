# AI 编译器 / NPU / MLIR 课程大纲

本课程以通用 NPU 硬件架构与 MLIR 编译器工具链实践（开源工业级框架）为核心平台，沿着“理论认知 → 工具掌握 → 技能进阶 → 项目实战”的主线，通过手写 MLIR 优化 Pass、模型转换全流程跟踪、INT8 量化精度对比、YOLOv8 Stream 实践，以及 Qwen 大语言模型 NPU 部署等六个由浅入深的 Lab 实战，系统性地打通从 AI 编译器底层原理到在真实边缘硬件上完成工业级模型高效部署的全链路工程能力。

## 课程安排

### AI 编译器基础

| 序号 | 课程子阶段 | 课程内容说明 | 课程时长 |
| --- | --- | --- | --- |
| 1 | AI 编译器中的 C/C++ | `SmallVector` / `StringRef` / `ArrayRef`，内存池技术 `BumpPtrAllocator`，RTTI（`isa` / `cast` / `dyn_cast`），CRTP（静态多态），Visitor，TableGen | 3 |
| 2 | AI 编译器概论（Graph vs Kernel） | 计算图表示、算子融合理论、AOT vs JIT 编译 | 2 |
| 3 | MLIR 基础概念：Dialect / Op / Type | MLIR 语法、中间表达格式、常用的 Builtin Dialect | 3 |
| 4 | MLIR Pass 管理与模式匹配 | PassManager 工作原理、GreedyPatternRewrite 机制 | 4 |
| 5 | Lab1：手写简单 MLIR 优化 Pass | 基于 MLIR 框架编写一个冗余算子消除的 Pass | 3 |

### 硬件与环境

| 序号 | 课程子阶段 | 课程内容说明 | 课程时长 |
| --- | --- | --- | --- |
| 1 | NPU 硬件架构与存储层次 | NPU 核心结构、计算单元与数据搬运（DMA）协同、局部存储空间管理 | 4 |
| 2 | WSL2 + Docker 开发环境搭建 | 配置 WSL2、Docker 映射、NPU 工具链安装 | 2 |
| 3 | Lab2：NPU 基础推理 Demo 运行 | 在 NPU 硬件平台上跑通 ResNet50 的 CPU / NPU 对比实验 | 3 |

### MLIR 实战剖析

| 序号 | 课程子阶段 | 课程内容说明 | 课程时长 |
| --- | --- | --- | --- |
| 1 | 编译器工具链架构与工作流概览 | 模型转换（Transform）与部署（Deploy）工具链内幕 | 3 |
| 2 | Frontend：TopDialect 功能及 Converter 介绍 | TopDialect 的定义及相关 Pass 的功能；如何从 ONNX / Torch / TFLite 转换到 TopDialect | 4 |
| 3 | Backend：TpuDialect 的设计、Conversion / Lowering 介绍 | `tpu-mlir` 里的 conversion：`TopToTpu` / `TopToTosa` 简介；`tpu-mlir` 中的 Pass 介绍 | 4 |
| 4 | `tpu-mlir` 中的 LayerGroup | LayerGroup 详解，基于 LayerGroup 进行算子调度 / 内存编排 | 4 |
| 5 | Lab3：模型转换流程全跟踪 | 完成从 ONNX -> MLIR -> BModel 的全手动转换过程 | 3 |

### 量化与性能优化

| 序号 | 课程子阶段 | 课程内容说明 | 课程时长 |
| --- | --- | --- | --- |
| 1 | INT8 量化数学原理与误差分析 | 对称与非对称量化、Per-channel 量化原理 | 3 |
| 2 | 校准算法（MinMax / KL / Percentile） | 使用校准集寻找最佳阈值、自动寻找精度敏感层 | 4 |
| 3 | Lab4：量化精度对比与精度回退 | 分析量化损失，使用混合精度提升模型表现 | 3 |
| 4 | 性能分析工具（Profiling）使用 | 使用 Profiling 工具定位算子耗时瓶颈与带宽利用率 | 2 |

### 模型部署实战（CV）

| 序号 | 课程子阶段 | 课程内容说明 | 课程时长 |
| --- | --- | --- | --- |
| 1 | YOLOv8 目标检测模型 NPU 部署 | 在 NPU 硬件平台上部署 YOLOv8 网络，实现 YOLOv8 的预处理（letterbox）与后处理 | 4 |
| 2 | Lab5：YOLOv8 Stream 实践 | 构建生产者-消费者模型，实现预处理、模型推理与后处理的并行化执行 | 4 |

### LLM 与 Transformer

| 序号 | 课程子阶段 | 课程内容说明 | 课程时长 |
| --- | --- | --- | --- |
| 1 | Attention 算子部署和优化 | Transformer 原理，MHA 部署，Softmax / Matmul 在 NPU 上的并行指令优化 | 4 |
| 2 | KV Cache NPU 管理机制 | 解决 LLM 推理过程中的显存瓶颈与 KV 缓存复用 | 4 |
| 3 | Qwen3 模型量化与部署 | W8A8 或 W4A16 量化方案在 NPU 平台上的实现 | 6 |
| 4 | Lab6：部署一个 NPU 加速聊天机器人 | 实现基于 Qwen3 的低延迟 NPU 推理应用 | 4 |
