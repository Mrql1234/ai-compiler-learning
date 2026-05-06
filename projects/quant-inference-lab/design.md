# 设计说明：Quant Inference Lab

## 总览

这个项目刻意把两个经常分开学习的话题放在一起：

- **模型侧效率问题**
  - 量化
  - 权重表示
  - 压缩后参考执行
- **运行时侧效率问题**
  - 调度
  - 批处理
  - KV-cache 内存管理

这种拆分方式和真实系统很像：

```text
模型 / 编译器侧
  FP32 模型
  -> observer / calibration
  -> quantized weights / activations
  -> 低精度执行路径

运行时 / 推理引擎侧
  requests
  -> scheduler
  -> KV-cache allocator
  -> decode loop
  -> latency / throughput behavior
```

## 为什么把两者放在一起

来自笔记 10 的关键信息：

- 低精度会降低模型大小和带宽压力
- per-channel weight quantization 很重要
- calibration 和 scale 选择会直接影响误差

来自笔记 14 的关键信息：

- 推理系统的性能很大程度取决于内存管理和 batching
- KV-cache 和调度是第一类瓶颈
- 推理引擎的性能不只是 kernel 快不快

因此，这个项目希望让你建立两个基本判断：

- 量化本质上是一个**表示与数值近似问题**
- 推理引擎本质上是一个**运行时资源管理问题**

## 模块设计

### 1. `quant/`

#### `observer.py`

- 收集最小值 / 最大值统计
- 负责计算量化参数
- 支持：
  - 对称 vs 非对称
  - per-tensor vs per-channel

#### `quantizer.py`

- 把浮点 tensor 转成量化 tensor
- 保存：
  - 整数数据
  - scale
  - zero point
  - axis 信息
- 支持反量化，用于和 reference 对照

#### `int8_linear.py`

- 提供参考版 `linear`：
  - FP32 reference 路径
  - weight-only quantized 路径
  - input + weight quantized 路径
- 故意保持实现直白，方便阅读

### 2. `engine/`

#### `request.py`

- 推理请求的数据模型
- 跟踪：
  - prompt 长度
  - 目标生成长度
  - 当前进度
  - TTFT step
  - completion step

#### `kv_cache.py`

- 模拟基于 page 的 KV 分配
- page 数量是核心资源
- 不实现真实 attention kernel，但表达 paged attention 的资源约束

#### `scheduler.py`

- 模拟 decode loop，包括：
  - waiting queue
  - active set
  - 每 step token 生成
  - page 分配与释放
- 输出运行摘要指标

## 当前简化假设

- token 只用数量表示，不表示真实 token ID
- KV-cache 用逻辑 page 数表示，不表示真实字节数
- 不做真实 transformer 数学计算
- 不接 CUDA 或 Triton 执行
- 不做模型解析

这些简化是刻意保留的。当前学习目标是理解系统行为，而不是先陷入大框架细节。

## 是否需要 GPU

当前版本**不需要 GPU 云服务器**。

原因很简单：

- 量化部分目前是数值表示实验
- 推理引擎部分目前是调度与资源模拟
- 两者都能在 CPU 上完整展示核心概念

只有在以下阶段，GPU 才会变得必要：

1. 做真实量化模型 benchmark
2. 接 TensorRT-LLM / vLLM 等后端
3. 比较不同 kernel 或不同精度的真实性能
4. 引入长上下文、高并发、真实模型加载

## 后续建议演进方向

1. 增加 activation calibration 数据集工具
2. 增加 AWQ 风格的 group quantization 实验
3. 在 scheduler 中显式拆分 prefill 与 decode 预算
4. 增加一个策略层，模拟不同引擎选择：
   - `quantized_cpu`
   - `generic_gpu`
   - `high_throughput_engine`
5. 后续再接真实后端，例如 TensorRT-LLM 或 vLLM
