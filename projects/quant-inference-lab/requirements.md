# 需求说明：Quant Inference Lab

## 项目目标

构建一个小型教学实验项目，把下面两类知识串起来：

- [notes/10-quantization-sparsity.md](/home/ql/code/ai-compiler-learning/notes/10-quantization-sparsity.md:1) 中的量化概念
- [notes/14-inference-engines-comparison.md](/home/ql/code/ai-compiler-learning/notes/14-inference-engines-comparison.md:1) 中的推理引擎概念

项目希望帮助学习者同时理解两件事：

1. 为什么量化能提升部署效率
2. 为什么即便模型已经压缩，vLLM / TensorRT-LLM 这类运行时系统仍然重要

## 功能需求

### 量化部分

- 提供 min-max observer
- 支持对称量化和非对称量化
- 支持 per-tensor 和 per-channel 量化
- 提供 INT8 `linear` 参考执行路径
- 能报告量化输出相对 FP32 reference 的误差

### 推理引擎部分

- 建模推理请求，包括 prompt 长度和生成长度
- 模拟 paged KV-cache 分配
- 模拟 continuous batching 和 decode step
- 收集每个请求的指标，例如 TTFT 和完成 step
- 提供整体运行摘要指标

## 非功能需求

- 以教学性和可读性优先
- 保持依赖最小
- 只用 CPU 也能运行
- 测试行为尽量稳定
- 代码规模足够小，可以在较短时间内读完

## 依赖

- 必需：
  - `numpy`
- 后续可选：
  - `torch`
  - `matplotlib`

## 交付物

- 可运行示例
- 单元测试
- 设计文档和任务文档
- 按 `quant/` 与 `engine/` 组织的代码结构
