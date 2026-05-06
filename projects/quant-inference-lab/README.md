# Quant Inference Lab

`Quant Inference Lab` 是一个教学型小项目，用来把两个经常分开学习的话题放到同一个最小实验里：

- 量化基础与 PTQ 风格流程
- 推理引擎运行时机制，例如 continuous batching 和 paged KV cache

这个项目的目标不是实现生产级推理服务，而是把 [notes/10-quantization-sparsity.md](/home/ql/code/ai-compiler-learning/notes/10-quantization-sparsity.md:1) 和 [notes/14-inference-engines-comparison.md](/home/ql/code/ai-compiler-learning/notes/14-inference-engines-comparison.md:1) 里的关键思想落成可执行、可改动、可观察的代码。

## 项目范围

当前重点：

- 基于 min-max observer 的量化
- 对称 / 非对称量化
- per-tensor / per-channel INT8 权重量化
- INT8 `linear` 参考执行路径
- 推理请求调度模拟
- paged KV-cache 分配模拟
- continuous batching 风格 decode 循环

第一版暂不覆盖：

- 完整 LLM 模型加载
- 真实 CUDA kernel
- TensorRT / vLLM 直接集成
- 分布式推理

## 目录结构

```text
projects/quant-inference-lab/
  quant/
    observer.py
    quantizer.py
    int8_linear.py
  engine/
    request.py
    kv_cache.py
    scheduler.py
  examples/
    ptq_demo.py
    engine_demo.py
  tests/
    test_quantizer.py
    test_scheduler.py
  requirements.md
  design.md
  tasks.md
  qa.md
```

## 快速开始

```bash
cd projects/quant-inference-lab
python3 -m examples.ptq_demo
python3 -m examples.engine_demo
python3 -m unittest discover -s tests
```

如果你想直接复用仓库里现有的 Python 虚拟环境：

```bash
/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/.venv/bin/python3.10 -m examples.ptq_demo
```

## 你会学到什么

`examples/ptq_demo.py` 展示：

- observer 如何收集数值范围
- scale / zero point 如何计算
- INT8 `linear` 输出和 FP32 reference 的差异
- 为什么 per-channel weight quantization 通常更稳

`examples/engine_demo.py` 展示：

- 请求如何进入 waiting queue
- KV page 如何分配和回收
- decode scheduler 如何混合处理多个请求
- page 预算和 batch 预算如何影响 TTFT 与整体完成时间

## 运行环境

当前项目**不需要 GPU 云服务器**。

第一版的代码和示例都是：

- CPU 可运行
- 依赖很少
- 重点在机制理解，不在真实 GPU 加速

如果后续你要扩展到下面这些方向，才建议迁移到 GPU 机器：

- 真实量化模型 benchmark
- TensorRT-LLM / vLLM 接入
- CUDA / Triton kernel 实验
- 长上下文或高并发真实推理服务

## 建议学习顺序

1. 先读 `requirements.md`
2. 再读 `design.md`
3. 跑 `examples/ptq_demo.py`
4. 跑 `examples/engine_demo.py`
5. 读 `qa.md`
6. 修改 `tests/` 和示例参数，观察行为变化
