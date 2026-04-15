# 14 - 推理引擎实战：vLLM 与 TensorRT-LLM 深度对比

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 3 - LLM 推理优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU（显存 24GB+）  
> 🔗 前置：[12-LLM 推理优化](./12-llm-inference-kv-cache.md), [13-高级推理优化](./13-speculative-decoding-moe.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 对比 **vLLM、TensorRT-LLM、TGI、SGLang** 等主流推理引擎
2. 理解各引擎的**核心优化技术**和**适用场景**
3. 能根据需求**选择合适的推理引擎**
4. 掌握**生产环境部署**的最佳实践
5. 能进行**性能调优和故障排查**

---

## 💡 推理引擎全景图

### 主流推理引擎对比

| 引擎 | 开发方 | 核心优化 | 易用性 | 性能 | 生态 |
|------|--------|---------|--------|------|------|
| **vLLM** | UC Berkeley | PagedAttention | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **TensorRT-LLM** | NVIDIA | Kernel 优化 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TGI** | HuggingFace | Continuous Batching | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **SGLang** | 清华 + 字节 | 程序引导解码 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **LMDeploy** | 商汤 | 量化 + 批处理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **DeepSpeed-MII** | Microsoft | ZeRO-Inference | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 选择指南

```
场景 1：快速原型 / 研究
  → vLLM (最简单，性能好)

场景 2：生产部署 / 企业级
  → TensorRT-LLM (最优化，NVIDIA 官方支持)

场景 3：HuggingFace 生态
  → TGI (无缝集成 HF 模型)

场景 4：复杂推理 / 程序引导
  → SGLang (支持复杂解码逻辑)

场景 5：量化部署 / 边缘设备
  → LMDeploy / TensorRT-LLM (INT4/INT8 支持好)
```

---

## 🔥 vLLM 深度实践

### 架构回顾

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM 架构                            │
├─────────────────────────────────────────────────────────┤
│  API Server (OpenAI 兼容)                               │
│       ↓                                                 │
│  Scheduler                                              │
│  - PagedAttention 显存管理                              │
│  - Continuous Batching                                  │
│  - 请求优先级调度                                       │
│       ↓                                                 │
│  Worker (GPU 执行)                                      │
│  - 模型执行                                             │
│  - KV Cache 管理                                        │
│       ↓                                                 │
│  GPU Memory                                             │
│  - 模型权重 (静态)                                      │
│  - KV Cache 池 (动态分页)                               │
│  - 激活值 (临时)                                        │
└─────────────────────────────────────────────────────────┘
```

### 完整部署示例

```bash
# ===== 1. 安装 vLLM =====
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"

# ===== 2. 启动服务（单卡）=====
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --quantization none

# ===== 3. 启动服务（多卡 Tensor Parallel）=====
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-num-seqs 256 \
    --max-model-len 4096

# ===== 4. 启动服务（量化 INT4 AWQ）=====
python -m vllm.entrypoints.api_server \
    --model TheBloke/Llama-2-70B-AWQ \
    --quantization awq \
    --tensor-parallel-size 4 \
    --max-num-seqs 256

# ===== 5. 启动服务（启用 Speculative Decoding）=====
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --speculative-model meta-llama/Llama-2-7b-hf \
    --num-speculative-tokens 5 \
    --tensor-parallel-size 8
```

### 关键参数详解

```bash
--model <model_name_or_path>
  模型名称或本地路径
  支持：HF Hub 名称、本地路径、GCS/S3 路径

--tensor-parallel-size <N>
  Tensor Parallel 大小（GPU 数量）
  推荐：7B 用 1 卡，13B 用 1-2 卡，70B 用 4-8 卡

--max-num-seqs <N>
  最大并发序列数（影响显存和吞吐）
  默认：256
  调优：显存充足可增大，延迟敏感可减小

--max-model-len <N>
  最大序列长度（包括 prompt + 生成）
  默认：模型的最大位置嵌入
  调优：减小可节省显存

--gpu-memory-utilization <0.0-1.0>
  GPU 显存使用比例
  默认：0.9
  调优：留 10-20% 给系统，避免 OOM

--dtype <auto|float16|bfloat16|float32>
  计算精度
  推荐：auto (自动选择) 或 bfloat16 (A100/H100)

--quantization <none|awq|gptq|squeezellm>
  量化方法
  推荐：awq (4-bit) 或 none (FP16)

--enforce-eager
  强制使用 eager 模式（不用 CUDA Graph）
  调试时用，生产环境不要加
```

### 性能调优

```bash
# 场景 1：低延迟优先
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-num-seqs 64 \        # 减小并发
    --max-model-len 2048 \     # 限制长度
    --gpu-memory-utilization 0.8

# 场景 2：高吞吐优先
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-num-seqs 512 \       # 增大并发
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95

# 场景 3：长上下文优先
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --max-model-len 16384 \    # 支持 16K 上下文
    --max-num-seqs 32 \        # 减小并发
    --gpu-memory-utilization 0.9
```

### 监控和指标

```bash
# 启用 Prometheus 指标
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --prometheus-port 9090

# 查看指标
curl http://localhost:9090/metrics

# 关键指标：
# vllm:num_requests_running - 正在处理的请求数
# vllm:num_requests_waiting - 等待的请求数
# vllm:gpu_cache_usage_perc - KV Cache 使用率
# vllm:time_to_first_token_seconds - 首 token 延迟
# vllm:time_per_output_token_seconds - 每 token 延迟
```

---

## 🔥 TensorRT-LLM 深度实践

### 核心优势

```
TensorRT-LLM vs vLLM:

TensorRT-LLM 优势：
  - NVIDIA 官方支持
  - 更深的 Kernel 优化
  - 支持更多硬件特性（FP8、稀疏化）
  - 生产级稳定性

vLLM 优势：
  - 更简单易用
  - 快速支持新模型
  - 更好的社区生态
  - OpenAI 兼容 API
```

### 完整部署流程

```bash
# ===== 1. 安装 TensorRT-LLM =====
# 方法 1：Docker（推荐）
docker pull nvcr.io/nvidia/tensorrt-llm/llm-gpu-benchmark
docker run --gpus all -it nvcr.io/nvidia/tensorrt-llm/llm-gpu-benchmark

# 方法 2：源码安装
pip install tensorrt-llm

# ===== 2. 构建模型引擎 =====
# 以 LLaMA-2-7B 为例
cd tensorrt-llm/examples/llama

# 下载模型
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

# 构建 TensorRT 引擎
python build.py \
    --model_dir Llama-2-7b-hf \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --use_weight_only false \
    --world_size 1 \
    --tp_size 1 \
    --output_dir llama2-7b-engine \
    --max_batch_size 32 \
    --max_input_len 4096 \
    --max_output_len 1024

# ===== 3. 运行推理 =====
python run.py \
    --max_output_len 100 \
    --input_text "Hello, my name is" \
    --engine_dir llama2-7b-engine \
    --tokenizer_dir Llama-2-7b-hf

# ===== 4. 部署为服务 =====
# TensorRT-LLM 提供 Triton Inference Server 集成
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/llama2-7b-engine:/model \
    nvcr.io/nvidia/tritonserver:23.12-py3 \
    tritonserver --model-repository=/model
```

### 量化支持

```bash
# INT8 量化（PTQ）
python build.py \
    --model_dir Llama-2-7b-hf \
    --dtype float16 \
    --use_int8 \
    --calib_dataset c4 \
    --calib_size 512 \
    --output_dir llama2-7b-int8-engine

# INT4 量化（AWQ）
python build.py \
    --model_dir TheBloke/Llama-2-7b-AWQ \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int4 \
    --output_dir llama2-7b-int4-engine

# FP8 量化（H100 专属）
python build.py \
    --model_dir Llama-2-7b-hf \
    --dtype float8 \
    --use_fp8 \
    --output_dir llama2-7b-fp8-engine
```

### 性能对比（A100）

| 模型 | 引擎 | 精度 | 吞吐 (tokens/s) | 延迟 (ms/token) |
|------|------|------|----------------|-----------------|
| LLaMA-7B | vLLM | FP16 | 150 | 6.7 |
| LLaMA-7B | TRT-LLM | FP16 | 180 | 5.5 |
| LLaMA-7B | TRT-LLM | INT8 | 280 | 3.6 |
| LLaMA-7B | TRT-LLM | INT4 | 350 | 2.9 |
| LLaMA-70B | vLLM (8 卡) | FP16 | 45 | 22 |
| LLaMA-70B | TRT-LLM (8 卡) | FP16 | 55 | 18 |
| LLaMA-70B | TRT-LLM (8 卡) | INT4 | 90 | 11 |

---

## 🔥 TGI（Text Generation Inference）

### 快速部署

```bash
# ===== 1. Docker 部署（推荐）=====
docker run --gpus all \
    -p 8080:80 \
    -v /data:/data \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 1 \
    --max-input-length 4096 \
    --max-total-tokens 4096 \
    --max-batch-size 256

# ===== 2. 多卡部署 =====
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id meta-llama/Llama-2-70b-hf \
    --num-shard 8 \
    --max-input-length 4096 \
    --max-total-tokens 4096

# ===== 3. 量化部署 =====
docker run --gpus all \
    -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:2.0 \
    --model-id TheBloke/Llama-2-70B-AWQ \
    --quantize awq \
    --num-shard 4
```

### TGI 特性

```
TGI 核心特性：
  - HuggingFace 原生集成
  - Continuous Batching
  - Tensor Parallelism
  - 量化支持（AWQ、GPTQ、bitsandbytes）
  - Prometheus 监控
  - OpenTelemetry 追踪

优势：
  - 与 HF Hub 无缝集成
  - 支持模型最多
  - 企业级支持

劣势：
  - 性能略低于 vLLM 和 TRT-LLM
  - 配置相对复杂
```

---

## 🔥 SGLang：程序引导解码

### 什么是 SGLang？

```
SGLang = Structured Generation Language

核心思想：
  - 用编程语言的方式描述解码逻辑
  - 支持复杂的多轮对话、函数调用、JSON 输出
  - 优化解码过程的显存和计算

适用场景：
  - 复杂推理任务
  - 多轮对话系统
  - 结构化输出（JSON、XML）
  - 函数调用 / Agent 场景
```

### SGLang 示例

```python
# sglang_example.py
import sglang as sgl

# 定义结构化生成
@sgl.function
def multi_turn_conversation(s, question1, question2):
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen("answer1"))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2"))

# 运行
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")
sgl.set_default_runtime(runtime)

state = multi_turn_conversation.run(
    question1="What is the capital of France?",
    question2="What is its population?"
)

print(state["answer1"])
print(state["answer2"])

# 结构化输出（JSON）
@sgl.function
def extract_json(s, text):
    s += "Extract information from the following text as JSON:\n\n"
    s += text + "\n\n"
    s += "JSON output:\n"
    s += sgl.gen("json_output", max_tokens=500, regex=r'\{.*\}')

# 用正则约束输出格式
state = extract_json.run(text="Paris is the capital of France with 2.1M people.")
print(state["json_output"])
```

### SGLang 性能

```
SGLang vs vLLM:

简单生成任务：
  - SGLang: 与 vLLM 相当
  - vLLM: 略简单

复杂结构化任务：
  - SGLang: 2-3x 快（优化了解码流程）
  - vLLM: 需要多次请求

SGLang 独特优势：
  - 正则约束解码
  - 多轮对话优化
  - 并行采样优化
```

---

## 📊 生产环境部署指南

### 部署 Checklist

```
□ 选择合适的推理引擎（vLLM / TRT-LLM / TGI）
□ 确定模型精度（FP16 / INT8 / INT4）
□ 配置 Tensor Parallel 大小
□ 设置合理的 max_num_seqs 和 max_model_len
□ 配置显存使用比例（0.8-0.95）
□ 启用监控指标（Prometheus）
□ 配置日志和告警
□ 压力测试（找到最大吞吐）
□ 配置自动扩缩容
□ 备份和恢复方案
```

### 性能基准测试脚本

```python
# benchmark_inference.py
import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def send_request(prompt, max_tokens=100, url="http://localhost:8000"):
    """发送单个请求"""
    start = time.time()
    
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "meta-llama/Llama-2-7b-hf",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
    )
    
    end = time.time()
    result = response.json()
    
    return {
        "total_time": end - start,
        "prompt_tokens": result.get('usage', {}).get('prompt_tokens', 0),
        "completion_tokens": result.get('usage', {}).get('completion_tokens', 0),
        "total_tokens": result.get('usage', {}).get('total_tokens', 0)
    }

def benchmark(concurrency=32, num_requests=100, url="http://localhost:8000"):
    """性能基准测试"""
    prompts = [f"Question {i}: Explain the concept of machine learning." 
               for i in range(num_requests)]
    
    print(f"开始基准测试：{num_requests} 请求，并发度 {concurrency}")
    
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda p: send_request(p), prompts))
    
    end = time.time()
    
    # 统计
    total_time = end - start
    total_tokens = sum(r['total_tokens'] for r in results)
    throughput = total_tokens / total_time
    
    latencies = [r['total_time'] for r in results]
    p50 = statistics.median(latencies)
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f"\n===== 基准测试结果 =====")
    print(f"总请求数：{num_requests}")
    print(f"总时间：{total_time:.2f}s")
    print(f"总 tokens: {total_tokens}")
    print(f"吞吐：{throughput:.2f} tokens/s")
    print(f"平均延迟：{statistics.mean(latencies)*1000:.2f}ms")
    print(f"P50 延迟：{p50*1000:.2f}ms")
    print(f"P90 延迟：{p90*1000:.2f}ms")
    print(f"P99 延迟：{p99*1000:.2f}ms")
    
    return {
        "throughput": throughput,
        "p50": p50,
        "p90": p90,
        "p99": p99
    }

# 运行基准测试
if __name__ == "__main__":
    # 测试不同并发度
    for concurrency in [1, 8, 16, 32, 64]:
        print(f"\n{'='*50}")
        print(f"测试并发度：{concurrency}")
        print(f"{'='*50}")
        benchmark(concurrency=concurrency, num_requests=50)
```

### 故障排查

```bash
# 问题 1：OOM（显存不足）
# 解决：减小 max-num-seqs 或 max-model-len
#      降低 gpu-memory-utilization

# 问题 2：请求排队过长
# 解决：增大 max-num-seqs
#      增加 GPU 数量
#      启用 Speculative Decoding

# 问题 3：延迟过高
# 解决：减小并发度
#      使用量化
#      检查网络延迟

# 问题 4：吞吐过低
# 解决：增大并发度
#      使用 TensorRT-LLM
#      启用量化

# 查看日志
docker logs <container_id>

# 查看 GPU 状态
watch -n 1 nvidia-smi

# 查看 vLLM 指标
curl http://localhost:9090/metrics | grep vllm
```

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 部署 vLLM 服务，跑通第一个请求
- [ ] 对比 vLLM 和 PyTorch 原生的性能
- [ ] 尝试不同的并发配置
- [ ] 在笔记里记录性能数据

### 选做（深入）

- [ ] 部署 TensorRT-LLM 引擎
- [ ] 尝试量化部署（INT4/INT8）
- [ ] 用 SGLang 实现结构化输出
- [ ] 阅读 vLLM/TensorRT-LLM 源码

---

## 📚 参考资料

- **推理引擎**：
  - vLLM: https://github.com/vllm-project/vllm
  - TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
  - TGI: https://github.com/huggingface/text-generation-inference
  - SGLang: https://github.com/sgl-project/sglang

- **部署文档**：
  - vLLM 文档：https://docs.vllm.ai/
  - TensorRT-LLM 文档：https://nvidia.github.io/TensorRT-LLM/
  - TGI 文档：https://huggingface.co/docs/text-generation-inference

- **性能优化**：
  - NVIDIA LLM 部署指南：https://docs.nvidia.com/deeplearning/triton-inference-server/
  - vLLM 性能调优：https://docs.vllm.ai/en/latest/performance/

---

_笔记创建：2026-04-15_  
_适合人群：想部署生产级 LLM 推理服务的开发者_  
_平台：Linux/Windows + NVIDIA GPU（推荐 A100/H100/RTX 4090）_  
_难度：⭐⭐⭐⭐（需要理解推理引擎和分布式系统）_
