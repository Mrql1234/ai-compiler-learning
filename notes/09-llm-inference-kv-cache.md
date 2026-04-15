# 09 - LLM 推理优化：KV Cache 与 Continuous Batching

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 3 - LLM 推理优化  
> ⏱️ 预计耗时：3-4 周  
> 💻 平台：Linux/Windows + NVIDIA GPU（显存 16GB+）  
> 🔗 前置：[07-算子融合](./07-operator-fusion.md), [08-量化与稀疏化](./08-quantization-sparsity.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **LLM 推理的特殊挑战**（显存、延迟、吞吐）
2. 掌握 **KV Cache 原理和实现**
3. 理解 **PagedAttention**（vLLM 核心技术）
4. 掌握 **Continuous Batching** 调度策略
5. 能部署和优化 **vLLM / TGI** 推理服务

---

## 💡 LLM 推理的特殊挑战

### 自回归生成的本质

**LLM 推理 = 重复执行以下步骤**：

```
输入："Hello, how are"
  ↓
模型预测下一个 token：" you"
  ↓
追加到输入："Hello, how are you"
  ↓
模型预测下一个 token："?"
  ↓
... 重复直到生成 EOS 或达到最大长度
```

**问题**：每次生成都要重新计算所有之前的 token！

```
第 1 步：计算 token 1
  - Q1, K1, V1 → Attention → Output1

第 2 步：计算 token 1, 2
  - Q1, K1, V1 → Attention → Output1  ← 重复计算！
  - Q2, K2, V2 → Attention → Output2

第 3 步：计算 token 1, 2, 3
  - Q1, K1, V1 → Attention → Output1  ← 重复计算！
  - Q2, K2, V2 → Attention → Output2  ← 重复计算！
  - Q3, K3, V3 → Attention → Output3
```

### KV Cache：避免重复计算

**核心思想**：缓存之前计算的 K 和 V，只计算新的 token。

```
第 1 步：
  - 计算 Q1, K1, V1
  - Attention(Q1, K1, V1) → Output1
  - 缓存：KV_Cache = [K1, V1]

第 2 步：
  - 计算 Q2, K2, V2（只计算新 token）
  - Attention(Q2, [K1, K2], [V1, V2]) → Output2  ← 复用缓存！
  - 缓存：KV_Cache = [K1, V1, K2, V2]

第 3 步：
  - 计算 Q3, K3, V3（只计算新 token）
  - Attention(Q3, [K1, K2, K3], [V1, V2, V3]) → Output3
  - 缓存：KV_Cache = [K1, V1, K2, V2, K3, V3]
```

**收益**：
- 计算量从 O(N²) 降到 O(N)
- 推理延迟降低 5-10x

### 📊 KV Cache 显存占用分析

**公式**：
```
KV Cache 大小 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × bytes_per_param
```

**示例**：LLaMA-7B，batch_size=1，seq_len=4096

```
LLaMA-7B 规格：
  - num_layers = 32
  - num_heads = 32
  - head_dim = 128
  - 精度：FP16 (2 bytes)

KV Cache 大小：
  = 2 × 32 × 32 × 128 × 4096 × 1 × 2 bytes
  = 2 × 32 × 32 × 128 × 4096 × 2
  = 2,147,483,648 bytes
  = 2 GB

如果 batch_size=32：
  = 2 GB × 32 = 64 GB  ← 显存爆炸！
```

**关键洞察**：
- KV Cache 是 LLM 推理的**主要显存瓶颈**
- 长序列 + 大批次 = 显存不足
- 需要**高效的显存管理**

---

## 📚 KV Cache 实现详解

### Attention 机制回顾

**标准 Attention**：

```python
def attention(Q, K, V, mask=None):
    # Q, K, V: (batch, seq_len, num_heads, head_dim)
    
    # 1. 计算 Attention 分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, N, N)
    
    # 2. Scale
    dim = Q.shape[-1]
    scores = scores / (dim ** 0.5)
    
    # 3. 应用 mask（防止看到未来）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 5. 应用 Attention
    output = torch.matmul(attn_weights, V)  # (B, H, N, D)
    
    return output
```

### 带 KV Cache 的 Attention

```python
class KVCacheAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        
        # KV Cache（预分配显存）
        self.register_buffer('kv_cache', None)
    
    def forward(self, x, position_ids, kv_cache=None):
        """
        Args:
            x: 输入 (batch, seq_len, hidden_dim)
            position_ids: 位置 ID (batch, seq_len)
            kv_cache: 之前的 KV 缓存 [(K, V), ...] per layer
        
        Returns:
            output: 输出 (batch, seq_len, hidden_dim)
            new_kv_cache: 更新后的 KV 缓存
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. 计算 Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 2. 如果有缓存，拼接到之前的 K/V
        if kv_cache is not None:
            prev_k, prev_v = kv_cache  # (batch, prev_seq_len, num_heads, head_dim)
            k = torch.cat([prev_k, k], dim=1)  # 拼接 K
            v = torch.cat([prev_v, v], dim=1)  # 拼接 V
        
        # 3. 更新缓存
        new_kv_cache = (k, v)
        
        # 4. 计算 Attention
        # Q: (B, N, H, D), K: (B, N+prev, H, D), V: (B, N+prev, H, D)
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)  # (B, H, N+prev, D)
        v = v.transpose(1, 2)  # (B, H, N+prev, D)
        
        # Attention 分数
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N, N+prev)
        scores = scores / (self.head_dim ** 0.5)
        
        # Causal mask（只看之前的 token）
        mask = torch.tril(torch.ones(seq_len, k.shape[2], device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 应用 Attention
        output = torch.matmul(attn_weights, v)  # (B, H, N, D)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        
        return self.o_proj(output), new_kv_cache

# 使用示例
config = type('Config', (), {
    'num_heads': 32,
    'head_dim': 128,
    'max_seq_len': 4096
})()

attention = KVCacheAttention(config)

# 第 1 步：prefill 阶段（处理输入 prompt）
x1 = torch.randn(1, 10, 4096)  # batch=1, seq_len=10
position_ids1 = torch.arange(10).unsqueeze(0)
output1, kv_cache1 = attention(x1, position_ids1, kv_cache=None)

# 第 2 步：decode 阶段（生成新 token）
x2 = torch.randn(1, 1, 4096)  # batch=1, seq_len=1（只生成 1 个 token）
position_ids2 = torch.tensor([[10]])  # 位置 ID 接续
output2, kv_cache2 = attention(x2, position_ids2, kv_cache=kv_cache1)

# 第 3 步：继续生成
x3 = torch.randn(1, 1, 4096)
position_ids3 = torch.tensor([[11]])
output3, kv_cache3 = attention(x3, position_ids3, kv_cache=kv_cache2)
```

### Prefill vs Decode 阶段

```
LLM 推理分为两个阶段：

1. Prefill 阶段（处理输入 prompt）
   - 输入：整个 prompt（可能几百个 token）
   - 计算：并行处理所有 token
   - 瓶颈：计算密集（需要快速处理完 prompt）
   - KV Cache：构建初始缓存

2. Decode 阶段（生成新 token）
   - 输入：每次 1 个 token
   - 计算：自回归，串行生成
   - 瓶颈：内存密集（每次只计算 1 个 token，但要用全部 KV Cache）
   - KV Cache：不断追加新 token 的 K/V
```

**性能对比**：

| 阶段 | 输入长度 | 计算类型 | 瓶颈 | 优化方向 |
|------|---------|---------|------|---------|
| Prefill | 长（100-1000） | 并行 | 计算 | Tensor Core、算子融合 |
| Decode | 短（1） | 串行 | 内存 | KV Cache 优化、显存管理 |

---

## 🔥 PagedAttention：vLLM 的核心技术

### 传统 KV Cache 的问题

**问题**：传统方法**预分配连续显存**，导致严重浪费。

```
传统方法：
  - 为每个请求预分配 max_seq_len 的 KV Cache
  - 实际使用：平均只有 30-50%
  - 浪费：50-70% 显存

示例：
  max_seq_len = 4096
  实际平均 seq_len = 1500
  显存利用率 = 1500 / 4096 = 37%
  浪费 = 63%
```

**碎片化问题**：
```
请求 1：实际长度 500，占用 4096 空间 → 浪费 3596
请求 2：实际长度 2000，占用 4096 空间 → 浪费 2096
请求 3：实际长度 3000，但只剩 2000 空间 → OOM（显存不足）

总浪费：5692 tokens 空间
总可用：12288 tokens 空间
利用率：46%
```

### PagedAttention 的核心思想

**灵感**：操作系统的**虚拟内存分页**。

```
操作系统虚拟内存：
  - 虚拟地址 → 物理地址（页表映射）
  - 按需分配物理页
  - 支持非连续物理内存

PagedAttention：
  - KV Cache 逻辑块 → 物理显存块（页表映射）
  - 按需分配物理块
  - 支持非连续物理显存
```

**实现**：

```
逻辑 KV Cache（连续视图）：
  Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | ...

物理显存（实际分配）：
  Block 2 → 物理页 0x1000
  Block 0 → 物理页 0x2000
  Block 4 → 物理页 0x3000
  Block 1 → 物理页 0x4000
  Block 3 → 物理页 0x5000

页表（Block Table）：
  逻辑块 ID → 物理页地址
  0 → 0x2000
  1 → 0x4000
  2 → 0x1000
  3 → 0x5000
  4 → 0x3000
```

**收益**：
- 显存利用率：**80-95%**（vs 传统 30-50%）
- 支持更多并发请求：**2-4x 吞吐提升**
- 无碎片化问题

### vLLM 架构

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM 架构                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  API Server (HTTP/OpenAI 兼容)                          │
│       ↓                                                 │
│  Scheduler (调度器)                                     │
│  - 管理请求队列                                         │
│  - Continuous Batching                                  │
│  - 显存管理                                             │
│       ↓                                                 │
│  Worker (GPU 执行)                                      │
│  - PagedAttention Kernel                                │
│  - KV Cache 管理                                        │
│  - 模型执行                                             │
│       ↓                                                 │
│  GPU Memory                                             │
│  - 模型权重（静态）                                     │
│  - KV Cache 池（动态分配）                              │
│  - 激活值（临时）                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 部署 vLLM

```bash
# 1. 安装 vLLM
pip install vllm

# 2. 启动服务（OpenAI 兼容 API）
python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --max-model-len 4096

# 3. 测试（OpenAI 兼容）
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-7B-v0.1",
        "prompt": "Hello, my name is",
        "max_tokens": 100,
        "temperature": 0.7
    }'

# 4. 或用 Python SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx"  # 任意字符串
)

response = client.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    prompt="Hello, my name is",
    max_tokens=100
)

print(response.choices[0].text)
```

### vLLM 性能对比

| 框架 | 吞吐 (tokens/s) | 显存利用率 | P99 延迟 |
|------|----------------|-----------|---------|
| 原生 PyTorch | 100 | 35% | 500ms |
| HuggingFace TGI | 180 | 55% | 300ms |
| **vLLM** | **450** | **85%** | **150ms** |

**测试条件**：LLaMA-7B, A100, batch_size=32, seq_len=512

---

## 🔥 Continuous Batching：动态批处理

### 传统 Batching 的问题

**问题**：传统批处理要求**所有请求同时开始、同时结束**。

```
传统 Static Batching：
  Batch 1:
    Request 1: [████████████████] 16 tokens
    Request 2: [████████████████] 16 tokens
    Request 3: [████████████████] 16 tokens
    Request 4: [████████████████] 16 tokens
  
  问题：
  - 所有请求必须等最长的完成
  - GPU 有空闲（短的请求先完成，但要等）
  - 吞吐低
```

### Continuous Batching

**核心思想**：**动态加入/完成**请求，最大化 GPU 利用率。

```
Continuous Batching：
  Iteration 1:
    Request 1: [█] (生成第 1 个 token)
    Request 2: [█] (生成第 1 个 token)
    Request 3: [█] (生成第 1 个 token)
    Request 4: [█] (生成第 1 个 token)
  
  Iteration 2:
    Request 1: [██] (生成第 2 个 token)
    Request 2: [██] (生成第 2 个 token)
    Request 3: [██] (生成第 2 个 token)
    Request 4: [██] (生成第 2 个 token)
  
  Iteration 3:
    Request 1: [███] (生成第 3 个 token)
    Request 2: [███] (生成第 3 个 token)
    Request 3: [███] (完成！EOS)
    Request 4: [███] (生成第 3 个 token)
    Request 5: [█]   ← 新请求加入！
  
  Iteration 4:
    Request 1: [████] (完成！EOS)
    Request 2: [████] (生成第 4 个 token)
    Request 4: [████] (生成第 4 个 token)
    Request 5: [██]   (生成第 2 个 token)
    Request 6: [█]    ← 新请求加入！
```

**收益**：
- GPU 利用率：**90%+**（vs 传统 50-60%）
- 吞吐提升：**2-3x**
- 延迟降低：短请求无需等待

### 调度器实现（简化）

```python
# continuous_batching_scheduler.py
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Request:
    id: str
    prompt: str
    max_tokens: int
    generated_tokens: List[int] = None
    is_finished: bool = False
    
    def __post_init__(self):
        if self.generated_tokens is None:
            self.generated_tokens = []

class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # 活跃请求
        self.active_requests: Dict[str, Request] = {}
        
        # 等待队列
        self.waiting_queue: List[Request] = []
        
        # KV Cache 管理
        self.kv_cache: Optional[torch.Tensor] = None
    
    def add_request(self, request: Request):
        """添加新请求"""
        if len(self.active_requests) < self.max_batch_size:
            # 直接加入活跃请求
            self.active_requests[request.id] = request
        else:
            # 加入等待队列
            self.waiting_queue.append(request)
    
    def step(self, model) -> Dict[str, List[int]]:
        """执行一步推理（生成一个 token）"""
        if not self.active_requests:
            return {}
        
        # 1. 准备输入（只包含未完成的请求）
        active_reqs = [r for r in self.active_requests.values() if not r.is_finished]
        
        if not active_reqs:
            return {}
        
        # 2. 构建 batch
        batch_size = len(active_reqs)
        # 获取最后一个 token（或 prompt 的第一个 token）
        input_ids = [r.generated_tokens[-1:] if r.generated_tokens else [r.prompt] 
                     for r in active_reqs]
        
        # 3. 模型推理（简化）
        # 实际实现中，这里会调用模型的 forward，传入 KV Cache
        outputs = model.generate_step(input_ids, kv_cache=self.kv_cache)
        
        # 4. 更新请求状态
        results = {}
        finished_ids = []
        
        for i, request in enumerate(active_reqs):
            new_token = outputs[i]
            request.generated_tokens.append(new_token)
            results[request.id] = request.generated_tokens.copy()
            
            # 检查是否完成
            if new_token == 2 or len(request.generated_tokens) >= request.max_tokens:
                # EOS token 或达到最大长度
                request.is_finished = True
                finished_ids.append(request.id)
        
        # 5. 移除完成的请求
        for req_id in finished_ids:
            del self.active_requests[req_id]
        
        # 6. 从等待队列加入新请求
        while len(self.active_requests) < self.max_batch_size and self.waiting_queue:
            new_req = self.waiting_queue.pop(0)
            self.active_requests[new_req.id] = new_req
        
        return results
    
    def run(self, model, max_iterations: int = 100):
        """运行直到所有请求完成"""
        all_results = {}
        
        for iteration in range(max_iterations):
            if not self.active_requests and not self.waiting_queue:
                break
            
            results = self.step(model)
            all_results.update(results)
            
            # 打印进度
            active_count = len([r for r in self.active_requests.values() if not r.is_finished])
            print(f"Iteration {iteration}: {active_count} active requests")
        
        return all_results

# 使用示例
scheduler = ContinuousBatchingScheduler(max_batch_size=4, max_seq_len=512)

# 添加请求
scheduler.add_request(Request(id="req1", prompt="Hello,", max_tokens=20))
scheduler.add_request(Request(id="req2", prompt="What is", max_tokens=30))
scheduler.add_request(Request(id="req3", prompt="Tell me", max_tokens=15))
scheduler.add_request(Request(id="req4", prompt="Explain", max_tokens=25))
scheduler.add_request(Request(id="req5", prompt="How to", max_tokens=20))  # 会进入等待队列

# 运行
# results = scheduler.run(model)
```

---

## 📊 性能优化 Checklist

### LLM 推理优化清单

```
□ 启用 KV Cache（必须）
□ 使用 PagedAttention（vLLM）
□ 启用 Continuous Batching
□ 选择合适的数据并行度
□ 使用量化（INT8/FP16）
□ 优化 prompt 长度（减少预填充时间）
□ 设置合理的 max_seq_len
□ 监控显存使用
□ 使用 FlashAttention（长序列）
□ 启用 GPU 直接通信（多卡）
```

### 常见性能陷阱

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| KV Cache 不足 | OOM 错误 | 减少 max_num_seqs 或 max_model_len |
| 批大小太小 | GPU 利用率低 | 增大 max_num_seqs |
| 批大小太大 | 延迟增加 | 限制 batch_size，启用 Continuous Batching |
| 显存碎片 | OOM（但总显存够） | 用 vLLM（PagedAttention） |
| CPU-GPU 传输慢 | 吞吐低 | 减少数据传输，用共享内存 |

---

## 🔍 性能分析实战

### 监控 vLLM 性能

```bash
# 1. 查看服务指标（Prometheus 格式）
curl http://localhost:8000/metrics

# 输出示例：
# vllm:num_requests_running 32.0
# vllm:num_requests_waiting 5.0
# vllm:gpu_cache_usage_perc 85.0
# vllm:time_to_first_token_seconds 0.15
# vllm:time_per_output_token_seconds 0.02

# 2. 用 Grafana 可视化
# vLLM 提供 Prometheus 指标，可用 Grafana 展示

# 3. 性能分析
python -m vllm.entrypoints.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --profile \
    --trace-dir ./profiling_trace
```

### 基准测试

```python
# benchmark_vllm.py
import time
import requests
from concurrent.futures import ThreadPoolExecutor

def send_request(prompt, max_tokens=100):
    """发送单个请求"""
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "mistralai/Mistral-7B-v0.1",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    )
    result = response.json()
    return {
        "prompt_len": len(prompt.split()),
        "completion_len": len(result['choices'][0]['text'].split()),
        "total_time": result.get('usage', {}).get('total_time', 0)
    }

# 并发测试
prompts = [f"Question {i}: What is the meaning of life?" for i in range(100)]

start = time.time()
with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(send_request, prompts))
end = time.time()

# 统计
total_tokens = sum(r['completion_len'] for r in results)
total_time = end - start
throughput = total_tokens / total_time

print(f"总请求数：{len(prompts)}")
print(f"总时间：{total_time:.2f}s")
print(f"总生成 tokens: {total_tokens}")
print(f"吞吐：{throughput:.2f} tokens/s")
print(f"平均延迟：{total_time / len(prompts) * 1000:.2f} ms/请求")
```

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 理解 KV Cache 的原理和显存计算
- [ ] 部署 vLLM 服务，跑通第一个请求
- [ ] 对比 vLLM 和原生 PyTorch 的性能
- [ ] 在笔记里记录性能数据

### 选做（深入）

- [ ] 实现简化的 KV Cache Attention
- [ ] 尝试不同的 batch size，找到最优配置
- [ ] 用 Prometheus + Grafana 监控 vLLM
- [ ] 阅读 vLLM 源码，理解 PagedAttention 实现

### 挑战任务

- [ ] 部署多卡 vLLM（Tensor Parallel）
- [ ] 优化自定义模型的推理性能
- [ ] 贡献性能优化到 vLLM 社区

---

## 📚 参考资料

- **vLLM**：
  - 项目地址：https://github.com/vllm-project/vllm
  - 论文：Efficient Memory Management for Large Language Model Serving (2023)
  - 文档：https://docs.vllm.ai/

- **TGI (HuggingFace)**：
  - 项目地址：https://github.com/huggingface/text-generation-inference
  - 文档：https://huggingface.co/docs/text-generation-inference

- **相关论文**：
  - FlashAttention: Fast and Memory-Efficient Exact Attention (NeurIPS 2022)
  - PagedAttention: vLLM 论文 (2023)
  - Continuous Batching: Orca 论文 (2022)

- **教程**：
  - vLLM 官方教程：https://docs.vllm.ai/en/latest/
  - LLM 推理优化指南：https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

---

## 🔗 下一篇预告

**笔记 10**：Triton 编程入门 - 用 Python 写 GPU 算子

- Triton 是什么，为什么比 CUDA 简单
- 第一个 Triton 程序（向量加法）
- 理解 block、thread、内存层次
- 用 Triton 实现 LayerNorm 和 Attention

---

_笔记创建：2026-04-15_  
_适合人群：想深入 LLM 推理优化的开发者_  
_平台：Linux/Windows + NVIDIA GPU（显存 16GB+，推荐 A100/H100）_  
_难度：⭐⭐⭐⭐（需要理解 LLM 架构和显存管理）_
