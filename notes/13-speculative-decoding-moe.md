# 13 - 高级推理优化：Speculative Decoding 与 MoE

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 3 - LLM 推理优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU（显存 24GB+）  
> 🔗 前置：[12-LLM 推理优化](./12-llm-inference-kv-cache.md), [11-Transformer 架构](./11-transformer-architecture-deep-dive.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **Speculative Decoding（推测解码）** 的原理和收益
2. 掌握 **MoE（Mixture of Experts）** 的架构和优化挑战
3. 了解 **多卡推理** 的并行策略
4. 能部署和优化 **大模型推理服务**

---

## 💡 Speculative Decoding：用小模型加速大模型

### 核心思想

**问题**：LLM 自回归生成是**串行**的，无法并行。

```
传统生成：
  Token 1 → Token 2 → Token 3 → Token 4 → ...
  (每次 1 步，无法并行)

推测解码：
  1. 用小模型（Draft Model）快速生成多个候选 token
  2. 用大模型（Target Model）并行验证这些 token
  3. 接受的 token 直接输出，拒绝的重新采样

  小模型：Token 1 → Token 2 → Token 3 → Token 4 → Token 5
  大模型：[Token 1, Token 2, Token 3, Token 4, Token 5]  ← 并行验证！
  
  结果：5 步 → 2 步（理论 2.5x 加速）
```

### 算法流程

```
Speculative Decoding 算法：

输入：prompt, draft model (小), target model (大), γ (候选 token 数)

1. 用小模型自回归生成 γ 个候选 token:
   x_{t+1}, x_{t+2}, ..., x_{t+γ} ~ p_draft(x | x_{1:t})

2. 用小模型计算每个 token 的概率:
   q(x_{t+i} | x_{1:t+i-1}) for i = 1, ..., γ

3. 用大模型并行计算所有候选 token 的概率:
   p(x_{t+i} | x_{1:t+i-1}) for i = 1, ..., γ

4. 从前往后验证每个 token:
   for i = 1 to γ:
     - 计算接受概率：α = min(1, p(x)/q(x))
     - 以概率α接受 x_{t+i}
     - 如果拒绝，从 p(x) 重新采样，停止验证

5. 输出所有接受的 token，回到步骤 1
```

### 📊 性能分析

```
传统生成：
  - 每 token 时间：T_large
  - 生成 N 个 token：N × T_large

推测解码：
  - 小模型生成γ个 token：γ × T_small
  - 大模型并行验证：T_large（一次验证所有）
  - 接受率：α (通常 60-80%)
  - 每轮期望输出：α × γ 个 token
  - 生成 N 个 token：(N / (α×γ)) × (γ×T_small + T_large)

加速比：
  Speedup = (N × T_large) / [(N / (α×γ)) × (γ×T_small + T_large)]
          = (α×γ × T_large) / (γ×T_small + T_large)
          = (α×γ) / (γ×(T_small/T_large) + 1)

示例（LLaMA-70B + LLaMA-7B）：
  - T_small / T_large ≈ 0.1 (小模型 10x 快)
  - γ = 5 (每次生成 5 个候选)
  - α = 0.7 (70% 接受率)
  
  Speedup = (0.7 × 5) / (5 × 0.1 + 1)
          = 3.5 / 1.5
          = 2.33x

实际加速：1.5-2.5x（取决于任务和小模型质量）
```

### 实现示例

```python
# speculative_decoding.py
import torch
import torch.nn.functional as F

def speculative_decoding(draft_model, target_model, tokenizer, 
                         prompt, max_tokens=100, gamma=5):
    """
    简化的推测解码实现
    
    Args:
        draft_model: 小模型（Draft）
        target_model: 大模型（Target）
        tokenizer: 分词器
        prompt: 输入 prompt
        max_tokens: 最大生成 token 数
        gamma: 每次生成的候选 token 数
    """
    # 编码 prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()
    
    with torch.no_grad():
        while len(generated[0]) < max_tokens:
            # 1. 用小模型生成γ个候选 token
            candidate_tokens = []
            candidate_probs = []
            
            current_ids = generated.clone()
            for i in range(gamma):
                draft_output = draft_model(current_ids)
                draft_logits = draft_output.logits[0, -1, :]
                draft_probs = F.softmax(draft_logits, dim=-1)
                
                # 采样
                next_token = torch.multinomial(draft_probs, 1)
                candidate_tokens.append(next_token.item())
                candidate_probs.append(draft_probs[next_token].item())
                
                # 追加到输入
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            
            # 2. 用大模型并行验证
            target_output = target_model(generated)
            target_logits = target_output.logits[0, -1, :]
            target_probs = F.softmax(target_logits, dim=-1)
            
            # 3. 验证每个候选 token
            accepted = 0
            for i in range(gamma):
                candidate = candidate_tokens[i]
                q = candidate_probs[i]  # 小模型概率
                p = target_probs[candidate].item()  # 大模型概率
                
                # 计算接受概率
                alpha = min(1.0, p / q) if q > 0 else 0.0
                
                # 以概率α接受
                if torch.rand(1).item() < alpha:
                    # 接受
                    generated = torch.cat([
                        generated, 
                        torch.tensor([[candidate]], device=generated.device)
                    ], dim=-1)
                    accepted += 1
                else:
                    # 拒绝：从大模型分布重新采样
                    new_token = torch.multinomial(target_probs, 1)
                    generated = torch.cat([
                        generated,
                        new_token.unsqueeze(0).unsqueeze(0)
                    ], dim=-1)
                    break
            
            # 如果所有候选都被接受，再生成一个 token
            if accepted == gamma:
                target_output = target_model(generated)
                target_logits = target_output.logits[0, -1, :]
                next_token = torch.multinomial(F.softmax(target_logits, dim=-1), 1)
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
    
    # 解码输出
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output


# 使用示例
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# 生成
prompt = "Once upon a time,"
output = speculative_decoding(
    draft_model, target_model, tokenizer,
    prompt, max_tokens=100, gamma=5
)
print(output)
```

### 实际部署：vLLM 的 Speculative Decoding

```bash
# vLLM 支持推测解码
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --speculative-model meta-llama/Llama-2-7b-hf \
    --num-speculative-tokens 5 \
    --tensor-parallel-size 4

# 性能对比
# 传统生成：~15 tokens/s
# 推测解码：~30 tokens/s (2x 加速)
```

---

## 🔥 MoE：Mixture of Experts

### MoE 核心思想

**问题**：增加模型参数会线性增加计算量。

**MoE 方案**：增加参数，但**每次只激活一部分**。

```
标准 Transformer:
  输入 → FFN → 输出
  所有参数都激活

MoE Transformer:
  输入 → Router → 选择 Top-K Experts → Expert_1/Expert_2/... → 输出
  只激活 K 个专家（通常 K=1 或 2）

优势：
  - 总参数可以很大（如 Mixtral 8×7B = 56B 参数）
  - 每次推理只激活一部分（如 2×7B = 14B 参数）
  - "稀疏激活，稠密效果"
```

### MoE 架构详解

```python
class MoELayer(nn.Module):
    """
    Mixture of Experts 层
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # 专家数量
        self.top_k = config.top_k  # 激活的专家数
        self.hidden_dim = config.hidden_dim
        
        # Router（门控网络）
        self.router = nn.Linear(self.hidden_dim, self.num_experts)
        
        # Experts（专家网络）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, config.intermediate_size),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, self.hidden_dim)
            )
            for _ in range(self.num_experts)
        ])
    
    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, hidden_dim = x.shape
        
        # 1. Router 计算专家权重
        router_logits = self.router(x)  # (B, N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 2. 选择 Top-K 专家
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 3. 归一化权重（只考虑 Top-K）
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 4. 并行执行所有专家
        # 将输入复制 num_experts 份，每个专家处理一份
        x_expanded = x.unsqueeze(1).expand(-1, self.num_experts, -1, -1)
        # (B, num_experts, N, D)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_expanded[:, i])  # (B, N, D)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, N, D)
        
        # 5. 加权求和（只考虑 Top-K 专家）
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, :, k]  # (B, N)
            expert_weights = top_k_probs[:, :, k].unsqueeze(-1)  # (B, N, 1)
            
            # 收集对应专家的输出
            for b in range(batch):
                for n in range(seq_len):
                    expert_idx = expert_indices[b, n]
                    output[b, n] += expert_outputs[b, expert_idx, n] * expert_weights[b, n]
        
        return output
```

### Mixtral 8×7B 架构

```
Mixtral-8x7B 配置：
  - 总层数：32
  - 每层专家数：8
  - 激活专家数：2 (Top-2)
  - 总参数：~56B (8×7B)
  - 激活参数：~14B (2×7B per layer)

结构：
  - 每个 FFN 层替换为 MoE 层
  - Attention 层保持不变（稠密）
  - Router 动态选择 2 个专家

性能：
  - 推理速度：接近 7B 模型
  - 效果：接近 70B 稠密模型
  - 显存：需要加载 56B 参数（需要多卡或量化）
```

### 📊 MoE vs 稠密模型对比

| 模型 | 总参数 | 激活参数 | 推理速度 | 效果 |
|------|--------|---------|---------|------|
| LLaMA-7B | 7B | 7B | 1x | 基准 |
| LLaMA-70B | 70B | 70B | 0.3x | 优秀 |
| Mixtral-8×7B | 56B | 14B | 0.8x | 接近 70B |

**关键洞察**：
- MoE 用**稀疏激活**换取**参数效率**
- 推理速度取决于**激活参数**，不是总参数
- 但显存占用取决于**总参数**（需要全部加载）

---

## 🔥 MoE 推理优化挑战

### 挑战 1：专家并行

```
问题：56B 参数无法单卡加载（需要~112GB FP16）

解决方案：专家并行（Expert Parallel）
  - 将不同专家分配到不同 GPU
  - 每个 GPU 只加载部分专家
  - 运行时根据 Router 选择通信

示例（Mixtral 8×7B，4 卡）：
  GPU 0: Experts 0, 1
  GPU 1: Experts 2, 3
  GPU 2: Experts 4, 5
  GPU 3: Experts 6, 7
  
  推理流程：
  1. 所有 GPU 有完整的输入
  2. 每个 GPU 计算自己负责的专家
  3. All-to-All 通信，收集所有专家输出
  4. 加权求和得到最终输出
```

### 挑战 2：负载不均衡

```
问题：Router 可能偏向某些专家，导致负载不均衡

示例：
  Expert 0: 30% 的请求
  Expert 1: 25% 的请求
  Expert 2: 5% 的请求   ← 闲置
  ...
  Expert 7: 20% 的请求

后果：
  - 某些 GPU 繁忙，某些空闲
  - 整体吞吐下降

解决方案：
  1. Router 负载均衡损失（训练时）
  2. 专家容量限制（推理时）
  3. 动态批处理
```

### 挑战 3：通信开销

```
All-to-All 通信模式：
  每个 GPU 发送数据给所有其他 GPU
  
对于 MoE：
  - 输入需要路由到正确的专家 GPU
  - 输出需要收集回所有 GPU
  
通信量：
  - 每层：O(batch × seq_len × hidden_dim)
  - 32 层累积：显著开销

优化：
  - 重叠通信和计算
  - 量化通信数据
  - 使用 NVLink/InfiniBand
```

---

## 🔥 多卡推理并行策略

### 三种并行模式

```
1. Tensor Parallel (TP):
   - 将单个算子拆分到多卡
   - 例如：矩阵乘法拆分到 4 卡
   - 适合：单模型太大，单卡放不下
   - 通信：每层都需要 All-Reduce
   - 延迟：低（适合延迟敏感）

2. Pipeline Parallel (PP):
   - 将模型层拆分到多卡
   - 例如：GPU0 负责层 1-8，GPU1 负责层 9-16
   - 适合：超大模型（100B+）
   - 通信：层间传递激活值
   - 延迟：高（流水线气泡）

3. Data Parallel (DP):
   - 每卡有完整模型副本
   - 不同请求分配到不同卡
   - 适合：高吞吐场景
   - 通信：无（请求级隔离）
   - 延迟：取决于负载
```

### 组合策略

```
实际部署通常组合使用：

示例：70B 模型，8 卡部署

方案 1 (TP=8, PP=1, DP=1):
  - 8 卡做 Tensor Parallel
  - 每卡处理模型的 1/8
  - 适合：单请求低延迟

方案 2 (TP=4, PP=2, DP=1):
  - 4 卡 Tensor Parallel × 2 段 Pipeline
  - 适合：平衡延迟和吞吐

方案 3 (TP=2, PP=2, DP=2):
  - 2 卡 TP × 2 段 PP × 2 份 DP 副本
  - 适合：高吞吐多请求

vLLM 默认：TP=显卡数，PP=1
```

### vLLM 多卡部署

```bash
# Tensor Parallel (推荐)
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95

# Pipeline Parallel (vLLM 0.4.0+)
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --max-num-seqs 256

# 监控多卡状态
watch -n 1 nvidia-smi
```

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 理解 Speculative Decoding 的原理
- [ ] 理解 MoE 的稀疏激活思想
- [ ] 理解三种并行策略的区别
- [ ] 在笔记里记录学习心得

### 选做（深入）

- [ ] 用 HuggingFace 实现简化的推测解码
- [ ] 部署 Mixtral-8×7B 模型
- [ ] 尝试 vLLM 的多卡部署
- [ ] 阅读 Mixtral 论文

---

## 📚 参考资料

- **论文**：
  - Speculative Decoding: Fast Inference from Large Language Models (2023)
  - Mixtral of Experts (2024)
  - GShard: Scaling Giant Models with Conditional Computation (2020)

- **开源项目**：
  - vLLM: https://github.com/vllm-project/vllm
  - DeepSpeed-MoE: https://www.deepspeed.ai/tutorials/moe/
  - Fairseq-MoE: https://github.com/pytorch/fairseq

- **模型**：
  - Mixtral-8x7B: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
  - Grok-1: https://github.com/xai-org/grok-1

---

_笔记创建：2026-04-15_  
_适合人群：想深入 LLM 高级推理优化的开发者_  
_平台：Linux/Windows + NVIDIA GPU（推荐多卡环境）_  
_难度：⭐⭐⭐⭐⭐（需要理解分布式系统和 MoE 架构）_
