# 12 - Transformer 架构深入：从原理到优化

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 3 - LLM 推理优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU  
> 🔗 前置：[12-LLM 推理优化](./12-llm-inference-kv-cache.md), [07-Triton 编程](./07-triton-programming.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 深入理解 **Transformer 架构的每个组件**
2. 掌握 **Self-Attention 的计算细节和优化点**
3. 理解 **RoPE、SwiGLU、RMSNorm** 等现代改进
4. 能分析 **LLaMA/Qwen** 等主流模型的架构差异
5. 为后续 **自定义算子优化** 打下基础

---

## 💡 Transformer 架构全景

### 原始 Transformer（Attention Is All You Need, 2017）

```
┌─────────────────────────────────────────────────────────┐
│                  Transformer Decoder                     │
├─────────────────────────────────────────────────────────┤
│  输入 Token Embeddings                                  │
│       ↓                                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Decoder Layer × N                    │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Masked Self-Attention                      │  │  │
│  │  │    - Q = X × W_Q                            │  │  │
│  │  │    - K = X × W_K                            │  │  │
│  │  │    - V = X × W_V                            │  │  │
│  │  │    - Attention = softmax(QK^T/√d) × V       │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │       ↓ (残差连接 + LayerNorm)                     │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Feed-Forward Network (FFN)                 │  │  │
│  │  │    - FC1: d_model → d_ff                    │  │  │
│  │  │    - Activation: ReLU                       │  │  │
│  │  │    - FC2: d_ff → d_model                    │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │       ↓ (残差连接 + LayerNorm)                     │  │
│  └───────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  输出 logits → Softmax → 下一个 Token                   │
└─────────────────────────────────────────────────────────┘
```

### 现代 LLM 架构演进（LLaMA/Qwen）

```
┌─────────────────────────────────────────────────────────┐
│              LLaMA-2 / Qwen-2 Decoder                    │
├─────────────────────────────────────────────────────────┤
│  Token Embeddings                                       │
│       ↓                                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Decoder Layer × N (32/40/80)         │  │
│  │                                                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  1. RMSNorm (Pre-Norm)                      │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │       ↓                                            │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  2. RoPE Self-Attention                     │  │  │
│  │  │    - RoPE: 旋转位置编码                      │  │  │
│  │  │    - Grouped Query Attention (GQA)          │  │  │
│  │  │    - KV Cache 优化                          │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │       ↓ (残差连接)                                  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  3. RMSNorm (Pre-Norm)                      │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │       ↓                                            │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  4. SwiGLU FFN                              │  │  │
│  │  │    - Gate: x × W_gate                       │  │  │
│  │  │    - Up: x × W_up                           │  │  │
│  │  │    - Down: (Gate ⊙ Swish(Up)) × W_down      │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │       ↓ (残差连接)                                  │  │
│  └───────────────────────────────────────────────────┘  │
│       ↓                                                 │
│  Final RMSNorm → LM Head → Vocab logits                │
└─────────────────────────────────────────────────────────┘
```

### 📊 主流模型架构对比

| 模型 | 层数 | 隐藏层 | 头数 | FFN | 位置编码 | 注意力 |
|------|------|--------|------|-----|---------|--------|
| BERT | 12/24 | 768/1024 | 12/16 | GeLU | 绝对位置 | 双向 |
| LLaMA-7B | 32 | 4096 | 32 | SwiGLU | RoPE | MHA |
| LLaMA-2-70B | 80 | 8192 | 64 | SwiGLU | RoPE | GQA |
| Qwen-7B | 32 | 4096 | 32 | SwiGLU | RoPE | MHA |
| Qwen-72B | 80 | 8192 | 64 | SwiGLU | RoPE | GQA |
| Mistral-7B | 32 | 4096 | 32 | SwiGLU | RoPE | GQA |

**缩写**：
- MHA: Multi-Head Attention
- GQA: Grouped Query Attention
- MQA: Multi-Query Attention

---

## 📚 Self-Attention 深入

### 标准 Attention 计算

**公式**：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

其中：
  Q = X × W_Q  (N, d_k)
  K = X × W_K  (N, d_k)
  V = X × W_V  (N, d_v)
  
  N = seq_len (序列长度)
  d_k = d_v = d_model / num_heads
```

**计算流程**：

```python
def self_attention(X, W_Q, W_K, W_V, W_O):
    """
    X: (batch, seq_len, d_model)
    W_Q, W_K, W_V: (d_model, d_k)
    W_O: (d_v, d_model)
    """
    # 1. 线性投影
    Q = X @ W_Q  # (B, N, d_k)
    K = X @ W_K  # (B, N, d_k)
    V = X @ W_V  # (B, N, d_v)
    
    # 2. 计算 Attention 分数
    scores = Q @ K.transpose(-2, -1)  # (B, N, N)
    
    # 3. Scale
    d_k = Q.shape[-1]
    scores = scores / (d_k ** 0.5)
    
    # 4. Causal Mask（防止看到未来）
    mask = torch.tril(torch.ones(N, N))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 5. Softmax
    attn = torch.softmax(scores, dim=-1)  # (B, N, N)
    
    # 6. 应用 Attention
    output = attn @ V  # (B, N, d_v)
    
    # 7. 输出投影
    output = output @ W_O  # (B, N, d_model)
    
    return output
```

### 计算复杂度分析

```
对于序列长度 N，隐藏维度 d：

1. Q/K/V 投影：
   - 3 × (N × d × d) = 3Nd² FLOPs

2. QK^T 矩阵乘法：
   - N × N × d = N²d FLOPs

3. Attention × V：
   - N × N × d = N²d FLOPs

4. 输出投影：
   - N × d × d = Nd² FLOPs

总计：2Nd² + 2N²d FLOPs

当 N >> d 时（长序列）：
  - 主导项：2N²d（Attention 矩阵）
  - 内存：O(N²) 存储 Attention 矩阵
  - 这就是为什么需要 FlashAttention！
```

### Multi-Head Attention

```python
def multi_head_attention(X, num_heads, W_Q, W_K, W_V, W_O):
    """
    将 d_model 分成 num_heads 个头，每个头独立计算 Attention
    """
    batch, seq_len, d_model = X.shape
    d_k = d_model // num_heads
    
    # 1. 投影并分割头
    Q = X @ W_Q  # (B, N, d_model)
    K = X @ W_K
    V = X @ W_V
    
    # 重塑：(B, N, num_heads, d_k) → (B, num_heads, N, d_k)
    Q = Q.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 2. 并行计算所有头的 Attention
    scores = Q @ K.transpose(-2, -1)  # (B, H, N, N)
    scores = scores / (d_k ** 0.5)
    
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)  # (B, H, N, N)
    output = attn @ V  # (B, H, N, d_k)
    
    # 3. 拼接头并投影
    output = output.transpose(1, 2).reshape(batch, seq_len, d_model)
    output = output @ W_O
    
    return output
```

**为什么需要 Multi-Head？**
- 每个头可以学习**不同的注意力模式**
- 类似 CNN 的**多个卷积核**
- 实验证明比单头效果好

---

## 🔥 RoPE：旋转位置编码

### 为什么需要位置编码？

**问题**：Self-Attention 是**排列等变**的（permutation equivariant）。

```
Attention 只看内容，不看位置：
  "我 爱 你" 和 "你 爱 我"
  - 词袋相同，Attention 输出相同
  - 但语义完全不同！

解决方案：加入位置信息
  - 绝对位置编码（Transformer 原始）
  - 相对位置编码（T5, RoPE）
```

### RoPE 核心思想

**Rotary Positional Embedding**（旋转位置编码）：

```
核心公式：
  将 Q 和 K 旋转一个角度，旋转量取决于位置

对于位置 m 的向量 q_m：
  RoPE(q_m, m) = R(m) × q_m
  
  其中 R(m) 是旋转矩阵：
  R(m) = [cos(mθ)  -sin(mθ)]
         [sin(mθ)   cos(mθ)]
  
  θ = 10000^(-2i/d), i = 0, 1, ..., d/2-1

关键性质：
  RoPE(q_m, m) · RoPE(k_n, n) = f(q_m, k_n, m-n)
  
  即：内积只依赖于相对位置 (m-n)！
```

### RoPE 实现

```python
import torch

def apply_rope(q, k, freqs_cis):
    """
    应用 RoPE 到 Q 和 K
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        freqs_cis: 预计算的频率 (seq_len, head_dim//2)
    
    Returns:
        q_rot, k_rot: 旋转后的 Q 和 K
    """
    # 将 head_dim 分成两半
    # q = [q0, q1, q2, q3, ...] → [q0, q1], [q2, q3], ...
    q_reshaped = q.reshape(*q.shape[:-1], -1, 2)  # (B, H, N, D/2, 2)
    k_reshaped = k.reshape(*k.shape[:-1], -1, 2)
    
    # 复数表示：q = q_real + i*q_imag
    q_complex = torch.view_as_complex(q_reshaped)  # (B, H, N, D/2)
    k_complex = torch.view_as_complex(k_reshaped)
    
    # 构建旋转复数：exp(i * m * θ)
    # freqs_cis: (N, D/2)，每行是 [m*θ_0, m*θ_1, ...]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D/2)
    rotate_complex = torch.polar(torch.ones_like(freqs_cis), freqs_cis)
    
    # 应用旋转
    q_rot = q_complex * rotate_complex
    k_rot = k_complex * rotate_complex
    
    # 转回实数
    q_out = torch.view_as_real(q_rot).flatten(3)  # (B, H, N, D)
    k_out = torch.view_as_real(k_rot).flatten(3)
    
    return q_out, k_out


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算 RoPE 频率
    
    Args:
        dim: head dimension
        max_seq_len: 最大序列长度
        theta: 基础频率（默认 10000）
    
    Returns:
        freqs_cis: (max_seq_len, dim//2)
    """
    # 计算θ_i = 10000^(-2i/d), i = 0, 1, ..., d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 计算每个位置的旋转角度：m * θ
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    
    return freqs


# 使用示例
batch, num_heads, seq_len, head_dim = 1, 32, 512, 128

# 预计算频率（只需一次）
freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=2048)

# 生成 Q 和 K
q = torch.randn(batch, num_heads, seq_len, head_dim)
k = torch.randn(batch, num_heads, seq_len, head_dim)

# 应用 RoPE
q_rot, k_rot = apply_rope(q, k, freqs_cis)

# 用旋转后的 Q 和 K 计算 Attention
scores = q_rot @ k_rot.transpose(-2, -1) / (head_dim ** 0.5)
```

### RoPE 的优势

| 特性 | 绝对位置编码 | RoPE |
|------|------------|------|
| 外推能力 | 差（超过训练长度失效） | 较好（可外推） |
| 相对位置感知 | 需要额外设计 | 内置 |
| 实现复杂度 | 简单 | 中等 |
| 主流模型采用 | BERT, T5 | LLaMA, Qwen, PaLM |

---

## 🔥 SwiGLU：现代 FFN 激活

### 传统 FFN vs SwiGLU

**传统 FFN**（Transformer）：

```python
def traditional_ffn(x, W1, W2):
    """
    x: (batch, seq_len, d_model)
    W1: (d_model, d_ff)
    W2: (d_ff, d_model)
    """
    h = x @ W1  # (B, N, d_ff)
    h = torch.relu(h)  # 或 GeLU
    output = h @ W2  # (B, N, d_model)
    return output
```

**SwiGLU FFN**（LLaMA, PaLM）：

```python
def swiglu_ffn(x, W_gate, W_up, W_down):
    """
    SwiGLU = Swish × Gated Linear Unit
    
    x: (batch, seq_len, d_model)
    W_gate, W_up: (d_model, d_ff)
    W_down: (d_ff, d_model)
    """
    # 门控分支
    gate = x @ W_gate  # (B, N, d_ff)
    
    # 上投影分支
    up = x @ W_up  # (B, N, d_ff)
    
    # Swish 激活 × 门控
    # Swish(x) = x × sigmoid(x)，但 LLaMA 用 SiLU = x × sigmoid(x)
    h = torch.nn.functional.silu(gate) * up  # (B, N, d_ff)
    
    # 下投影
    output = h @ W_down  # (B, N, d_model)
    
    return output
```

### 为什么 SwiGLU 更好？

```
传统 FFN：
  - 1 个投影矩阵 (d_model → d_ff)
  - 1 个激活函数
  - 1 个投影矩阵 (d_ff → d_model)
  - 参数量：2 × d_model × d_ff

SwiGLU：
  - 2 个投影矩阵 (d_model → d_ff) [gate 和 up]
  - 1 个逐元素乘法 + Swish
  - 1 个投影矩阵 (d_ff → d_model)
  - 参数量：3 × d_model × d_ff

性能提升：
  - 实验表明 SwiGLU 收敛更快
  - 最终效果更好（perplexity 更低）
  - 代价：参数量增加 50%

LLaMA 的补偿策略：
  - 减小 d_ff 比例（从 4×d_model 降到 8/3×d_model）
  - 保持总参数量相近
```

### 参数量对比（LLaMA-7B）

```
LLaMA-7B 配置：
  - d_model = 4096
  - num_layers = 32
  - d_ff (SwiGLU) = 11008 (约 8/3 × d_model)

传统 FFN 参数量/层：
  - 2 × 4096 × 16384 = 134M

SwiGLU 参数量/层：
  - 3 × 4096 × 11008 = 135M

总参数量相近，但 SwiGLU 表达力更强！
```

---

## 🔥 RMSNorm：更高效的归一化

### LayerNorm vs RMSNorm

**LayerNorm**（原始 Transformer）：

```python
def layernorm(x, gamma, beta, eps=1e-6):
    """
    x: (..., hidden_dim)
    gamma, beta: (hidden_dim,)
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

**RMSNorm**（LLaMA, PaLM）：

```python
def rmsnorm(x, gamma, eps=1e-6):
    """
    Root Mean Square Layer Normalization
    
    去掉均值中心化，只用 RMS
    """
    # 计算 RMS（均方根）
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    
    # 归一化
    x_norm = x / rms
    
    # 只有 gamma（没有 beta）
    return gamma * x_norm
```

### 为什么 RMSNorm 更好？

```
LayerNorm：
  - 计算 mean：O(d)
  - 计算 var：O(d)
  - 减去 mean：O(d)
  - 参数：gamma + beta (2d)

RMSNorm：
  - 计算 RMS：O(d)
  - 归一化：O(d)
  - 参数：gamma (d)

优势：
  - 计算量减少约 15%
  - 参数减少 50%
  - 效果相当或略好

LLaMA 采用 RMSNorm 的原因：
  - Pre-Norm 架构（归一化在 Attention 之前）
  - 残差连接足够稳定训练
  - 不需要 LayerNorm 的完整归一化
```

---

## 🔥 GQA：分组查询注意力

### MHA vs MQA vs GQA

```
Multi-Head Attention (MHA):
  - num_heads 个 Q 头
  - num_heads 个 K 头
  - num_heads 个 V 头
  - KV Cache: num_heads × d_k × seq_len

Multi-Query Attention (MQA):
  - num_heads 个 Q 头
  - 1 个 K 头（共享）
  - 1 个 V 头（共享）
  - KV Cache: 1 × d_k × seq_len
  - 推理快，但质量下降

Grouped Query Attention (GQA):
  - num_heads 个 Q 头
  - num_kv_heads 个 K 头（num_kv_heads < num_heads）
  - num_kv_heads 个 V 头
  - 每个 KV 头被多个 Q 头共享
  - KV Cache: num_kv_heads × d_k × seq_len
  - 平衡质量和速度
```

### 示意图

```
MHA (num_heads=8):
  Q0  K0  V0  → Attn0
  Q1  K1  V1  → Attn1
  Q2  K2  V2  → Attn2
  ...
  Q7  K7  V7  → Attn7
  KV Cache: 8 个头

MQA (num_heads=8):
  Q0  K   V   → Attn0
  Q1  K   V   → Attn1
  Q2  K   V   → Attn2
  ...
  Q7  K   V   → Attn7
  KV Cache: 1 个头

GQA (num_heads=8, num_kv_heads=2):
  Q0  Q1  Q2  Q3  → K0  V0  → Attn0-3
  Q4  Q5  Q6  Q7  → K1  V1  → Attn4-7
  KV Cache: 2 个头
```

### PyTorch 实现

```python
def grouped_query_attention(X, num_heads, num_kv_heads, W_Q, W_K, W_V, W_O):
    """
    GQA 实现
    
    Args:
        X: (batch, seq_len, d_model)
        num_heads: Q 的头数
        num_kv_heads: K/V 的头数（必须整除 num_heads）
    """
    batch, seq_len, d_model = X.shape
    d_k = d_model // num_heads
    groups = num_heads // num_kv_heads
    
    # 1. 投影
    Q = X @ W_Q  # (B, N, d_model)
    K = X @ W_K  # (B, N, num_kv_heads × d_k)
    V = X @ W_V  # (B, N, num_kv_heads × d_k)
    
    # 2. 重塑头
    Q = Q.view(batch, seq_len, num_heads, d_k).transpose(1, 2)  # (B, H, N, d_k)
    K = K.view(batch, seq_len, num_kv_heads, d_k).transpose(1, 2)  # (B, KV_H, N, d_k)
    V = V.view(batch, seq_len, num_kv_heads, d_k).transpose(1, 2)
    
    # 3. 重复 K/V 以匹配 Q 的头数
    # K: (B, KV_H, N, d_k) → (B, H, N, d_k)
    K = K.repeat_interleave(groups, dim=1)
    V = V.repeat_interleave(groups, dim=1)
    
    # 4. 计算 Attention（与 MHA 相同）
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)
    output = attn @ V  # (B, H, N, d_k)
    
    # 5. 拼接并投影
    output = output.transpose(1, 2).reshape(batch, seq_len, d_model)
    output = output @ W_O
    
    return output
```

### KV Cache 节省

```
LLaMA-2-70B 配置：
  - num_heads = 64
  - num_kv_heads = 8 (GQA)
  - d_k = 128
  - seq_len = 4096

MHA KV Cache:
  - 64 × 128 × 4096 × 2 bytes (FP16) × 2 (K+V)
  - = 128 MB / layer
  - 80 层 = 10.2 GB

GQA KV Cache:
  - 8 × 128 × 4096 × 2 bytes × 2
  - = 16 MB / layer
  - 80 层 = 1.28 GB

节省：10.2 GB → 1.28 GB = 8x 减少！
这就是为什么 70B 模型能用 GQA 在单卡上推理。
```

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 理解 Self-Attention 的计算流程
- [ ] 实现 RoPE 位置编码
- [ ] 对比 LayerNorm 和 RMSNorm
- [ ] 理解 GQA 的 KV Cache 优化

### 选做（深入）

- [ ] 实现 SwiGLU FFN
- [ ] 实现 GQA Attention
- [ ] 阅读 LLaMA 论文
- [ ] 用 Triton 实现优化的 Attention

---

## 📚 参考资料

- **论文**：
  - Attention Is All You Need (Transformer, 2017)
  - RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
  - GLU Variants Improve Transformer (SwiGLU, 2020)
  - RMSNorm: Root Mean Square Layer Normalization (2019)
  - GQA: Training Generalized Multi-Query Transformer Models (2023)

- **模型代码**：
  - LLaMA: https://github.com/meta-llama/llama
  - Qwen: https://github.com/QwenLM/Qwen
  - Mistral: https://github.com/mistralai/mistral-src

---

_笔记创建：2026-04-15_  
_适合人群：想深入理解 LLM 架构的开发者_  
_难度：⭐⭐⭐⭐（需要理解 Transformer 和线性代数）_
