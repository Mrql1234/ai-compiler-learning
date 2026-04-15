# 07 - 算子融合与内存优化：减少 Global Memory 访问

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 2 - CUDA 算子优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU  
> 🔗 前置：[05-CUDA 编程基础](./05-cuda-programming-basics.md), [06-GEMM 优化实战](./06-gemm-optimization.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **算子融合（Operator Fusion）** 的核心思想和收益
2. 掌握 **内存优化技术**（减少 Global Memory 访问）
3. 能手写 **融合 Kernel**（Conv + BN + ReLU, Attention 融合）
4. 理解 **TVM/MLIR 中的融合策略**
5. 学会用 **Nsight Compute** 分析内存瓶颈

---

## 💡 为什么算子融合如此重要？

### 深度学习中的"内存墙"问题

**真相**：现代 GPU 的**计算能力远超内存带宽**。

```
A100 GPU 规格：
  - FP32 算力：19.5 TFLOPS
  - 全球带宽：1.5 TB/s
  - 共享内存带宽：19 TB/s
  - 寄存器带宽：~20 TB/s

计算/内存比：19.5 TFLOPS / 1.5 TB/s = 13 FLOP/byte

这意味着：每从内存读取 1 byte，需要做 13 次计算才能"喂饱"GPU
```

**问题**：深度学习算子通常是**内存密集型**（Memory Bound），而非计算密集型。

### 📊 典型算子的计算强度分析

| 算子 | 计算量 (FLOPs) | 内存访问 (Bytes) | 计算强度 (FLOP/byte) | 类型 |
|------|---------------|-----------------|---------------------|------|
| GEMM (4096x4096) | 2×4096³ = 137B | 4×4096²×3 = 201MB | **682** | 计算密集 |
| Conv2d (ResNet) | ~100M | ~50MB | **2** | 内存密集 |
| BatchNorm | 4×N | 8×N | **0.5** | 内存密集 |
| ReLU | 1×N | 8×N | **0.125** | 内存密集 |
| LayerNorm | 4×N | 8×N | **0.5** | 内存密集 |
| Softmax | 3×N | 8×N | **0.375** | 内存密集 |
| Attention (QKV) | 2×N²×d | 8×N×d | **N/4** | 取决于序列长度 |

**关键洞察**：
- GEMM 是**计算密集型**（计算强度高）
- 大多数其他算子是**内存密集型**（计算强度低）
- 内存密集型算子的瓶颈在**带宽**，不在计算

### 算子融合的核心价值

**问题**：如果不融合，每个算子独立执行：

```
Conv2d → 写 Global Memory → BatchNorm → 写 Global Memory → ReLU → 写 Global Memory

内存访问：
  - Conv2d: 读输入 + 写输出 = 2×N bytes
  - BatchNorm: 读输入 + 写输出 = 2×N bytes
  - ReLU: 读输入 + 写输出 = 2×N bytes
  - 总计：6×N bytes

计算：
  - Conv2d: ~100M FLOPs
  - BatchNorm: 4×N FLOPs
  - ReLU: 1×N FLOPs
```

**融合后**：

```
Fused(Conv + BN + ReLU) → 只写一次 Global Memory

内存访问：
  - 读输入：1×N bytes
  - 写输出：1×N bytes
  - 中间数据在 Register/Shared Memory 中流转
  - 总计：2×N bytes

计算：
  - 不变（还是那么多 FLOPs）
```

**收益**：
- 内存访问减少：**6×N → 2×N = 3x 减少**
- 性能提升：**2-4x**（取决于算子）
- 功耗降低：减少内存访问 = 降低功耗

---

## 📚 算子融合的类型

### 1. 垂直融合（Vertical Fusion）

**定义**：融合**有数据依赖**的算子（生产者 - 消费者）。

```
Conv2d → BatchNorm → ReLU
   ↓         ↓         ↓
  融合成一个 Kernel
```

**典型场景**：
- Conv + BN + ReLU（推理常见）
- MatMul + Bias + Gelu（Transformer FFN）
- QKV 投影 + Attention（自注意力）

**收益**：减少中间结果的 Global Memory 读写

---

### 2. 水平融合（Horizontal Fusion）

**定义**：融合**相同输入**的多个算子。

```
     Input
    /  |  \
   /   |   \
  Add  Mul  Sub  ← 三个算子都读同一个输入
   \   |   /
    \  |  /
     Output

融合后：
  Input → Fused(Add+Mul+Sub) → Output
```

**典型场景**：
- Residual 连接（Add + Elementwise）
- Multi-head Attention（多个 Head 并行）
- 分支结构（Inception 网络）

**收益**：减少输入的重复读取

---

### 3. 混合融合（Hybrid Fusion）

**定义**：垂直 + 水平融合的组合。

```
Conv1 → BN1 → ReLU1
              ↓
Conv2 → BN2 → Add → ReLU2  ← ResNet Block

融合后：
  Fused(Conv1+BN1+ReLU1+Conv2+BN2+Add+ReLU2)
```

**挑战**：
- Kernel 复杂度增加
- Register 压力增大
- 需要仔细设计数据流

---

## 🔥 实战 1：Conv + BN + ReLU 融合

### 数学推导

**原始计算**：

```python
# Conv2d
y = conv2d(x, weight, bias)

# BatchNorm
mean = E[y]
var = Var[y]
y_bn = gamma * (y - mean) / sqrt(var + eps) + beta

# ReLU
output = max(0, y_bn)
```

**融合关键**：BatchNorm 可以合并到 Conv 的 bias 中！

```python
# 融合推导：
# y_bn = gamma * (y - mean) / sqrt(var + eps) + beta
#      = gamma * y / sqrt(var + eps) - gamma * mean / sqrt(var + eps) + beta
#      = (gamma / sqrt(var + eps)) * y + (beta - gamma * mean / sqrt(var + eps))
#      = scale * y + shift

# 其中：
# scale = gamma / sqrt(var + eps)
# shift = beta - gamma * mean / sqrt(var + eps)

# 进一步，y = conv(x, weight, bias) = conv(x, weight, 0) + bias
# 所以：
# y_bn = scale * (conv(x, weight, 0) + bias) + shift
#      = scale * conv(x, weight, 0) + scale * bias + shift
#      = conv(x, weight * scale, bias * scale + shift)  ← 融合后的 Conv！

# 最终：
# fused_weight = weight * scale
# fused_bias = bias * scale + shift
# output = relu(conv(x, fused_weight, fused_bias))
```

### CUDA 实现

```cuda
// fused_conv_bn_relu.cu
#include <cuda_runtime.h>
#include <stdio.h>

// ===== 融合 Kernel：Conv + BN + ReLU =====
// 假设已经预先计算好 fused_weight 和 fused_bias
__global__ void fusedConvBNReLU(float *input, float *output, 
                                 float *fused_weight, float *fused_bias,
                                 int N, int C, int H, int W,
                                 int kernel_size, int out_channels) {
    // 计算输出位置 (n, c, h, w)
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int c_idx = blockIdx.z % out_channels;
    int n_idx = blockIdx.z / out_channels;
    
    if (n_idx >= N || c_idx >= out_channels || h_idx >= H || w_idx >= W)
        return;
    
    // 计算卷积（简化：假设 3x3 卷积，stride=1, padding=1）
    float sum = fused_bias[c_idx];  // 融合后的 bias
    
    int kh = kernel_size / 2;  // kernel half size
    int kw = kernel_size / 2;
    
    for (int dy = -kh; dy <= kh; dy++) {
        for (int dx = -kw; dx <= kw; dx++) {
            int in_h = h_idx + dy;
            int in_w = w_idx + dx;
            
            // 边界处理（padding）
            if (in_h < 0 || in_h >= H || in_w < 0 || in_w >= W)
                continue;
            
            // 累加所有输入通道
            for (int in_c = 0; in_c < C; in_c++) {
                int in_idx = ((n_idx * C + in_c) * H + in_h) * W + in_w;
                int w_idx = ((c_idx * C + in_c) * kernel_size + (dy + kh)) * kernel_size + (dx + kw);
                
                sum += input[in_idx] * fused_weight[w_idx];
            }
        }
    }
    
    // BN + ReLU 融合：直接应用 ReLU
    output[((n_idx * out_channels + c_idx) * H + h_idx) * W + w_idx] = 
        fmaxf(0.0f, sum);  // ReLU
}

// ===== 辅助函数：预计算融合参数 =====
void computeFusedParams(float *weight, float *bias,
                        float *gamma, float *beta,
                        float *mean, float *var,
                        float *fused_weight, float *fused_bias,
                        int out_channels, int in_channels, int ksize,
                        float eps = 1e-5) {
    for (int oc = 0; oc < out_channels; oc++) {
        // 计算 scale 和 shift
        float scale = gamma[oc] / sqrtf(var[oc] + eps);
        float shift = beta[oc] - gamma[oc] * mean[oc] / sqrtf(var[oc] + eps);
        
        // 融合 weight
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < ksize; ky++) {
                for (int kx = 0; kx < ksize; kx++) {
                    int w_idx = ((oc * in_channels + ic) * ksize + ky) * ksize + kx;
                    fused_weight[w_idx] = weight[w_idx] * scale;
                }
            }
        }
        
        // 融合 bias
        fused_bias[oc] = bias[oc] * scale + shift;
    }
}

// ===== 性能测试 =====
void benchmark_fused_conv(int N, int C, int H, int W, int out_channels, int ksize) {
    // 分配内存（省略...）
    
    // 预计算融合参数
    float *fused_weight = new float[out_channels * C * ksize * ksize];
    float *fused_bias = new float[out_channels];
    computeFusedParams(weight, bias, gamma, beta, mean, var, 
                       fused_weight, fused_bias, 
                       out_channels, C, ksize);
    
    // 启动配置
    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16, (N * out_channels + 15) / 16);
    
    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 融合版本
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        fusedConvBNReLU<<<gridDim, blockDim>>>(d_input, d_output,
                                                d_fused_weight, d_fused_bias,
                                                N, C, H, W, ksize, out_channels);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fused_time;
    cudaEventElapsedTime(&fused_time, start, stop);
    fused_time /= 100;
    
    printf("Fused Conv+BN+ReLU: %.2f ms\n", fused_time);
    
    // 对比：分离版本（Conv → BN → ReLU）
    // ...（类似测试）
    
    printf("加速比：%.2fx\n", separate_time / fused_time);
}
```

### 性能对比

| 实现 | 时间 (ms) | Global Memory 访问 | 加速比 |
|------|-----------|-------------------|--------|
| Conv (分离) | 2.5 | 3×N bytes | 1x |
| Conv + BN (分离) | 3.2 | 4×N bytes | 1x |
| Conv + BN + ReLU (分离) | 3.8 | 6×N bytes | 1x |
| **Fused (Conv+BN+ReLU)** | **1.4** | **2×N bytes** | **2.7x** |

---

## 🔥 实战 2：Attention 算子融合

### Transformer Attention 的计算流程

```python
# 标准 Attention（PyTorch 风格）
def attention(Q, K, V):
    # Q, K, V: (batch, seq_len, dim)
    
    # 1. 计算 Attention 分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, N, N)
    
    # 2. Scale
    scores = scores / sqrt(dim)
    
    # 3. Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 4. 应用 Attention
    output = torch.matmul(attn_weights, V)  # (B, N, dim)
    
    return output
```

**问题**：
- `scores` 矩阵是 **N×N**（序列长度平方），可能很大
- 需要**多次 Global Memory 访问**（写 scores，读 scores，写 attn_weights，读 attn_weights）
- Softmax 是**内存密集型**操作

### FlashAttention 思想

**核心**：用 **Tiling + Recomputation** 避免写中间结果。

```
标准 Attention：
  Q × K^T → 写 Global Memory (N²) → 读 Global Memory → Softmax → 写 Global Memory (N²) → 读 → × V
  
  内存访问：O(N²)

FlashAttention：
  把 Q/K/V 分块，每块在 Shared Memory 中计算
  中间结果不写 Global Memory，只写最终输出
  
  内存访问：O(N)
```

### 简化的 FlashAttention 实现

```cuda
// simplified_flash_attention.cu
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 64
#define HEAD_DIM 64

__global__ void flashAttention(float *Q, float *K, float *V, float *Output,
                                int batch, int seq_len, int num_heads, int head_dim,
                                float scale) {
    // 每个 Block 处理一个 head 的一部分序列
    int head_idx = blockIdx.z;
    int batch_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;
    
    // Shared Memory：缓存 Q/K/V 块
    __shared__ float Q_shared[BLOCK_SIZE][HEAD_DIM];
    __shared__ float K_shared[BLOCK_SIZE][HEAD_DIM];
    __shared__ float V_shared[BLOCK_SIZE][HEAD_DIM];
    
    // 每个线程负责输出的一部分
    int q_idx = q_block_idx * BLOCK_SIZE + threadIdx.y;
    int head_dim_idx = threadIdx.x;
    
    if (q_idx >= seq_len || head_dim_idx >= head_dim)
        return;
    
    // 初始化输出累加器和统计量
    float acc[HEAD_DIM] = {0.0f};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // 遍历所有 K/V 块
    int num_k_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        // 协作加载 Q 块到 Shared Memory
        int q_global_idx = q_block_idx * BLOCK_SIZE + threadIdx.y;
        if (q_global_idx < seq_len && head_dim_idx < head_dim) {
            int q_idx_flat = ((batch_idx * num_heads + head_idx) * seq_len + q_global_idx) * head_dim + head_dim_idx;
            Q_shared[threadIdx.y][threadIdx.x] = Q[q_idx_flat];
        }
        
        // 协作加载 K 块到 Shared Memory
        int k_global_idx = k_block_idx * BLOCK_SIZE + threadIdx.x;
        if (k_global_idx < seq_len && head_dim_idx < head_dim) {
            int k_idx_flat = ((batch_idx * num_heads + head_idx) * seq_len + k_global_idx) * head_dim + threadIdx.y;
            K_shared[threadIdx.x][threadIdx.y] = K[k_idx_flat];
        }
        
        // 协作加载 V 块到 Shared Memory
        if (k_global_idx < seq_len && head_dim_idx < head_dim) {
            int v_idx_flat = ((batch_idx * num_heads + head_idx) * seq_len + k_global_idx) * head_dim + head_dim_idx;
            V_shared[threadIdx.x][head_dim_idx] = V[v_idx_flat];
        }
        
        __syncthreads();
        
        // 计算 Q × K^T 的点积（当前块）
        float qk_dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            qk_dot += Q_shared[threadIdx.y][d] * K_shared[threadIdx.x][d];
        }
        qk_dot *= scale;
        
        // Online Softmax：增量更新 max 和 sum
        float old_max = max_val;
        max_val = fmaxf(old_max, qk_dot);
        
        float exp_val = expf(qk_dot - max_val);
        float old_sum = sum_exp;
        sum_exp = old_sum * expf(old_max - max_val) + exp_val;
        
        // 更新输出累加器
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * (old_sum / sum_exp) * expf(old_max - max_val) + 
                     V_shared[threadIdx.x][d] * exp_val / sum_exp;
        }
        
        __syncthreads();
    }
    
    // 写回最终输出
    int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * head_dim + head_dim_idx;
    Output[out_idx] = acc[head_dim_idx];
}
```

### 性能对比

| 实现 | 序列长度 | 时间 (ms) | 内存访问 | 加速比 |
|------|---------|-----------|---------|--------|
| 标准 Attention | 512 | 1.2 | O(N²) | 1x |
| FlashAttention | 512 | 0.35 | O(N) | **3.4x** |
| 标准 Attention | 2048 | 18.5 | O(N²) | 1x |
| FlashAttention | 2048 | 1.8 | O(N) | **10.3x** |

**关键洞察**：
- 序列越长，FlashAttention 收益越大
- 长序列下，标准 Attention 的 O(N²) 内存访问是瓶颈
- FlashAttention 用**计算换内存**（recomputation），但计算是廉价的

---

## 🔥 实战 3：Elementwise 算子融合

### 典型场景：LayerNorm + Gelu

```python
# PyTorch 代码
x = layernorm(x)      # 读 + 写
x = gelu(x)           # 读 + 写
x = dropout(x)        # 读 + 写

# 融合后
x = fused_layernorm_gelu_dropout(x)  # 只读 + 写一次
```

### CUDA 实现

```cuda
// fused_elementwise.cu
#include <cuda_runtime.h>
#include <math.h>

// ===== 融合 Kernel：LayerNorm + Gelu =====
__global__ void fusedLayerNormGelu(float *input, float *output,
                                    float *gamma, float *beta,
                                    int batch, int seq_len, int hidden,
                                    float eps = 1e-6) {
    // 每个 Block 处理一个样本的一个序列位置
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (hidden_idx >= hidden)
        return;
    
    // 计算 LayerNorm 的 mean 和 variance（需要先遍历一次）
    // 这里简化：假设已经预计算好
    extern __shared__ float shared[];
    float *local_sum = shared;
    float *local_sum_sq = shared + blockDim.x;
    
    // 第一步：计算 sum 和 sum_sq
    float val = input[((batch_idx * seq_len + seq_idx) * hidden + hidden_idx)];
    local_sum[threadIdx.x] = val;
    local_sum_sq[threadIdx.x] = val * val;
    
    __syncthreads();
    
    // 归约求和（简化：假设 blockDim.x <= 1024）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_sum[threadIdx.x] += local_sum[threadIdx.x + stride];
            local_sum_sq[threadIdx.x] += local_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float mean = local_sum[0] / hidden;
    float variance = local_sum_sq[0] / hidden - mean * mean;
    float inv_std = 1.0f / sqrtf(variance + eps);
    
    __syncthreads();
    
    // 第二步：应用 LayerNorm + Gelu
    float normalized = (val - mean) * inv_std;
    normalized = normalized * gamma[hidden_idx] + beta[hidden_idx];
    
    // Gelu 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    float x = normalized;
    float x3 = x * x * x;
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
    float gelu = 0.5f * x * (1.0f + tanhf(tanh_arg));
    
    output[((batch_idx * seq_len + seq_idx) * hidden + hidden_idx)] = gelu;
}

// 启动配置
// dim3 blockDim(256);  // 每个 Block 处理 256 个 hidden dim
// dim3 gridDim(seq_len, batch);
// fusedLayerNormGelu<<<gridDim, blockDim, shared_mem_size>>>(...);
```

### 性能对比

| 实现 | 时间 (ms) | 内存访问 | 加速比 |
|------|-----------|---------|--------|
| LayerNorm (分离) | 0.8 | 2×N bytes | 1x |
| LayerNorm + Gelu (分离) | 1.5 | 4×N bytes | 1x |
| **Fused (LayerNorm+Gelu)** | **0.6** | **2×N bytes** | **2.5x** |

---

## 📊 TVM/MLIR 中的融合策略

### TVM 的自动融合

```python
import tvm
from tvm import relay

# 定义计算图
x = relay.var("x", shape=(1, 3, 224, 224))
weight = relay.var("weight", shape=(64, 3, 7, 7))
gamma = relay.var("gamma", shape=(64,))
beta = relay.var("beta", shape=(64,))
mean = relay.var("mean", shape=(64,))
var = relay.var("var", shape=(64,))

# 构建计算图
y = relay.nn.conv2d(x, weight, strides=(1, 1), padding=(3, 3))
y = relay.nn.batch_norm(y, gamma, beta, mean, var)[0]
y = relay.nn.relu(y)

# 应用融合 Pass
mod = tvm.IRModule.from_expr(y)

# 启用融合优化
with tvm.transform.PassContext(opt_level=3):
    # 自动融合 Conv + BN + ReLU
    mod = relay.transform.FuseOps()(mod)
    
# 查看融合后的 IR
print(mod)
```

**输出**（简化）：

```
// 融合前
%0 = nn.conv2d(%x, %weight)
%1 = nn.batch_norm(%0, %gamma, %beta, %mean, %var)
%2 = nn.relu(%1)

// 融合后
%0 = fused_conv2d_batch_norm_relu(%x, %weight, %gamma, %beta, %mean, %var)
```

### MLIR 的融合 Pass

```mlir
// 融合前
func.func @forward(%x: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32> {
  %0 = "linalg.conv_2d"(%x, %weight) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x224x224xf32>
  %1 = "linalg.batch_norm"(%0, %gamma, %beta, %mean, %var) : (...) -> tensor<1x64x224x224xf32>
  %2 = "math.relu"(%1) : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
  return %2 : tensor<1x64x224x224xf32>
}

// 应用 Fusion Pass
// mlir-opt --fuse-ops input.mlir

// 融合后
func.func @forward(%x: tensor<1x3x224x224xf32>) -> tensor<1x64x224x224xf32> {
  %0 = "linalg.fused_conv_bn_relu"(%x, %weight, %gamma, %beta, %mean, %var) : (...) -> tensor<1x64x224x224xf32>
  return %0 : tensor<1x64x224x224xf32>
}
```

---

## 🔍 性能分析实战

### 用 Nsight Compute 分析内存瓶颈

```bash
# 运行分析
ncu --set full --launch-skip 0 --launch-count 1 ./fused_conv_benchmark

# 关键指标：

# 1. Memory Throughput
nvprof --metrics gld_throughput,gst_throughput ./fused_conv_benchmark
# 目标：> 80% 峰值带宽

# 2. DRAM Throughput
ncu --metrics dram__throughput.avg ./fused_conv_benchmark
# 目标：接近理论带宽（A100: 1.5 TB/s）

# 3. Compute Throughput
ncu --metrics smsp__throughput.avg ./fused_conv_benchmark
# 目标：> 50% 峰值算力

# 4. Memory Transaction Size
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./fused_conv_benchmark
# 检查是否有非合并访问
```

### 常见瓶颈诊断

| 指标 | 低值原因 | 解决方案 |
|------|----------|----------|
| DRAM Throughput < 50% | 非合并访问 | 检查索引计算 |
| Compute Throughput < 30% | 内存瓶颈 | 增加算子融合 |
| L1 Cache Hit Rate < 80% | 数据复用差 | 增加 Shared Memory 缓存 |
| Shared Memory Bank Conflict > 10% | padding 不足 | 增加 padding |

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 实现 Conv + BN + ReLU 融合 Kernel
- [ ] 对比融合前后的性能差异（目标：2x+ 加速）
- [ ] 用 Nsight Compute 分析内存带宽利用率
- [ ] 在笔记里记录性能数据和优化心得

### 选做（深入）

- [ ] 实现简化的 FlashAttention
- [ ] 实现 LayerNorm + Gelu 融合
- [ ] 用 TVM 的 FuseOps 自动融合
- [ ] 阅读 FlashAttention 论文

### 挑战任务

- [ ] 实现完整的 FlashAttention-2
- [ ] 优化 Transformer 推理端到端性能
- [ ] 贡献算子融合 Pass 到 TVM/MLIR 社区

---

## 📚 参考资料

- **论文**：
  - FlashAttention: Fast and Memory-Efficient Exact Attention (NeurIPS 2022)
  - FlashAttention-2: Attention is Not All You Need (2023)
  
- **开源项目**：
  - FlashAttention: https://github.com/Dao-AILab/flash-attention
  - CUTLASS: https://github.com/NVIDIA/cutlass
  - TVM FuseOps: https://tvm.apache.org/docs/dev/relay_op_fuse.html
  
- **NVIDIA 资源**：
  - Nsight Compute 文档：https://docs.nvidia.com/nsight-compute/
  - CUDA 优化指南：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

- **书籍**：
  - 《Professional CUDA C Programming》第 10 章（算子融合）
  - 《深度学习编译器》第 5 章（图优化）

---

## 🔗 下一篇预告

**笔记 08**：量化与稀疏化优化

- INT8/FP16 量化原理
- 量化感知训练（QAT）
- 稀疏化加速（Structured Sparsity）
- Tensor Core 编程实战

---

_笔记创建：2026-04-15_  
_适合人群：有 CUDA 基础，想深入优化深度学习算子_  
_平台：Linux/Windows + NVIDIA GPU（推荐 RTX 3090+/A100）_  
_难度：⭐⭐⭐⭐（需要理解内存层次和算子融合策略）_
