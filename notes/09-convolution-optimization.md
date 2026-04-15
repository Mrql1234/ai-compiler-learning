# 11 - 卷积算子优化：im2col 与 Winograd 算法

> 📅 学习日期：2026-04-15  
> 📚 阶段：阶段 2 - CUDA 算子优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU  
> 🔗 前置：[06-GEMM 优化实战](./06-gemm-optimization.md), [08-算子融合](./08-operator-fusion.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **卷积计算的本质和优化挑战**
2. 掌握 **im2col + GEMM** 优化方法
3. 理解 **Winograd 算法** 的数学原理
4. 能用 **CUDA/Triton** 实现优化卷积
5. 理解 **深度可分离卷积** 的优化策略

---

## 💡 卷积的本质与挑战

### 卷积计算定义

**2D 卷积公式**：

```
Output[n, oc, oh, ow] = ΣΣΣ Input[n, ic, ih, iw] × Weight[oc, ic, kh, kw]
                         ic  kh  kw

其中：
  - n: batch size
  - ic: input channels (C)
  - oc: output channels (K)
  - ih, iw: input height/width (H, W)
  - oh, ow: output height/width (H', W')
  - kh, kw: kernel height/width (R, S)
```

**朴素实现**（7 重循环）：

```c
// 朴素卷积 O(N × K × C × H' × W' × R × S)
for (int n = 0; n < N; n++) {
    for (int oc = 0; oc < K; oc++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < C; ic++) {
                    for (int kh = 0; kh < R; kh++) {
                        for (int kw = 0; kw < S; kw++) {
                            int ih = oh * stride + kh * dilation - pad;
                            int iw = ow * stride + kw * dilation - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input[n][ic][ih][iw] * weight[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                output[n][oc][oh][ow] = sum + bias[oc];
            }
        }
    }
}
```

**问题**：
- 7 重循环，**计算复杂度极高**
- 大量**重复内存访问**（权重被重复读取）
- **分支判断多**（padding 检查）

### 📊 卷积计算量分析

**示例**：ResNet-50 第一层卷积

```
输入：N=32, C=3, H=224, W=224
权重：K=64, R=7, S=7, stride=2, padding=3
输出：N=32, K=64, H'=112, W'=112

计算量：
  = N × K × H' × W' × C × R × S
  = 32 × 64 × 112 × 112 × 3 × 7 × 7
  = 32 × 64 × 12,544 × 3 × 49
  = 3,758,080,512 FLOPs
  ≈ 3.76 GFLOPs

内存访问：
  - 输入：32 × 3 × 224 × 224 × 4 bytes = 4.8 MB
  - 权重：64 × 3 × 7 × 7 × 4 bytes = 0.36 MB
  - 输出：32 × 64 × 112 × 112 × 4 bytes = 10.3 MB
  - 总计：~15.5 MB

计算强度：3.76 GFLOPs / 15.5 MB = 243 FLOPs/byte
→ 计算密集型（适合 GPU 加速）
```

---

## 🔥 im2col + GEMM 优化

### 核心思想

**im2col** = Image To Column

```
将卷积转换为矩阵乘法：

输入图像 (C, H, W)           im2col 后 (C×R×S, H'×W')
┌─────────────────┐          ┌─────────────────┐
│ C=3, H=4, W=4   │          │ C×R×S=12, H'×W'=4│
│                 │          │                 │
│ [通道 0]        │          │ [col 0] [col 1] │
│ 1 2 3 4         │  3x3 卷积 │  ...    ...    │
│ 5 6 7 8         │   →      │ [col 2] [col 3] │
│ 9 10 11 12      │          │                 │
│ 13 14 15 16     │          │                 │
└─────────────────┘          └─────────────────┘

权重 (K, C, R, S)            转置后 (K, C×R×S)
┌─────────────────┐          ┌─────────────────┐
│ K=2, C=3, R=3   │          │ K=2, C×R×S=27   │
│ S=3             │          │                 │
└─────────────────┘          └─────────────────┘

输出：GEMM(K, C×R×S) × im2col(C×R×S, H'×W') = (K, H'×W')
```

### 数学推导

```
标准卷积：
  Y = X * W  (卷积操作)

im2col 转换：
  X_col = im2col(X, kernel_size, stride, padding)
  Y = W_reshaped × X_col  (矩阵乘法)
  Y = reshape(Y, (N, K, H_out, W_out))

优势：
  - 利用高度优化的 GEMM 库（cuBLAS）
  - 内存访问模式规则化
  - 批量处理更高效
```

### CUDA 实现

```cuda
// im2col_kernel.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void im2col_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride, int pad, int dilation,
    int H_out, int W_out
) {
    // 每个线程处理 output 的一个元素
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col_idx >= H_out * W_out || row_idx >= C * R * S)
        return;
    
    // 计算输出位置
    int n = col_idx / (H_out * W_out);
    int hw_out = col_idx % (H_out * W_out);
    int oh = hw_out / W_out;
    int ow = hw_out % W_out;
    
    int c = row_idx / (R * S);
    int rs = row_idx % (R * S);
    int kr = rs / S;
    int ks = rs % S;
    
    // 计算输入位置
    int ih = oh * stride - pad + kr * dilation;
    int iw = ow * stride - pad + ks * dilation;
    
    // 边界检查
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        output[row_idx * H_out * W_out + col_idx] = input[in_idx];
    } else {
        output[row_idx * H_out * W_out + col_idx] = 0.0f;
    }
}

// im2col + GEMM 卷积
void conv2d_im2col_gemm(
    const float* input,   // (N, C, H, W)
    const float* weight,  // (K, C, R, S)
    float* output,        // (N, K, H_out, W_out)
    const float* bias,    // (K,)
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride, int pad, int dilation
) {
    // 1. 计算输出尺寸
    int H_out = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1;
    int W_out = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1;
    
    // 2. 分配 im2col 缓冲区
    int col_rows = C * R * S;
    int col_cols = N * H_out * W_out;
    float* col_buffer;
    cudaMalloc(&col_buffer, col_rows * col_cols * sizeof(float));
    
    // 3. 启动 im2col kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((col_cols + 15) / 16, (col_rows + 15) / 16);
    im2col_kernel<<<gridDim, blockDim>>>(
        input, col_buffer,
        N, C, H, W, K, R, S,
        stride, pad, dilation,
        H_out, W_out
    );
    
    // 4. 调用 cuBLAS GEMM
    // Y = W × X_col
    // (K, col_cols) = (K, C*R*S) × (C*R*S, col_cols)
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        K, col_cols, col_rows,
        &alpha,
        weight, K,           // W: (K, C*R*S)
        col_buffer, col_rows, // X_col: (C*R*S, col_cols)
        &beta,
        output, K            // Y: (K, col_cols)
    );
    
    // 5. 加 bias（融合到 GEMM 或单独 kernel）
    // ...（省略，可参考笔记 07 的融合技术）
    
    // 6. 清理
    cudaFree(col_buffer);
    cublasDestroy(handle);
}
```

### 性能对比

| 实现方式 | 相对速度 | 显存占用 | 适用场景 |
|----------|---------|---------|---------|
| 朴素卷积 | 1x | 低 | 教学/小 kernel |
| im2col + GEMM | 5-10x | 高（需要 col 缓冲） | 通用卷积 |
| Winograd | 10-15x | 中 | 3x3 小卷积 |
| 深度可分离 | 20-30x | 低 | MobileNet 等 |

---

## 🔥 Winograd 算法

### 核心思想

**Winograd 最小算法**：用更少的乘法计算卷积。

```
标准 2D 卷积 F(2×2, 3×3)：
  - 输出 2×2，kernel 3×3
  - 需要 2×2×3×3 = 36 次乘法

Winograd F(2×2, 3×3)：
  - 变换输入和权重到 Winograd 域
  - 逐元素乘法：只需 4×4 = 16 次乘法
  - 逆变换回空间域
  - 加速比：36/16 = 2.25x
```

### 数学原理

**1D Winograd 示例**：

```
标准卷积：y = h * x (h 是 kernel, x 是输入)
  y[0] = h[0]*x[0] + h[1]*x[1]
  y[1] = h[0]*x[1] + h[1]*x[2]
  需要 4 次乘法

Winograd 变换：
  1. 变换输入：x' = B^T × x
  2. 变换权重：h' = G × h
  3. 逐元素乘：y' = h' ⊙ x'
  4. 逆变换：y = A^T × y'

  其中 B, G, A 是预定义的变换矩阵
  
  乘法次数：2 次（逐元素乘）
  加速比：4/2 = 2x
```

**2D Winograd F(m×m, r×r)**：

```
算法：F(m×m, r×r)
  - 输出块大小：m×m
  - kernel 大小：r×r
  - 变换矩阵大小：(m+r-1) × (m+r-1)

常用配置：
  - F(2×2, 3×3): 4×4 变换，16 次乘法
  - F(4×4, 3×3): 6×6 变换，36 次乘法
  - F(6×6, 3×3): 8×8 变换，64 次乘法
```

### CUDA 实现（简化）

```cuda
// winograd_2d_kernel.cu
#include <cuda_runtime.h>

// Winograd F(2x2, 3x3) 变换矩阵（预计算）
__device__ const float G[6][3] = {
    {1.0,  0.0,  0.0},
    {1.0/3, 1.0/3, 1.0/3},
    {1.0/3, -1.0/3, 1.0/3},
    {1.0/12, 1.0/6, 1.0/3},
    {1.0/12, -1.0/6, 1.0/3},
    {0.0,  0.0,  1.0}
};

__device__ const float B_T[6][6] = {
    {4.0,  0.0, -5.0,  0.0,  1.0,  0.0},
    {0.0, -4.0, -4.0,  1.0,  1.0,  0.0},
    {0.0,  4.0, -4.0, -1.0,  1.0,  0.0},
    {0.0, -2.0, -1.0,  2.0,  1.0,  0.0},
    {0.0,  2.0, -1.0, -2.0,  1.0,  0.0},
    {0.0,  4.0,  0.0, -5.0,  0.0,  1.0}
};

__device__ const float A_T[4][6] = {
    {1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
    {0.0, 1.0, -1.0, 2.0, -2.0, 0.0},
    {0.0, 1.0, 1.0, 4.0, 4.0, 0.0},
    {0.0, 1.0, -1.0, 8.0, -8.0, 1.0}
};

__global__ void winograd_transform_input(
    const float* input,
    float* V,
    int N, int C, int H, int W,
    int tile_H, int tile_W
) {
    // 每个 block 处理一个 tile
    int n = blockIdx.z;
    int c = blockIdx.y;
    int tile_idx = blockIdx.x;
    int tile_y = tile_idx / tile_W;
    int tile_x = tile_idx % tile_W;
    
    // 加载输入块到 Shared Memory（6x6）
    __shared__ float d[6][6];
    
    // 边界检查 + 加载
    for (int i = threadIdx.y; i < 6; i += blockDim.y) {
        for (int j = threadIdx.x; j < 6; j += blockDim.x) {
            int in_y = tile_y * 2 + i;
            int in_x = tile_x * 2 + j;
            
            if (in_y < H && in_x < W) {
                d[i][j] = input[((n * C + c) * H + in_y) * W + in_x];
            } else {
                d[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    // 应用 B^T 变换：V = B^T × d × B
    float temp[6][6];
    for (int i = threadIdx.y; i < 6; i += blockDim.y) {
        for (int j = threadIdx.x; j < 6; j += blockDim.x) {
            temp[i][j] = 0.0f;
            for (int k = 0; k < 6; k++) {
                temp[i][j] += B_T[i][k] * d[k][j];
            }
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.y; i < 6; i += blockDim.y) {
        for (int j = threadIdx.x; j < 6; j += blockDim.x) {
            d[i][j] = 0.0f;
            for (int k = 0; k < 6; k++) {
                d[i][j] += temp[i][k] * B_T[j][k];  // B 是对称的
            }
        }
    }
    __syncthreads();
    
    // 存储变换后的 V
    for (int i = threadIdx.y; i < 6; i += blockDim.y) {
        for (int j = threadIdx.x; j < 6; j += blockDim.x) {
            int v_idx = ((n * C + c) * tile_H * tile_W * 36 + 
                        tile_idx * 36 + i * 6 + j);
            V[v_idx] = d[i][j];
        }
    }
}

__global__ void winograd_gemm(
    const float* V,      // 变换后的输入
    const float* G,      // 变换后的权重
    float* M,            // 中间结果
    int N, int C, int K,
    int tile_H, int tile_W
) {
    // 每个 block 处理一个输出 tile 的一个通道
    int n = blockIdx.z;
    int k = blockIdx.y;
    int tile_idx = blockIdx.x;
    
    // 逐元素乘法（在 Winograd 域）
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                int v_idx = ((n * C + c) * tile_H * tile_W * 36 + 
                            tile_idx * 36 + i * 6 + j);
                int g_idx = ((k * C + c) * 36 + i * 6 + j);
                int m_idx = ((n * K + k) * tile_H * tile_W * 36 + 
                            tile_idx * 36 + i * 6 + j);
                
                if (c == 0) M[m_idx] = 0.0f;
                M[m_idx] += V[v_idx] * G[g_idx];
            }
        }
    }
}

__global__ void winograd_inverse_transform(
    const float* M,
    float* output,
    int N, int K, int tile_H, int tile_W,
    int H_out, int W_out
) {
    // 逆变换：Y = A^T × M × A
    // 实现类似 transform_input，用 A^T 矩阵
    // ...（省略，原理相同）
}
```

### 性能对比

| 算法 | 3x3 卷积相对速度 | 适用 kernel 大小 |
|------|----------------|----------------|
| 朴素 | 1x | 任意 |
| im2col + GEMM | 5-8x | 任意（通用） |
| Winograd F(2x2, 3x3) | 10-12x | 3x3 最优 |
| Winograd F(4x4, 3x3) | 12-15x | 3x3 最优 |

**Winograd 的局限**：
- 只适合小 kernel（3x3, 5x5）
- 大 kernel 变换开销大
- 数值精度略低（多次浮点变换）

---

## 🔥 深度可分离卷积

### 原理

**标准卷积 vs 深度可分离卷积**：

```
标准卷积：
  输入 (C, H, W) → [Conv K×C×R×S] → 输出 (K, H', W')
  计算量：K × C × H' × W' × R × S

深度可分离卷积（两步）：
  1. Depthwise Conv:
     输入 (C, H, W) → [Conv 1×C×R×S] → 中间 (C, H', W')
     计算量：C × H' × W' × R × S
  
  2. Pointwise Conv (1x1):
     中间 (C, H', W') → [Conv K×C×1×1] → 输出 (K, H', W')
     计算量：K × C × H' × W' × 1 × 1
  
  总计算量：C × H' × W' × (R × S + K)
  
加速比（R=S=3）：
  (K × C × H' × W' × 9) / (C × H' × W' × (9 + K))
  = 9K / (9 + K)
  ≈ 8-9x (当 K >> 9)
```

### CUDA 实现

```cuda
// depthwise_conv_kernel.cu
__global__ void depthwise_conv_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N, int C, int H, int W,
    int R, int S,
    int stride, int pad, int dilation,
    int H_out, int W_out
) {
    // 每个 block 处理一个 channel 的一个 tile
    int n = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oh >= H_out || ow >= W_out)
        return;
    
    float sum = 0.0f;
    
    for (int kr = 0; kr < R; kr++) {
        for (int ks = 0; ks < S; ks++) {
            int ih = oh * stride - pad + kr * dilation;
            int iw = ow * stride - pad + ks * dilation;
            
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int in_idx = ((n * C + c) * H + ih) * W + iw;
                int w_idx = (c * R + kr) * S + ks;  // depthwise: 每个 channel 独立 kernel
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    
    int out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
    output[out_idx] = sum;
}

__global__ void pointwise_conv_kernel(
    const float* input,
    const float* weight,
    float* output,
    const float* bias,
    int N, int C, int H, int W,
    int K
) {
    // 1x1 卷积 = 通道混合
    int n = blockIdx.z;
    int k = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hw >= H * W)
        return;
    
    int h = hw / W;
    int w = hw % W;
    
    float sum = bias ? bias[k] : 0.0f;
    
    for (int c = 0; c < C; c++) {
        int in_idx = ((n * C + c) * H + h) * W + w;
        int w_idx = (k * C + c);
        sum += input[in_idx] * weight[w_idx];
    }
    
    int out_idx = ((n * K + k) * H + h) * W + w;
    output[out_idx] = sum;
}
```

### 应用场景

| 模型 | 卷积类型 | 加速效果 |
|------|---------|---------|
| ResNet-50 | 标准卷积 | - |
| MobileNetV2 | 深度可分离 | 8-10x |
| EfficientNet | 深度可分离 + MBConv | 10-12x |
| ShuffleNet | 深度可分离 + 通道混洗 | 8-10x |

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 理解 im2col 的转换原理
- [ ] 用 cuBLAS 实现 im2col + GEMM 卷积
- [ ] 对比朴素卷积和 im2col 的性能
- [ ] 在笔记里记录性能数据

### 选做（深入）

- [ ] 实现 Winograd F(2x2, 3x3) 变换
- [ ] 实现深度可分离卷积
- [ ] 用 Triton 实现卷积算子
- [ ] 阅读 cuDNN 论文了解工业级优化

---

## 📚 参考资料

- **论文**：
  - Fast Algorithms for Convolutional Neural Networks (Winograd, CVPR 2016)
  - cuDNN: Efficient Primitives for Deep Learning (NVIDIA, 2014)
  
- **开源项目**：
  - cuDNN: https://developer.nvidia.com/cudnn
  - CUTLASS Convolution: https://github.com/NVIDIA/cutlass
  
- **教程**：
  - NVIDIA GTC: "High-Performance Convolutional Neural Networks"
  - Deep Learning Systems 课程（CMU 10-414/714）

---

_笔记创建：2026-04-15_  
_适合人群：有 GEMM 优化基础，想深入卷积算子_  
_平台：Linux/Windows + NVIDIA GPU_  
_难度：⭐⭐⭐⭐（需要理解卷积数学和内存优化）_
