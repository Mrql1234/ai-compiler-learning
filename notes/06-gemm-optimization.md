# 06 - GEMM 优化实战：从朴素实现到接近 cuBLAS

> 📅 学习日期：2026-04-07  
> 📚 阶段：阶段 2 - CUDA 算子优化  
> ⏱️ 预计耗时：2-3 周  
> 💻 平台：Linux/Windows + NVIDIA GPU  
> 🔗 前置：[05-CUDA 编程基础](./05-cuda-programming-basics.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **GEMM（通用矩阵乘法）** 为什么是 AI 计算的核心
2. 掌握 **多级分块（Tiling）** 优化策略
3. 理解 **Register 重用** 和 **Shared Memory 优化**
4. 理解手写 GEMM 从朴素实现走向**高性能分块 kernel**所需的关键优化
5. 学会用 **Nsight Compute** 分析性能瓶颈

---

## 💡 为什么 GEMM 如此重要？

### GEMM 往往占据主要计算量

**经验规律**：在很多以稠密线性层为主的模型里，矩阵乘法往往占据很大一部分计算量，但具体占比会随着模型结构、batch size、序列长度和算子融合策略变化。

```
Transformer 层：
  - Q/K/V 投影：GEMM
  - Attention 矩阵：GEMM
  - FFN 层：GEMM
  - 输出投影：GEMM

卷积层（im2col 后）：GEMM

全连接层：GEMM
```

**结论**：对于很多 dense workload，优化好 GEMM 往往就抓住了最主要的性能热点之一。

### 📊 GEMM 性能对比（A100 GPU, FP16）

> 说明：下表用于说明**性能趋势**，不同实现、数据类型、是否启用 Tensor Core、矩阵尺寸和 CUDA 版本都会显著影响实际数值。

| 实现方式 | 4096x4096 矩阵 | 性能（TFLOPS） | 相对 cuBLAS |
|----------|----------------|----------------|-------------|
| 朴素 CPU (单核) | 数十秒级 | < 0.1 | < 0.1% |
| 朴素 CPU (多核 AVX2) | 数秒级 | < 1 | < 1% |
| 朴素 GPU (Global Memory) | 亚秒级 | 5-20 | 2-10% |
| Shared Memory 优化 | 0.1-0.3s | 40-90 | 20-45% |
| Register Tiling + 向量化 | 0.05-0.08s | 120-180 | 60-85% |
| **cuBLAS (同精度配置)** | **约 0.045s** | **约 210** | **100%** |
| CUTLASS (模板库) | 接近 cuBLAS | 同量级 | 95-105% |

**关键洞察**：
- 朴素 GPU 实现通常只有 cuBLAS 的个位数百分比性能
- 经过系统优化，手写 kernel 有机会逼近 cuBLAS，但通常依赖 GPU 型号、尺寸和数据类型
- 最后的一截性能往往还需要 **Tensor Core、异步拷贝、流水线和更细的调参**

---

## 📚 GEMM 基础回顾

### 矩阵乘法定义

```
C = A × B

其中：
  A: M × K 矩阵
  B: K × N 矩阵
  C: M × N 矩阵

计算：C[i,j] = Σ(A[i,k] × B[k,j])  for k = 0..K-1
```

### 朴素实现（CPU 版本）

```c
// 三重循环，O(N³) 复杂度
void matmul_naive(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

**问题**：
- 缓存命中率极低（每次访问都可能是 cache miss）
- 没有利用 SIMD/并行
- 内存带宽利用率 < 10%

---

## 🔥 CUDA GEMM 优化之旅

### 版本 1：朴素 GPU 实现（Global Memory）

```cuda
__global__ void gemm_naive(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C 的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C 的列
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动配置：blockDim = (16, 16), gridDim = ((N+15)/16, (M+15)/16)
```

**性能分析**：

```
每次计算 C[i,j] 需要：
  - 读取 K 个 A 元素（A 的行）
  - 读取 K 个 B 元素（B 的列）
  - 写入 1 个 C 元素

全局内存访问次数：2 * M * N * K 次读取 + M * N 次写入

对于 1024x1024 矩阵：
  - 读取：2 * 1024³ 个 float ≈ 8GB 数据（FP32）
  - A100 Global Memory 带宽：~1.5 TB/s
  - 理论下界：8GB / 1.5TB/s ≈ 5.3ms
  - 实际时间：~8ms（说明远不止算力，还受到访存模式和缓存行为影响）
```

**瓶颈**：
1. **Global Memory 带宽限制** - 每次访问都走慢速全局内存
2. **重复读取** - 每个 A[i,k] 被读取 N 次，每个 B[k,j] 被读取 M 次
3. **局部性差** - 对单个线程来说，`B[k * N + col]` 是沿 `k` 方向的 stride-`N` 访问，复用和缓存友好性都很差

---

### 版本 2：Shared Memory 分块

**核心思想**：把矩阵分成小块，每块加载到 Shared Memory，重复使用。

```cuda
#define TILE_SIZE 32

__global__ void gemm_shared_1d(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块循环
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A 的块到 Shared Memory
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (aRow < M && aCol < K) ? 
                                        A[aRow * K + aCol] : 0.0f;
        
        // 加载 B 的块到 Shared Memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && bCol < N) ?
                                        B[bRow * N + bCol] : 0.0f;
        
        __syncthreads();  // 等待所有线程加载完成
        
        // 在 Shared Memory 中计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();  // 等待所有线程计算完成
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**优化效果**：

```
Shared Memory 带宽：~19 TB/s（A100）
Global Memory 带宽：~1.5 TB/s（A100）

带宽提升：19 / 1.5 = 12.7x

实际性能提升：
  - 朴素 GPU：8ms
  - Shared Memory：1.2ms
  - 加速比：6.7x
```

**问题**：
- `__syncthreads()` 开销大（每轮都要同步）
- Shared Memory 有 bank conflict（后面优化）

---

### 版本 3：线程级分块 + Register 累加器

**核心思想**：在 block 级分块之外，再让**每个线程计算一个小的输出子块**，把多个累加器放进 Register 中。

> 下面这段代码是**教学版示意实现**，重点展示“线程 tile 的索引关系”和“Register 外积更新”。
> 它比生产级 kernel 简单，但索引关系是自洽的。

```cuda
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16
#define THREADS_Y 8
#define THREADS_X 8
#define THREAD_TILE_M 2
#define THREAD_TILE_N 2

__global__ void gemm_register_tiling_demo(const float *A, const float *B, float *C,
                                          int M, int K, int N) {
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;
    
    // 每个线程负责 2x2 输出块
    int rowBase = blockRow + ty * THREAD_TILE_M;
    int colBase = blockCol + tx * THREAD_TILE_N;
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    
    int numTiles = (K + BLOCK_K - 1) / BLOCK_K;
    for (int t = 0; t < numTiles; t++) {
        // 8x8 个线程协作填满 16x16 的 A/B tile：每个线程各加载 4 个元素
        #pragma unroll
        for (int li = 0; li < 2; li++) {
            #pragma unroll
            for (int lj = 0; lj < 2; lj++) {
                int sRow = ty + li * THREADS_Y;
                int sCol = tx + lj * THREADS_X;

                int aRow = blockRow + sRow;
                int aCol = t * BLOCK_K + sCol;
                As[sRow][sCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

                int bRow = t * BLOCK_K + sRow;
                int bCol = blockCol + sCol;
                Bs[sRow][sCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
            }
        }
        
        __syncthreads();
        
        // Register 外积更新：一次从 Shared Memory 取出一小组 A/B 片段
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float aFrag[THREAD_TILE_M];
            float bFrag[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                aFrag[i] = As[ty * THREAD_TILE_M + i][k];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                bFrag[j] = Bs[k][tx * THREAD_TILE_N + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    acc[i][j] += aFrag[i] * bFrag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回这个线程负责的 2x2 输出块
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int row = rowBase + i;
            int col = colBase + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}
```

**Register 重用分析**：

```
每个线程计算 2x2 = 4 个输出元素
每个输出元素需要 K 次乘加
总共：4 * K 次乘加

从 Shared Memory 读取：
  - A: 同一个 A 片段会被这个线程复用到多个输出列
  - B: 同一个 B 片段会被这个线程复用到多个输出行

Register 中的累加器：
  - 4 个 float = 16 bytes（教学版 thread tile）
  - 访问代价远低于 Shared / Global Memory
  - 真正高性能实现通常会扩大 thread tile，但也会提高 register 压力

性能趋势（示意）：
  - Shared Memory V2：约 1.2ms
  - Register Tiling：约 0.3-0.5ms
  - 通常能继续获得数倍提升，但会开始受 register 数量和 occupancy 约束
```

---

### 版本 4：避免 Bank Conflict

**问题**：Shared Memory 按地址交错映射到多个 bank；同一 warp 中如果多个线程访问落到同一 bank 的不同地址，就会发生 bank conflict，导致访问串行化。

```
以常见 NVIDIA GPU 的 FP32 访问为例：
  - Shared Memory 逻辑上分成 32 个 bank
  - 连续的 4-byte 字通常映射到连续 bank（按 bank 编号取模）
  - warp 中 32 个线程理想情况下分别访问不同 bank

32x32 的 float 矩阵：
  As[32][32]
  
如果按行优先布局：
  - 行访问通常更自然：相邻元素地址连续
  - 列访问可能出现 stride=32 的模式

问题示例：
  - 如果 warp 里的线程访问 `As[row][fixed_col]`
  - 连续两行在内存里相隔 32 个 float
  - `32 mod 32 = 0`，就可能映射到同一个 bank，形成冲突
```

**解决方案**：添加 padding，打散 bank 映射。

```cuda
// 错误：32x32 会有 bank conflict
__shared__ float As[32][32];

// 正确：添加 padding
__shared__ float As[32][33];  // 每行多 1 个元素

// 或者更通用：
#define SHARED_PADDING 1
__shared__ float As[TILE_M][TILE_K + SHARED_PADDING];
```

**效果**：
- padding 的收益依访问模式而定，常见收益是几个百分点到十几个百分点
- 它本质上是在改变每一行的 stride，打散原来容易冲突的 bank 映射

---

### 版本 5：Vectorized Memory Access

**核心思想**：在连续、对齐的维度上用 `float4` 一次读取 4 个 float，提高带宽利用率。

> 下面是**示意代码**。`float4` 只适合用在地址连续且满足 16-byte 对齐的访存路径上。

```cuda
const float4* B4 = reinterpret_cast<const float4*>(B + base_offset);
float4 vec = B4[vec_idx];  // 一次取 4 个连续 float

// 等价于：
// vec.x, vec.y, vec.z, vec.w
```

**要求**：
- 内存地址必须 16-byte 对齐
- 被向量化的那一维最好是 4 的倍数
- 只适用于连续维度，不能机械地替换所有标量加载

**效果**：
- 理想情况下可减少指令条数并改善带宽利用率
- 实际收益取决于对齐情况、访存合并和 kernel 其余瓶颈

---

## 📊 完整性能对比

> 说明：下表按 `2*M*K*N` 计算 FLOPs，数值用于展示趋势，不代表所有 GPU/编译选项下的固定结果。

### 1024x1024 矩阵乘法（FP32, A100，示意）

| 版本 | 技术 | 时间 (ms) | GFLOPS | vs cuBLAS |
|------|------|-----------|--------|-----------|
| CPU 单核 | 朴素 | 850 | 2.5 | 0.8% |
| CPU 多核 | AVX2 + OpenMP | 45 | 47 | 15% |
| GPU V1 | Global Memory | 8.2 | 260 | 1.8% |
| GPU V2 | Shared Memory | 1.2 | 1780 | 12.5% |
| GPU V3 | Register Tiling | 0.35 | 6130 | 43% |
| GPU V4 | + Bank Conflict Fix | 0.30 | 7160 | 50% |
| GPU V5 | + Vectorized Load | 0.24 | 8950 | 63% |
| **cuBLAS** | **官方优化** | **0.15** | **14200** | **100%** |

### 4096x4096 矩阵乘法（FP32, A100，示意）

| 版本 | 时间 (ms) | GFLOPS | vs cuBLAS |
|------|-----------|--------|-----------|
| GPU V5 (手写) | 2.8-3.2 | 10800-12400 | 80-90% |
| cuBLAS | 2.4-2.8 | 13000-14500 | 100% |

**关键洞察**：
- 大矩阵更容易达到高利用率（并行度更高）
- 手写优化在大矩阵、合适 GPU 和充分调参下，可能逼近 cuBLAS
- 最后的一截性能通常还需要 Tensor Core、异步流水线和更细粒度的工程优化

---

## 🔍 性能分析实战（Nsight Compute）

### 关键指标

```bash
# 运行分析
ncu --set full --kernel-name regex:gemm_optimized ./gemm_benchmark

# 查看当前环境支持哪些 metrics / sections
ncu --query-metrics
```

建议重点查看的 Nsight Compute 页面 / 指标类别：

- `Launch Statistics`：看 occupancy、block 配置是否合理
- `Memory Workload Analysis`：看 DRAM / L2 / Shared Memory 的吞吐和瓶颈
- `Scheduler Statistics`：看 stall 原因是否主要来自 memory dependency 或同步
- `Source / SASS`：看关键循环有没有被展开、是否出现过多重放或长延迟指令

> 不同 GPU 架构和 CUDA 版本下，具体 metric 名称会变化；相比已经逐步淘汰的 `nvprof`，现在应优先使用 `ncu`。

### 常见瓶颈诊断

| 指标 | 低值原因 | 解决方案 |
|------|----------|----------|
| Occupancy < 30% | Register/Shared Memory 太多 | 减少每线程资源 |
| Global Load Efficiency < 50% | 非合并访问 | 检查索引计算 |
| Shared Bank Conflict > 10% | padding 不足 | 增加 padding |
| Compute Efficiency < 50% | 分支/同步过多 | 减少 `__syncthreads()` |

---

## 💻 实战代码

### 可运行的分块 + Register Tiling 版本

```cuda
// gemm_optimized.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16
#define THREADS_Y 8
#define THREADS_X 8
#define THREAD_TILE_M 2
#define THREAD_TILE_N 2
#define SHARED_PADDING 1

__global__ void gemm_optimized(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    __shared__ float As[BLOCK_M][BLOCK_K + SHARED_PADDING];
    __shared__ float Bs[BLOCK_K][BLOCK_N + SHARED_PADDING];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;
    
    int rowBase = blockRow + ty * THREAD_TILE_M;
    int colBase = blockCol + tx * THREAD_TILE_N;
    
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    
    int numTiles = (K + BLOCK_K - 1) / BLOCK_K;
    
    for (int t = 0; t < numTiles; t++) {
        // 每个线程为 A tile 和 B tile 各加载 2x2 个元素
        #pragma unroll
        for (int li = 0; li < 2; li++) {
            #pragma unroll
            for (int lj = 0; lj < 2; lj++) {
                int sRow = ty + li * THREADS_Y;
                int sCol = tx + lj * THREADS_X;

                int aRow = blockRow + sRow;
                int aCol = t * BLOCK_K + sCol;
                As[sRow][sCol] = (aRow < M && aCol < K) ?
                                 A[aRow * K + aCol] : 0.0f;

                int bRow = t * BLOCK_K + sRow;
                int bCol = blockCol + sCol;
                Bs[sRow][sCol] = (bRow < K && bCol < N) ?
                                 B[bRow * N + bCol] : 0.0f;
            }
        }
        
        __syncthreads();
        
        // 计算这个线程负责的 2x2 输出子块
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float aFrag[THREAD_TILE_M];
            float bFrag[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                aFrag[i] = As[ty * THREAD_TILE_M + i][k];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                bFrag[j] = Bs[k][tx * THREAD_TILE_N + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    acc[i][j] += aFrag[i] * bFrag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int row = rowBase + i;
            int col = colBase + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// 性能测试代码
void benchmark_gemm(int M, int K, int N) {
    // 分配内存
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    // 初始化测试数据
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    
    // 设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // 启动配置
    dim3 blockDim(THREADS_X, THREADS_Y);
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N,
                 (M + BLOCK_M - 1) / BLOCK_M);
    
    // 预热
    gemm_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    
    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        gemm_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsedMs;
    cudaEventElapsedTime(&elapsedMs, start);
    elapsedMs /= iterations;
    
    // 计算性能
    float gflops = (2.0f * M * K * N) / (elapsedMs * 1e6);
    printf("GEMM %dx%dx%d: %.2f ms, %.1f GFLOPS\n", M, K, N, elapsedMs, gflops);
    
    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}

int main() {
    benchmark_gemm(1024, 1024, 1024);
    benchmark_gemm(2048, 2048, 2048);
    benchmark_gemm(4096, 4096, 4096);
    return 0;
}
```

### 编译和运行

```bash
# 编译
nvcc -O3 -arch=sm_80 gemm_optimized.cu -o gemm_benchmark

# 运行
./gemm_benchmark

# 性能分析
ncu --set full ./gemm_benchmark
```

---

## 🎓 进阶优化方向

### 1. Tensor Core (FP16/INT8)

```cuda
// A100/V100 支持 Tensor Core
#include <mma.h>
using namespace nvcuda;

// WMMA 矩阵片段
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// 一次计算 16x16x16 的矩阵乘法
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**性能提升**：FP16 下可达 **312 TFLOPS**（A100）

### 2. 多级流水线（Double Buffering）

```cuda
// 伪代码：真实实现通常使用 cp.async / cuda::pipeline / cooperative groups
// 这里沿用前文的 tile 宏，例如 BLOCK_M / BLOCK_N / BLOCK_K
// 用两个 Shared Memory 块，重叠加载和计算
__shared__ float As[2][BLOCK_M][BLOCK_K];
__shared__ float Bs[2][BLOCK_K][BLOCK_N];

int writePhase = 0;
int readPhase = 1;

for (int t = 0; t < numTiles; t++) {
    // 异步加载下一块
    load_async(As[writePhase], Bs[writePhase], t);
    
    // 计算当前块
    compute(As[readPhase], Bs[readPhase]);
    
    // 切换相位
    writePhase = 1 - writePhase;
    readPhase = 1 - readPhase;
    
    __syncthreads();
}
```

### 3. 自动调优（AutoTVM / Ansor）

不同 TVM 版本在 `auto_scheduler` / `meta_schedule` 的 API 上差异较大，这里不再内嵌一段容易失效的“半正确代码”。

更准确的理解是：

- TVM 可以把 GEMM 表达成 `TE/TIR` 或更高层 IR
- 然后针对 `cuda` target 自动搜索 block size、thread tile、unroll、vectorize 等参数
- 如果你想在当前仓库里建立直觉，建议先回看 `04` 里已经适配本机环境的调度实验
- 真正跑 CUDA target 的 TVM 自动调优，需要在有 NVIDIA GPU 的环境里参考对应 TVM 版本的官方示例

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 实现朴素 GPU GEMM，记录性能基线
- [ ] 实现 Shared Memory 分块版本，对比性能提升
- [ ] 实现 Register Tiling 版本，观察 thread tile 对性能和 occupancy 的影响
- [ ] 用 Nsight Compute 分析性能瓶颈

### 选做（深入）

- [ ] 添加 Bank Conflict 优化，观察 padding 是否带来额外收益
- [ ] 实现 Vectorized Memory Access
- [ ] 尝试 Tensor Core（如果有 A100/V100）
- [ ] 阅读 CUTLASS 源码，学习工业级实现

### 挑战任务

- [ ] 实现卷积的 im2col + GEMM
- [ ] 优化 Transformer 的 Attention 矩阵乘法
- [ ] 参与 TVM / MLIR 的 GEMM 优化社区

---

## 📚 参考资料

- **NVIDIA GTC 演讲**：
  - "CUDA C++ 编程入门"
  - "Optimizing Matrix Multiplication"
  - "Tensor Core Programming"
  
- **开源项目**：
  - CUTLASS：https://github.com/NVIDIA/cutlass
  - cuBLAS 文档：https://docs.nvidia.com/cuda/cublas/
  
- **论文**：
  - "Anatomy of High-Performance Matrix Multiplication" (ACM ToMS 2008)
  - "CUTLASS: Composable CUDA Templates for Linear Algebra" (NVIDIA 2019)

- **书籍**：
  - 《CUDA C 编程权威指南》第 7-9 章
  - 《Professional CUDA C Programming》

---

## 🔗 下一篇预告

**笔记 07**：算子融合与内存优化

- 算子融合（Operator Fusion）原理
- 减少 Global Memory 访问
- TVM / MLIR 中的融合策略
- 实战：融合 Conv + BN + ReLU

---

_笔记创建：2026-04-07_  
_适合人群：有 CUDA 基础，想深入优化 GPU 算子_  
_平台：Linux/Windows + NVIDIA GPU（推荐 A100/V100/RTX 3090+）_  
_难度：⭐⭐⭐⭐（需要理解 GPU 架构和性能分析）_
