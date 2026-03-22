# 05 - CUDA 编程基础：GPU 并行模型入门

> 📅 学习日期：2026-03-22  
> 📚 阶段：阶段 2 - CUDA 算子优化  
> ⏱️ 预计耗时：3-4 周  
> 💻 平台：Linux/Windows + NVIDIA GPU（或 macOS + 云 GPU）  
> 🔗 前置：[01-AI 编译器入门](./01-ai-compiler-intro.md), [04-TVM 调度优化](./04-tvm-scheduling-optimization.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 理解 **GPU 并行模型**（Thread/Block/Grid）
2. 理解 **GPU 内存层次**（Global/Shared/Register）
3. 写出 **第一个 CUDA 程序**（向量加法）
4. 理解 **Warp、Occupancy、内存合并访问** 等核心概念
5. 能分析和优化简单的 CUDA kernel 性能

---

## 💡 为什么需要 CUDA？

### 从 Java 的视角理解 GPU

想象一下，如果你的 Java 程序可以这样写：

```java
// 串行版本（CPU 思维）
for (int i = 0; i < 1000000; i++) {
    c[i] = a[i] + b[i];  // 每次处理 1 个元素
}

// 并行版本（GPU 思维）
parallel_for (int i = 0; i < 1000000; i++) {
    c[i] = a[i] + b[i];  // 100 万个线程同时处理！
}
```

**GPU 的本质**：用**数量换时间**——用成千上万个简单核心，同时做相同的事。

### CPU vs GPU 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                      CPU (Intel/AMD)                        │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                        │
│  │ Core│  │ Core│  │ Core│  │ Core│  ← 4-16 个强核心       │
│  │ 大缓存 │  │ 大缓存 │  │ 大缓存 │  │ 大缓存 │                        │
│  └─────┘  └─────┘  └─────┘  └─────┘                        │
│                                                             │
│  特点：低延迟、强单核、适合复杂逻辑                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      GPU (NVIDIA)                           │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┐                 │
│  │SM  │SM  │SM  │SM  │SM  │SM  │SM  │SM  │  ← 几十到上百个  │
│  │┌──┐│┌──┐│┌──┐│┌──┐│┌──┐│┌──┐│┌──┐│┌──┐│    流式多处理器  │
│  ││██│││██│││██│││██│││██│││██│││██│││██││                 │
│  │└──┘│└──┘│└──┘│└──┘│└──┘│└──┘│└──┘│└──┘│                 │
│  └────┴────┴────┴────┴────┴────┴────┴────┘                 │
│                                                             │
│  特点：高吞吐、弱单核、适合大规模并行                         │
└─────────────────────────────────────────────────────────────┘
```

### 📊 真实性能对比

| 操作 | CPU (单核) | CPU (多核) | GPU | 加速比 |
|------|-----------|-----------|-----|--------|
| 向量加法 (100M 元素) | 250ms | 45ms | 3ms | **83x** |
| 矩阵乘法 (1024x1024) | 800ms | 150ms | 8ms | **100x** |
| 矩阵乘法 (4096x4096) | 50s | 10s | 0.3s | **167x** |

**关键洞察**：
- GPU 适合**数据并行**（相同操作，大量数据）
- GPU 不适合**复杂逻辑**（分支多、依赖多）
- **内存传输开销**是关键瓶颈（CPU↔GPU 数据传输）

---

## 📚 核心概念详解

### 1. GPU 并行层次：Thread/Block/Grid

**定义**：CUDA 用三层层次组织并行线程。

```
Grid (网格) - 整个 kernel 启动
    ├── Block (块) 0,0
    │   ├── Thread (0,0)  Thread (1,0)  ...  Thread (31,0)
    │   ├── Thread (0,1)  Thread (1,1)  ...  Thread (31,1)
    │   └── ...
    ├── Block (1,0)
    │   ├── Thread (0,0)  Thread (1,0)  ...
    │   └── ...
    └── ...
```

**代码表示**：

```cuda
// 启动配置：gridDim x blockDim
myKernel<<<gridDim, blockDim>>>(args...);

// 例如：
// gridDim = (32, 1, 1)   // 32 个 Block
// blockDim = (256, 1, 1) // 每个 Block 256 个 Thread
// 总线程数 = 32 * 256 = 8192
```

**Java 类比**：

```java
// Java 的线程池类比
ExecutorService pool = Executors.newFixedThreadPool(256);  // Block 大小

// 启动 32 个任务，每个任务用 256 个线程
for (int blockId = 0; blockId < 32; blockId++) {
    pool.submit(() -> {
        // 每个任务内有 256 个"虚拟线程"
        for (int threadId = 0; threadId < 256; threadId++) {
            // 处理数据
        }
    });
}

// CUDA 的区别：硬件级支持，线程切换开销几乎为零
```

**线程索引计算**：

```cuda
__global__ void myKernel(float* data) {
    // 1D 情况（最常用）
    int threadId = threadIdx.x;           // 块内线程 ID (0-255)
    int blockId = blockIdx.x;             // 块 ID (0-31)
    int blockDim = blockDim.x;            // 块大小 (256)
    
    // 全局线程 ID
    int globalId = blockId * blockDim + threadId;
    
    // 处理数据
    data[globalId] = ...;
}
```

**2D/3D 索引**（处理图像/矩阵时常用）：

```cuda
__global__ void matrixKernel(float* matrix, int width, int height) {
    // 2D 线程索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    
    if (col < width && row < height) {
        int idx = row * width + col;
        matrix[idx] = ...;
    }
}

// 启动 2D grid
dim3 blockDim(16, 16);  // 16x16 = 256 线程/块
dim3 gridDim((width + 15) / 16, (height + 15) / 16);
matrixKernel<<<gridDim, blockDim>>>(...);
```

---

### 2. GPU 内存层次

**GPU 有多种内存，速度和作用域不同**：

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 内存层次结构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Register (寄存器)                                          │
│  - 速度：最快（1 周期）                                      │
│  - 容量：每个线程 255 个 32 位寄存器                           │
│  - 作用域：线程私有                                          │
│  - 用途：局部变量、中间计算                                   │
│                                                             │
│  L1 Cache / Shared Memory (共享内存)                        │
│  - 速度：快（~20 周期）                                       │
│  - 容量：每 Block 48KB（可配置）                              │
│  - 作用域：Block 内共享                                       │
│  - 用途：线程间通信、数据缓存                                 │
│                                                             │
│  L2 Cache (二级缓存)                                        │
│  - 速度：中等（~200 周期）                                    │
│  - 容量：几 MB                                                │
│  - 作用域：整个 GPU 共享                                       │
│  - 用途：缓存全局内存访问                                     │
│                                                             │
│  Global Memory (全局内存/HBM)                               │
│  - 速度：慢（~400 周期）                                      │
│  - 容量：几 GB 到几十 GB                                       │
│  - 作用域：整个 GPU 共享                                       │
│  - 用途：输入输出数据、大数组                                 │
│                                                             │
│  Constant Memory (常量内存)                                  │
│  - 速度：中（缓存命中时快）                                   │
│  - 容量：64KB                                                │
│  - 作用域：整个 GPU 共享（只读）                               │
│  - 用途：不变的常量、权重                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Java 类比**：

```java
// Register = 局部变量（最快，JIT 优化到寄存器）
int localSum = 0;  // 可能在寄存器中

// Shared Memory = ThreadLocal + 共享缓存
ThreadLocal<Integer> threadLocal = ...;  // 线程私有
// 但 Shared Memory 可以被 Block 内所有线程访问

// Global Memory = Heap 内存
int[] globalArray = new int[1000000];  // 在堆上，访问慢
```

**性能差异**（以 A100 GPU 为例）：

| 内存类型 | 带宽 | 延迟 | 使用建议 |
|----------|------|------|----------|
| Register | ~20 TB/s | 1 周期 | 尽量用局部变量 |
| Shared Memory | ~20 TB/s | ~20 周期 | 线程间共享数据 |
| L2 Cache | ~1.5 TB/s | ~200 周期 | 自动缓存 |
| Global Memory | ~1.5 TB/s | ~400 周期 | 减少访问，合并访问 |

---

### 3. Warp（线程束）

**定义**：Warp 是 GPU 调度的基本单位，包含 **32 个线程**。

**关键特性**：
- Warp 内的线程**同步执行**（SIMT：单指令多线程）
- 如果 Warp 内线程有**分支分歧**，会串行执行（性能下降）
- Warp 是**免费调度**的，没有线程切换开销

**分支分歧示例**：

```cuda
// ❌ 有分支分歧（性能差）
__global__ void branchKernel(float* data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id % 2 == 0) {
        data[id] = data[id] * 2;  // 偶数线程执行
    } else {
        data[id] = data[id] + 1;  // 奇数线程执行
    }
    // Warp 需要执行两次：一次执行 if，一次执行 else
}

// ✅ 无分支分歧（性能好）
__global__ void noBranchKernel(float* data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 用数学表达式代替分支
    float factor = (id % 2 == 0) ? 2.0f : 1.0f;
    float add = (id % 2 == 0) ? 0.0f : 1.0f;
    data[id] = data[id] * factor + add;
}
```

**Java 类比**：

```java
// SIMD 向量化（类似 Warp 执行）
// Java 的 Vector API（Project Panama）
IntVector a = IntVector.fromArray(SPECIES, array1, 0);
IntVector b = IntVector.fromArray(SPECIES, array2, 0);
IntVector c = a.add(b);  // 8 个元素同时计算
c.intoArray(result, 0);
```

---

### 4. Occupancy（占用率）

**定义**：Occupancy = 活跃 Warp 数 / 最大可能 Warp 数

**影响因素**：
- **寄存器使用量**：每个线程用的寄存器越多，能并发的线程越少
- **Shared Memory 使用量**：每 Block 用的共享内存越多，能并发的 Block 越少
- **Block 大小**：太小的 Block 无法充分利用 SM

**优化建议**：
- 目标 Occupancy：**50-100%**（不是越高越好！）
- Block 大小：常用 **128、256、512**（32 的倍数）
- 减少寄存器压力：避免大数组、复杂循环

---

### 5. 内存合并访问（Coalesced Access）

**定义**：当 Warp 内的线程**连续访问**全局内存时，硬件会合并成**单次事务**。

**示例**：

```cuda
// ✅ 合并访问（性能好）
__global__ void coalescedRead(float* data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[id];  // 连续访问：0, 1, 2, 3, ...
}

// ❌ 非合并访问（性能差）
__global__ void uncoalescedRead(float* data) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[id * 32];  // 跳跃访问：0, 32, 64, 96, ...
    // 每个线程访问不同的内存段，无法合并
}
```

**Java 类比**：

```java
// 合并访问 = 顺序遍历数组（缓存友好）
for (int i = 0; i < array.length; i++) {
    sum += array[i];  // 顺序访问，缓存命中率高
}

// 非合并访问 = 跳跃访问（缓存不友好）
for (int i = 0; i < array.length; i += 32) {
    sum += array[i];  // 跳跃访问，缓存命中率低
}
```

---

## 🔧 实践 1：环境准备

### 检查 GPU

```bash
# 查看 GPU 信息
nvidia-smi

# 查看 CUDA 版本
nvcc --version

# 查看设备详情
deviceQuery  # CUDA samples 中的工具
```

**预期输出**：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 On |                  N/A |
|  0%   45C    P8    10W / 250W |    500MiB /  8192MiB |      5%      Default |
+-------------------------------+----------------------+----------------------+
```

### 安装 CUDA Toolkit

**Ubuntu/Debian**：

```bash
# 方法 1：用系统包管理器
sudo apt install nvidia-cuda-toolkit

# 方法 2：从 NVIDIA 官网下载（推荐，版本更新）
# https://developer.nvidia.com/cuda-downloads
```

**macOS**：
- Apple Silicon (M1/M2) **不支持 CUDA**
- 可以用云 GPU（Colab、Lambda Labs、RunPod）
- 或用 Metal 后端（但本教程专注 CUDA）

**Windows**：
- 从 NVIDIA 官网下载 CUDA Toolkit
- 安装时包含 Visual Studio 集成

### 验证安装

```bash
# 编译并运行 deviceQuery
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

---

## 🔧 实践 2：第一个 CUDA 程序

### 向量加法

```cuda
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

// ===== CUDA Kernel =====
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // 计算全局线程 ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查（防止越界）
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// ===== 辅助函数：检查 CUDA 错误 =====
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ===== 主函数 =====
int main() {
    // 1. 设置问题规模
    int n = 1000000;  // 100 万元素
    size_t size = n * sizeof(float);
    
    printf("向量加法：n = %d\n", n);
    printf("数据大小：%.2f MB\n", size / 1024.0 / 1024.0);
    
    // 2. 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 3. 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // 4. 分配设备内存
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    
    // 5. 复制数据到 GPU
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // 6. 配置并启动 kernel
    int blockSize = 256;  // 每个 Block 的线程数
    int gridSize = (n + blockSize - 1) / blockSize;  // 向上取整
    
    printf("启动配置：gridSize = %d, blockSize = %d\n", gridSize, blockSize);
    printf("总线程数：%d\n", gridSize * blockSize);
    
    // 启动 kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // 7. 检查 kernel 执行错误
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());  // 等待 GPU 完成
    
    // 8. 复制结果回主机
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    // 9. 验证结果（检查前 10 个）
    printf("\n验证结果（前 10 个元素）：\n");
    for (int i = 0; i < 10; i++) {
        printf("  c[%d] = %.1f (期望：%.1f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }
    
    // 10. 性能测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("\n性能测试：\n");
    printf("  Kernel 执行时间：%.3f ms\n", milliseconds);
    printf("  带宽：%.2f GB/s\n", (size * 3) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / milliseconds);
    
    // 11. 清理
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\n✅ 完成！\n");
    return 0;
}
```

### 编译和运行

```bash
# 编译
nvcc -o vector_add vector_add.cu

# 运行
./vector_add
```

**预期输出**（GTX 1080 Ti）：

```
向量加法：n = 1000000
数据大小：3.81 MB
启动配置：gridSize = 3907, blockSize = 256
总线程数：1000192

验证结果（前 10 个元素）：
  c[0] = 0.0 (期望：0.0)
  c[1] = 3.0 (期望：3.0)
  c[2] = 6.0 (期望：6.0)
  ...

性能测试：
  Kernel 执行时间：0.342 ms
  带宽：33.42 GB/s

✅ 完成！
```

---

## 🔧 实践 3：矩阵乘法（朴素版本）

### 2D 线程索引示例

```cuda
// matrix_mul_naive.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

// ===== 朴素矩阵乘法 Kernel =====
__global__ void matrixMulNaive(float *A, float *B, float *C, int M, int N, int K) {
    // 计算行列索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 输出矩阵的列
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 输出矩阵的行
    
    // 边界检查
    if (col < N && row < M) {
        float sum = 0.0f;
        // 计算点积
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ===== CPU 版本（用于对比）=====
void matrixMulCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// ===== 验证结果 =====
bool verifyResult(float *gpu, float *cpu, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(gpu[i] - cpu[i]) > 1e-3) {
            printf("结果不匹配：gpu[%d] = %f, cpu[%d] = %f\n", 
                   i, gpu[i], i, cpu[i]);
            return false;
        }
    }
    return true;
}

// ===== 主函数 =====
int main() {
    // 矩阵大小：M x K * K x N = M x N
    int M = 512, N = 512, K = 512;
    
    printf("矩阵乘法：%d x %d * %d x %d = %d x %d\n", M, K, K, N, M, N);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_gpu = (float*)malloc(sizeC);
    float *h_C_cpu = (float*)malloc(sizeC);
    
    // 初始化随机数据
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // 复制数据
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // 配置 kernel
    dim3 blockDim(16, 16);  // 16x16 = 256 线程/块
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    
    printf("启动配置：gridDim = (%d, %d), blockDim = (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("总线程数：%d\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    // ===== GPU 性能测试 =====
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // 正式测试
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    gpuTime /= 10;  // 平均
    
    // 复制结果
    cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // ===== CPU 性能测试 =====
    clock_t cpuStart = clock();
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpuEnd = clock();
    float cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000;
    
    // 输出结果
    printf("\n性能对比：\n");
    printf("  CPU 时间：%.2f ms\n", cpuTime);
    printf("  GPU 时间：%.2f ms\n", gpuTime);
    printf("  加速比：%.2fx\n", cpuTime / gpuTime);
    
    // 验证
    printf("\n结果验证：%s\n", verifyResult(h_C_gpu, h_C_cpu, M, N) ? "✅ 正确" : "❌ 错误");
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    
    return 0;
}
```

### 编译和运行

```bash
nvcc -o matrix_mul_naive matrix_mul_naive.cu -O3
./matrix_mul_naive
```

**预期输出**（GTX 1080 Ti）：

```
矩阵乘法：512 x 512 * 512 x 512 = 512 x 512
启动配置：gridDim = (32, 32), blockDim = (16, 16)
总线程数：262144

性能对比：
  CPU 时间：2850.32 ms
  GPU 时间：45.23 ms
  加速比：63.02x

结果验证：✅ 正确
```

---

## 🔧 实践 4：用 Shared Memory 优化

### 理解 Shared Memory 的作用

```
朴素版本的问题：
- 每个线程从 Global Memory 读取 K 次 A 和 K 次 B
- 大量重复读取（相邻线程读取相同数据）
- Global Memory 带宽是瓶颈

优化思路：
- 用 Shared Memory 缓存数据块
- Block 内线程协作加载数据
- 减少 Global Memory 访问
```

### 优化版本

```cuda
// matrix_mul_shared.cu
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 声明 Shared Memory（每个 Block 一份）
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    float sum = 0.0f;
    
    // 分块计算
    int numTiles = (K + 15) / 16;
    for (int t = 0; t < numTiles; t++) {
        // 协作加载数据到 Shared Memory
        int tiledCol = t * 16 + threadIdx.x;
        int tiledRow = t * 16 + threadIdx.y;
        
        // 加载 A 的块
        if (row < M && tiledCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        // 加载 B 的块
        if (tiledRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        // 同步：确保所有线程都加载完成
        __syncthreads();
        
        // 计算当前块的贡献
        for (int i = 0; i < 16; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        // 同步：确保所有线程都计算完成
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 性能对比

```bash
nvcc -o matrix_mul_shared matrix_mul_shared.cu -O3
./matrix_mul_shared
```

**预期输出**：

```
性能对比：
  CPU 时间：2850.32 ms
  GPU (朴素) 时间：45.23 ms
  GPU (Shared Memory) 时间：12.45 ms
  
  加速比 (vs CPU)：229x
  加速比 (vs 朴素 GPU)：3.6x
```

---

## 📊 性能优化 Checklist

### Kernel 优化清单

```
□ 选择合适的 Block 大小（128/256/512，32 的倍数）
□ 确保内存合并访问（连续访问模式）
□ 减少分支分歧（避免 if-else 或确保 Warp 内一致）
□ 用 Shared Memory 缓存重复数据
□ 减少寄存器使用（提高 Occupancy）
□ 用常量内存存储不变的参数
□ 异步内存传输（cudaMemcpyAsync + Stream）
□ 多 GPU 并行（如果适用）
```

### 常见性能陷阱

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 非合并访问 | 带宽只有预期 1/4 | 检查线程索引计算 |
| 分支分歧 | Warp 执行时间翻倍 | 重构代码减少分支 |
| Shared Memory 不足 | Occupancy 低 | 减少每 Block 用量 |
| 寄存器溢出 | 性能急剧下降 | 减少局部变量，用 `__launch_bounds__` |
| 同步开销大 | `__syncthreads()` 太多 | 减少同步次数 |
| PCIe 传输瓶颈 | GPU 利用率低 | 用异步传输，重叠计算和传输 |

---

## 🤔 常见问题

### Q1: 没有 NVIDIA GPU 怎么学？

**A**: 
- **Google Colab**：免费 T4 GPU（https://colab.research.google.com）
- **Kaggle Kernels**：免费 P100 GPU
- **云 GPU 服务**：Lambda Labs、RunPod、Vast.ai（按小时计费）
- **本地模拟**：用 CPU 后端学习语法（性能差但能跑）

### Q2: CUDA 和 OpenCL 选哪个？

**A**: 
| 特性 | CUDA | OpenCL |
|------|------|--------|
| 生态 | NVIDIA 独占，生态完善 | 跨厂商，生态分散 |
| 性能 | 通常更好 | 依赖实现 |
| 易用性 | 文档好，工具多 | 文档分散 |
| 适用场景 | NVIDIA GPU 优先 | 需要跨平台 |

**建议**：学 CUDA（生态好），概念可迁移到 OpenCL。

### Q3: Block 大小怎么选？

**A**: 
- **起点**：256 线程/块
- **尝试**：128、256、512
- **原则**：
  - 必须是 32 的倍数（Warp 大小）
  - 不要超过 1024（硬件限制）
  - 根据寄存器/Shared Memory 用量调整

### Q4: 如何调试 CUDA 代码？

**A**: 
```bash
# 1. 用 cuda-memcheck 检查内存错误
cuda-memcheck ./your_program

# 2. 在 kernel 中加 printf（需要 compute capability >= 2.0）
__global__ void debugKernel() {
    printf("Thread %d: value = %f\n", threadIdx.x, value);
}

# 3. 用 Nsight 工具（图形化调试器）
nsight-compute ./your_program
nsight-systems ./your_program

# 4. 同步检查
cudaDeviceSynchronize();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
```

### Q5: CUDA 和 PyTorch/TVM 的关系？

**A**: 
```
PyTorch / TVM (高层框架)
    ↓
调用 CUDA 库（cuBLAS、cuDNN）
    ↓
CUDA Kernel (底层实现)
    ↓
GPU 硬件
```

**学习路径**：
1. 先学 CUDA 基础（理解底层）
2. 再用 PyTorch/TVM（高效开发）
3. 需要优化时手写 CUDA/Triton

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 安装 CUDA Toolkit，跑通 `deviceQuery`
- [ ] 编译运行 `vector_add.cu`，修改 Block 大小观察性能变化
- [ ] 编译运行 `matrix_mul_naive.cu`，记录 CPU/GPU 加速比
- [ ] 在笔记里记录遇到的问题和解决方案

### 选做（深入）

- [ ] 实现 `matrix_mul_shared.cu`，对比朴素版本性能
- [ ] 用 Nsight 分析 kernel 性能瓶颈
- [ ] 尝试不同的 Block 大小（128/256/512），找到最优配置
- [ ] 阅读 NVIDIA GTC 演讲 "CUDA C++ 编程入门"

---

## 📚 参考资料

- CUDA C 编程指南：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Samples：https://github.com/NVIDIA/cuda-samples
- NVIDIA GTC 演讲：https://www.nvidia.com/gtc/
- 《CUDA C 编程权威指南》- 机械工业出版社
- CUDA Zone（学习中心）：https://developer.nvidia.com/cuda-zone

---

## 📅 下次预告

**笔记 06**：GEMM 优化实战 - 从朴素实现到接近 cuBLAS

- 深入理解矩阵乘法优化
- 多级分块（Tiling）策略
- Register 重用技巧
- 性能分析与调优

---

_笔记创建：2026-03-22_  
_适合人群：Java 应用开发背景，GPU 编程零基础_  
_平台：Linux/Windows + NVIDIA GPU（或云 GPU）_  
_难度：⭐⭐⭐（需要理解并行思维和内存层次）_
