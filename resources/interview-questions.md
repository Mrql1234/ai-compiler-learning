# AI 编译器面试高频问题

> 整理自各大厂 JD 和面试经验 | 适合 Java 背景转行  
> 最后更新：2026-03-22

---

## 📋 目录

- [基础概念](#基础概念)
- [TVM/MLIR](#tvmmlir)
- [CUDA/GPU 优化](#cudagpu-优化)
- [LLM 推理](#llm-推理)
- [工程实践](#工程实践)
- [编程题](#编程题)

---

## 基础概念

### Q1: 什么是 AI 编译器？和传统编译器有什么区别？

**参考答案**：

```
AI 编译器 = 深度学习模型 → 优化的机器码

传统编译器：
  源码 (C/C++) → IR → 优化 → 机器码

AI 编译器：
  计算图 (PyTorch/TF) → 图优化 → 算子融合 → 
  代码生成 → 机器码 (CPU/GPU/NPU)

核心区别：
1. 处理对象不同：AI 编译器处理计算图，传统编译器处理程序代码
2. 优化目标不同：AI 编译器关注算子融合、内存复用，传统编译器关注指令调度、寄存器分配
3. 后端多样性：AI 编译器需要支持多种硬件（CPU/GPU/NPU/TPU）
```

**Java 类比**：
```java
// 传统编译器 = javac + JIT
Java 源码 → 字节码 → JIT 编译 → 机器码

// AI 编译器 = TVM/TorchInductor
PyTorch 模型 → 计算图 → 图优化 → 机器码
```

---

### Q2: 解释一下算子融合（Operator Fusion）

**参考答案**：

```
算子融合 = 合并多个算子为一个 kernel，减少内存访问

为什么需要融合？
- 每个算子单独执行需要读写全局内存
- 融合后数据在寄存器/共享内存中流转，只读写一次

典型融合模式：
1. Conv + BN + ReLU → FusedConv
2. MatMul + Bias + Gelu → FusedGemm
3. Q + K + V Projection → FusedQKV

性能收益：2-5x（减少内存带宽压力）
```

**代码示例**：
```python
# 融合前：3 次全局内存读写
conv_out = conv2d(x, w)        # 读 x, 写 conv_out
bn_out = batch_norm(conv_out)  # 读 conv_out, 写 bn_out
relu_out = relu(bn_out)        # 读 bn_out, 写 relu_out

# 融合后：1 次全局内存读写
# BN 可以合并到 Conv 的 bias 中
fused_out = fused_conv_bn_relu(x, w_fused, b_fused)
```

---

### Q3: 什么是计算图（Compute Graph）？

**参考答案**：

```
计算图 = 用图结构表示神经网络的计算流程

节点 = 算子（Conv、MatMul、ReLU 等）
边 = 张量（数据流）

两种表示方式：
1. 动态图（PyTorch 默认）：即时执行，灵活但难优化
2. 静态图（TensorFlow/XLA）：先构建图再执行，易优化但灵活性差

PyTorch 2.x 的 torch.compile 通过 TorchDynamo 捕获动态图为静态图，
然后进行优化。
```

---

## TVM/MLIR

### Q4: TVM 的工作流程是什么？

**参考答案**：

```
1. 模型导入：PyTorch/TF/ONNX → Relay IR
2. 图优化：算子融合、常量折叠、死代码消除
3. Lowering：Relay IR → TE (Tensor Expression) → TIR (Tensor IR)
4. 调度优化：分块、并行、向量化
5. 代码生成：LLVM/CUDA 后端 → 机器码

关键组件：
- Relay IR：高层计算图表示
- AutoTVM：自动搜索最优调度
- TVMC：命令行工具
```

---

### Q5: MLIR 是什么？为什么需要它？

**参考答案**：

```
MLIR (Multi-Level IR) = 多级中间表示框架

问题：传统 LLVM IR 太"低"，丢失高层语义
- 循环、张量、向量操作都变成了 load/store
- 领域特定优化难做（如神经网络的算子融合）

MLIR 的解决方案：多层 Dialect
- Tensor Dialect（高层）：张量操作
- Affine Dialect（中层）：循环优化
- LLVM Dialect（底层）：接近机器码

优势：
1. 在合适的层次做合适的优化
2. 复用底层基础设施（LLVM Codegen）
3. 易于扩展新 Dialect
```

**Java 类比**：
```java
// 多层 IR 类似 Java 的多层抽象
Stream API (高层) → 字节码 (中层) → 机器码 (底层)

// MLIR 的 Dialect 类似不同的编程范式
Stream Dialect / Reactive Dialect / Imperative Dialect
```

---

### Q6: 解释一下 MLIR 的 Dialect、Operation、Pass

**参考答案**：

```
Dialect（方言）：一组相关的 Operation 集合，代表某个抽象层次
- 如：Tensor Dialect、Math Dialect、GPU Dialect

Operation（操作）：MLIR 的基本计算单元，类似指令
- 格式：%result = dialect.op %operand {attr} : type

Pass（优化遍）：对 IR 进行转换或优化的模块
- 如：常量折叠、死代码消除、算子融合
```

**示例**：
```mlir
// Tensor Dialect 的 Operation
%0 = tensor.empty() : tensor<4xf32>
%1 = tensor.insert %v into %0[0] : tensor<4xf32>

// Math Dialect 的 Operation
%2 = math.exp %1 : f32
```

---

## CUDA/GPU 优化

### Q7: GPU 的内存层次结构是什么？

**参考答案**：

```
从快到慢：

1. Register（寄存器）
   - 速度：1 周期
   - 容量：每线程 255 个 32 位寄存器
   - 作用域：线程私有

2. Shared Memory（共享内存）
   - 速度：~20 周期
   - 容量：每 Block 48KB
   - 作用域：Block 内共享
   - 用途：线程间通信、数据缓存

3. L2 Cache
   - 速度：~200 周期
   - 容量：几 MB
   - 作用域：整个 GPU 共享

4. Global Memory（HBM）
   - 速度：~400 周期
   - 容量：几 GB 到几十 GB
   - 作用域：整个 GPU 共享
   - 用途：输入输出数据

优化原则：尽量让数据待在快的内存里
```

---

### Q8: 什么是 Warp？分支分歧（Branch Divergence）是什么？

**参考答案**：

```
Warp = GPU 调度的基本单位，包含 32 个线程

SIMT（单指令多线程）：
- Warp 内的线程同步执行同一条指令
- 如果有分支，线程会分化执行

分支分歧：
当 Warp 内线程执行不同分支时，硬件需要串行执行每个分支

示例：
__global__ void kernel(float* data) {
    if (threadIdx.x % 2 == 0) {
        data[id] = data[id] * 2;  // 偶数线程
    } else {
        data[id] = data[id] + 1;  // 奇数线程
    }
    // Warp 需要执行两次：一次 if，一次 else
}

优化：减少分支或用数学表达式代替分支
```

---

### Q9: 什么是内存合并访问（Coalesced Access）？

**参考答案**：

```
合并访问 = Warp 内线程连续访问全局内存时，硬件合并成单次事务

✅ 合并访问（好）：
thread 0 访问 addr[0]
thread 1 访问 addr[1]
thread 2 访问 addr[2]
...
→ 一次事务读取连续地址

❌ 非合并访问（差）：
thread 0 访问 addr[0]
thread 1 访问 addr[32]
thread 2 访问 addr[64]
...
→ 多次事务，带宽利用率低

性能差异：合并访问可达非合并访问的 4-8x 带宽
```

---

### Q10: GEMM 优化的核心技术有哪些？

**参考答案**：

```
GEMM（通用矩阵乘法）是深度学习最核心的算子

优化技术：

1. 分块（Tiling）
   - 将大矩阵切分为小块，提高缓存命中率
   - 多级分块：L1/L2/Shared Memory

2. Shared Memory 缓存
   - 将数据从 Global Memory 加载到 Shared Memory
   - Block 内线程共享数据，减少重复读取

3. Register 重用
   - 将数据加载到寄存器，多次复用
   - 减少 Shared Memory 访问

4. 向量化（Vectorize）
   - 使用 SIMD 指令（如 float4）
   - 一次加载/存储多个元素

5. 流水线（Pipeline）
   - 重叠内存加载和计算
   - 隐藏内存延迟

6. Warp-level Matrix Multiply
   - 使用 Tensor Core（Volta 及以上）
   - 16x16 矩阵乘法硬件加速

性能提升：朴素实现 → 优化后可达 100x+
```

---

## LLM 推理

### Q11: 什么是 KV Cache？为什么需要它？

**参考答案**：

```
KV Cache = 缓存 Attention 的 Key/Value 状态，加速自回归生成

问题：LLM 生成时，每一步都要重新计算所有 token 的 K/V
- 第 1 步：计算 token 1 的 K/V
- 第 2 步：计算 token 1,2 的 K/V（token 1 重复计算）
- 第 3 步：计算 token 1,2,3 的 K/V（token 1,2 重复计算）

解决：缓存已计算的 K/V
- 第 1 步：计算并缓存 token 1 的 K/V
- 第 2 步：复用 token 1 的 K/V，只计算 token 2 的
- 第 3 步：复用 token 1,2 的 K/V，只计算 token 3 的

性能收益：减少 50-70% 的计算量
内存开销：序列长度 × 隐藏层维度 × 层数
```

---

### Q12: 什么是 Continuous Batching？

**参考答案**：

```
Continuous Batching（动态批处理）= 动态调整批处理大小，提升 GPU 利用率

传统批处理问题：
- 固定 batch size，必须等所有请求完成才能处理下一批
- 短请求完成后，GPU 空闲等待长请求

Continuous Batching：
- 请求完成后立即插入新请求
- 不同请求可以处于不同的生成阶段
- GPU 始终保持高利用率

实现要点：
1. 动态调度器：管理请求队列
2. KV Cache 管理：支持动态插入/删除
3. 内存管理：PagedAttention 等技术

性能收益：吞吐提升 2-4x
```

---

### Q13: FlashAttention 的核心思想是什么？

**参考答案**：

```
FlashAttention = IO 感知的 Attention 实现，减少 HBM 访问

问题：标准 Attention 需要多次读写 HBM
- Q、K、V 从 HBM 加载
- 计算 Attention 矩阵（N×N，很大）
- 写回结果到 HBM

FlashAttention 的思想：
1. 分块计算：将 Q、K、V 分块，每块在 Shared Memory 中计算
2. 避免物化 Attention 矩阵：不显式存储 N×N 矩阵
3. 重计算（Recomputation）：用计算换内存，减少 HBM 访问

性能收益：
- 速度提升 2-4x
- 内存减少 5-10x（支持更长序列）
```

---

### Q14: 什么是 PagedAttention？

**参考答案**：

```
PagedAttention = 借鉴操作系统分页思想的 KV Cache 管理技术

问题：传统 KV Cache 预分配连续内存，浪费严重
- 无法预测序列长度，只能按最大值分配
- 不同序列长度差异大，碎片化严重

PagedAttention 的思想：
1. 分页管理：将 KV Cache 分为固定大小的 page
2. 非连续存储：page 可以分散在 GPU 内存中
3. 页表映射：用页表记录逻辑→物理地址映射

优势：
- 内存利用率提升 50-80%
- 支持更长的序列
- 动态扩缩容

vLLM 的核心技术之一
```

---

## 工程实践

### Q15: 如何分析 CUDA kernel 的性能瓶颈？

**参考答案**：

```
工具：
1. Nsight Compute：分析单个 kernel
2. Nsight Systems：分析整个应用
3. cuda-memcheck：检查内存错误

分析步骤：
1. 检查 Occupancy：活跃 Warp 数 / 最大 Warp 数
2. 检查内存带宽：Global Memory 利用率
3. 检查计算利用率：SM 利用率
4. 检查分支分歧：Warp 内线程是否执行不同路径

常见瓶颈：
- 内存带宽不足：优化合并访问，用 Shared Memory
- Occupancy 低：减少寄存器/Shared Memory 用量
- 分支分歧：重构代码减少分支
- 计算瓶颈：用 Tensor Core，优化算法
```

---

### Q16: torch.compile 和 TVM 有什么区别？

**参考答案**：

```
| 特性 | torch.compile | TVM |
|------|--------------|-----|
| 定位 | PyTorch 内置编译器 | 独立编译器栈 |
| 易用性 | 一行代码 | 需要学习 API |
| 后端 | Inductor (Triton) | 多后端 (CPU/GPU/NPU) |
| 图优化 | 基础 | 丰富（算子融合、常量折叠等） |
| 自动调优 | 有限 | AutoTVM/Ansor |
| 生态 | PyTorch 生态 | 跨框架 |

选择建议：
- PyTorch 项目、快速原型 → torch.compile
- 跨框架、生产部署、极致性能 → TVM
```

---

## 编程题

### 题 1：手写 CUDA 向量加法

```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// 启动配置
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

---

### 题 2：手写 CUDA 矩阵乘法（朴素版本）

```cuda
__global__ void matrixMul(float *A, float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < N && row < M) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

---

### 题 3：用 Shared Memory 优化矩阵乘法

```cuda
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    float sum = 0.0f;
    int numTiles = (K + 15) / 16;
    
    for (int t = 0; t < numTiles; t++) {
        // 加载数据到 Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * K + (t * 16 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        
        __syncthreads();
        
        // 计算
        for (int i = 0; i < 16; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

---

### 题 4：解释并实现算子融合

```python
# 融合 Conv + BN + ReLU

# BN 公式：
# y = gamma * (x - mean) / sqrt(var + eps) + beta

# 融合到 Conv：
# fused_weight = conv_weight * (gamma / sqrt(var + eps))
# fused_bias = (conv_bias - mean) * (gamma / sqrt(var + eps)) + beta

def fuse_conv_bn(conv, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    fused_weight = conv.weight * (bn.weight / std).view(-1, 1, 1, 1)
    if conv.bias is not None:
        fused_bias = (conv.bias - bn.running_mean) * (bn.weight / std) + bn.bias
    else:
        fused_bias = -bn.running_mean * (bn.weight / std) + bn.bias
    return fused_weight, fused_bias
```

---

## 📚 推荐准备

### 必读论文
1. TVM: An Automated End-to-End Optimizing Compiler (OSDI 2018)
2. MLIR: Scaling Compiler Infrastructure (CGO 2020)
3. FlashAttention (NeurIPS 2022)
4. PagedAttention/vLLM (2023)

### 实践项目
- TVM 编译一个模型到 CPU/GPU
- 手写 CUDA 矩阵乘法并优化
- 用 Triton 实现 LayerNorm/Attention
- 部署 vLLM，跑通 LLM 推理

### 简历亮点
- GitHub 仓库（有 README 和性能对比）
- 开源贡献（TVM/MLIR/vLLM 等）
- 性能优化案例（数据说话）

---

_最后更新：2026-03-22 | 整理：cx330 ✨_
