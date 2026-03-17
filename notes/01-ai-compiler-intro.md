# 01 - AI 编译器入门：给 Java 开发者的第一堂课

> 📅 学习日期：2026-03-17  
> 📚 阶段：阶段 1 - AI 编译器入门  
> ⏱️ 预计耗时：1-2 周

---

## 🎯 学习目标

学完这篇，你应该能：

1. 说清楚 **AI 编译器是干什么的**
2. 理解 **计算图、算子、IR** 这些核心概念
3. 用 TVM 编译一个简单模型到 CPU/GPU
4. 理解 MLIR 的基本思想（不用深究实现）
5. 用 `torch.compile` 优化 PyTorch 模型

---

## 💡 为什么需要 AI 编译器？

### 从 Java 的角度理解

想象一下，如果你写的 Java 代码**每次运行都直接解释执行**，没有 JIT 编译：

```java
// 没有 JIT，每次都解释执行
for (int i = 0; i < 1000000; i++) {
    sum += i;  // 每次都要解析这行代码
}
```

是不是很慢？**AI 模型推理也是同样的问题**。

| 场景 | 问题 | 解决方案 |
|------|------|----------|
| Java 代码 | 解释执行太慢 | **JIT 编译器** (C1/C2) |
| Python 模型 | 动态图执行开销大 | **AI 编译器** (TVM/MLIR/XLA) |
| 神经网络 | 算子之间内存访问多 | **算子融合** |

### AI 编译器的核心价值

```
原始模型 (PyTorch/TensorFlow)
    ↓
【图优化】合并算子、删除冗余计算
    ↓
【代码生成】生成优化的 CPU/GPU 代码
    ↓
高性能推理 (比原生快 2-10 倍)
```

---

## 📚 核心概念解析

### 1. 计算图 (Compute Graph)

**Java 类比**：想象一个 `Stream` 流水线

```java
// Java Stream - 类似计算图
list.stream()
    .filter(x -> x > 0)      // 节点 1
    .map(x -> x * 2)         // 节点 2
    .collect(toList());      // 节点 3
```

**神经网络计算图**：

```
输入 → [Conv2d] → [ReLU] → [MaxPool] → [Linear] → 输出
         ↓           ↓          ↓           ↓
       算子 1      算子 2      算子 3      算子 4
```

**关键点**：
- 每个框是一个**算子 (Operator)**
- 箭头是**张量 (Tensor)** 数据流
- 编译器可以**整体优化**这个图

---

### 2. 中间表示 (IR)

**Java 类比**：JVM 字节码

```java
// Java 源码
int sum = a + b * c;

// JVM 字节码 (简化)
LOAD a
LOAD b
LOAD c
MUL
ADD
STORE sum
```

**AI 编译器的 IR**：

```
// 高层 IR (类似 Relay IR)
%0 = conv2d(%input, %weight)
%1 = relu(%0)
%2 = maxpool(%1)
return %2

// 底层 IR (类似 LLVM IR)
%3 = call @cuda_conv2d(%input, %weight)
%4 = call @cuda_relu(%3)
...
```

**为什么需要 IR？**
- 统一表示不同框架的模型（PyTorch、TensorFlow 都能转成 IR）
- 在 IR 上做优化（与框架无关）
- 生成不同后端的代码（CPU、GPU、NPU）

---

### 3. 算子融合 (Operator Fusion)

**问题**：每个算子单独执行，需要多次读写内存

```
[Conv2d] → 写内存 → [ReLU] → 写内存 → [MaxPool]
   ↓                    ↓                    ↓
 读输入                读中间结果          读中间结果
```

**融合后**：

```
[Conv2d + ReLU + MaxPool] → 一次写内存
   ↓
 只在最后写一次
```

**Java 类比**：Stream 的惰性求值 + 融合

```java
// 不融合 - 多次遍历
list.stream().filter(...).collect();  // 遍历 1
list.stream().map(...).collect();     // 遍历 2

// 融合 - 一次遍历
list.stream().filter(...).map(...).collect();  // 遍历 1 次
```

**性能提升**：减少内存访问，提升 2-5 倍

---

### 4. Dialect (方言)

**MLIR 的核心概念**，**Java 类比**：不同 DSL

```java
// 类似不同的 Dialect
Stream API      // 一种"方言"
CompletableFuture  // 另一种"方言"
Reactive Stream    // 又一种"方言"
```

**MLIR Dialect**：

```
// Tensor Dialect - 描述张量操作
%0 = tensor.empty() : tensor<4xf32>

// Math Dialect - 描述数学运算
%1 = math.exp %0 : f32

// GPU Dialect - 描述 GPU 执行
gpu.launch_func ... 
```

**为什么需要多个 Dialect？**
- 不同抽象层次用不同方言
- 逐步 lowering（高层 → 底层）
- 模块化，易于扩展

---

## 🔧 实践 1：TVM 入门

### 环境准备

```bash
# 创建虚拟环境（推荐）
python3 -m venv ai-compiler-env
source ai-compiler-env/bin/activate

# 安装 TVM
pip install apache-tvm -U
```

### 第一个 TVM 示例：矩阵乘法

```python
import tvm
from tvm import te
import numpy as np

# 1. 定义计算（类似定义神经网络层）
M, N, K = 128, 128, 128
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
k = te.reduce_axis((0, K), 'k')
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

# 2. 创建调度（类似 JVM 的优化策略）
s = te.create_schedule(C.op)

# 3. 编译（生成机器码）
target = 'llvm'  # CPU 后端
f = tvm.build(s, [A, B, C], target, name='matmul')

# 4. 执行
ctx = tvm.cpu(0)
a = tvm.nd.array(np.random.rand(M, K).astype('float32'), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype('float32'), ctx)
c = tvm.nd.array(np.zeros((M, N), dtype='float32'), ctx)

f(a, b, c)
print(f"计算完成，结果形状：{c.shape}")
```

**运行看看效果**：

```bash
python tvm_matmul.py
```

---

## 🔧 实践 2：PyTorch 2.x torch.compile

这是**最容易上手**的 AI 编译器实践！

### 环境

```bash
pip install torch>=2.0
```

### 示例代码

```python
import torch
import time

# 定义一个简单模型
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 创建模型和输入
model = SimpleNet()
x = torch.randn(32, 128)

# 🔥 关键：编译模型
compiled_model = torch.compile(model)

# 第一次运行会编译（有开销）
print("第一次运行（包含编译）...")
start = time.time()
y1 = compiled_model(x)
print(f"耗时：{time.time() - start:.4f}s")

# 第二次运行（用编译后的代码）
print("第二次运行（编译后）...")
start = time.time()
y2 = compiled_model(x)
print(f"耗时：{time.time() - start:.4f}s")

# 对比原生模型
print("原生模型...")
start = time.time()
y3 = model(x)
print(f"耗时：{time.time() - start:.4f}s")

# 验证结果正确
print(f"结果一致：{torch.allclose(y1, y3)}")
```

**预期输出**：
```
第一次运行（包含编译）... 耗时：2.5s  (包含编译开销)
第二次运行（编译后）...   耗时：0.3s  (编译后加速)
原生模型...              耗时：0.8s
结果一致：True
```

---

## 📝 Java 开发者视角：关键对比

| JVM 概念 | AI 编译器对应 | 说明 |
|----------|--------------|------|
| Java 字节码 | Relay IR / MLIR | 中间表示 |
| JIT 编译器 (C1/C2) | TVM / XLA | 运行时编译优化 |
| 热点代码探测 | 图优化 Pass | 识别可优化的部分 |
| 内联优化 | 算子融合 | 减少调用开销 |
| 逃逸分析 | 内存复用 | 减少内存分配 |
| GC | 张量内存池 | 管理内存生命周期 |

**你的优势**：
- ✅ 理解编译优化思想（JIT 你肯定听过）
- ✅ 有性能调优经验（GC 调优、JVM 参数）
- ✅ 工程能力强（Java 项目通常更复杂）

**需要补的**：
- ⚠️ Python 基础（1 周能搞定）
- ⚠️ 神经网络基础（知道前向传播就行）
- ⚠️ GPU 编程模型（后续阶段再深入）

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 安装 TVM，跑通矩阵乘法示例
- [ ] 安装 PyTorch 2.x，用 `torch.compile` 优化一个模型
- [ ] 画出计算图示例（手绘或 draw.io）
- [ ] 用自己的话解释：什么是算子融合？

### 选做（深入）

- [ ] 阅读 TVM 官方教程前 3 章
- [ ] 对比 `torch.compile` 前后的性能差异
- [ ] 在笔记里记录遇到的问题和解决方案

---

## 🤔 常见问题

### Q1: AI 编译器和传统编译器（gcc/clang）有什么区别？

**A**: 
- 传统编译器：源代码 → 机器码
- AI 编译器：计算图 → 优化的机器码
- 核心思想类似（IR、优化 Pass、代码生成），但 AI 编译器处理的是**动态图**和**张量计算**

### Q2: 我需要先学完 C++ 再学 AI 编译器吗？

**A**: **不需要！** 
- TVM 有 Python API
- PyTorch 2.x 也是 Python
- 先会用，再深入源码（C++）

### Q3: 没有 GPU 能学吗？

**A**: **可以！**
- TVM 支持 CPU 后端（llvm）
- `torch.compile` 也能在 CPU 上跑
- GPU 优化是后续阶段的内容

---

## 📚 参考资料

- TVM 官方教程：https://tvm.apache.org/docs/tutorial/
- PyTorch 2.x 编译：https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- 本文对应的攻略：`../resources/ai-compiler-study-guide.md`

---

## 📅 下次预告

**笔记 02**：深入 MLIR - 理解编译器的事实标准

- MLIR 是什么，为什么重要
- Dialect、Operation、Pass 详解
- 写一个超简单的 MLIR Pass

---

_笔记创建：2026-03-17_  
_适合人群：Java 应用开发背景，AI 编译器零基础_
