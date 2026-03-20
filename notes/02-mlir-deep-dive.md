# 02 - 深入 MLIR：编译器的事实标准

> 📅 学习日期：2026-03-20  
> 📚 阶段：阶段 1 - AI 编译器入门  
> ⏱️ 预计耗时：2-3 周  
> 🔗 前置：[01-AI 编译器入门](./01-ai-compiler-intro.md)

---

## 🎯 学习目标

学完这篇，你应该能：

1. 说清楚 **MLIR 是什么，为什么需要它**
2. 理解 **Dialect、Operation、Pass** 三大核心概念
3. 读懂简单的 MLIR 代码
4. 用 Python 写一个 toy Dialect
5. 写一个超简单的 MLIR Pass（常量折叠）

---

## 💡 MLIR 是什么？

### 一句话定义

**MLIR (Multi-Level IR)** = 编译器界的"瑞士军刀"——一套框架，多层抽象，统一表示。

### 为什么需要 MLIR？

**问题：传统编译器的困境**

```
传统编译器栈：
┌─────────────────────────────────────┐
│  前端：Clang (C/C++)                │
│       ↓                              │
│  中层：LLVM IR (单一抽象层)          │  ← 问题：太"低"了！
│       ↓                              │
│  后端：X86/ARM/NVPTX                │
└─────────────────────────────────────┘

问题：
- 高层语义丢失（循环、张量、向量操作都变成了 load/store）
- 领域特定优化难做（比如神经网络的算子融合）
- 每个新领域都要重复造轮子
```

**MLIR 的解决方案：多层 IR**

```
MLIR 编译器栈：
┌─────────────────────────────────────┐
│  高层：Tensor Dialect (张量操作)     │  ← 保留高层语义
│       ↓ [Lowering Pass]             │
│  中层：Affine Dialect (循环 + 内存)  │  ← 做循环优化
│       ↓ [Lowering Pass]             │
│  低层：LLVM Dialect (接近机器码)     │  ← 生成机器码
│       ↓                              │
│  后端：LLVM / CUDA / ...            │
└─────────────────────────────────────┘

优势：
- 每层保留该层次的语义
- 在合适的层次做合适的优化
- 复用底层基础设施（LLVM Codegen）
```

### 📊 Java 类比：帮你理解 MLIR

| MLIR 概念 | Java 世界对应 | 说明 |
|-----------|--------------|------|
| **Dialect** | 不同的 API/DSL | Stream API、CompletableFuture、Reactive Stream |
| **Operation** | 方法调用 | `stream.map()`, `future.thenApply()` |
| **Type** | Java 类型 | `List<String>`, `CompletableFuture<Integer>` |
| **Attribute** | 注解/元数据 | `@Deprecated`, `@Nullable` |
| **Pass** | 字节码转换器 | JIT 优化 Pass、ByteBuddy 拦截器 |
| **Lowering** | 编译过程 | Java 源码 → 字节码 → 机器码 |

**形象理解**：

```
// Java 世界的"多层 IR"
// 高层：Stream API（声明式，保留语义）
list.stream().filter(x -> x > 0).map(x -> x * 2).collect(...)

// 中层：字节码（保留部分结构）
ALOAD 0
INVOKESPECIAL java/util/stream/Stream.filter
INVOKESPECIAL java/util/stream/Stream.map

// 底层：机器码（失去高层语义）
MOV RAX, [RBX]
CMP RAX, 0
JLE ...
```

MLIR 做的就是：**让编译器能在每一层做优化**，而不是等到变成机器码才动手。

---

## 📚 核心概念详解

### 1. Dialect（方言）

**定义**：Dialect 是一组相关的 Operation 的集合，代表某个抽象层次或领域。

**类比**：Java 里的不同"语言风格"

```java
// 不同的 Dialect = 不同的编程范式
// Stream Dialect - 函数式风格
list.stream().filter(...).map(...).collect(...)

// Imperative Dialect - 命令式风格
for (int i = 0; i < list.size(); i++) {
    if (list.get(i) > 0) {
        result.add(list.get(i) * 2);
    }
}

// Reactive Dialect - 响应式风格
Flux.fromIterable(list)
    .filter(x -> x > 0)
    .map(x -> x * 2)
    .collectList();
```

**MLIR 常见 Dialect**：

```
// Tensor Dialect - 张量操作（高层）
%0 = tensor.empty() : tensor<4xf32>
%1 = tensor.insert %v into %0[%idx] : tensor<4xf32>

// Math Dialect - 数学运算（中层）
%2 = math.exp %1 : f32
%3 = math.sin %2 : f32

// Affine Dialect - 仿射循环（中层，适合循环优化）
affine.for %i = 0 to 128 {
  affine.load %A[%i, %j] : memref<128x128xf32>
}

// LLVM Dialect - LLVM IR（底层）
%4 = call @llvm.exp.f32(%3) : (f32) -> f32

// GPU Dialect - GPU 操作
gpu.launch_func ...
```

**为什么需要多个 Dialect？**

```
编译流程 = 从高层 Dialect 逐步 lowering 到低层 Dialect

Tensor Dialect (神经网络层)
    ↓ [Lowering: 展开张量操作]
Affine Dialect (循环层)
    ↓ [Lowering: 展开循环]
MemRef Dialect (内存层)
    ↓ [Lowering: 生成内存访问]
LLVM Dialect (机器码层)
    ↓ [Codegen]
二进制代码
```

每 lowering 一层，就**丢失一些高层语义**，但**获得一些底层细节**。

---

### 2. Operation（操作）

**定义**：Operation 是 MLIR 的基本计算单元，类似指令。

**通用格式**：

```mlir
// 一般形式
%result = dialect.operation %operand1, %operand2 {attr1 = value1} : (type1, type2) -> (result_type)

// 分解：
// %result     - 结果的 SSA 名字（类似变量）
// dialect.    - Dialect 前缀
// operation   - 操作名
// %operand    - 操作数（用 % 引用之前的结果）
// {attr}      - 属性（元数据）
// : (type)    - 类型签名
// -> (type)   - 返回类型
```

**实际例子**：

```mlir
// 1. 创建空张量
%0 = tensor.empty() : tensor<4xf32>

// 2. 张量插入
%1 = tensor.insert %v into %0[%idx] : tensor<4xf32>

// 3. 数学运算
%2 = math.exp %1 : f32

// 4. 仿射循环（带属性）
affine.for %i = 0 to 128 step 1 {
  %3 = affine.load %A[%i, %j] : memref<128x128xf32>
  %4 = affine.load %B[%i, %j] : memref<128x128xf32>
  %5 = arith.addf %3, %4 : f32
  affine.store %5, %C[%i, %j] : memref<128x128xf32>
}

// 5. 函数定义
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
```

**SSA 是什么？**

```
// SSA (Static Single Assignment) - 静态单赋值
// 每个变量只能赋值一次

// ❌ 错误（多次赋值）
%x = arith.addf %a, %b : f32
%x = arith.mulf %x, %c : f32  // x 被重新赋值了！

// ✅ 正确（每次新名字）
%x1 = arith.addf %a, %b : f32
%x2 = arith.mulf %x1, %c : f32  // 新变量 x2
```

**Java 类比**：

```java
// Java 不是 SSA（可以重复赋值）
int x = a + b;
x = x * c;  // OK

// MLIR 是 SSA（类似 Java 的 final 变量）
final int x1 = a + b;
final int x2 = x1 * c;  // 必须新名字
```

---

### 3. Type（类型）

**MLIR 类型系统**：

```mlir
// 基本类型
i32         // 32 位整数
f32, f64    // 32/64 位浮点
index       // 索引类型（类似 size_t）

// 张量类型
tensor<4xf32>           // 4 个 f32 的张量
tensor<4x8xf32>         // 4x8 的 2D 张量
tensor<?x?xf32>         // 动态形状的 2D 张量

// MemRef 类型（带内存布局）
memref<128x128xf32>              // 128x128 的内存引用
memref<128x128xf32, #map>        // 带仿射映射的内存引用

// 函数类型
(tensor<4xf32>) -> tensor<4xf32>  // 函数签名
```

**Tensor vs MemRef**：

```mlir
// Tensor - 值语义（不可变，类似 Java 的 String）
%0 = tensor.empty() : tensor<4xf32>
%1 = tensor.insert %v into %0[0] : tensor<4xf32>  // 返回新 tensor

// MemRef - 引用语义（可变，类似 Java 的数组）
%0 = memref.alloc() : memref<4xf32>
memref.store %v, %0[0] : memref<4xf32>  // 原地修改
```

**Java 类比**：

```java
// Tensor = Immutable List（值语义）
List<Integer> t0 = List.of(1, 2, 3);
List<Integer> t1 = List.of(0, ...t0);  // 新列表

// MemRef = ArrayList（引用语义）
ArrayList<Integer> m0 = new ArrayList<>(List.of(1, 2, 3));
m0.set(0, 0);  // 原地修改
```

---

### 4. Attribute（属性）

**定义**：Attribute 是附加在 Operation 上的元数据。

**常见属性**：

```mlir
// 1.  dense 属性（密集数组）
%dense<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>

// 2. 仿射映射属性
#map = affine_map<(i, j) -> (i, j)>
%0 = affine.load %A[%i, %j] {affine_map = #map} : memref<128x128xf32>

// 3. 字符串属性
func.func @main() {
  "unknown.operation"() {label = "my_op"} : () -> ()
}

// 4. 整数属性
%0 = arith.constant 42 : i32 {some_attr = 42 : i32}
```

**Java 类比**：注解

```java
// MLIR Attribute
"unknown.operation"() {label = "my_op"} : () -> ()

// Java Annotation
@Label("my_op")
void operation() { ... }
```

---

### 5. Pass（优化遍）

**定义**：Pass 是对 IR 进行转换或优化的模块。

**Pass 的工作流程**：

```
输入 IR
    ↓
[Pass 1: 常量折叠]
    42 + 0 → 42
    ↓
[Pass 2: 死代码消除]
    删除未使用的计算
    ↓
[Pass 3: 算子融合]
    Conv + ReLU → FusedConvReLU
    ↓
输出 IR（优化后）
```

**实际例子：常量折叠 Pass**

```mlir
// 优化前
%0 = arith.constant 42 : i32
%1 = arith.constant 0 : i32
%2 = arith.addi %0, %1 : i32  // 42 + 0

// 经过 ConstantFold Pass 后
%0 = arith.constant 42 : i32  // %1 和 %2 被折叠了
```

**Pass 管理器**：

```cpp
// C++ API（了解即可）
PassManager pm;
pm.addPass(createCSEPass());           // 公共子表达式消除
pm.addPass(createConstantFoldPass());  // 常量折叠
pm.addPass(createDeadCodeElimPass());  // 死代码消除
pm.run(module);
```

**Java 类比**：JIT 优化 Pass

```
JVM JIT 编译器也有类似的 Pass：
- 内联 Pass (Inline Pass)
- 逃逸分析 Pass (Escape Analysis)
- 锁消除 Pass (Lock Elimination)
- 常量折叠 Pass (Constant Folding)
```

---

## 🔧 实践 1：用 Python 理解 MLIR

MLIR 原生是 C++，但我们可以用 Python 来理解概念。

### 安装 Toy 教程环境

```bash
# Toy 是 MLIR 官方的教学语言（类似一个简单的矩阵语言）
# GitHub: https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy

# 方法 1：用预编译的 Python 绑定（推荐）
pip install mlir-python-bindings

# 方法 2：自己编译 LLVM（不推荐，耗时 2 小时+）
```

### 第一个 MLIR 示例（Python）

```python
# 创建一个简单的 MLIR 模块
from mlir.dialects import func, arith, math

# 定义一个函数：f(x) = exp(x * 2) + 1
def create_function():
    # 这里只是示意，实际 MLIR Python 绑定还在开发中
    # 我们主要看 MLIR 文本表示
    
    mlir_code = """
func.func @my_func(%arg0: f32) -> f32 {
  %c2 = arith.constant 2.0 : f32
  %0 = arith.mulf %arg0, %c2 : f32
  %1 = math.exp %0 : f32
  %c1 = arith.constant 1.0 : f32
  %2 = arith.addf %1, %c1 : f32
  return %2 : f32
}
"""
    return mlir_code

print(create_function())
```

**输出**：

```mlir
func.func @my_func(%arg0: f32) -> f32 {
  %c2 = arith.constant 2.0 : f32
  %0 = arith.mulf %arg0, %c2 : f32
  %1 = math.exp %0 : f32
  %c1 = arith.constant 1.0 : f32
  %2 = arith.addf %1, %c1 : f32
  return %2 : f32
}
```

**可视化**：

```
函数：my_func(f32) -> f32

%arg0 (输入)
   ↓
%c2 = 2.0
   ↓
%0 = %arg0 * %c2
   ↓
%1 = exp(%0)
   ↓
%c1 = 1.0
   ↓
%2 = %1 + %c1
   ↓
return %2
```

---

## 🔧 实践 2：写一个超简单的 MLIR Pass

**目标**：实现常量折叠（Constant Folding）

### Pass 设计思路

```
识别模式：
  %0 = arith.constant 42 : i32
  %1 = arith.constant 0 : i32
  %2 = arith.addi %0, %1 : i32

折叠为：
  %2 = arith.constant 42 : i32
```

### Python 伪代码实现

```python
# 简化的常量折叠 Pass 示例
# 实际 MLIR Pass 需要用 C++ 写，这里用 Python 示意逻辑

class ConstantFoldPass:
    def run(self, module):
        for op in module.operations:
            if isinstance(op, AddOp):
                # 检查两个操作数是否都是常量
                lhs = self.get_defining_op(op.lhs)
                rhs = self.get_defining_op(op.rhs)
                
                if isinstance(lhs, ConstantOp) and isinstance(rhs, ConstantOp):
                    # 可以折叠！计算结果
                    result = lhs.value + rhs.value
                    
                    # 替换原操作
                    self.replace_op_with_constant(op, result)
                    
        return module
    
    def get_defining_op(self, value):
        # 找到定义这个值的操作
        return value.defining_op
    
    def replace_op_with_constant(self, op, value):
        # 用常量替换原操作
        new_const = ConstantOp(value)
        op.replace_all_uses_with(new_const)
        op.erase()
```

### 实际效果

```mlir
// 优化前
func.func @test() -> i32 {
  %0 = arith.constant 42 : i32
  %1 = arith.constant 0 : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}

// 运行 ConstantFold Pass
// $ mlir-opt --constant-fold input.mlir

// 优化后
func.func @test() -> i32 {
  %0 = arith.constant 42 : i32
  return %0 : i32
}
```

**编译并运行**（如果你编译了 LLVM）：

```bash
# 1. 保存输入 IR 到 input.mlir
# 2. 运行优化
mlir-opt --constant-fold input.mlir -o output.mlir

# 3. 查看结果
cat output.mlir
```

---

## 🔧 实践 3：用 TorchDynamo 看 MLIR 实战

PyTorch 2.x 的 `torch.compile` 底层就用到了 MLIR！

### 环境

```bash
pip install torch>=2.0
```

### 捕获计算图

```python
import torch
import torch._dynamo as dynamo

# 定义一个简单函数
def my_func(x):
    return torch.exp(x * 2) + 1

# 用 Dynamo 捕获图
compiled_func = dynamo.optimize("eager")(my_func)

# 运行一次（触发图捕获）
x = torch.randn(4)
y = compiled_func(x)

# 查看捕获的图
print(dynamo.explain(compiled_func)(x))
```

**输出**（简化）：

```
Captured Graph:
graph():
    %x : [num_users=1] = placeholder[target=x]
    %c2 : [num_users=1] = call_function[target=torch.tensor](args = (2.0,))
    %mul : [num_users=1] = call_function[target=operator.mul](args = (%x, %c2))
    %exp : [num_users=1] = call_function[target=torch.exp](args = (%mul,))
    %add : [num_users=1] = call_function[target=operator.add](args = (%exp, 1))
    return %add
```

**这就是一个计算图！** 类似 MLIR 的表示。

### 查看编译后的代码

```python
import torch

def my_func(x):
    return torch.exp(x * 2) + 1

# 用 torch.compile 编译
compiled = torch.compile(my_func, backend="inductor")

# 运行
x = torch.randn(4)
y = compiled(x)

# 查看生成的代码（Inductor 后端用 Triton）
print(compiled.get_graph_breaks())
```

**生成的 Triton 代码**（类似 MLIR lowering 后的结果）：

```python
# Triton kernel（简化）
@triton.jit
def kernel(x_ptr, out_ptr, n):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    tmp = x * 2.0
    exp_val = tl.exp(tmp)
    out = exp_val + 1.0
    tl.store(out_ptr + pid, out)
```

**看到了吗？**这就是一个 lowering 过程：

```
PyTorch 源码
    ↓ [TorchDynamo 捕获]
计算图（类似高层 Dialect）
    ↓ [TorchInductor 优化]
Triton 代码（类似底层 Dialect）
    ↓ [Triton 编译]
GPU 机器码
```

---

## 📝 Java 开发者视角：关键对比

| MLIR 概念 | JVM 对应 | 深度对比 |
|-----------|---------|----------|
| **Dialect** | 字节码指令集 | JVM 只有一套指令集，MLIR 有多套（Dialect） |
| **Operation** | 字节码指令 | `aload_0`, `invokevirtual` vs `arith.addf` |
| **Type** | JVM 类型描述符 | `Ljava/lang/String;` vs `tensor<4xf32>` |
| **Pass** | JIT 优化 Pass | 思想完全一致！JVM 也有 C1/C2 优化 Pass |
| **Lowering** | 解释→JIT 编译 | 多层 lowering vs 两层（字节码→机器码） |
| **SSA** | JIT 内部表示 | HotSpot JIT 内部也用 SSA 做优化 |

**你的优势**：
- ✅ 理解 JIT 编译思想（MLIR 的 Pass 和 JIT 优化很像）
- ✅ 有性能调优经验（GC 调优 ↔ MLIR 内存优化）
- ✅ 工程能力强（MLIR 是大工程，需要工程思维）

**需要适应的**：
- ⚠️ C++ 语法（MLIR 主要是 C++）
- ⚠️ 函数式思维（SSA、不可变）
- ⚠️ 多层抽象（JVM 只有两层，MLIR 有 N 层）

---

## 🤔 常见问题

### Q1: MLIR 和 LLVM IR 是什么关系？

**A**: 
```
MLIR 是 LLVM 的"上层建筑"

MLIR (多层 IR)
    ↓ [Lowering]
LLVM Dialect (MLIR 的一种 Dialect)
    ↓ [LLVM Codegen]
LLVM IR (传统 LLVM 的输入)
    ↓
机器码
```

MLIR 可以 lowering 到 LLVM IR，也可以 lowering 到其他后端（CUDA、SPIR-V 等）。

### Q2: 我需要学完 C++ 才能学 MLIR 吗？

**A**: **不需要！**
- 先理解概念（Dialect、Operation、Pass）
- 用现成的工具（TorchDynamo、TVM）
- 需要改源码时再深入 C++

### Q3: MLIR 在实际项目中怎么用？

**A**: 大多数人不直接写 MLIR，而是：
- 用 **PyTorch 2.x**（底层用 MLIR）
- 用 **TVM**（有自己的 IR，但受 MLIR 影响）
- 用 **IREE**（基于 MLIR 的推理引擎）
- 写 **Triton** 算子（基于 MLIR）

直接开发 MLIR Dialect/Pass 的是编译器团队，应用层开发用不到那么深。

### Q4: 学习 MLIR 的最佳路径是什么？

**A**: 
```
1. 理解概念（本文）
2. 用 PyTorch 2.x / TVM（感受编译器优化）
3. 读 MLIR Toy 教程（官方入门）
4. 写一个简单的 Pass（实践）
5. 深入某个 Dialect（如 Tensor Dialect）
```

---

## ✅ 本周任务清单

### 必做（核心）

- [ ] 读懂本文所有 MLIR 代码示例
- [ ] 用 `torch.compile` 编译一个模型，查看捕获的图
- [ ] 画出 MLIR lowering 流程图（手绘或 draw.io）
- [ ] 用自己的话解释：什么是 Dialect？

### 选做（深入）

- [ ] 编译 LLVM，跑通 Toy 教程（https://mlir.llvm.org/docs/Tutorials/Toy/）
- [ ] 尝试写一个简单的 MLIR Pass（常量折叠）
- [ ] 阅读 TorchDynamo 源码，理解图捕获机制
- [ ] 在笔记里记录遇到的问题和解决方案

---

## 📚 参考资料

- MLIR 官方教程：https://mlir.llvm.org/docs/Tutorials/
- MLIR Toy 示例：https://mlir.llvm.org/docs/Tutorials/Toy/
- TorchDynamo 文档：https://pytorch.org/docs/stable/torch.compiler.html
- 本文对应的攻略：`../resources/ai-compiler-study-guide.md`
- LLVM Discourse（提问的好地方）：https://discourse.llvm.org/

---

## 📅 下次预告

**笔记 03**：CUDA 编程入门 - 理解 GPU 并行模型

- GPU 架构基础（Thread/Block/Grid）
- 第一个 CUDA 程序（向量加法）
- 理解共享内存和线程同步
- 用 CUDA 实现矩阵乘法

---

_笔记创建：2026-03-20_  
_适合人群：Java 应用开发背景，AI 编译器零基础_  
_深度：⭐⭐⭐⭐（比笔记 01 更深，需要多读几遍）_
