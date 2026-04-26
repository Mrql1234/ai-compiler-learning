# 04 template（模板）

## 讲解

Java 对照：
- Java 泛型是类型擦除，运行时类型信息有限。
- C++ 模板是编译期展开，类型能力更强，性能更好，但报错可能更复杂。

典型形式：

```cpp
template <typename T>
T add(T a, T b) { return a + b; }
```

MLIR 常见模板写法：

```cpp
op.getDefiningOp<arith::ConstantOp>()
```

## 常见坑

- 模板报错很长时，先看你调用模板的那一行，再逐步回溯。

## 示例代码

见：`examples/04_templates.cpp`

```
模板有函数模板 类模板
    不是运行时多态，在编译期生成具体代码
    把类型/常量当参数

```