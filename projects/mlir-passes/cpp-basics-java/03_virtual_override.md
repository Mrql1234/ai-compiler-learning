# 03 virtual 与 override（多态）

## 讲解

Java 对照：
- Java 普通实例方法默认可被重写。
- C++ 只有标记为 `virtual` 的函数才有运行时多态。

`override` 的作用：
- 明确“我就是要重写父类虚函数”。
- 如果函数签名写错，编译器会直接报错，避免隐藏 bug。

MLIR 里常见：

```cpp
LogicalResult matchAndRewrite(...) const override;
```

## 常见坑

- 父类析构函数如果是多态基类，建议写 `virtual ~Base() = default;`。

## 示例代码

见：`examples/03_virtual_override.cpp`
