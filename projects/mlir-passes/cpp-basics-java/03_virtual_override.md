# 03 virtual 与 override（多态）

## 讲解

Java 对照：
- Java 普通实例方法默认可被重写。
- C++ 只有标记为 `virtual` 的函数才有运行时多态。

`override` 的作用：
- 明确“我就是要重写父类虚函数”。
- 如果函数签名写错，编译器会直接报错，避免隐藏 bug。

virtual 的核心作用就是：

让函数调用走“运行时多态”而不是“按变量声明类型静态决定”。

也就是说，调用哪个函数实现，不只看“指针/引用是什么类型”，还看“对象真实类型是什么”。

1. 不加 virtual 会怎样
比如：  
```
class Base {
public:
    void speak() {
        std::cout << "Base\n";
    }
};
class Derived : public Base {
public:
    void speak() {
        std::cout << "Derived\n";
    }
};
```
然后：

Base* p = new Derived();
p->speak();
输出是：

Base
因为这里不加 virtual，编译器按 p 的声明类型 Base* 来决定调用哪个函数。

这叫静态绑定。



MLIR 里常见：

```cpp
LogicalResult matchAndRewrite(...) const override;
```

## 常见坑

- 父类析构函数如果是多态基类，建议写 `virtual ~Base() = default;`。

## 示例代码

见：`examples/03_virtual_override.cpp`
