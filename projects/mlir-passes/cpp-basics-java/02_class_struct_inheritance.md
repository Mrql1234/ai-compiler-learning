# 02 class/struct 与继承

## 讲解

Java 对照：
- Java 里通常只写 `class`。
- C++ 有 `class` 和 `struct`，主要区别是默认访问权限：
  - `class` 默认 `private`
  - `struct` 默认 `public`

继承语法：

```cpp
struct Child : public Parent { ... };
```

这在 MLIR Pass 里很常见，例如：
- `struct MyPass : public PassWrapper<...> { ... }`

## 常见坑

- 忘了写 `public` 会导致继承成员访问受限。

## 示例代码

见：`examples/02_class_struct_inheritance.cpp`
