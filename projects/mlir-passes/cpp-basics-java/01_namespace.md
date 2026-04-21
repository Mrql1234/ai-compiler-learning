# 01 namespace（命名空间）

## 讲解

`namespace` 可以理解为 C++ 里的“名字分组机制”。

Java 对照：
- Java 的 `package` 主要用于文件/类组织。
- C++ 的 `namespace` 更轻量，可以在同一文件中灵活使用。

在 LLVM/MLIR 代码中你会经常看到：
- `namespace mlir { ... }`
- 匿名命名空间 `namespace { ... }`（仅当前 `.cpp` 文件可见，类似“文件私有”）

## 常见坑

- 不同命名空间有同名函数，调用时要写清楚前缀，如 `mlir::foo()`。

## 示例代码

见：`examples/01_namespace.cpp`
