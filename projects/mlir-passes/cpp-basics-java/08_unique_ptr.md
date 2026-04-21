# 08 std::unique_ptr（独占智能指针）

## 讲解

Java 对照：
- Java 里对象生命周期通常交给 GC。
- C++ 常用智能指针管理堆对象，其中 `unique_ptr` 表示“唯一所有者”。

关键点：
- 不能拷贝，只能移动（`std::move`）。
- 离开作用域自动释放资源。

在 MLIR Pass 工厂函数中很常见：

```cpp
std::unique_ptr<Pass> createPass() {
  return std::make_unique<MyPass>();
}
```

## 示例代码

见：`examples/08_unique_ptr.cpp`
