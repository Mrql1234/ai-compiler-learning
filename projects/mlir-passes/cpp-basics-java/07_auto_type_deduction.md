# 07 auto（类型推导）

## 讲解

Java 对照：
- 类似 Java 的 `var`，但 C++ 里使用更广。

典型收益：
- 减少模板类型冗长代码。
- 提高阅读主流程的效率（这在 LLVM/MLIR 代码里非常重要）。

## 常见坑

- `auto` 会发生值拷贝，若你想避免拷贝，考虑 `auto&`。

```
 for (const auto &[name, score] : scores) {
    std::cout << name << " => " << score << "\n";
  }

  遍历 scores 中每个元素，把每个元素按引用、只读地拆成 name 和 score 两个变量来用。
  const auto& : 只读引用，不拷贝

```

## 示例代码

见：`examples/07_auto_type_deduction.cpp`
