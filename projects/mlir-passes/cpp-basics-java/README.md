# C++ 基础学习包（Java 开发者版）

这套资料面向 Java 背景，目标是帮助你快速读懂 MLIR/LLVM 风格 C++17 代码。

## 目录说明

- `01_namespace.md` + `examples/01_namespace.cpp`
- `02_class_struct_inheritance.md` + `examples/02_class_struct_inheritance.cpp`
- `03_virtual_override.md` + `examples/03_virtual_override.cpp`
- `04_templates.md` + `examples/04_templates.cpp`
- `05_reference_pointer.md` + `examples/05_reference_pointer.cpp`
- `06_const.md` + `examples/06_const.cpp`
- `07_auto_type_deduction.md` + `examples/07_auto_type_deduction.cpp`
- `08_unique_ptr.md` + `examples/08_unique_ptr.cpp`
- `09_move_semantics.md` + `examples/09_move_semantics.cpp`
- `10_raii.md` + `examples/10_raii.cpp`

## 建议学习顺序

1. 先看 `05_reference_pointer.md` 和 `10_raii.md`（最关键）
2. 再看 `06_const.md`、`08_unique_ptr.md`、`09_move_semantics.md`
3. 最后看 `04_templates.md`（模板通常最难）

## 编译运行示例

在 `projects/mlir-passes/cpp-basics-java` 目录下：

```bash
clang++ -std=c++17 examples/01_namespace.cpp -o /tmp/cpp_demo_01 && /tmp/cpp_demo_01
```

如果你没有 `clang++`，用 `g++` 也可以：

```bash
g++ -std=c++17 examples/01_namespace.cpp -o /tmp/cpp_demo_01 && /tmp/cpp_demo_01
```
