# MLIR 项目 C++ 版本与 C++ 基础速查（Java 开发者版）

## 1. 当前项目使用的 C++ 版本

### 结论
- 当前仓库 `projects/mlir-passes`：使用 **C++17**。
- 你本机 `~/codex_chat/llvm-project-personal` 中的 LLVM/MLIR 主仓：也要求 **C++17**（最低要求）。

### 依据（本机实际配置）
- `projects/mlir-passes/CMakeLists.txt`：
  - `set(CMAKE_CXX_STANDARD 17 ...)`
  - `set(CMAKE_CXX_STANDARD_REQUIRED ON)`
- `~/codex_chat/llvm-project-personal/mlir/CMakeLists.txt`：
  - `set(CMAKE_CXX_STANDARD 17 ...)`
- `~/codex_chat/llvm-project-personal/llvm/CMakeLists.txt`：
  - `set(LLVM_REQUIRED_CXX_STANDARD 17)`
  - 并将 `CMAKE_CXX_STANDARD` 设为该值，低于 17 会报错。

## 2. 你现在最需要的 C++ 基础（面向 MLIR Pass 开发）

下面不是“完整 C++ 教程”，而是你阅读 `ConstantFoldPass.cpp` 这类文件时最常用的语法。

## 3. 心智模型：先把 Java 映射到 C++

- Java 的“对象都在堆上，引用统一管理”
  - C++：对象既可以在栈上，也可以在堆上，生命周期管理更灵活，也更容易踩坑。
- Java 有 GC
  - C++ 没有默认 GC，通常用 RAII + 智能指针管理资源。
- Java 泛型（类型擦除）
  - C++ 模板是编译期展开，能力更强，报错也更复杂。
- Java 方法重载 + 虚方法
  - C++ 同样有重载、虚函数、多态，但还多了值语义、移动语义等概念。

## 4. 读 MLIR 代码时最常见的语法点

### 4.1 命名空间（namespace）

```cpp
namespace mlir {
namespace {
// 匿名命名空间：当前 .cpp 内部可见
}
} // namespace mlir
```

- 类似 Java 的 `package`，但更灵活。
- `namespace {}`（匿名命名空间）可理解为“文件内 private”。

### 4.2 类继承与 override

```cpp
struct ConstantFoldAddI : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(...) const override { ... }
};
```

- `:` 表示继承。
- `public` 类似 Java `extends` 后的公开继承。
- `override` 和 Java 一样：明确重写父类虚函数。

### 4.3 模板（template）与泛型差异

```cpp
OpRewritePattern<arith::AddIOp>
op.getDefiningOp<arith::ConstantOp>()
```

- 角括号是模板参数，编译期确定类型。
- 在 MLIR/LLVM 中模板大量出现，是常态。

### 4.4 引用与指针（Java 开发者必看）

```cpp
PatternRewriter &rewriter  // 引用，不能为空，必须绑定对象
MLIRContext *context       // 指针，可为空，需要判空语义
```

- `&`（声明里）是“引用类型”。
- `*` 是“指针类型”。
- 经验：能用引用就先用引用，表达“必须存在”。

### 4.5 `auto` 类型推导

```cpp
auto lhs = op.getLhs().getDefiningOp<arith::ConstantOp>();
```

- 类似 Java 10 的 `var`，但 C++ 更常用。
- 在 MLIR 代码中大量使用，用来减少冗长类型名。

### 4.6 `std::unique_ptr`

```cpp
std::unique_ptr<Pass> createConstantFoldPass() {
  return std::make_unique<ConstantFoldPass>();
}
```

- 独占所有权智能指针，离开作用域自动释放。
- 可类比“受控资源句柄”，不要手写 `new/delete`。

### 4.7 `const` 与只读语义

```cpp
LogicalResult matchAndRewrite(...) const override
```

- 函数尾部 `const`：表示该成员函数不修改对象状态。
- 指针/引用上的 `const` 还有多种组合，先掌握“只读约束”这个核心语义即可。

## 5. C++17 在这个项目中常见的能力

- `auto` 与更强的类型推导
- `std::make_unique`
- 结构化绑定（`auto [a, b] = ...`，你后续代码里可能会用到）
- `if`/`switch` 初始化语句（部分 LLVM 代码会见到）

> 你当前这份 `mlir-passes` 代码即使用 C++17 编译，建议新增代码也保持 C++17 语法，不要提前引入 C++20 特性。

## 6. Java 开发者最容易踩的坑（结合 MLIR 场景）

- 误把 C++ 指针当 Java 引用
  - 指针可能为空、可能悬空，使用前要确认生命周期与所有权。
- 忽略值语义与拷贝成本
  - C++ 传值可能发生拷贝，优先用 `const T&` 传大型对象。
- 过度手动内存管理
  - 优先智能指针（如 `unique_ptr`），少用裸 `new/delete`。
- 模板报错读不懂
  - 先定位“你写的第一行触发点”，不要从最底层错误看起。

## 7. 一段“能看懂 MLIR Pass”所需最小语法清单

你先掌握这 10 个点就足够进入实战：

1. `namespace` 与匿名命名空间  
2. `struct/class` 与继承  
3. 虚函数与 `override`  
4. 模板基础（`Foo<Bar>`）  
5. 引用 `&` 与指针 `*`  
6. `const`（函数 const、参数 const）  
7. `auto` 类型推导  
8. 智能指针 `std::unique_ptr`  
9. `std::move`（看到时知道是“转移所有权”）  
10. RAII（对象离开作用域自动释放资源）  

## 8. 给你的建议学习顺序（高效版）

1. 先读 `ConstantFoldPass.cpp`，只关注“匹配 + 替换”主流程。  
2. 遇到语法不懂，回到本文第 4 节查。  
3. 再补 `DeadCodeElim` / `Fusion` 时，重点关注模板和重写模式。  
4. 最后再系统补 C++ 语法细节，而不是一开始全学完。  

---

如果你愿意，我下一步可以基于你仓库里的 `ConstantFoldPass.cpp` 再做一份“逐行注释版（Java 对照）”，直接帮你把每行 C++ 语义翻成 Java 思维模型。
