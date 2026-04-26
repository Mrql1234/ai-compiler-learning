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

```
智能指针 std::unique_ptr<int> p
  独占，不能拷贝 p2 = p1,即两个unique_ptr指向同一个对象， 只能 std::move(p1)
  是对象，封装了指针，可以执行指针的操作 *p   p.a  p->f()
  作用：避免拷贝
    void set_child(std::unique_ptr<Node> child);
    我不再让 p 拥有这个对象，把它交给 set_child
    set_child(std::move(p));
    函数参数 child 会在函数结束时销毁
    如果函数内部没有把 child 再移动到别处，对象就会被释放
    如果函数内部保存了它，对象继续存活

```

```
函数参数传递是否会拷贝, 看定义
  值传递
    void f(Node node);
    Node n;
    f(n);            // 拷贝
    f(std::move(n)); // 移动， std::move(n) 返回的是一个 右值引用类型 Node&&,见11_move.cpp
    f(Node{});       // 构造临时对象，通常移动/省略
  引用传递，不拷贝
  指针传递，不拷贝
  智能指针，

```

## 示例代码

见：`examples/08_unique_ptr.cpp`
