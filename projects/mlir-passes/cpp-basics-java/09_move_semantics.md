# 09 移动语义（std::move）

## 讲解

Java 对照：
- Java 常见的是“引用传递语义 + GC”。
- C++ 为了性能，支持“移动”而不是“深拷贝”。

核心思想：
- 当对象即将被丢弃时，把它内部资源“转移”给新对象。
- 常见于返回大对象、容器扩容、智能指针转移所有权。

## 常见坑

- `std::move(x)` 后，`x` 仍“有效但状态未指定”，不要依赖其旧值。

## 示例代码

见：`examples/09_move_semantics.cpp`

```
创建对象或赋值方法：
    默认构造函数
    普通构造函数（带参数）
    拷贝构造
    移动构造
    返回值构造
    动态创建对象
    容器内原地构造
    拷贝赋值
    移动赋值
```

```
1. 默认构造
Node a;
调用：

Node();
含义：

创建一个新对象
不带参数
2. 普通构造/带参构造
Node a(10);
Node b{"abc"};
调用：

Node(int);
Node(std::string);
含义：

用你提供的参数创建对象
3. 拷贝构造
Node a;
Node b = a;
调用：

Node(const Node&);
含义：

用一个已有左值对象，创建一个新对象
4. 移动构造
Node a;
Node b = std::move(a);
调用：

Node(Node&&);
含义：

用一个“可被搬走资源”的对象，创建一个新对象
5. 返回值构造
Node make_node() {
    return Node(1);
}

Node a = make_node();
这里通常涉及：

直接构造
移动构造
或拷贝省略（RVO / NRVO）
现代 C++ 里很多时候会直接构造到目标位置，甚至看不到拷贝/移动。

6. 动态创建对象
Node* p = new Node();
auto p2 = std::make_unique<Node>();
auto p3 = std::make_shared<Node>();
本质上还是调用构造函数，只是对象建在堆上。

7. 容器内原地构造
vec.emplace_back(10);
这也是创建对象，只不过：

对象直接构造在容器内部内存上
尽量避免额外临时对象

8. 拷贝赋值
Node a;
Node b;
b = a;
调用：
Node& operator=(const Node&);
含义：
b 已经存在
用 a 的内容覆盖 b

9. 移动赋值
Node a;
Node b;
b = std::move(a);
调用：
Node& operator=(Node&&);
含义：
b 已经存在
把 a 的资源搬给 b

```

```
这些方式之间最核心的区别
1. 构造 vs 赋值
这是最大区别。

构造：对象还不存在
Node a;        // 构造
Node b = a;    // 拷贝构造
Node c = std::move(a); // 移动构造
赋值：对象已经存在
Node a, b;
b = a;             // 拷贝赋值
b = std::move(a);  // 移动赋值
你可以记一句：

带定义的是构造
对象先存在，再 = 是赋值

2. 拷贝 vs 移动
拷贝
保留源对象
复制资源
std::string a = "hello";
std::string b = a;
结果：

a 还保持原值
b 得到一份副本
移动
转移资源
源对象通常变成“有效但内容不确定”或空状态
std::string a = "hello";
std::string b = std::move(a);
结果：

b 拿到资源
a 还能用，但内容别依赖
```