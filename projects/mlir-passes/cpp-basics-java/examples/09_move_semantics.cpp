#include <iostream>
#include <string>
#include <utility>

class BigData {
public:

  // 默认构造函数
  BigData() {
    std::cout << "default ctor\n";
  };
  // 普通构造函数
  explicit BigData(std::string payload) : data(std::move(payload)) {
    std::cout << "ctor\n";
  }
  
  // 拷贝构造
  BigData(const BigData &other) : data(other.data) {
    std::cout << "copy ctor\n";
  }

  // 移动构造
  BigData(BigData &&other) noexcept : data(std::move(other.data)) {
    std::cout << "move ctor\n";
  }
  
  // 拷贝赋值运算符
  BigData& operator=(const BigData& other) {
    data = other.data;
    std::cout << "copy assign\n";
    return *this;
  };

  // 移动赋值
  BigData& operator=(BigData&& other) {
    data = std::move(other.data);
    std::cout << "move assign\n";
    return *this;
  }

  const std::string &get() const { return data; }

private:
  std::string data;
};


BigData makeData() {
  BigData d("very large payload ...");
  return d;
}
 
/**
区分：
  构造还是赋值，看是否已存在
  对已有对象显式使用 std::move 时，通常会优先匹配移动构造/移动赋值

出现 赋值= 不一定是赋值，可能是构造
出现 T a() 不一定是构造，可能是函数声明
*/

int main() {
  // 返回值初始化常伴随返回值优化，不适合拿来稳定演示拷贝/移动构造
  // 在多数现代编译器和当前编译设置下，这里通常只看到普通构造；
  // 拷贝构造和移动构造常被返回值优化省略
  BigData a = makeData();

  BigData a1("new large payload");
  
  // std::move(a) 把 a 转成右值引用，随后用它初始化 b，触发移动构造
  BigData b = std::move(a1);
  std::cout << "b = " << b.get() << "\n";
  
  BigData c("hello world");
  // 触发拷贝构造
  BigData d1 = c;

  BigData d2; //默认构造
  // 触发拷贝赋值
  d2 = c;
  // 移动赋值
  d2 = std::move(c);

  BigData c2("new instance");
  // 触发拷贝构造
  BigData e(c2);
  // 触发移动构造
  BigData f(std::move(c2));


  return 0;
}
