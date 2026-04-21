#include <iostream>
#include <string>

template <typename T>
T add(T a, T b) {
  return a + b;
}

template <typename T>
class Box {
public:
  explicit Box(T v) : value(v) {}
  T get() const { return value; }

private:
  T value;
};

int main() {
  std::cout << "add<int>(3, 4) = " << add<int>(3, 4) << "\n";
  std::cout << "add<double>(1.5, 2.0) = " << add<double>(1.5, 2.0) << "\n";

  Box<std::string> b("mlir");
  std::cout << "Box value = " << b.get() << "\n";
  return 0;
}
