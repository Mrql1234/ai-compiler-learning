#include <iostream>

int sumConstRef(const int &a, const int &b) {
  return a + b;
}

class Counter {
public:
  explicit Counter(int v) : value(v) {}
  int get() const { return value; }
  void inc() { value += 1; }

private:
  int value;
};

int main() {
  int x = 3;
  int y = 4;
  std::cout << "sum = " << sumConstRef(x, y) << "\n";

  const int *ptrToConst = &x;
  int *const constPtr = &x;
  *constPtr = 99;

  Counter c(10);
  std::cout << "counter = " << c.get() << "\n";
  c.inc();
  std::cout << "counter after inc = " << c.get() << "\n";
  std::cout << "ptrToConst points to = " << *ptrToConst << "\n";
  return 0;
}
