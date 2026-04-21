#include <iostream>

void addOneByRef(int &x) {
  x += 1;
}

void addOneByPtr(int *x) {
  if (x != nullptr) {
    *x += 1;
  }
}

int main() {
  int a = 10;
  int b = 20;

  addOneByRef(a);
  addOneByPtr(&b);

  std::cout << "a = " << a << ", b = " << b << "\n";

  int *maybeNull = nullptr;
  addOneByPtr(maybeNull); // safe because we checked nullptr
  return 0;
}
