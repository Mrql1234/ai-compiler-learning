#include <iostream>

namespace math_v1 {
int add(int a, int b) { return a + b; }
}

namespace math_v2 {
int add(int a, int b) { return a + b + 100; }
}

namespace {
void internalLog(const char *msg) {
  std::cout << "[internal] " << msg << "\n";
}
} // namespace

int main() {
  internalLog("namespace demo start");
  std::cout << "math_v1::add(1, 2) = " << math_v1::add(1, 2) << "\n";
  std::cout << "math_v2::add(1, 2) = " << math_v2::add(1, 2) << "\n";
  return 0;
}
