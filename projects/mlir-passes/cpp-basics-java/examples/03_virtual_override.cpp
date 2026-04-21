#include <iostream>
#include <memory>

class Pass {
public:
  virtual ~Pass() = default;
  virtual void run() const { std::cout << "Base pass\n"; }
};

class ConstantFoldPass : public Pass {
public:
  void run() const override { std::cout << "ConstantFoldPass running\n"; }
};

class DCEPass : public Pass {
public:
  void run() const override { std::cout << "DCEPass running\n"; }
};

void execute(const Pass &p) {
  p.run();
}

int main() {
  ConstantFoldPass c;
  DCEPass d;
  execute(c);
  execute(d);
  return 0;
}
