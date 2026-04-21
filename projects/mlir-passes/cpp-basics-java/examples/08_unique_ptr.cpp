#include <iostream>
#include <memory>

class Resource {
public:
  Resource() { std::cout << "Resource acquired\n"; }
  ~Resource() { std::cout << "Resource released\n"; }
  void work() const { std::cout << "Resource working\n"; }
};

int main() {
  std::unique_ptr<Resource> p1 = std::make_unique<Resource>();
  p1->work();

  std::unique_ptr<Resource> p2 = std::move(p1);
  if (!p1) {
    std::cout << "p1 is now empty after move\n";
  }
  p2->work();
  return 0;
}
