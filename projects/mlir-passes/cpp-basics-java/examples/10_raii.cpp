#include <iostream>
#include <stdexcept>

class ScopeGuard {
public:
  explicit ScopeGuard(const char *n) : name(n) {
    std::cout << "[acquire] " << name << "\n";
  }

  ~ScopeGuard() {
    std::cout << "[release] " << name << "\n";
  }

private:
  const char *name;
};

void runTask(bool fail) {
  ScopeGuard guard("task_resource");
  std::cout << "task running\n";
  if (fail) {
    throw std::runtime_error("task failed");
  }
  std::cout << "task finished\n";
}

int main() {
  try {
    runTask(false);
    runTask(true);
  } catch (const std::exception &e) {
    std::cout << "caught: " << e.what() << "\n";
  }
  return 0;
}
