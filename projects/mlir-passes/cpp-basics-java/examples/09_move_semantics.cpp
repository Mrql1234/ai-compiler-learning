#include <iostream>
#include <string>
#include <utility>

class BigData {
public:
  explicit BigData(std::string payload) : data(std::move(payload)) {
    std::cout << "ctor\n";
  }

  BigData(const BigData &other) : data(other.data) {
    std::cout << "copy ctor\n";
  }

  BigData(BigData &&other) noexcept : data(std::move(other.data)) {
    std::cout << "move ctor\n";
  }

  const std::string &get() const { return data; }

private:
  std::string data;
};

BigData makeData() {
  BigData d("very large payload ...");
  return d;
}

int main() {
  BigData a = makeData();
  BigData b = std::move(a);
  std::cout << "b = " << b.get() << "\n";
  return 0;
}
