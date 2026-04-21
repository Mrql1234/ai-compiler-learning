#include <iostream>
#include <string>

struct Config {
  int threads = 4;
};

class Animal {
public:
  explicit Animal(std::string n) : name(std::move(n)) {}
  void printName() const { std::cout << "Animal name: " << name << "\n"; }

protected:
  std::string name;
};

class Dog : public Animal {
public:
  explicit Dog(std::string n) : Animal(std::move(n)) {}
  void bark() const { std::cout << name << " says: woof\n"; }
};

int main() {
  Config cfg;
  std::cout << "threads = " << cfg.threads << "\n";

  Dog d("buddy");
  d.printName();
  d.bark();
  return 0;
}
