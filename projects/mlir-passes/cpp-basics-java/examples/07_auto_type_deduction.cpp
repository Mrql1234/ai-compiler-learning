#include <iostream>
#include <map>
#include <string>
#include <vector>

int main() {
  std::vector<int> nums = {1, 2, 3};

  int sumByCopy = 0;
  for (auto n : nums) {
    sumByCopy += n;
  }

  for (auto &n : nums) {
    n *= 10;
  }

  std::map<std::string, int> scores = {{"alice", 90}, {"bob", 95}};
  for (const auto &[name, score] : scores) {
    std::cout << name << " => " << score << "\n";
  }

  std::cout << "sumByCopy = " << sumByCopy << "\n";
  std::cout << "nums[0] = " << nums[0] << "\n";
  return 0;
}
