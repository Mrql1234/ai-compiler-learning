#include <iostream>
#include <utility>

struct Node {
    Node() {
        std::cout << "default\n";
    }

    Node(const Node&) {
        std::cout << "copy\n";
    }

    Node(Node&&) noexcept {
        std::cout << "move\n";
    }
};

void f(Node node) {
    std::cout << "inside f\n";
}

int main() {
    Node n;

    f(n);            // copy
    f(std::move(n)); // move  std::move(n) 返回的是一个 右值引用类型 Node&&
}
