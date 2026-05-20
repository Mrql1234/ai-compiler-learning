#include <iostream>
#include <utility>

struct Node {
    Node() {
        std::cout << "default\n";
    }

    Node(const Node&) {
        std::cout << "copy\n";
    }

    Node(Node&& other) noexcept {
        std::cout << "move\n";
        //value = std::move(other.value);
    }

    std::string value;
};

void f(Node node) {
    std::cout << "inside f\n";
}

int main() {
    Node n;

    f(n);            // copy
    f(std::move(n)); // move  std::move(n) 返回的是一个 右值引用类型 Node&&

    Node n1;
    n1.value = "hello";
    Node n2 = std::move(n1);
    std::cout << "n1.value = " << n1.value << "\n";
    std::cout << "n2.value = " << n2.value << "\n";
    
}
