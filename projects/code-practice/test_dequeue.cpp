#include <iostream>
#include <bits/stdc++.h>

int main() {
    std::deque<int> dq;
    dq.push_back(1);
    dq.push_front(2);
    for(int x : dq) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    std::cout << dq.front() << " " << dq.back() << std::endl;

    dq.pop_back();
    dq.pop_front();

    std::cout << dq.size() << std::endl;

    /**
    是未定义行为，因为不能对空 deque 调用 front() 或 back()
     */
    int a = dq.front();
    int b = dq.back();

    std::cout << a << " " << b << std::endl;

}