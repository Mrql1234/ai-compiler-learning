#include <iostream>
#include <bits/stdc++.h>

int main() {
    std::queue<int> q;
    q.push(1);
    int x = q.front();
    std::cout << x << std::endl;
    q.pop();
    bool e = q.empty();
    std::cout << e << std::endl;
}