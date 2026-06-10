#include <iostream>
#include <bits/stdc++.h>

int main(){
    std::priority_queue<int> pq;
    pq.push(1);
    pq.push(5);
    std::cout << pq.top() << std::endl;
    // 
    std::priority_queue<int, std::vector<int>, std::greater<int>> small_pq;
    small_pq.push(10);
    small_pq.push(20);
    std::cout << small_pq.top() << std::endl;
}