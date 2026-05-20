#include <vector>
#include <iostream>
#include <bits/stdc++.h>
/**
    vector方法
    size()
    push_back()
    pop_back()
    v[i]
    back()
    begin()
    end()

    可以当作栈用，push_back()和pop_back()
*/
int main() {
    std::vector<int> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
    //
    v.pop_back();

    sort(v.begin(), v.end());
    reverse(v.begin(), v.end());
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;

    
    return 0;
}