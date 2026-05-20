#include <iostream>
#include <bits/stdc++.h>

int main() {
    std::unordered_map<std::string, int> mp;
    mp["a"] = 1;
    mp["a"]++;
    
    if (mp.count("a")) {
        std::cout << "a is in mp" << std::endl;
    }
    if (mp.find("a") != mp.end()) {
        std::cout << "a is in mp" << std::endl;
    }
    
    // 遍历
    for (auto &[k, v] : mp) {
        std::cout << k << " " << v << std::endl;
    }

    mp.erase("a");

    mp["first"] = 1;
    if(mp.find("first") != mp.end()) {
        std::cout << mp.find("first")->first << " " << mp.find("first")->second << std::endl;
    }

    std::cout << mp.size() << std::endl;

}