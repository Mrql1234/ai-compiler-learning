#include <iostream>
#include <bits/stdc++.h>

/**
lower_bound和upper_bound是map的成员函数，用于查找第一个大于等于或大于给定值的键。
有序容器：set、map、multiset、multimap 都支持， 返回的是迭代器
map的迭代器是pair<const key, value>
set的迭代器是const key

*/
int main() {
    std::map<int, std::string> mp;
    mp[3] = "c";
    mp[1] = "a";
    
    auto it = mp.lower_bound(2); // 第一个 key >= 2
    auto it2 = mp.upper_bound(2); // 第一个 key > 2
    std::cout << it->first << " " << it->second << std::endl;
    std::cout << it2->first << " " << it2->second << std::endl;
    for (auto &[k, v] : mp) {
        std::cout << k << " " << v << std::endl;
    }
}
