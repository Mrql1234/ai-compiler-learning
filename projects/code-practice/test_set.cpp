#include <iostream>
#include <bits/stdc++.h>

int main() {
    std::unordered_set<int> st;
    st.insert(1);
    st.erase(1);
    if (st.count(1)) {
        std::cout << "1 is in st" << std::endl;
    }

    std::set<int> s;
    s.insert(10);
    auto it = s.lower_bound(7);
    std::cout << *it << std::endl;
}