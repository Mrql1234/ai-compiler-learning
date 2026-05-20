#include <iostream>
#include <bits/stdc++.h>

int main() {
    std::string s = "abc";
    s.push_back('d');
    s.pop_back();
    int n = s.size();
    
    s.substr(1, 2);        // "bc"
    std::cout << s[0] << std::endl;
    std::cout << s << std::endl;
    reverse(s.begin(), s.end());
    std::cout << s << std::endl;
    sort(s.begin(), s.end());
    std::cout << s << std::endl;
}