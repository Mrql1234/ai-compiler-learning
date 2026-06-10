#include <iostream>
#include <bits/stdc++.h>

int main() {


}

bool CheckRules(const std::string &input) {
    int n = input.size();
    std::string s = input.substr(1, n-1);
    int res = true;
    bool isLowerCase = s[0] >= 'a' && s[0] <= 'z'  ;
    bool isUpperCase = s[0] >= 'A' && s[0] <= 'Z' ;
    
    for (int i = 0; i < n-1; i++) {
        if (isLowerCase) {
            if(!(s[i] >= 'a' && s[0] <= 'z')){
                res = false;
                break;
            }
        } else if( isUpperCase){
            if (!(s[0] >= 'A' && s[0] <= 'Z')) {
                res = false;
                break;
            }
        }
    }
    return res;
}