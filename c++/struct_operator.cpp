#include <iostream>

// A struct with members and an overloaded operator()
struct MyStruct {
    int x;
    int y;

    MyStruct(int a, int b) : x(a), y(b) {} // Constructor

    int operator()(int l, int m) {
        return x+y+l+m;
    }
};

int main() {
    MyStruct adder(5, 3); // Create an instance of MyStruct with members x and y

    int result = adder(1,2); // Use it like a function
    std::cout << "Result: " << result << std::endl; // Output: Result: 8

    return 0;
}