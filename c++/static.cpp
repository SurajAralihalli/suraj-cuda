#include <iostream>

class MyClass {
public:
    static int staticMember; // Static member variable

    static void staticFunction() { // Static member function
        std::cout << "Static member function called." << std::endl;
    }
};

int MyClass::staticMember = 0; // Definition of static member variable

void regularFunction() {
    static int staticLocalVariable = 0; // Local static variable
    staticLocalVariable++;
    std::cout << "Static local variable: " << staticLocalVariable << std::endl;
}

int main() {
    // Accessing static member variable and static member function of class
    std::cout << "Static member variable: " << MyClass::staticMember << std::endl;
    MyClass::staticFunction();
    
    MyClass obj;
    obj.staticMember++;
    std::cout << "obj.staticMember: " << obj.staticMember << std::endl;

    // Using regular function with static local variable
    for (int i = 0; i < 5; ++i) {
        regularFunction();
    }
    
    // Static member variable: 0
    // Static member function called.
    // obj.staticMember: 1
    // Static local variable: 1
    // Static local variable: 2
    // Static local variable: 3
    // Static local variable: 4
    // Static local variable: 5

    return 0;
}
