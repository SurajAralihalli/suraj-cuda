#include <iostream>
#include <cstring>

class MyString {
private:
    char* str;

public:
    // Constructor that initializes the string
    MyString(const char* s) {
        str = new char[strlen(s) + 1];
        strcpy(str, s);
    }

    // Copy constructor (deep copy)
    MyString(const MyString& other) {
        str = new char[strlen(other.str) + 1];
        strcpy(str, other.str);
    }

    // Move constructor
    MyString(MyString&& other) noexcept {
        str = other.str;
        other.str = nullptr;
    }

    // Destructor to release memory
    ~MyString() {
        delete[] str;
    }

    // Print the string
    void print() const {
        std::cout << str << std::endl;
    }
};

int main() {
    // Create a MyString object using the regular constructor
    MyString str1("Hello, World!");

    // Create a MyString object using the copy constructor
    MyString str2 = str1; // Calls the copy constructor

    // Create a MyString object using the move constructor
    MyString str3 = std::move(str1); // Calls the move constructor

    // Print the strings to verify the results
    std::cout << "str1: ";
    str1.print(); // Should be empty because resources were moved
    std::cout << "str2: ";
    str2.print(); // Should be a copy of str1
    std::cout << "str3: ";
    str3.print(); // Should be the original str1

    return 0;
}
