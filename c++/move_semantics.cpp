#include <iostream>
#include <string>

class MyString {
public:
    MyString(std::string&& str) : data(std::move(str)) {
        // Constructor takes an rvalue reference and moves it to 'data'
    }

    void print() {
        std::cout << data << std::endl;
    }

private:
    std::string data;
};

int main() {
    std::string text = "Hello, World!";
    MyString myStr(std::move(text)); // Move 'text' into 'myStr'

    std::cout << "Original string: " << text << std::endl; // 'text' is now in a valid but unspecified state
    myStr.print(); // Output the moved string




    //example 2
    std::string source = "Hello, World!";  // Create an lvalue 'source'
    
    std::string destination = std::move(source);  // Use 'std::move' to transfer ownership
    
    std::cout << "Source: " << source << std::endl;  // 'source' is now in a valid but unspecified state
    std::cout << "Destination: " << destination << std::endl;  // Output: "Hello, World!"

    return 0;
}
