#include <iostream>
#include <utility> // for std::move

class MyClass {
private:
    int* data; // Pointer to dynamically allocated data

public:
    // Default constructor
    MyClass() : data(nullptr) {}

    // Constructor
    explicit MyClass(int value) : data(new int(value)) {}

    // Destructor
    ~MyClass() {
        delete data;
    }

    // Copy constructor
    MyClass(const MyClass& other) : data(new int(*other.data)) {
        std::cout << "Copy constructor called" << std::endl;
    }

    // Move constructor
    MyClass(MyClass&& other) noexcept : data(other.data) {
        other.data = nullptr; // Reset the source object's pointer
        std::cout << "Move constructor called" << std::endl;
    }

    // Copy assignment operator
    MyClass& operator=(const MyClass& other) {
        if (this != &other) {
            delete data; // Release current data

            // Allocate new data and copy from the source
            data = new int(*other.data);
        }
        std::cout << "Copy assignment operator called" << std::endl;
        return *this;
    }

    // Move assignment operator
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            delete data; // Release current data

            // Move data pointer from the source and reset source pointer
            data = other.data;
            other.data = nullptr;
        }
        std::cout << "Move assignment operator called" << std::endl;
        return *this;
    }

    // Getter function
    int getValue() const {
        return (data ? *data : -1); // Return data if not null, otherwise -1
    }
};

int main() {
    // Example of copy constructor
    MyClass obj1(10);
    MyClass obj2(obj1); // Calls copy constructor
    std::cout << "Value of obj2: " << obj2.getValue() << std::endl;

    // Example of move constructor using std::move()
    MyClass obj3 = std::move(obj1); // Calls move constructor
    std::cout << "Value of obj3: " << obj3.getValue() << std::endl;

    // Example of copy assignment operator
    MyClass obj4;
    obj4 = obj2; // Calls copy assignment operator
    std::cout << "Value of obj4: " << obj4.getValue() << std::endl;

    // Example of move assignment operator using std::move()
    MyClass obj5;
    obj5 = std::move(obj3); // Calls move assignment operator
    std::cout << "Value of obj5: " << obj5.getValue() << std::endl;

    return 0;
}
