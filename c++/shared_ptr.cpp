#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass(int val) : value(val) {}
    void setValue(int val) { value = val; }
    int getValue() const { return value; }

private:
    int value;
};

int main() {
    // Creating a shared pointer to a dynamically allocated instance of MyClass
    std::shared_ptr<MyClass> sharedObjPtr = std::make_shared<MyClass>(42);

    // Using the shared pointer to access and modify the object
    std::cout << "Initial value: " << sharedObjPtr->getValue() << std::endl;
    sharedObjPtr->setValue(99);
    std::cout << "Modified value: " << sharedObjPtr->getValue() << std::endl;

    // Creating another shared pointer pointing to the same object
    std::shared_ptr<MyClass> anotherSharedObjPtr = sharedObjPtr;

    // Using the other shared pointer to modify the object
    anotherSharedObjPtr->setValue(123);
    std::cout << "Value modified via another shared pointer: " << sharedObjPtr->getValue() << std::endl;

    // The object will be automatically deallocated when all shared pointers go out of scope
    return 0;
}
