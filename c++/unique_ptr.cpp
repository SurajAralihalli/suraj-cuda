#include <iostream>
#include <memory>

class Animal {
public:
    Animal(const std::string& name) : name(name) {}
    void speak() const {
        std::cout << name << " says hello!" << std::endl;
    }

private:
    std::string name;
};

int main() {
    // Creating an Animal object on the stack
    Animal cat("Whiskers");
    cat.speak();  // Calls the speak method on the Animal object

    // Creating an Animal object using std::unique_ptr
    std::unique_ptr<Animal> dog = std::make_unique<Animal>("Buddy");
    dog->speak();  // Calls the speak method on the Animal object managed by the unique_ptr

    return 0;
}
