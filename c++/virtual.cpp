class Animal {
public:
    virtual void makeSound() {
        std::cout << "Animal makes a generic sound" << std::endl;
    }
};

class Dog : public Animal {
public:
    void makeSound() override {
        std::cout << "Dog barks" << std::endl;
    }
};

class Cat : public Animal {
public:
    void makeSound() override {
        std::cout << "Cat meows" << std::endl;
    }
};

int main() {
    Animal* myAnimal;
    Dog myDog;
    Cat myCat;

    myAnimal = &myDog;
    myAnimal->makeSound(); // Calls Dog's makeSound

    myAnimal = &myCat;
    myAnimal->makeSound(); // Calls Cat's makeSound

    return 0;
}
