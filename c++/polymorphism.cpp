// Online C++ compiler to run C++ program online
#include <iostream>

class Shape {
public:
    virtual void draw() {
        std::cout << "Drawing a generic shape." << std::endl;
    }
};

class Circle : public Shape {
public:
    void draw() override {
        std::cout << "Drawing a circle." << std::endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() override {
        std::cout << "Drawing a rectangle." << std::endl;
    }
};

int main() {
    Shape* shapes[2]; // Create an array of pointers to the base class

    shapes[0] = new Circle();    // Assign a Circle object
    shapes[1] = new Rectangle(); // Assign a Rectangle object

    for (int i = 0; i < 2; i++) {
        shapes[i]->draw(); // Call the draw() method (polymorphism)
        delete shapes[i]; // Clean up the allocated objects
    }

    return 0;
}
