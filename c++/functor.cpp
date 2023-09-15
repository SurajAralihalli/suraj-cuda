#include <iostream>

class SquareFunctor {
public:
    int operator()(int x) const {
        return x * x;
    }
};

int main() {
    SquareFunctor square;

    int result = square(5); // Use the functor like a function
    std::cout << "The square of 5 is: " << result << std::endl;

    return 0;
}
