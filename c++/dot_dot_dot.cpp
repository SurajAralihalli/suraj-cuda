#include <cstdarg>
#include <iostream>

// Variadic function that calculates the sum of integers
int sum_integers(int count, ...) {
    va_list args;
    va_start(args, count);

    int sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += va_arg(args, int);
    }

    va_end(args);
    return sum;
}

// Variadic template function that prints elements
template<typename T>
void print_elements(T value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Args>
void print_elements(T value, Args... args) {
    std::cout << value << ", ";
    print_elements(args...);
}

// Variadic macro to print values
#define PRINT_VALUES(...) \
    do { \
        std::cout << "Values: "; \
        print_values(__VA_ARGS__); \
    } while(0)

// Helper function for the variadic macro
template<typename T>
void print_values(T value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Args>
void print_values(T value, Args... args) {
    std::cout << value << ", ";
    print_values(args...);
}

int main() {
    // Example of the variadic function
    std::cout << "Sum: " << sum_integers(3, 1, 2, 3) << std::endl;  // Output: Sum: 6

    // Example of the variadic template
    print_elements(1, 2, 3, "four", 5.5);  // Output: 1, 2, 3, four, 5.5

    // Example of the variadic macro
    PRINT_VALUES(1, 2, "three", 4.4);  // Output: Values: 1, 2, three, 4.4

    return 0;
}
