#include <stdio.h>

// Define a class with an array in CUDA device code
class MyClass {
public:
    __device__ void printArray() {
        for (int i = 0; i < arraySize; ++i) {
            printf("Element %d: %d\n", i, array[i]);
        }
    }

private:
    static const int arraySize = 5;
    int array[arraySize] = {1, 2, 3, 4, 5};
};

// CUDA kernel function
__global__ void myKernel() {
    MyClass obj;
    obj.printArray();
}

int main() {
    // Launch CUDA kernel
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for kernel to complete

    return 0;
}
