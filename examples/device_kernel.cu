#include <iostream>
#include <cuda_runtime.h>

__device__ int addOnDevice(int a, int b) {
    return a + b;
}

__global__ void myKernel(int *result) {
    // Call the __device__ function from the kernel
    *result = addOnDevice(3, 5);
}

int main() {
    // This is host code (executed on the CPU)
    
    // Allocate memory on the device
    int *d_result;
    cudaMalloc((void**)&d_result, sizeof(int));

    // Launch a kernel that uses the __device__ function
    myKernel<<<1, 1>>>(d_result);

    // Copy the result back from the device
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result: " << h_result << std::endl;

    // Free allocated memory on the device
    cudaFree(d_result);

    return 0;
}
