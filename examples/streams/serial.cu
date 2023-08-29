#include <iostream>
#include <cuda_runtime.h>

__global__ void busy() {
    int start = clock();
    while ((clock() - start) <100'000'000);
    printf("I'm awake!\n");
}

int main() {
    for (int i = 0; i <5; i++) {
        busy<<<1, 1>>>();
        cudaDeviceSynchronize(); 
    }
}