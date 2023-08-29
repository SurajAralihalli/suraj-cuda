#include <iostream>
#include <cuda_runtime.h>

__global__ void busy() {
    int start = clock();
    while ((clock() - start) < 100'000'000);
    printf("I'm awake!\n");
}

int main()
{
    cudaStream_t streams[5];
    for (int i = 0; i <5; i++) {
        cudaStreamCreate(&streams[i]);
        busy<<<1, 1,0,streams[i]>>>();
    }

    cudaDeviceSynchronize();
    
    for (int i = 0; i <5; i++) {
        cudaStreamDestroy(streams[i]);
    }
}