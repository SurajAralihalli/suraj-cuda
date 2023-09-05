#include <iostream>
#include <cuda_runtime.h>

__global__ void busy(int i, int* arr) {
    int start = clock();
    while ((clock() - start) < 100'000'000);
    printf("I'm awake: %d \n", i);
    arr[0]=i;
}


int main()
{
    cudaStream_t streams[5];
    int* arr;
    cudaMallocManaged((int**)&arr, 5 * sizeof(int));
    for(int i=0;i<5;i++) {
        arr[i] = -1;
    }
    
    for (int i = 0; i <5; i++) {
        cudaStreamCreate(&streams[i]);
        busy<<<1, 1,0,streams[i]>>>(i, arr);
    }

    cudaDeviceSynchronize();
    
    for (int i = 0; i <5; i++) {
        cudaStreamDestroy(streams[i]);
    }

    for(int i=0;i<5;i++) {
        std::cout << arr[i] << " ";
    }

    std::cout << std::endl;
}