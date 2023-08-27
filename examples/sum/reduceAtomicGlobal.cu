#include <iostream>
#include <cuda_runtime.h>

__device__ float result = 0;

__global__ void reduceAtomicGlobal(const float* input, int N) {
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < N)
    atomicAdd(&result, input[id]);
}

int main() {
    int N = 4000;
    float* array = NULL;
    cudaMallocManaged((float**)&array, sizeof(float) * N);
    
    for(int i=0;i<N;i++) {
        array[i] = i+0.5;
    }

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    reduceAtomicGlobal<<<gridSize,blockSize>>>(array, N);
    cudaDeviceSynchronize();

    float hostResult;
    cudaMemcpyFromSymbol(&hostResult, result, sizeof(float));

    std::cout << "sum: " << hostResult << std::endl;

    cudaFree(array);

}

