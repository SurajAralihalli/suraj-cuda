#include <iostream>
#include <cuda_runtime.h>

__device__ float result = 0;

__global__ void reduceShared(const float* input, int N)
{
    extern __shared__ float data[];
    
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    data[threadIdx.x] = (id < N ? input[id] : 0);

    for (int s = blockDim.x/2; s > 0; s/=2)
    {
        __syncthreads();
        if (threadIdx.x < s) {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result, data[0]);
    }
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
    float sharedMemSize = blockSize * sizeof(float);

    reduceShared<<<gridSize,blockSize,sharedMemSize>>>(array, N);
    cudaDeviceSynchronize();

    float hostResult;
    cudaMemcpyFromSymbol(&hostResult, result, sizeof(float));

    std::cout << "sum: " << hostResult << std::endl;

    cudaFree(array);

}