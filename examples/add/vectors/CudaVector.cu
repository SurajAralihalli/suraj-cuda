#include <iostream>
#include <cuda_runtime.h>

class CudaVector {
private:
    float *deviceData;
    size_t size;

public:
    CudaVector(size_t _size) : size(_size) {
        cudaMalloc(&deviceData, size * sizeof(float));
    }

    ~CudaVector() {
        cudaFree(deviceData);
    }

    void setData(const float *hostData) {
        cudaMemcpy(deviceData, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    void getData(float *hostData) {
        cudaMemcpy(hostData, deviceData, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void add(const CudaVector &other) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        vectorAddKernel<<<gridSize, blockSize>>>(deviceData, other.deviceData, size);
        cudaDeviceSynchronize();
    }

    __global__ void vectorAddKernel(float *a, const float *b, size_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            a[idx] += b[idx];
        }
    }
};

int main() {
    const size_t vectorSize = 1000;
    float hostVectorA[vectorSize];
    float hostVectorB[vectorSize];

    // Initialize hostVectorA and hostVectorB...

    CudaVector cudaVecA(vectorSize);
    CudaVector cudaVecB(vectorSize);

    cudaVecA.setData(hostVectorA);
    cudaVecB.setData(hostVectorB);

    cudaVecA.add(cudaVecB);

    float resultVector[vectorSize];
    cudaVecA.getData(resultVector);

    // Print resultVector...

    return 0;
}
