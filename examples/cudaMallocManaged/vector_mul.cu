#include <iostream>

class Vector {
private:
    int size;
    float* elements;

public:
    // Constructor
    __host__ Vector(int s) : size(s) {
        cudaMallocManaged(&elements, size * sizeof(float));
        for (int i = 0; i < size; ++i) {
            elements[i] = i;
        }
    }

    // Destructor
    __host__ ~Vector() {
        cudaFree(elements);
    }

    // Device method: Multiply vector elements by a scalar
    __device__ void scale(float scalar) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            elements[idx] *= scalar;
        }
    }

    // Host method: Print vector elements
    __host__ void print() {
        for (int i = 0; i < size; ++i) {
            std::cout << elements[i] << " ";
        }
        std::cout << std::endl;
    }
};

__global__ void scaleVector(Vector v, float scalar) {
    v.scale(scalar);
}

int main() {
    const int size = 10;
    const float scalar = 2.0f;

    // Create vector
    Vector v(size);

    // Print original vector
    std::cout << "Original vector: ";
    v.print();

    // Launch kernel to scale vector elements
    scaleVector<<<(size + 255) / 256, 256>>>(v, scalar);
    cudaDeviceSynchronize();

    // Print scaled vector
    std::cout << "Scaled vector: ";
    v.print();

    return 0;
}
