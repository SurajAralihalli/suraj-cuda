#include <iostream>
#include <memory>
#include <cuda_runtime.h>

class CudaBuffer {
public:
    CudaBuffer(size_t size) : size_(size) {
        cudaMalloc(&data_, size_);
        if (data_ == nullptr) {
            throw std::runtime_error("Failed to allocate CUDA memory");
        }
    }

    ~CudaBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void FillWithZeros() {
        cudaMemset(data_, 0, size_);
    }

private:
    size_t size_;
    int* data_;  // Just an example, you can use appropriate data type
};

int main() {
    try {
        CudaBuffer cudaBuffer(100 * sizeof(int));  // Allocate CUDA memory

        cudaBuffer.FillWithZeros();  // Fill with zeros

        // 'cudaBuffer' goes out of scope here and destructor is automatically called (because variable is on stack)

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        // Handle the exception
    }

    return 0;
}
