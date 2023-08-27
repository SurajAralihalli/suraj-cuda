#include <iostream>
#include <cuda_runtime.h>

using namespace std;


template<typename T>
__global__ void addKernel(T *vecData, T *otherVecData, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        vecData[idx] += otherVecData[idx];
    }
}

template<typename T>
class cudaVec {
    private:
        int N;

    public:
        T* vecData;
        cudaVec(int count);
        ~cudaVec();
        void add(const cudaVec<T>* otherVec);
};

template<typename T>
cudaVec<T>::cudaVec(int count) {
    this->N = count;
    cudaMallocManaged((T**)&vecData, count * sizeof(T));
}

template<typename T>
cudaVec<T>::~cudaVec() {
    cudaFree(vecData);
}

template<typename T>
void cudaVec<T>::add(const cudaVec *otherVec) {
    int blockSize = 50;
    int gridSize = (N + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize>>>(this->vecData, otherVec->vecData, N);
    cudaDeviceSynchronize();
    cout << "add completed" << endl;
}


int main() {
    int N = 500;

    cudaVec<float> vec1(N);
    cudaVec<float>* vec2 = new cudaVec<float>(N);

    for(int i=0; i<N; i++) {
        vec1.vecData[i] = 1;
        vec2->vecData[i] = i;
    }

    vec1.add(vec2);

    for(int i=0;i<N;i++) {
        cout << vec1.vecData[i] << " ";
    }
    cout << endl;

    delete vec2;

    return 0;
}