#include <stdio.h>
#include <cassert>
#include <iostream>



using namespace std;

__global__ void stencil_1d(int *in, int *out, int BLOCK_SIZE, int RADIUS, int N)
{ 

  extern __shared__ int temp[];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
  int lindex = threadIdx.x + RADIUS;
  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {

    if(gindex - RADIUS >= 0) {
      temp[lindex - RADIUS] = in[gindex - RADIUS]; 
    }
    if(gindex + BLOCK_SIZE < N) {
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }
  }

  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
    result += temp[lindex + offset];
  }
  // Store the result
  out[gindex] = result;
}

int main () {
  int* in;
  int* out;

  int BLOCK_SIZE = 256;
  int RADIUS = 3;

  int N = 256 * 5;
  int bytes = sizeof(int) * N;

  cudaMallocManaged((int **)&in,bytes);
  cudaMallocManaged((int **)&out,bytes);

  cudaMemAdvise(in, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  for (int i = 0; i < N; i++) {
    in[i] = 1;
    out[i] = 0;
  }

  int GRID_SIZE = (N + BLOCK_SIZE - 1)/ BLOCK_SIZE;

  int id = cudaGetDevice(&id);

  cudaMemAdvise(in, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemAdvise(out, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemPrefetchAsync(in, bytes, id);
  cudaMemPrefetchAsync(out, bytes, id);

  int sharedMemSize = sizeof(int) * (BLOCK_SIZE + (2 * RADIUS));
  stencil_1d<<<GRID_SIZE, BLOCK_SIZE,sharedMemSize>>>(in, out, BLOCK_SIZE, RADIUS, N);

  cudaDeviceSynchronize();
  cudaMemPrefetchAsync(out, bytes, cudaCpuDeviceId);


  for (int i = 0; i < N; i++) {
    // assert(out[i] == (2*RADIUS + 1));
    cout << out[i] << " ";
  }
  cout << endl;

  cout << "stencil completed" << endl;

  cudaFree(in);
  cudaFree(out);

  return 0;
}