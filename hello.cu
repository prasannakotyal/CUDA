#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel that runs on the GPU
__global__ void hello_cuda() {
    printf("Hello World from GPU! (Thread %d)\n", threadIdx.x);
}

int main() {
    // Print from CPU
    printf("Hello World from CPU!\n");

    // Launch kernel with 1 block and 1 thread
    hello_cuda<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}