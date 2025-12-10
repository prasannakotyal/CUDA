#include <iostream>
#include <cuda_runtime.h>

#define N         4096
#define RADIUS    3
#define BLOCK_SIZE 16

__global__
void stencil_1d(int *in, int *out, int n)
{
    extern __shared__ int temp[]; 
    // dynamic shared memory used so size can be BLOCK_SIZE + 2*RADIUS

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Load central values
    if (gindex < n) {
        temp[lindex] = in[gindex];
    }

    // Load halo left and right
    if (threadIdx.x < RADIUS) {
        int left_index = gindex - RADIUS;
        temp[lindex - RADIUS] = (left_index >= 0 ? in[left_index] : 0);

        int right_index = gindex + BLOCK_SIZE;
        temp[lindex + BLOCK_SIZE] = (right_index < n ? in[right_index] : 0);
    }

    __syncthreads();

    // Only compute for valid global indices
    if (gindex < n) {
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
            result += temp[lindex + offset];
        }
        out[gindex] = result;
    }
}

int main(void) {
    int *h_in = nullptr, *h_out = nullptr;
    int *d_in = nullptr, *d_out = nullptr;
    int size = N * sizeof(int);

    // Allocate host arrays
    h_in  = (int *)malloc(size);
    h_out = (int *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_in[i]  = 1;     // e.g., all ones
        h_out[i] = 0;
    }

    // Allocate device memory
    cudaMalloc(&d_in,  size);
    cudaMalloc(&d_out, size);

    // Copy data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel: note grid = ceil(N / BLOCK_SIZE)
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<grid, BLOCK_SIZE, (BLOCK_SIZE + 2*RADIUS)*sizeof(int)>>>(d_in, d_out, N);

    // Copy result back
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; ++i) {
        int expected = 0;
        // sum over RADIUS each side
        for (int j = -RADIUS; j <= RADIUS; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < N) {
                expected += 1;  // since h_in[...] = 1
            }
        }
        if (h_out[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": got " << h_out[i]
                      << " expected " << expected << "\n";
            return -1;
        }
    }

    std::cout << "Success!\n";

    // Cleanup
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
