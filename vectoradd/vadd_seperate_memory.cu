#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// constants
const int N = 1 << 20;
const int BLOCK_SZ = 256;
const float A_VAL = 1.0f;
const float B_VAL = 2.0f;

__global__
void add(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    // host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = A_VAL;
        h_b[i] = B_VAL;
        h_c[i] = 0.0f;
    }

    // device pointers
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // host -> device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK_SZ - 1) / BLOCK_SZ;

    // launch kernel
    add<<<grid, BLOCK_SZ>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // device -> host (result is in d_a)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // results verification
    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_error = fmax(max_error, fabs(h_c[i] - (A_VAL + B_VAL)));
    }
    cout << "Max error: " << max_error << endl;

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
