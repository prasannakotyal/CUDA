#include <iostream>
#include <cuda_runtime.h>
using namespace std;

const int DSIZE     = 4096;
const int BLOCK_SZ  = 16;
const float A_VAL   = 1.0f;
const float B_VAL   = 2.0f;

__global__
void mmul(const float *A, const float *B, float *C, int ds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < ds && idy < ds) {
        float sum = 0.0f;
        for (int i = 0; i < ds; ++i) {
            sum += A[idy * ds + i] * B[i * ds + idx];
        }
        C[idy * ds + idx] = sum;
    }
}

int main()
{
    size_t bytes = DSIZE * DSIZE * sizeof(float);

    float *h_A = new float[DSIZE * DSIZE];
    float *h_B = new float[DSIZE * DSIZE];
    float *h_C = new float[DSIZE * DSIZE];

    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        h_A[i] = A_VAL;
        h_B[i] = B_VAL;
        h_C[i] = 0.0f;
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid((DSIZE + BLOCK_SZ - 1) / BLOCK_SZ,
              (DSIZE + BLOCK_SZ - 1) / BLOCK_SZ);

    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    float expected = A_VAL * B_VAL * DSIZE;
    float max_error = 0.0f;
    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        max_error = fmax(max_error, fabs(h_C[i] - expected));
    }
    cout << "Max error: " << max_error << endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
