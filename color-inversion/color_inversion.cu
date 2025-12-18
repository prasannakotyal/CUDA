#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

/* CUDA kernel */
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx >= total) return;

    int base = idx * 4;

    image[base + 0] = 255 - image[base + 0]; // R
    image[base + 1] = 255 - image[base + 1]; // G
    image[base + 2] = 255 - image[base + 2]; // B
    // Alpha unchanged
}

/* Initialize image */
void init_image(std::vector<unsigned char>& image) {
    for (int i = 0; i < image.size(); i += 4) {
        image[i + 0] = rand() % 256;
        image[i + 1] = rand() % 256;
        image[i + 2] = rand() % 256;
        image[i + 3] = 255;
    }
}

/* Print image */
void print_image(const std::vector<unsigned char>& image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int base = (y * width + x) * 4;
            std::cout << "("
                      << (int)image[base + 0] << ","
                      << (int)image[base + 1] << ","
                      << (int)image[base + 2] << ","
                      << (int)image[base + 3] << ") ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/* CUDA wrapper */
void invert_image(std::vector<unsigned char>& image, int width, int height) {
    unsigned char* d_image;
    int bytes = image.size();

    cudaMalloc(&d_image, bytes);
    cudaMemcpy(d_image, image.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;

    invert_kernel<<<blocks, threads>>>(d_image, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(image.data(), d_image, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}

int main() {
    srand(42);

    int width = 2;
    int height = 2;

    std::vector<unsigned char> image(width * height * 4);

    init_image(image);

    std::cout << "Original image:\n";
    print_image(image, width, height);

    invert_image(image, width, height);

    std::cout << "Inverted image:\n";
    print_image(image, width, height);

    return 0;
}
