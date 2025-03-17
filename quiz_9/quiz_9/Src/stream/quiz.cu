#include <iostream>
#include "image_loader.h"  // Include the header file

using namespace std;

#define KERNEL_SIZE 3
#define BLOCK_SIZE 16  // CUDA block size

// Edge Detection Kernel (Stored in Constant Memory for Faster Access)
__constant__ float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};

// CUDA Kernel for Image Convolution
__global__ void convolveCUDA(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Ensure within bounds

    int sum = 0;
    int offset = KERNEL_SIZE / 2;

    // Apply 3x3 convolution filter
    for (int ky = -offset; ky <= offset; ky++) {
        for (int kx = -offset; kx <= offset; kx++) {
            int pixelX = min(max(x + kx, 0), width - 1);
            int pixelY = min(max(y + ky, 0), height - 1);
            int neighbor_index = pixelY * width + pixelX;
            sum += input[neighbor_index] * kernel[ky + offset][kx + offset];
        }
    }

    int output_index = (y * width + x);
    output[output_index] = min(max(sum, 0), 255);  // Clamp result to valid pixel range
}

// Main Function
int main() {
    int width, height, channels;

    // Load Image (Using image_loader.cpp)
    unsigned char* h_input = load_image("image.jpg", &width, &height, &channels);
    if (!h_input) {
        cerr << "Failed to load image!" << endl;
        return -1;
    }

    size_t img_size = width * height * sizeof(unsigned char);
    unsigned char* d_input, * d_output;

    // Allocate Memory on GPU
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);

    // Copy Input Image to GPU
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // Launch CUDA Kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convolveCUDA << <gridSize, blockSize >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy Output Image from GPU to Host
    unsigned char* h_output = new unsigned char[width * height];
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Save Image (Using image_loader.cpp)
    save_image("output.jpg", h_output, width, height);

    // Cleanup
    free_image(h_input);
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    cout << "CUDA Convolution Applied. Output saved as output.jpg" << endl;
    return 0;
}
