#include <iostream>
#include <chrono>  // For measuring CPU execution time
#include "image_loader.h"  // Image loading utilities

using namespace std;
using namespace std::chrono;

#define KERNEL_SIZE 3
#define BLOCK_SIZE 16  // CUDA block size

// Edge Detection Kernel (Stored in Constant Memory for Faster Access)
__constant__ float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};

// CPU Function for Image Convolution
void convolveCPU(const unsigned char* input, unsigned char* output, int width, int height) {
    int offset = KERNEL_SIZE / 2;

    // ✅ Declare CPU-side kernel
    float cpu_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    int pixelX = min(max(x + kx, 0), width - 1);
                    int pixelY = min(max(y + ky, 0), height - 1);
                    int index = pixelY * width + pixelX;

                    // ✅ Use cpu_kernel instead of device kernel
                    sum += input[index] * cpu_kernel[ky + offset][kx + offset];
                }
            }

            int output_index = y * width + x;
            output[output_index] = min(max(int(sum), 0), 255);  // Clamp to valid range
        }
    }
}

// CUDA Kernel for Image Convolution
__global__ void convolveCUDA(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Ensure within bounds

    int sum = 0;
    int offset = KERNEL_SIZE / 2;

    for (int ky = -offset; ky <= offset; ky++) {
        for (int kx = -offset; kx <= offset; kx++) {
            int pixelX = min(max(x + kx, 0), width - 1);
            int pixelY = min(max(y + ky, 0), height - 1);
            int neighbor_index = pixelY * width + pixelX;
            sum += input[neighbor_index] * kernel[ky + offset][kx + offset];
        }
    }

    int output_index = (y * width + x);
    output[output_index] = min(max(sum, 0), 255);  // Clamp result to valid range
}

// Main Function
int main() {
    int width, height, channels;

    // Load Image
    unsigned char* h_input = load_image("image.jpg", &width, &height, &channels);
    if (!h_input) {
        cerr << "Failed to load image!" << endl;
        return -1;
    }

    size_t img_size = width * height * sizeof(unsigned char);
    unsigned char* h_output_cpu = new unsigned char[width * height];
    unsigned char* h_output_gpu = new unsigned char[width * height];

    // Measure CPU Execution Time
    auto start_cpu = high_resolution_clock::now();
    convolveCPU(h_input, h_output_cpu, width, height);
    auto stop_cpu = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(stop_cpu - start_cpu);
    cout << "CPU Convolution Time: " << cpu_duration.count() << " ms" << endl;

    // Allocate Memory on GPU
    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);

    // Copy Input Image to GPU
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    // ✅ CUDA Event Timing for GPU Execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // ✅ Run the kernel 100 times to prevent GPU power throttling
    for (int i = 0; i < 100; i++) {
        convolveCUDA << <dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (d_input, d_output, width, height);
    }

    // ✅ Synchronize and Stop Timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "GPU Convolution Time (100 runs avg): " << milliseconds / 100 << " ms" << endl;

    // Copy Output Image from GPU to Host
    cudaMemcpy(h_output_gpu, d_output, img_size, cudaMemcpyDeviceToHost);

    // Save Results
    save_image("output_cpu.jpg", h_output_cpu, width, height);
    save_image("output_gpu.jpg", h_output_gpu, width, height);

    // Cleanup
    free_image(h_input);
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "CUDA Convolution Applied. Results saved as output_cpu.jpg and output_gpu.jpg" << endl;
    return 0;
}
