#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <vector>
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

#define KERNEL_SIZE 3

// Sample Edge Detection Kernel
float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {-1, -1, -1},
    {-1,  8, -1},
    {-1, -1, -1}
};

// Convolution function
void convolve(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int offset = KERNEL_SIZE / 2;

    for (int y = offset; y < height - offset; ++y) {
        for (int x = offset; x < width - offset; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    int pixelX = x + kx - offset;
                    int pixelY = y + ky - offset;
                    int index = (pixelY * width + pixelX) * channels;
                    sum += input[index] * kernel[ky][kx];
                }
            }
            int output_index = (y * width + x) * channels;
            output[output_index] = max(0, min(255, static_cast<int>(sum))); // Ensure valid pixel range
        }
    }
}

int main() {
    int width, height, channels;

    // Load image (grayscale mode)
    unsigned char* image = stbi_load("image.jpg", &width, &height, &channels, 1);
    if (!image) {
        cerr << "Failed to load image!" << endl;
        return -1;
    }

    // Allocate memory for output image
    unsigned char* output = new unsigned char[width * height];

    // Apply convolution
    convolve(image, output, width, height, 1);

    // Save output image
    stbi_write_jpg("output.jpg", width, height, 1, output, 100);

    // Cleanup
    stbi_image_free(image);
    delete[] output;

    cout << "Convolution applied. Output saved as output.jpg" << endl;
    return 0;
}
