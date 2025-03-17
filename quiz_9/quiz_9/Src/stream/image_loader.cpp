#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "image_loader.h"

unsigned char* load_image(const char* filename, int* width, int* height, int* channels) {
    return stbi_load(filename, width, height, channels, 1);  // Force grayscale
}

void free_image(unsigned char* image) {
    stbi_image_free(image);
}

void save_image(const char* filename, unsigned char* image, int width, int height) {
    stbi_write_jpg(filename, width, height, 1, image, 100);
}
