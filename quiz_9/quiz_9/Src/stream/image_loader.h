#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

	unsigned char* load_image(const char* filename, int* width, int* height, int* channels);
	void free_image(unsigned char* image);
	void save_image(const char* filename, unsigned char* image, int width, int height);

#ifdef __cplusplus
}
#endif

#endif  // IMAGE_LOADER_H

