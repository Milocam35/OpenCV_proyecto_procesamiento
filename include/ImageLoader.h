#ifndef IMAGE_LOADER
#define IMAGE_LOADER

#include <string>

class ImageLoader {
public:
    ImageLoader() = default;
    bool loadAndShowImage(const std::string& imagePath);
};

    #endif // IMAGE_LOADER