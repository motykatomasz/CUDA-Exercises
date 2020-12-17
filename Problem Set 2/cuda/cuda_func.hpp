#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


void blurImageCuda(cv::Mat& originalImgage, cv::Mat& blurredImage, float* filter,
                    size_t rows, size_t cols, size_t step, int kernelWidth);

void prepare(unsigned char **h_ptrImage,
            cv::Mat& originalImage);

void mergeImages(cv::Mat& blurredImage, unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel,
                    size_t rows, size_t cols);
