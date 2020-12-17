#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



void rgbToGreyscale(cv::Mat& rgbImage, cv::Mat& greyImage, 
                    int rows, int cols);

void prepare(unsigned char **h_ptrRgbImage, unsigned char **h_ptrGreyImage, 
            cv::Mat& rgbImage, cv::Mat& greyImage);