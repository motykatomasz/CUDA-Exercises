#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

void rgbToGreyscaleOpenCV(cv::Mat& rgbImage, cv::Mat& greyImage, 
                    int rows, int cols, int step);