#ifndef REFERENCE_H__
#define REFERENCE_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void blurImageOpenCV(cv::Mat& rgbImage, cv::Mat& outputImage,
                          size_t numRows, size_t numCols, size_t step,
                          const float* const filter, const int filterWidth);

#endif