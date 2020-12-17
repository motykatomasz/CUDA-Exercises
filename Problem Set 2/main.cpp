#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/utils.hpp"
#include "utils/timer.hpp"
#include "utils/helpers.hpp"

#include "cuda/cuda_func.hpp"
#include "opencv/opencv_func.hpp"


int main(int argc, char **argv) {
    cv::Mat originalImage, blurredImageCuda, blurredImageOpenCV;
    float* filter;

    originalImage = readImage("../images/abc.jpg");

    size_t numRows = getRows(originalImage);
    size_t numCols = getCols(originalImage);
    size_t step = getStep(originalImage);

    const int blurKernelWidth = 9;
    const float blurKernelSigma = 2.;

    filter = new float[blurKernelWidth * blurKernelWidth];

    createFilter(filter, blurKernelWidth, blurKernelSigma);
    // printFilter(filter, blurKernelWidth);

    blurredImageCuda = cv::Mat::zeros(cv::Size(numCols,numRows), CV_8UC3);
    blurredImageOpenCV = cv::Mat::zeros(cv::Size(numCols,numRows), CV_8UC3);

    if (!originalImage.isContinuous() || !blurredImageCuda.isContinuous() || !blurredImageOpenCV.isContinuous()) {
        throw std::runtime_error("Images must be stored in continuous fashion in order for this solution to work!!!");
    } else {
        std::cout << "Images are continuous." << std::endl;
    }

    blurImageCuda(originalImage, blurredImageCuda, filter, numRows, numCols, step, blurKernelWidth);

    saveImage(blurredImageCuda, "../images/abc_blur_cuda.jpg");

    blurImageOpenCV(originalImage, blurredImageOpenCV, numRows, numCols, step, filter, blurKernelWidth);

    saveImage(blurredImageOpenCV, "../images/abc_blur_opencv.jpg");

    return 0;
}
