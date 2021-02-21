#include <iostream>
#include <chrono> 

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/utils.hpp"
#include "cuda/cuda_func.hpp"
#include "opencv/opencv_func.hpp"


int main(int argc, char **argv) {
    cv::Mat rgbImage, greyImageCuda, greyImageOpenCV;

    rgbImage = readImage("../images/abc.jpg");

    int numRows = getRows(rgbImage);
    int numCols = getCols(rgbImage);

    std::cout << "Num rows: " << numRows << std::endl;
    std::cout << "Num cols: " << numCols << std::endl;

    greyImageCuda = cv::Mat::zeros(cv::Size(numCols,numRows), CV_8UC1);
    greyImageOpenCV = cv::Mat::zeros(cv::Size(numCols,numRows), CV_8UC1);

    if (!rgbImage.isContinuous() || !greyImageCuda.isContinuous() || !greyImageOpenCV.isContinuous()) {
        throw std::runtime_error("Images must be stored in continuous fashion in order for this solution to work!!!");
    } else {
        std::cout << "Images are continuous." << std::endl;
    }

    rgbToGreyscale(rgbImage, greyImageCuda, numRows, numCols);

    saveImage(greyImageCuda, "../images/abc_grey_cuda.jpg");

    int step = getStep(rgbImage);

    auto start = std::chrono::high_resolution_clock::now();

    rgbToGreyscaleOpenCV(rgbImage, greyImageOpenCV, numRows, numCols, step);

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;

    std::cout << "OpenCV code ran in: " << elapsed.count() * 1000 << " msecs." << std::endl;

    saveImage(greyImageCuda, "../images/abc_grey_opencv.jpg");

    return 0;
}
