#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "utils.hpp"

using namespace cv;

Mat readImage(std::string imgPath){
    Mat img = imread(imgPath, IMREAD_COLOR);

    if(img.empty()) {
        throw std::runtime_error("Could not read image from path: " + imgPath);
    } else {
        std::cout << "Image read successfully!" << std::endl;
    }

    return img;
}

void saveImage(const Mat& img, std::string imgPath) {
  cv::imwrite(imgPath.c_str(), img);
}

int getRows(const Mat& img) {
    return img.rows;
}

int getCols(const Mat& img) {
    return img.cols;
}

int getStep(const Mat& img){
    return img.step;
}

