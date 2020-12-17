#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


cv::Mat readImage(std::string imgPath) {
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

    if(img.empty()) {
        throw std::runtime_error("Could not read image from path: " + imgPath);
    } else {
        std::cout << "Image read successfully!" << std::endl;
    }

    return img;
}

void saveImage(const cv::Mat& img, std::string imgPath) {
  cv::imwrite(imgPath.c_str(), img);
}

int getRows(const cv::Mat& img) {
    return img.rows;
}

int getCols(const cv::Mat& img) {
    return img.cols;
}

int getStep(const cv::Mat& img){
    return img.step;
}

void createFilter(float* filter, int blurKernelWidth, float blurKernelSigma) {

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }
}

void printFilter(float* filter, int blurKernelWidth) {
  for (int r = 0;r < blurKernelWidth; ++r) {
    for (int c = 0;c < blurKernelWidth; ++c) {
      std::cout << filter[r*blurKernelWidth + c] << "-";
    }
    std::cout << std::endl;
  }
}
