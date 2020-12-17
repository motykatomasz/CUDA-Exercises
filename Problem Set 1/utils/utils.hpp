#include <opencv2/core.hpp>

using namespace cv;

Mat readImage(std::string imgPath);

void saveImage(const Mat& img, std::string imgPath);

int getRows(const Mat& img);

int getCols(const Mat& img);

int getStep(const Mat& img);