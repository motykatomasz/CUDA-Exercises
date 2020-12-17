#include "opencv_func.hpp"

void rgbToGreyscaleOpenCV(cv::Mat& rgbImage, cv::Mat& greyImage, 
                    int rows, int cols, int step) {

unsigned char* rgbImagePtr = rgbImage.ptr<unsigned char>(0);
unsigned char* greyImagePtr = rgbImage.ptr<unsigned char>(0);



for (int i=0; i<rows; ++i){
    for (int j=0; j<step; j=j+3){

        int colIndex = j/3;
        int indexGrey = i * cols + colIndex;

        greyImagePtr[indexGrey] = .299f * rgbImagePtr[j + 2] + 
                               .587f * rgbImagePtr[j + 1] + 
                               .114f * rgbImagePtr[j + 0];

    }
}

}