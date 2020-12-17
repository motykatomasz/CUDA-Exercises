#include <algorithm>
#include <cassert>
#include <iostream>
#include <chrono>

#include "opencv_func.hpp"

void channelConvolution(const unsigned char* const channel,
                        unsigned char* const channelBlurred,
                        const size_t numRows, const size_t numCols,
                        const float *filter, const int filterWidth)
{
  //Dealing with an even width filter is trickier
  assert(filterWidth % 2 == 1);

  //For every pixel in the image
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
		      int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value = static_cast<float>(channel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result += image_value * filter_value;
        }
      }

      channelBlurred[r * numCols + c] = result;
    }
  }
}

void blurImageOpenCV(cv::Mat& rgbImage, cv::Mat& outputImage,
                          size_t numRows, size_t numCols, size_t step,
                          const float* const filter, const int filterWidth){

  const unsigned char* const originalImagePtr = rgbImage.ptr<unsigned char>(0);
  unsigned char* const blurImageOpenCVPtr = outputImage.ptr<unsigned char>(0);

  unsigned char *red   = new unsigned char[numRows * numCols];
  unsigned char *blue  = new unsigned char[numRows * numCols];
  unsigned char *green = new unsigned char[numRows * numCols];

  unsigned char *redBlurred   = new unsigned char[numRows * numCols];
  unsigned char *blueBlurred  = new unsigned char[numRows * numCols];
  unsigned char *greenBlurred = new unsigned char[numRows * numCols];

  //First we separate the incoming RGBA image into three separate channels
  //for Red, Green and Blue
  for (size_t i = 0; i < numRows * numCols; ++i) {
    int pixelStart = i*3;
    red[i]   = originalImagePtr[pixelStart+2];
    green[i] = originalImagePtr[pixelStart+1];
    blue[i]  = originalImagePtr[pixelStart];
  }

  auto start = std::chrono::high_resolution_clock::now();

  //Now we can do the convolution for each of the color channels
  channelConvolution(red, redBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(green, greenBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(blue, blueBlurred, numRows, numCols, filter, filterWidth);

  auto finish = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = finish - start;

  std::cout << "OpenCV code ran in: " << elapsed.count() * 1000 << " msecs." << std::endl;

  //now recombine into the output image - Alpha is 255 for no transparency
  for (size_t i = 0; i < numRows * numCols; ++i) {
    int pixelStart = i*3;
    blurImageOpenCVPtr[pixelStart] = blueBlurred[i];
    blurImageOpenCVPtr[pixelStart+1] = greenBlurred[i];
    blurImageOpenCVPtr[pixelStart+2] = redBlurred[i];
  }

  delete[] red;
  delete[] green;
  delete[] blue;

  delete[] redBlurred;
  delete[] greenBlurred;
  delete[] blueBlurred; 
}
