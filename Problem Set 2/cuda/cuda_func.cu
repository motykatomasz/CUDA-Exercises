// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../utils/utils.hpp"
#include "../utils/timer.hpp"

#include "cuda_func.hpp"


__global__ void splitChannels(unsigned char* const originalImage,  unsigned char* const redChannel, 
                            unsigned char* const greenChannel,  unsigned char* const blueChannel, size_t cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelIndexSingleChannel = x * cols + y;
    int pixelIndex = 3 * (x * cols + y);

    blueChannel[pixelIndexSingleChannel] = originalImage[pixelIndex];
    greenChannel[pixelIndexSingleChannel] = originalImage[pixelIndex + 1];
    redChannel[pixelIndexSingleChannel] = originalImage[pixelIndex + 2];
}

__global__ void blurringKernelNaive(unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel, float* filter, 
                                size_t rows, size_t cols, size_t kernelWidth, size_t apron) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float newValueRed = 0.0;
    float newValueGreen = 0.0;
    float newValueBlue = 0.0;

    int xStartIndex = x-apron;
    int yStartIndex = y-apron;

    // This is naive implementation.
    for (int i=0; i<kernelWidth; ++i) {
        int k = i + xStartIndex;
        for (int j=0; j<=kernelWidth; ++j) {
            int l = j + yStartIndex;

            if ((k>=0 && k<rows) && (l>=0 && l<cols)) {
                float val = filter[i*kernelWidth + j];
                newValueRed += redChannel[k*cols + l] * val;
                newValueGreen += greenChannel[k*cols + l] * val;
                newValueBlue += blueChannel[k*cols + l] * val;
            }
        }
    }

    __syncthreads();

    redChannel[x*cols + y] = newValueRed;
    greenChannel[x*cols + y] = newValueGreen;
    blueChannel[x*cols + y] = newValueBlue;
    }

__global__ void blurringKernelBetter(unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel, float* filter, 
        size_t rows, size_t cols, size_t kernelWidth, size_t apron) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // 1. Read part of image corresponding to the filter. (Coalescing)

    __shared__ float redBlock[rows * cols];
    __shared__ float greenBlock[rows * cols];
    __shared__ float blueBlock[rows * cols];

    // TODO This is still a bit naive usage of shared memory. Part of pixels for consolution will be read from global memory (Need to include apron.)
    redBlock[x*cols + y] = redChannel[x*cols + y];
    greenBlock[x*cols + y] = greenChannel[x*cols + y];
    blueBlock[x*cols + y] = blueChannel[x*cols + y];

    __syncthreads();

    float newValueRed = 0.0;
    float newValueGreen = 0.0;
    float newValueBlue = 0.0;

    int xStartIndex = x-apron;
    int yStartIndex = y-apron;

    for (int i=0; i<kernelWidth; ++i) {
        int k = i + xStartIndex;
        for (int j=0; j<=kernelWidth; ++j) {
            int l = j + yStartIndex;

            if ((k>=0 && k<rows) && (l>=0 && l<cols)) {
                float val = filter[i*kernelWidth + j];
                newValueRed += redChannel[k*cols + l] * val;
                newValueGreen += greenChannel[k*cols + l] * val;
                newValueBlue += blueChannel[k*cols + l] * val;
            }
        }
    }

    __syncthreads();

    redChannel[x*cols + y] = newValueRed;
    greenChannel[x*cols + y] = newValueGreen;
    blueChannel[x*cols + y] = newValueBlue;
}

void blurImageCuda(cv::Mat& originalImage, cv::Mat& blurredImage, float* h_filter, size_t rows, size_t cols, size_t step, int kernelWidth) {
    unsigned char *h_redChannel = new unsigned char[rows*cols];
    unsigned char *h_greenChannel = new unsigned char[rows*cols];
    unsigned char *h_blueChannel = new unsigned char[rows*cols];

    unsigned char *d_redChannel, *d_greenChannel, *d_blueChannel;
    unsigned char *h_ptrImage, *d_ptrImage;
    float *d_filter;

    int CHANNEL_BYTES = rows * cols;
    int IMAGE_BYTES = rows * step;
    int FILTER_BYTES = kernelWidth * kernelWidth * sizeof(float);

    prepare(&h_ptrImage, originalImage);

    // This can be swapped to cudaMallocManaged to simplify the memory management 
    // but right now it's done this way for the educational purposes.
    checkCudaErrors(cudaMalloc(&d_redChannel, CHANNEL_BYTES));
    checkCudaErrors(cudaMalloc(&d_greenChannel, CHANNEL_BYTES));
    checkCudaErrors(cudaMalloc(&d_blueChannel, CHANNEL_BYTES));    

    checkCudaErrors(cudaMalloc(&d_ptrImage, IMAGE_BYTES));
    checkCudaErrors(cudaMalloc(&d_filter, FILTER_BYTES));

    // Copy the input data from host to device
    checkCudaErrors(cudaMemcpy(d_ptrImage, h_ptrImage, IMAGE_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, FILTER_BYTES, cudaMemcpyHostToDevice));

    const dim3 blockSize(32, 32, 1);
  
    // Now size of the grid 
    int numGrid_x = ceil((float)(rows)/32);
    int numGrid_y = ceil((float)(cols)/32);

    std::cout << "Grid size: " << numGrid_x << ":" << numGrid_y << std::endl;

    const dim3 gridSize(numGrid_x, numGrid_y, 1);

    // Call splitting channels kernel
    splitChannels<<<gridSize, blockSize>>>(d_ptrImage, d_redChannel, d_greenChannel, d_blueChannel, cols);

    // Wait for completion of all threads.
    checkCudaErrors(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.Start();

    int apron = kernelWidth/2;
    // Call the kernel
    blurringKernelBetter<<<gridSize, blockSize>>>(d_redChannel, d_greenChannel, d_blueChannel, d_filter, rows, cols, kernelWidth, apron);

    // Wait for completion of all threads.
    checkCudaErrors(cudaDeviceSynchronize());

    timer.Stop();

    std::cout << "Cuda code ran in: " << timer.Elapsed() << " msecs." << std::endl;

    // Copy the output data from device to host
    checkCudaErrors(cudaMemcpy(h_redChannel, d_redChannel, CHANNEL_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_greenChannel, d_greenChannel, CHANNEL_BYTES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_blueChannel, d_blueChannel, CHANNEL_BYTES, cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_ptrImage));
    checkCudaErrors(cudaFree(d_filter));
    checkCudaErrors(cudaFree(d_redChannel));
    checkCudaErrors(cudaFree(d_greenChannel));
    checkCudaErrors(cudaFree(d_blueChannel));

    mergeImages(blurredImage, h_redChannel, h_greenChannel, h_blueChannel, rows, cols);
}

void prepare(unsigned char **h_ptrImage,
            cv::Mat& originalImage) {
    *h_ptrImage = originalImage.ptr<unsigned char>(0);
}

void mergeImages(cv::Mat& blurredImage, unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel,
                    size_t rows, size_t cols) {
    cv::Mat r, g, b, fin;
    std::vector<cv::Mat> channels;

    fin = cv::Mat::zeros(cv::Size(cols,rows), CV_8UC3);

    r = cv::Mat(rows, cols, CV_8UC1, (void*) redChannel);
    g = cv::Mat(rows, cols, CV_8UC1, (void*) greenChannel);
    b = cv::Mat(rows, cols, CV_8UC1, (void*) blueChannel);

    // cv::imshow("red", r);
    // cv::waitKey(0);
    // cv::imshow("green", g);
    // cv::waitKey(0);
    // cv::imshow("blue", b);

    channels.push_back(b);
    channels.push_back(g);
    channels.push_back(r);

    blurredImage = fin;
    fin.release();
}