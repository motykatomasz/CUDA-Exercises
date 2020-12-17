#include <cmath>
#include <iostream>

#include "../utils/utils.hpp"
#include "../utils/timer.hpp"

#include "cuda_func.hpp"

__global__ void rgbToGreyscaleKernel(unsigned char* const greyImage, unsigned char* const rgbImage,
                                    int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < rows) && (y < cols)) {
        int indexGrey = x * cols + y;
        int startingIndexRgb = indexGrey * 3;

        greyImage[indexGrey] = .299f * rgbImage[startingIndexRgb + 2] + 
                               .587f * rgbImage[startingIndexRgb + 1] + 
                               .114f * rgbImage[startingIndexRgb + 0];
    }
}

void rgbToGreyscale( cv::Mat& rgbImage, cv::Mat& greyImage,
                    int rows, int cols) {
    unsigned char *h_ptrRgbImage, *d_ptrRgbImage, *h_ptrGreyImage, *d_ptrGreyImage;

    int INPUT_BYTES = rows * rgbImage.step;
    int OUTPUT_BYTES = rows * greyImage.step;

    prepare(&h_ptrRgbImage, &h_ptrGreyImage, rgbImage, greyImage);

    // This can be swapped to cudaMallocManaged to simplify the memory management 
    // but right now it's done this way for the educational purposes.
    cudaMalloc(&d_ptrRgbImage, INPUT_BYTES);
    cudaMalloc(&d_ptrGreyImage, OUTPUT_BYTES);

    // Copy the input data from host to device
    cudaMemcpy(d_ptrRgbImage, h_ptrRgbImage, INPUT_BYTES, cudaMemcpyHostToDevice);

    const dim3 blockSize(32, 32, 1);
  
    // Now size of the grid 
    int numGrid_x = ceil((float)(rows)/32);
    int numGrid_y = ceil((float)(cols)/32);

    std::cout << "Grid size: " << numGrid_x << ":" << numGrid_y << std::endl;

    const dim3 gridSize(numGrid_x, numGrid_y, 1);

    GpuTimer timer;
    timer.Start();

    // Call the kernel
    rgbToGreyscaleKernel<<<gridSize, blockSize>>>(d_ptrGreyImage, d_ptrRgbImage, rows, cols);

    // Wait for completion of all threads.
    cudaDeviceSynchronize();

    timer.Stop();

    std::cout << "Cuda code ran in: " << timer.Elapsed() << " msecs." << std::endl;

    // Copy the output data from device to host
    cudaMemcpy(h_ptrGreyImage, d_ptrGreyImage, OUTPUT_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_ptrRgbImage);
    cudaFree(d_ptrGreyImage);
    
}

void prepare(unsigned char **h_ptrRgbImage, unsigned char **h_ptrGreyImage, 
            cv::Mat& rgbImage, cv::Mat& greyImage) {
            
    *h_ptrRgbImage = rgbImage.ptr<unsigned char>(0);
    *h_ptrGreyImage = greyImage.ptr<unsigned char>(0);
}