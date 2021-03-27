#ifndef CUDA_UTILS_H__
#define CUDA_UTILS_H__

void preProcess(float** d_luminance, unsigned int** d_cdf,
                size_t *numRows, size_t *numCols,
                unsigned int *numberOfBins,
                const std::string &filename);

void postProcess(const std::string& output_file, 
                 size_t numRows, size_t numCols,
                 float min_log_Y, float max_log_Y);


void cleanupGlobalMemory(void);

#endif