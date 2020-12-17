#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer();

  ~GpuTimer();

  void Start();

  void Stop();

  float Elapsed();

};

#endif
