#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    const cudaError_t accelerateworld_cuda_error = (call);                          \
    if (accelerateworld_cuda_error != cudaSuccess) {                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "          \
                << cudaGetErrorString(accelerateworld_cuda_error) << " ("            \
                << static_cast<int>(accelerateworld_cuda_error) << ")" << std::endl; \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                               \
  } while (0)

#define CUDA_KERNEL_CHECK()        \
  do {                             \
    CUDA_CHECK(cudaGetLastError()); \
  } while (0)
