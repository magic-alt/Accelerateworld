#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

#include "accelerateworld/cuda_check.hpp"

int main() {
  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  std::cout << "CUDA driver version: " << driver_version << '\n';
  std::cout << "CUDA runtime version: " << runtime_version << '\n';
  std::cout << "CUDA device count: " << device_count << '\n';

  if (device_count == 0) {
    std::cerr << "No CUDA device is visible to the runtime.\n";
    return 2;
  }

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const double global_memory_gib =
        static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "\nDevice " << device << ": " << prop.name << '\n';
    std::cout << "  Compute capability: " << prop.major << '.' << prop.minor << '\n';
    std::cout << "  Global memory: " << std::fixed << std::setprecision(2) << global_memory_gib
              << " GiB\n";
    std::cout << "  SM count: " << prop.multiProcessorCount << '\n';
    std::cout << "  Warp size: " << prop.warpSize << '\n';
    std::cout << "  Max threads/block: " << prop.maxThreadsPerBlock << '\n';
    std::cout << "  Max threads/SM: " << prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "  Shared memory/block: " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits\n";
  }

  return 0;
}
