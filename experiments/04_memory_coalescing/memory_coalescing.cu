#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {

struct Options {
  std::size_t elements = 1u << 20;
  int stride = 32;
  int iterations = 30;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--elements" && i + 1 < argc) {
      options.elements = std::stoull(argv[++i]);
    } else if (arg == "--stride" && i + 1 < argc) {
      options.stride = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_memory_coalescing [--elements N] [--stride N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.elements == 0 || options.stride <= 0 || options.iterations <= 0) {
    throw std::invalid_argument("elements, stride and iterations must be positive");
  }
  return options;
}

__global__ void InitKernel(float* data, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = static_cast<float>(index % 1024) * 0.001f;
  }
}

__global__ void CoalescedReadKernel(const float* input, float* output, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = input[index];
  }
}

__global__ void StridedReadKernel(const float* input, float* output, std::size_t n, int stride) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = input[index * static_cast<std::size_t>(stride)];
  }
}

template <typename Launch>
double TimeKernel(int iterations, Launch&& launch) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch();
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / iterations;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::size_t physical_elements = options.elements * static_cast<std::size_t>(options.stride);
    const std::size_t input_bytes = physical_elements * sizeof(float);
    const std::size_t output_bytes = options.elements * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), output_bytes));

    constexpr int kThreads = 256;
    const int init_blocks = static_cast<int>((physical_elements + kThreads - 1) / kThreads);
    const int blocks = static_cast<int>((options.elements + kThreads - 1) / kThreads);
    InitKernel<<<init_blocks, kThreads>>>(d_input, physical_elements);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    const double coalesced_ms = TimeKernel(options.iterations, [&] {
      CoalescedReadKernel<<<blocks, kThreads>>>(d_input, d_output, options.elements);
    });
    const double strided_ms = TimeKernel(options.iterations, [&] {
      StridedReadKernel<<<blocks, kThreads>>>(d_input, d_output, options.elements, options.stride);
    });

    std::vector<float> output(options.elements);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < options.elements; ++i) {
      const std::size_t source_index = i * static_cast<std::size_t>(options.stride);
      const double expected = static_cast<double>(source_index % 1024) * 0.001;
      max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(output[i]) - expected));
    }

    const double useful_bytes = 2.0 * static_cast<double>(output_bytes);
    const double coalesced_gbs = useful_bytes / (coalesced_ms / 1000.0) / 1.0e9;
    const double strided_gbs = useful_bytes / (strided_ms / 1000.0) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Global-memory coalescing\n";
    std::cout << "  Logical elements: " << options.elements << '\n';
    std::cout << "  Stride: " << options.stride << '\n';
    std::cout << "  Coalesced: " << coalesced_ms << " ms, " << coalesced_gbs << " useful GB/s\n";
    std::cout << "  Strided: " << strided_ms << " ms, " << strided_gbs << " useful GB/s\n";
    std::cout << "  Coalescing speedup: " << strided_ms / coalesced_ms << "x\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    if (max_abs_error > 1.0e-5) {
      std::cerr << "Validation failed.\n";
      return 3;
    }
    std::cout << "  Validation: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
