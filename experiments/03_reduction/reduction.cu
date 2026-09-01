#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "accelerateworld/cuda_check.hpp"

namespace {

struct Options {
  std::size_t elements = 1u << 23;
  int iterations = 20;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--elements" && i + 1 < argc) {
      options.elements = std::stoull(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_reduction [--elements N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.elements == 0 || options.iterations <= 0) {
    throw std::invalid_argument("elements and iterations must be positive");
  }
  return options;
}

__global__ void FillKernel(float* data, std::size_t n, float value) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = value;
  }
}

__global__ void AtomicReductionKernel(const float* input, float* output, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    atomicAdd(output, input[index]);
  }
}

template <int kBlockSize>
__global__ void SharedReductionKernel(const float* input, float* output, std::size_t n) {
  __shared__ float partial[kBlockSize];
  const unsigned int tid = threadIdx.x;
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0f;
  while (index < n) {
    local_sum += input[index];
    index += stride;
  }
  partial[tid] = local_sum;
  __syncthreads();

  for (unsigned int offset = kBlockSize / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partial[tid] += partial[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, partial[0]);
  }
}

template <typename Launch>
double TimeReduction(float* d_output, int iterations, Launch&& launch) {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(float)));
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
    const std::size_t bytes = options.elements * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(float)));

    constexpr int kThreads = 256;
    const int fill_blocks = static_cast<int>((options.elements + kThreads - 1) / kThreads);
    const int reduction_blocks = std::min(fill_blocks, 4096);
    FillKernel<<<fill_blocks, kThreads>>>(d_input, options.elements, 1.0f);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    const double atomic_ms = TimeReduction(d_output, options.iterations, [&] {
      AtomicReductionKernel<<<fill_blocks, kThreads>>>(d_input, d_output, options.elements);
    });
    const double shared_ms = TimeReduction(d_output, options.iterations, [&] {
      SharedReductionKernel<kThreads><<<reduction_blocks, kThreads>>>(d_input, d_output,
                                                                      options.elements);
    });

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    const double expected = static_cast<double>(options.elements);
    const double abs_error = std::abs(static_cast<double>(result) - expected);
    const double tolerance = std::max(1.0, expected * 1.0e-6);

    const double useful_bytes = static_cast<double>(bytes);
    const double shared_bandwidth_gbs = useful_bytes / (shared_ms / 1000.0) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reduction\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  Atomic per-element: " << atomic_ms << " ms\n";
    std::cout << "  Shared block reduction: " << shared_ms << " ms\n";
    std::cout << "  Speedup: " << atomic_ms / shared_ms << "x\n";
    std::cout << "  Shared useful bandwidth: " << shared_bandwidth_gbs << " GB/s\n";
    std::cout << "  Result: " << result << " (expected " << expected << ")\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    if (abs_error > tolerance) {
      std::cerr << "Validation failed: abs error = " << abs_error << '\n';
      return 3;
    }
    std::cout << "  Validation: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
