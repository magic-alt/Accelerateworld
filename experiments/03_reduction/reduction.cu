#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "accelerateworld/cuda_check.hpp"

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

struct Options {
  std::size_t elements = 1u << 23;
  int iterations = 20;
};

struct Measurement {
  double milliseconds = 0.0;
  float result = 0.0f;
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

__device__ __forceinline__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

template <int kBlockSize>
__global__ void WarpShuffleReductionKernel(const float* input, float* output, std::size_t n) {
  static_assert(kBlockSize % kWarpSize == 0, "block size must contain complete warps");
  constexpr int kWarpCount = kBlockSize / kWarpSize;
  __shared__ float warp_sums[kWarpCount];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0f;
  while (index < n) {
    local_sum += input[index];
    index += stride;
  }

  local_sum = WarpReduceSum(local_sum);
  if (lane == 0) {
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < kWarpCount ? warp_sums[lane] : 0.0f;
    block_sum = WarpReduceSum(block_sum);
    if (lane == 0) {
      atomicAdd(output, block_sum);
    }
  }
}

template <typename Launch>
Measurement MeasureReduction(float* d_output, int iterations, Launch&& launch) {
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
  launch();
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());

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

  Measurement measurement;
  measurement.milliseconds = static_cast<double>(elapsed_ms) / iterations;
  CUDA_CHECK(cudaMemcpy(&measurement.result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return measurement;
}

double AbsError(float result, double expected) {
  return std::abs(static_cast<double>(result) - expected);
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

    const Measurement atomic = MeasureReduction(d_output, options.iterations, [&] {
      AtomicReductionKernel<<<fill_blocks, kThreads>>>(d_input, d_output, options.elements);
    });
    const Measurement shared = MeasureReduction(d_output, options.iterations, [&] {
      SharedReductionKernel<kThreads><<<reduction_blocks, kThreads>>>(d_input, d_output,
                                                                      options.elements);
    });
    const Measurement warp = MeasureReduction(d_output, options.iterations, [&] {
      WarpShuffleReductionKernel<kThreads><<<reduction_blocks, kThreads>>>(d_input, d_output,
                                                                           options.elements);
    });

    const double expected = static_cast<double>(options.elements);
    const double tolerance = std::max(1.0, expected * 1.0e-6);
    const double atomic_error = AbsError(atomic.result, expected);
    const double shared_error = AbsError(shared.result, expected);
    const double warp_error = AbsError(warp.result, expected);
    const double max_error = std::max({atomic_error, shared_error, warp_error});

    const double useful_bytes = static_cast<double>(bytes);
    const double shared_bandwidth_gbs = useful_bytes / (shared.milliseconds / 1000.0) / 1.0e9;
    const double warp_bandwidth_gbs = useful_bytes / (warp.milliseconds / 1000.0) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Reduction\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  Atomic per-element: " << atomic.milliseconds << " ms\n";
    std::cout << "  Shared block reduction: " << shared.milliseconds << " ms\n";
    std::cout << "  Warp shuffle reduction: " << warp.milliseconds << " ms\n";
    std::cout << "  Shared speedup vs atomic: " << atomic.milliseconds / shared.milliseconds << " x\n";
    std::cout << "  Warp shuffle speedup vs shared: " << shared.milliseconds / warp.milliseconds << " x\n";
    std::cout << "  Shared useful bandwidth: " << shared_bandwidth_gbs << " GB/s\n";
    std::cout << "  Warp shuffle useful bandwidth: " << warp_bandwidth_gbs << " GB/s\n";
    std::cout << "  Atomic result: " << atomic.result << '\n';
    std::cout << "  Shared result: " << shared.result << '\n';
    std::cout << "  Warp shuffle result: " << warp.result << '\n';
    std::cout << "  Max abs error: " << max_error << '\n';

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    if (max_error > tolerance) {
      std::cerr << "Validation failed: max abs error = " << max_error << '\n';
      return 3;
    }
    std::cout << "  Validation: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
