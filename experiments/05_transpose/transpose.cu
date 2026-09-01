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
constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

struct Options {
  int size = 2048;
  int iterations = 30;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--size" && i + 1 < argc) {
      options.size = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_transpose [--size N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.size <= 0 || options.iterations <= 0) {
    throw std::invalid_argument("size and iterations must be positive");
  }
  return options;
}

__global__ void NaiveTransposeKernel(const float* input, float* output, int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < n && y < n) {
    output[x * n + y] = input[y * n + x];
  }
}

__global__ void TiledTransposeKernel(const float* input, float* output, int n) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    if (x < n && y + offset < n) {
      tile[threadIdx.y + offset][threadIdx.x] = input[(y + offset) * n + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * kTileDim + threadIdx.x;
  y = blockIdx.x * kTileDim + threadIdx.y;
  for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
    if (x < n && y + offset < n) {
      output[(y + offset) * n + x] = tile[threadIdx.x][threadIdx.y + offset];
    }
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
    const std::size_t elements = static_cast<std::size_t>(options.size) * options.size;
    const std::size_t bytes = elements * sizeof(float);
    std::vector<float> input(elements);
    std::vector<float> output(elements);
    for (std::size_t i = 0; i < elements; ++i) {
      input[i] = static_cast<float>(i % 8192) * 0.001f;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 naive_block(32, 8);
    const dim3 naive_grid((options.size + naive_block.x - 1) / naive_block.x,
                          (options.size + naive_block.y - 1) / naive_block.y);
    const dim3 tiled_block(kTileDim, kBlockRows);
    const dim3 tiled_grid((options.size + kTileDim - 1) / kTileDim,
                          (options.size + kTileDim - 1) / kTileDim);

    const double naive_ms = TimeKernel(options.iterations, [&] {
      NaiveTransposeKernel<<<naive_grid, naive_block>>>(d_input, d_output, options.size);
    });
    const double tiled_ms = TimeKernel(options.iterations, [&] {
      TiledTransposeKernel<<<tiled_grid, tiled_block>>>(d_input, d_output, options.size);
    });

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    double max_abs_error = 0.0;
    for (int row = 0; row < options.size; ++row) {
      for (int col = 0; col < options.size; ++col) {
        const double expected = input[static_cast<std::size_t>(col) * options.size + row];
        const double actual = output[static_cast<std::size_t>(row) * options.size + col];
        max_abs_error = std::max(max_abs_error, std::abs(actual - expected));
      }
    }

    const double transferred = 2.0 * static_cast<double>(bytes);
    const double naive_gbs = transferred / (naive_ms / 1000.0) / 1.0e9;
    const double tiled_gbs = transferred / (tiled_ms / 1000.0) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Matrix transpose\n";
    std::cout << "  Size: " << options.size << " x " << options.size << '\n';
    std::cout << "  Naive: " << naive_ms << " ms, " << naive_gbs << " GB/s\n";
    std::cout << "  Shared-memory tiled+padding: " << tiled_ms << " ms, " << tiled_gbs << " GB/s\n";
    std::cout << "  Speedup: " << naive_ms / tiled_ms << "x\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    if (max_abs_error > 1.0e-6) {
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
