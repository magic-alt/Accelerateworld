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

constexpr int kTile = 16;

struct Options {
  int size = 512;
  int iterations = 10;
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
      std::cout << "Usage: aw_matmul [--size N] [--iterations N]\n";
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

__global__ void MatMulNaiveKernel(const float* a, const float* b, float* c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= n || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < n; ++k) {
    sum += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = sum;
}

__global__ void MatMulTiledKernel(const float* a, const float* b, float* c, int n) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float sum = 0.0f;
  const int tile_count = (n + kTile - 1) / kTile;

  for (int tile = 0; tile < tile_count; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k) {
      sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}

double TimeNaive(const float* a, const float* b, float* c, int n, int iterations) {
  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  MatMulNaiveKernel<<<grid, block>>>(a, b, c, n);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    MatMulNaiveKernel<<<grid, block>>>(a, b, c, n);
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

double TimeTiled(const float* a, const float* b, float* c, int n, int iterations) {
  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  MatMulTiledKernel<<<grid, block>>>(a, b, c, n);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    MatMulTiledKernel<<<grid, block>>>(a, b, c, n);
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

std::vector<float> CpuReference(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
  std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
      }
      c[row * n + col] = sum;
    }
  }
  return c;
}

double MaxAbsError(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  double max_error = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_error =
        std::max(max_error, std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
  }
  return max_error;
}

double Gflops(int n, double milliseconds) {
  const double operations = 2.0 * static_cast<double>(n) * n * n;
  return operations / (milliseconds * 1.0e6);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const int n = options.size;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> h_a(elements);
    std::vector<float> h_b(elements);
    std::vector<float> h_naive(elements);
    std::vector<float> h_tiled(elements);

    for (std::size_t i = 0; i < elements; ++i) {
      h_a[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.03125f;
      h_b[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.015625f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_naive = nullptr;
    float* d_tiled = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_naive), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tiled), bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const double naive_ms = TimeNaive(d_a, d_b, d_naive, n, options.iterations);
    const double tiled_ms = TimeTiled(d_a, d_b, d_tiled, n, options.iterations);

    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_tiled, bytes, cudaMemcpyDeviceToHost));

    double max_error = MaxAbsError(h_naive, h_tiled);
    std::string oracle = "naive GPU kernel";

    if (n <= 256) {
      const auto h_reference = CpuReference(h_a, h_b, n);
      max_error = std::max(MaxAbsError(h_reference, h_naive), MaxAbsError(h_reference, h_tiled));
      oracle = "CPU reference";
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Square matrix multiply\n";
    std::cout << "  N: " << n << '\n';
    std::cout << "  Iterations: " << options.iterations << '\n';
    std::cout << "  Naive: " << naive_ms << " ms, " << Gflops(n, naive_ms) << " GFLOP/s\n";
    std::cout << "  Tiled: " << tiled_ms << " ms, " << Gflops(n, tiled_ms) << " GFLOP/s\n";
    std::cout << "  Speedup (naive/tiled): " << naive_ms / tiled_ms << "x\n";
    std::cout << "  Correctness oracle: " << oracle << '\n';
    std::cout << "  Max abs error: " << max_error << '\n';

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_naive));
    CUDA_CHECK(cudaFree(d_tiled));

    if (max_error > 1.0e-2) {
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
