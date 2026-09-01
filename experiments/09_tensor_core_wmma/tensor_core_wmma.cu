#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

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
namespace wmma = nvcuda::wmma;
constexpr int kTile = 16;

struct Options {
  int size = 1024;
  int iterations = 20;
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
      std::cout << "Usage: aw_tensor_core_wmma [--size N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.size <= 0 || options.size % kTile != 0 || options.iterations <= 0) {
    throw std::invalid_argument("size must be a positive multiple of 16; iterations must be positive");
  }
  return options;
}

__global__ void FillHalfKernel(half* data, std::size_t n, float value) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = __float2half(value);
  }
}

__global__ void WmmaGemmKernel(const half* a, const half* b, float* c, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;

  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  const int row = tile_row * kTile;
  const int col = tile_col * kTile;
  for (int k = 0; k < n; k += kTile) {
    wmma::load_matrix_sync(a_frag, a + row * n + k, n);
    wmma::load_matrix_sync(b_frag, b + k * n + col, n);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  wmma::store_matrix_sync(c + row * n + col, acc_frag, n, wmma::mem_row_major);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::size_t elements = static_cast<std::size_t>(options.size) * options.size;
    const std::size_t half_bytes = elements * sizeof(half);
    const std::size_t float_bytes = elements * sizeof(float);

    half* d_a = nullptr;
    half* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), half_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), half_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), float_bytes));

    constexpr int kThreads = 256;
    const int fill_blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    FillHalfKernel<<<fill_blocks, kThreads>>>(d_a, elements, 0.5f);
    FillHalfKernel<<<fill_blocks, kThreads>>>(d_b, elements, 0.25f);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 grid(options.size / kTile, options.size / kTile);
    const dim3 block(32, 1, 1);
    WmmaGemmKernel<<<grid, block>>>(d_a, d_b, d_c, options.size);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < options.iterations; ++i) {
      WmmaGemmKernel<<<grid, block>>>(d_a, d_b, d_c, options.size);
    }
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = static_cast<double>(elapsed_ms) / options.iterations;

    std::vector<float> output(elements);
    CUDA_CHECK(cudaMemcpy(output.data(), d_c, float_bytes, cudaMemcpyDeviceToHost));
    const double expected = static_cast<double>(options.size) * 0.125;
    double max_abs_error = 0.0;
    for (float value : output) {
      max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(value) - expected));
    }
    const double flops = 2.0 * options.size * options.size * static_cast<double>(options.size);
    const double gflops = flops / (average_ms / 1000.0) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Tensor Core WMMA GEMM (FP16 inputs, FP32 accumulate)\n";
    std::cout << "  Size: " << options.size << " x " << options.size << '\n';
    std::cout << "  Average time: " << average_ms << " ms\n";
    std::cout << "  Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (max_abs_error > 1.0e-2) {
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
