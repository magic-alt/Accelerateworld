#include <cublas_v2.h>
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

#define CUBLAS_CHECK(call)                                                                    \
  do {                                                                                        \
    const cublasStatus_t status__ = (call);                                                    \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                                   \
      throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" +           \
                               std::to_string(__LINE__) + " status=" +                       \
                               std::to_string(static_cast<int>(status__)));                    \
    }                                                                                         \
  } while (false)

struct Options {
  int size = 1024;
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
      std::cout << "Usage: aw_cublas_gemm [--size N] [--iterations N]\n";
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

__global__ void FillKernel(float* data, std::size_t n, float value) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = value;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::size_t elements = static_cast<std::size_t>(options.size) * options.size;
    const std::size_t bytes = elements * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    FillKernel<<<blocks, kThreads>>>(d_a, elements, 0.5f);
    FillKernel<<<blocks, kThreads>>>(d_b, elements, 0.25f);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, options.size, options.size,
                            options.size, &alpha, d_a, options.size, d_b, options.size, &beta,
                            d_c, options.size));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < options.iterations; ++i) {
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, options.size, options.size,
                              options.size, &alpha, d_a, options.size, d_b, options.size, &beta,
                              d_c, options.size));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = static_cast<double>(elapsed_ms) / options.iterations;

    std::vector<float> output(elements);
    CUDA_CHECK(cudaMemcpy(output.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    const double expected = static_cast<double>(options.size) * 0.125;
    double max_abs_error = 0.0;
    for (float value : output) {
      max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(value) - expected));
    }
    const double flops = 2.0 * options.size * options.size * static_cast<double>(options.size);
    const double gflops = flops / (average_ms / 1000.0) / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "cuBLAS SGEMM baseline\n";
    std::cout << "  Size: " << options.size << " x " << options.size << '\n';
    std::cout << "  Average time: " << average_ms << " ms\n";
    std::cout << "  Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (max_abs_error > 1.0e-3) {
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
