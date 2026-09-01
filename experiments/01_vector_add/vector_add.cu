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
  std::size_t elements = 1u << 24;
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
      std::cout << "Usage: aw_vector_add [--elements N] [--iterations N]\n";
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

__global__ void VectorAddKernel(const float* a, const float* b, float* c, std::size_t n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::size_t bytes = options.elements * sizeof(float);

    std::vector<float> h_a(options.elements);
    std::vector<float> h_b(options.elements);
    std::vector<float> h_c(options.elements);

    for (std::size_t i = 0; i < options.elements; ++i) {
      h_a[i] = static_cast<float>((i % 1024) * 0.001);
      h_b[i] = static_cast<float>((i % 257) * 0.002);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((options.elements + kThreads - 1) / kThreads);

    VectorAddKernel<<<blocks, kThreads>>>(d_a, d_b, d_c, options.elements);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      VectorAddKernel<<<blocks, kThreads>>>(d_a, d_b, d_c, options.elements);
    }
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = elapsed_ms / options.iterations;

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < options.elements; ++i) {
      const double expected = static_cast<double>(h_a[i]) + static_cast<double>(h_b[i]);
      max_abs_error =
          std::max(max_abs_error, std::abs(static_cast<double>(h_c[i]) - expected));
    }

    const double seconds = average_ms / 1000.0;
    const double transferred_bytes = 3.0 * static_cast<double>(bytes);
    const double effective_bandwidth_gbs = transferred_bytes / seconds / 1.0e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Vector add\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  Iterations: " << options.iterations << '\n';
    std::cout << "  Average kernel time: " << average_ms << " ms\n";
    std::cout << "  Effective bandwidth: " << effective_bandwidth_gbs << " GB/s\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

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
