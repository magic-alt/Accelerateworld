#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
  std::size_t elements = 1u << 16;
  int iterations = 1000;
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
      std::cout << "Usage: aw_cuda_graph [--elements N] [--iterations N]\n";
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

__global__ void AffineKernel(const float* input, float* temp, std::size_t n, float alpha, float beta) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    temp[index] = alpha * input[index] + beta;
  }
}

__global__ void ReluKernel(const float* temp, float* output, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = temp[index] > 0.0f ? temp[index] : 0.0f;
  }
}

double MillisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::size_t bytes = options.elements * sizeof(float);
    constexpr float kAlpha = 1.1f;
    constexpr float kBeta = -0.2f;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((options.elements + kThreads - 1) / kThreads);

    std::vector<float> input(options.elements);
    std::vector<float> output(options.elements);
    for (std::size_t i = 0; i < options.elements; ++i) {
      input[i] = static_cast<float>(static_cast<int>(i % 1024) - 512) * 0.001f;
    }

    float* d_input = nullptr;
    float* d_temp = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iterations; ++i) {
      AffineKernel<<<blocks, kThreads, 0, stream>>>(d_input, d_temp, options.elements, kAlpha, kBeta);
      ReluKernel<<<blocks, kThreads, 0, stream>>>(d_temp, d_output, options.elements);
    }
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double direct_ms = MillisecondsSince(start) / options.iterations;

    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    AffineKernel<<<blocks, kThreads, 0, stream>>>(d_input, d_temp, options.elements, kAlpha, kBeta);
    ReluKernel<<<blocks, kThreads, 0, stream>>>(d_temp, d_output, options.elements);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iterations; ++i) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double graph_ms = MillisecondsSince(start) / options.iterations;

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < options.elements; ++i) {
      const double affine = kAlpha * static_cast<double>(input[i]) + kBeta;
      const double expected = std::max(0.0, affine);
      max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(output[i]) - expected));
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CUDA Graph launch overhead\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  Iterations: " << options.iterations << '\n';
    std::cout << "  Direct two-kernel sequence: " << direct_ms * 1000.0 << " us/iteration\n";
    std::cout << "  CUDA Graph sequence: " << graph_ms * 1000.0 << " us/iteration\n";
    std::cout << "  Speedup: " << direct_ms / graph_ms << "x\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_temp));
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
