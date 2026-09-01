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
  std::size_t elements = 1u << 24;
  int streams = 4;
  int iterations = 10;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--elements" && i + 1 < argc) {
      options.elements = std::stoull(argv[++i]);
    } else if (arg == "--streams" && i + 1 < argc) {
      options.streams = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_streams_pinned [--elements N] [--streams N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.elements == 0 || options.streams <= 0 || options.iterations <= 0) {
    throw std::invalid_argument("elements, streams and iterations must be positive");
  }
  return options;
}

__global__ void ScaleKernel(float* data, std::size_t n, float scale) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] *= scale;
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
    constexpr float kScale = 1.25f;
    constexpr int kThreads = 256;

    std::vector<float> pageable_in(options.elements);
    std::vector<float> pageable_out(options.elements);
    for (std::size_t i = 0; i < options.elements; ++i) {
      pageable_in[i] = static_cast<float>(i % 4096) * 0.001f;
    }

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), bytes));
    const int blocks = static_cast<int>((options.elements + kThreads - 1) / kThreads);

    auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      CUDA_CHECK(cudaMemcpy(d_data, pageable_in.data(), bytes, cudaMemcpyHostToDevice));
      ScaleKernel<<<blocks, kThreads>>>(d_data, options.elements, kScale);
      CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaMemcpy(pageable_out.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double pageable_ms = MillisecondsSince(start) / options.iterations;

    float* pinned_in = nullptr;
    float* pinned_out = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_in), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_out), bytes));
    std::copy(pageable_in.begin(), pageable_in.end(), pinned_in);

    std::vector<cudaStream_t> streams(options.streams);
    for (auto& stream : streams) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    const std::size_t chunk = (options.elements + options.streams - 1) / options.streams;
    start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      for (int s = 0; s < options.streams; ++s) {
        const std::size_t offset = static_cast<std::size_t>(s) * chunk;
        if (offset >= options.elements) {
          continue;
        }
        const std::size_t count = std::min(chunk, options.elements - offset);
        const std::size_t chunk_bytes = count * sizeof(float);
        const int chunk_blocks = static_cast<int>((count + kThreads - 1) / kThreads);
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, pinned_in + offset, chunk_bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        ScaleKernel<<<chunk_blocks, kThreads, 0, streams[s]>>>(d_data + offset, count, kScale);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpyAsync(pinned_out + offset, d_data + offset, chunk_bytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double overlapped_ms = MillisecondsSince(start) / options.iterations;

    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < options.elements; ++i) {
      const double expected = static_cast<double>(pageable_in[i]) * kScale;
      max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(pinned_out[i]) - expected));
    }

    const double transfer_bytes = 2.0 * static_cast<double>(bytes);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Streams + pinned memory\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  Streams: " << options.streams << '\n';
    std::cout << "  Pageable sequential end-to-end: " << pageable_ms << " ms\n";
    std::cout << "  Pinned async multi-stream end-to-end: " << overlapped_ms << " ms\n";
    std::cout << "  End-to-end speedup: " << pageable_ms / overlapped_ms << "x\n";
    std::cout << "  Effective transfer+compute throughput: "
              << transfer_bytes / (overlapped_ms / 1000.0) / 1.0e9 << " GB/s\n";
    std::cout << "  Max abs error: " << max_abs_error << '\n';

    for (auto stream : streams) {
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
    CUDA_CHECK(cudaFreeHost(pinned_in));
    CUDA_CHECK(cudaFreeHost(pinned_out));
    CUDA_CHECK(cudaFree(d_data));

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
