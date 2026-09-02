#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 5;
  int iterations = 30;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--m" && i + 1 < argc) {
      options.m = std::stoi(argv[++i]);
    } else if (arg == "--n" && i + 1 < argc) {
      options.n = std::stoi(argv[++i]);
    } else if (arg == "--k" && i + 1 < argc) {
      options.k = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_triton_cublas_fp16_gemm [--m M] [--n N] [--k K] "
                   "[--warmup N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.warmup < 0 ||
      options.iterations <= 0) {
    throw std::invalid_argument("matrix dimensions and iterations must be positive");
  }
  return options;
}

std::vector<__half> BuildMatrix(int rows, int cols, int mul_a, int mul_b) {
  std::vector<__half> values(static_cast<std::size_t>(rows) * cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int code = (row * mul_a + col * mul_b + 5) % 29 - 14;
      values[static_cast<std::size_t>(row) * cols + col] =
          __float2half_rn(static_cast<float>(code) * 0.015625f);
    }
  }
  return values;
}

void LaunchGemm(cublasHandle_t handle,
                const __half* a,
                const __half* b,
                __half* c,
                int m,
                int n,
                int k) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  // cuBLAS is column-major.  A row-major C = A * B is equivalent to
  // C^T = B^T * A^T in column-major storage without explicit transposes.
  CUBLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            b,
                            CUDA_R_16F,
                            n,
                            a,
                            CUDA_R_16F,
                            k,
                            &beta,
                            c,
                            CUDA_R_16F,
                            n,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

double MeasureMaxNormalizedError(const Options& options,
                                 const std::vector<__half>& a,
                                 const std::vector<__half>& b,
                                 const std::vector<__half>& c) {
  constexpr int kSamples = 64;
  double max_normalized = 0.0;
  for (int sample = 0; sample < kSamples; ++sample) {
    const int row = (sample * 37 + 3) % options.m;
    const int col = (sample * 53 + 7) % options.n;
    double sum = 0.0;
    for (int inner = 0; inner < options.k; ++inner) {
      const float av = __half2float(a[static_cast<std::size_t>(row) * options.k + inner]);
      const float bv = __half2float(b[static_cast<std::size_t>(inner) * options.n + col]);
      sum += static_cast<double>(av) * bv;
    }
    const float reference_fp16 = __half2float(__float2half_rn(static_cast<float>(sum)));
    const float got = __half2float(c[static_cast<std::size_t>(row) * options.n + col]);
    const double abs_error = std::abs(static_cast<double>(got) - reference_fp16);
    const double normalized = abs_error / (1.0 + std::abs(static_cast<double>(reference_fp16)));
    max_normalized = std::max(max_normalized, normalized);
  }
  return max_normalized;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::vector<__half> host_a = BuildMatrix(options.m, options.k, 17, 13);
    const std::vector<__half> host_b = BuildMatrix(options.k, options.n, 11, 7);
    std::vector<__half> host_c(static_cast<std::size_t>(options.m) * options.n);

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    __half* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), host_a.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), host_b.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), host_c.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), host_a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), host_b.size() * sizeof(__half), cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    for (int i = 0; i < options.warmup; ++i) {
      LaunchGemm(handle, d_a, d_b, d_c, options.m, options.n, options.k);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < options.iterations; ++i) {
      LaunchGemm(handle, d_a, d_b, d_c, options.m, options.n, options.k);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = static_cast<double>(elapsed_ms) / options.iterations;
    const double flops = 2.0 * static_cast<double>(options.m) * options.n * options.k;
    const double gflops = flops / (average_ms / 1000.0) / 1.0e9;

    CUDA_CHECK(cudaMemcpy(host_c.data(), d_c, host_c.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    const double max_normalized_error =
        MeasureMaxNormalizedError(options, host_a, host_b, host_c);
    const bool passed = max_normalized_error <= 0.02;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Direct cuBLAS FP16 GEMM\n";
    std::cout << "GPU: " << properties.name << '\n';
    std::cout << "Compute capability: " << properties.major << '.' << properties.minor << '\n';
    std::cout << "Shape: " << options.m << " x " << options.n << " x " << options.k << '\n';
    std::cout << "Input dtype: FP16\n";
    std::cout << "Output dtype: FP16\n";
    std::cout << "Accumulation: FP32\n";
    std::cout << "Latency: " << average_ms << " ms\n";
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "Max normalized error: " << max_normalized_error << '\n';
    std::cout << "Validation: " << (passed ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
