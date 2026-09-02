#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {
struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 2;
  int iterations = 10;
};

Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--m" && i + 1 < argc) o.m = std::stoi(argv[++i]);
    else if (a == "--n" && i + 1 < argc) o.n = std::stoi(argv[++i]);
    else if (a == "--k" && i + 1 < argc) o.k = std::stoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) o.warmup = std::stoi(argv[++i]);
    else if (a == "--iterations" && i + 1 < argc) o.iterations = std::stoi(argv[++i]);
    else throw std::invalid_argument("unknown or incomplete argument: " + a);
  }
  if (o.m <= 0 || o.n <= 0 || o.k <= 0 || o.warmup < 0 || o.iterations <= 0) {
    throw std::invalid_argument("invalid shape or iteration count");
  }
  return o;
}

std::vector<float> BuildA(const Options& o) {
  std::vector<float> x(static_cast<std::size_t>(o.m) * o.k);
  for (int m = 0; m < o.m; ++m)
    for (int k = 0; k < o.k; ++k)
      x[static_cast<std::size_t>(m) * o.k + k] = float((m * 17 + k * 13 + 3) % 17 - 8) * 0.0625f;
  return x;
}

std::vector<float> BuildB(const Options& o) {
  std::vector<float> x(static_cast<std::size_t>(o.k) * o.n);
  for (int n = 0; n < o.n; ++n)
    for (int k = 0; k < o.k; ++k)
      x[static_cast<std::size_t>(k) + static_cast<std::size_t>(n) * o.k] =
          float((k * 11 + n * 7 + 5) % 15 - 7) * 0.0625f;
  return x;
}

double Validate(const Options& o, const std::vector<float>& a, const std::vector<float>& b,
                const std::vector<float>& c) {
  double worst = 0.0;
  for (int s = 0; s < 16; ++s) {
    const int row = (s * 97 + 11) % o.m;
    const int col = (s * 53 + 7) % o.n;
    double ref = 0.0;
    for (int k = 0; k < o.k; ++k)
      ref += double(a[static_cast<std::size_t>(row) * o.k + k]) *
             double(b[static_cast<std::size_t>(k) + static_cast<std::size_t>(col) * o.k]);
    const double got = c[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * o.m];
    worst = std::max(worst, std::abs(got - ref) / std::max(1.0, std::abs(ref)));
  }
  return worst;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = Parse(argc, argv);
    const auto h_a = BuildA(o);
    const auto h_b = BuildB(o);
    const std::size_t c_count = static_cast<std::size_t>(o.m) * o.n;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, c_count * sizeof(float)));

    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75>;
    typename Gemm::Arguments args{{o.m, o.n, o.k}, {d_a, o.k}, {d_b, o.k},
                                  {d_c, o.m}, {d_c, o.m}, {1.0f, 0.0f}};
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
      throw std::runtime_error("CUTLASS SIMT cannot implement shape");
    const std::size_t workspace_bytes = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_bytes) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    Gemm gemm;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int i = 0; i < o.warmup; ++i)
      if (gemm(args, workspace, stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS SIMT warmup failed");
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t begin{}, end{};
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin, stream));
    for (int i = 0; i < o.iterations; ++i)
      if (gemm(args, workspace, stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("CUTLASS SIMT timed launch failed");
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    const double ms = double(elapsed) / o.iterations;
    const double gflops = (2.0 * double(o.m) * o.n * o.k) / (ms / 1000.0) / 1.0e9;

    if (gemm(args, workspace, stream) != cutlass::Status::kSuccess)
      throw std::runtime_error("CUTLASS SIMT validation launch failed");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> h_c(c_count);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    const double error = Validate(o, h_a, h_b, h_c);
    const bool passed = std::isfinite(error) && error <= 2.0e-4;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUTLASS version: 4.7.0\n";
    std::cout << "CUTLASS operator class: SIMT\n";
    std::cout << "CUTLASS SIMT latency: " << ms << " ms\n";
    std::cout << "CUTLASS SIMT throughput: " << gflops << " GFLOP/s\n";
    std::cout << "CUTLASS SIMT validation max normalized error: " << error << "\n";
    std::cout << "Validation: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
