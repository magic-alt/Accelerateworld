#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct CandidateResult {
  std::string id;
  std::string threadblock;
  std::string warp;
  std::string instruction;
  int stages = 0;
  bool valid = false;
  double average_ms = std::numeric_limits<double>::infinity();
  double gflops = 0.0;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--m" && i + 1 < argc) options.m = std::stoi(argv[++i]);
    else if (arg == "--n" && i + 1 < argc) options.n = std::stoi(argv[++i]);
    else if (arg == "--k" && i + 1 < argc) options.k = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc) options.warmup = std::stoi(argv[++i]);
    else if (arg == "--iterations" && i + 1 < argc) options.iterations = std::stoi(argv[++i]);
    else if (arg == "--help") {
      std::cout << "Usage: aw_cutlass_gemm_sm75 [--m M] [--n N] [--k K] "
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
  if (options.m % 64 != 0 || options.n % 64 != 0 || options.k % 32 != 0) {
    throw std::invalid_argument("SM75 tile lab requires M/N multiples of 64 and K multiple of 32");
  }
  return options;
}

std::vector<cutlass::half_t> BuildA(const Options& options, std::vector<float>* reconstructed) {
  const std::size_t count = static_cast<std::size_t>(options.m) * options.k;
  std::vector<cutlass::half_t> values(count);
  reconstructed->resize(count);
  for (int row = 0; row < options.m; ++row) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (row * 17 + inner * 13 + 3) % 31 - 15;
      const float source = static_cast<float>(code) * 0.03125f;
      const std::size_t index = static_cast<std::size_t>(row) * options.k + inner;
      values[index] = cutlass::half_t(source);
      (*reconstructed)[index] = static_cast<float>(values[index]);
    }
  }
  return values;
}

std::vector<cutlass::half_t> BuildB(const Options& options, std::vector<float>* reconstructed) {
  const std::size_t count = static_cast<std::size_t>(options.k) * options.n;
  std::vector<cutlass::half_t> values(count);
  reconstructed->resize(count);
  for (int col = 0; col < options.n; ++col) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (inner * 11 + col * 7 + 5) % 29 - 14;
      const float source = static_cast<float>(code) * 0.03515625f;
      const std::size_t index = static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k;
      values[index] = cutlass::half_t(source);
      (*reconstructed)[index] = static_cast<float>(values[index]);
    }
  }
  return values;
}

template <typename Gemm>
CandidateResult BenchmarkCandidate(const std::string& id,
                                   const std::string& threadblock,
                                   const std::string& warp,
                                   const std::string& instruction,
                                   int stages,
                                   const Options& options,
                                   const cutlass::half_t* d_a,
                                   const cutlass::half_t* d_b,
                                   float* d_c,
                                   cudaStream_t stream) {
  CandidateResult result{id, threadblock, warp, instruction, stages};
  typename Gemm::Arguments args{
      {options.m, options.n, options.k},
      {d_a, options.k},
      {d_b, options.k},
      {d_c, options.m},
      {d_c, options.m},
      {1.0f, 0.0f}};

  if (Gemm::can_implement(args) != cutlass::Status::kSuccess) return result;
  const std::size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

  Gemm gemm;
  for (int i = 0; i < options.warmup; ++i) {
    if (gemm(args, workspace, stream) != cutlass::Status::kSuccess) {
      if (workspace) CUDA_CHECK(cudaFree(workspace));
      return result;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < options.iterations; ++i) {
    if (gemm(args, workspace, stream) != cutlass::Status::kSuccess) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      if (workspace) CUDA_CHECK(cudaFree(workspace));
      return result;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  if (workspace) CUDA_CHECK(cudaFree(workspace));

  result.average_ms = static_cast<double>(elapsed_ms) / options.iterations;
  const double flops = 2.0 * static_cast<double>(options.m) * options.n * options.k;
  result.gflops = flops / (result.average_ms / 1000.0) / 1.0e9;
  result.valid = std::isfinite(result.average_ms) && result.average_ms > 0.0;
  return result;
}

double ValidateSamples(const Options& options,
                       const std::vector<float>& a,
                       const std::vector<float>& b,
                       const std::vector<float>& output) {
  constexpr int kSamples = 32;
  double max_norm = 0.0;
  for (int sample = 0; sample < kSamples; ++sample) {
    const int row = (sample * 97 + 11) % options.m;
    const int col = (sample * 53 + 7) % options.n;
    double reference = 0.0;
    for (int inner = 0; inner < options.k; ++inner) {
      reference += static_cast<double>(a[static_cast<std::size_t>(row) * options.k + inner]) *
                   static_cast<double>(b[static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k]);
    }
    const double actual = output[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * options.m];
    const double error = std::abs(actual - reference) / std::max(1.0, std::abs(reference));
    max_norm = std::max(max_norm, error);
  }
  return max_norm;
}

void PrintCandidate(const CandidateResult& result) {
  std::cout << result.id << " threadblock shape: " << result.threadblock << "\n";
  std::cout << result.id << " warp shape: " << result.warp << "\n";
  std::cout << result.id << " instruction shape: " << result.instruction << "\n";
  std::cout << result.id << " pipeline stages: " << result.stages << " stages\n";
  if (!result.valid) {
    std::cout << result.id << ": SKIPPED\n";
    return;
  }
  std::cout << result.id << " latency: " << result.average_ms << " ms\n";
  std::cout << result.id << " throughput: " << result.gflops << " GFLOP/s\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major * 10 + props.minor < 75) {
      std::cout << "Validation: PASS (SM75 TensorOp unsupported on this GPU)\n";
      return EXIT_SUCCESS;
    }

    std::vector<float> reconstructed_a;
    std::vector<float> reconstructed_b;
    const auto host_a = BuildA(options, &reconstructed_a);
    const auto host_b = BuildB(options, &reconstructed_b);
    const std::size_t c_count = static_cast<std::size_t>(options.m) * options.n;

    cutlass::half_t* d_a = nullptr;
    cutlass::half_t* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, host_a.size() * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_b, host_b.size() * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), host_a.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), host_b.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, c_count * sizeof(float)));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    using Config = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::half_t, cutlass::half_t, float, float>;
    using Instruction = typename Config::InstructionShape;
    using Epilogue = typename Config::EpilogueOutputOp;
    using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using Square = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        Instruction, Epilogue, Swizzle, Config::kStages,
        Config::kAlignmentA, Config::kAlignmentB>;
    using WideN = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        Instruction, Epilogue, Swizzle, Config::kStages,
        Config::kAlignmentA, Config::kAlignmentB>;
    using WideM = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        float, cutlass::layout::ColumnMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<128, 64, 32>,
        cutlass::gemm::GemmShape<64, 32, 32>,
        Instruction, Epilogue, Swizzle, Config::kStages,
        Config::kAlignmentA, Config::kAlignmentB>;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUTLASS version: 4.7.0\n";
    std::cout << "CUTLASS API: device::Gemm explicit SM75 TensorOp configuration\n";
    std::cout << "GPU compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "M: " << options.m << "\nN: " << options.n << "\nK: " << options.k << "\n";

    std::vector<CandidateResult> results;
    results.push_back(BenchmarkCandidate<Square>("CUTLASS square", "128x128x32", "64x64x32", "16x8x8", Config::kStages, options, d_a, d_b, d_c, stream));
    results.push_back(BenchmarkCandidate<WideN>("CUTLASS wide N", "64x128x32", "32x64x32", "16x8x8", Config::kStages, options, d_a, d_b, d_c, stream));
    results.push_back(BenchmarkCandidate<WideM>("CUTLASS wide M", "128x64x32", "64x32x32", "16x8x8", Config::kStages, options, d_a, d_b, d_c, stream));

    const CandidateResult* best = nullptr;
    for (const auto& result : results) {
      PrintCandidate(result);
      if (result.valid && (!best || result.average_ms < best->average_ms)) best = &result;
    }
    if (!best) throw std::runtime_error("no executable CUTLASS SM75 candidate");

    if (best->id == "CUTLASS square") {
      (void)BenchmarkCandidate<Square>("validation", "", "", "", Config::kStages, Options{options.m, options.n, options.k, 0, 1}, d_a, d_b, d_c, stream);
    } else if (best->id == "CUTLASS wide N") {
      (void)BenchmarkCandidate<WideN>("validation", "", "", "", Config::kStages, Options{options.m, options.n, options.k, 0, 1}, d_a, d_b, d_c, stream);
    } else {
      (void)BenchmarkCandidate<WideM>("validation", "", "", "", Config::kStages, Options{options.m, options.n, options.k, 0, 1}, d_a, d_b, d_c, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> host_output(c_count);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    const double max_norm_error = ValidateSamples(options, reconstructed_a, reconstructed_b, host_output);
    const bool passed = std::isfinite(max_norm_error) && max_norm_error <= 5.0e-3;

    std::cout << "Best CUTLASS config: " << best->id << "\n";
    std::cout << "Best CUTLASS latency: " << best->average_ms << " ms\n";
    std::cout << "Best CUTLASS throughput: " << best->gflops << " GFLOP/s\n";
    std::cout << "Best CUTLASS validation max normalized error: " << max_norm_error << "\n";
    std::cout << "Validation: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
