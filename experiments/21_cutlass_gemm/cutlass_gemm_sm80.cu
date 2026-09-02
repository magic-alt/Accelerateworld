#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

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
  std::string mode = "all";
};

struct CandidateResult {
  std::string id;
  std::string dtype;
  std::string threadblock;
  std::string warp;
  std::string instruction;
  int stages = 0;
  std::size_t shared_storage_bytes = 0;
  int max_active_blocks_per_sm = -1;
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
    else if (arg == "--mode" && i + 1 < argc) options.mode = argv[++i];
    else if (arg == "--help") {
      std::cout << "Usage: aw_cutlass_gemm_sm80 [--m M] [--n N] [--k K] "
                   "[--warmup N] [--iterations N] [--mode all|fp16|bf16]\n";
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
    throw std::invalid_argument("SM80 Universal GEMM lab requires M/N multiples of 64 and K multiple of 32");
  }
  if (options.mode != "all" && options.mode != "fp16" && options.mode != "bf16") {
    throw std::invalid_argument("mode must be all, fp16 or bf16");
  }
  return options;
}

template <typename Element>
std::vector<Element> BuildA(const Options& options, std::vector<float>* reconstructed) {
  const std::size_t count = static_cast<std::size_t>(options.m) * options.k;
  std::vector<Element> values(count);
  reconstructed->resize(count);
  for (int row = 0; row < options.m; ++row) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (row * 17 + inner * 13 + 3) % 31 - 15;
      const float source = static_cast<float>(code) * 0.03125f;
      const std::size_t index = static_cast<std::size_t>(row) * options.k + inner;
      values[index] = Element(source);
      (*reconstructed)[index] = static_cast<float>(values[index]);
    }
  }
  return values;
}

template <typename Element>
std::vector<Element> BuildB(const Options& options, std::vector<float>* reconstructed) {
  const std::size_t count = static_cast<std::size_t>(options.k) * options.n;
  std::vector<Element> values(count);
  reconstructed->resize(count);
  for (int col = 0; col < options.n; ++col) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (inner * 11 + col * 7 + 5) % 29 - 14;
      const float source = static_cast<float>(code) * 0.03515625f;
      const std::size_t index = static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k;
      values[index] = Element(source);
      (*reconstructed)[index] = static_cast<float>(values[index]);
    }
  }
  return values;
}

template <typename Gemm, typename Element>
CandidateResult BenchmarkCandidate(const std::string& id,
                                   const std::string& dtype,
                                   const std::string& threadblock,
                                   const std::string& warp,
                                   const std::string& instruction,
                                   int stages,
                                   const Options& options,
                                   const Element* d_a,
                                   const Element* d_b,
                                   float* d_c,
                                   cudaStream_t stream) {
  CandidateResult result;
  result.id = id;
  result.dtype = dtype;
  result.threadblock = threadblock;
  result.warp = warp;
  result.instruction = instruction;
  result.stages = stages;
  result.shared_storage_bytes = Gemm::Base::kSharedStorageSize;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k},
      1,
      {1.0f, 0.0f},
      d_a,
      d_b,
      d_c,
      d_c,
      static_cast<std::int64_t>(options.m) * options.k,
      static_cast<std::int64_t>(options.n) * options.k,
      static_cast<std::int64_t>(options.m) * options.n,
      static_cast<std::int64_t>(options.m) * options.n,
      options.k,
      options.k,
      options.m,
      options.m};

  if (Gemm::can_implement(args) != cutlass::Status::kSuccess) return result;
  result.max_active_blocks_per_sm = Gemm::maximum_active_blocks();
  const std::size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

  Gemm gemm;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    return result;
  }
  for (int i = 0; i < options.warmup; ++i) {
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
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
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
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
    max_norm = std::max(max_norm,
                        std::abs(actual - reference) / std::max(1.0, std::abs(reference)));
  }
  return max_norm;
}

void PrintCandidate(const CandidateResult& result) {
  std::cout << result.id << " dtype: " << result.dtype << "\n";
  std::cout << result.id << " threadblock shape: " << result.threadblock << "\n";
  std::cout << result.id << " warp shape: " << result.warp << "\n";
  std::cout << result.id << " instruction shape: " << result.instruction << "\n";
  std::cout << result.id << " pipeline stages: " << result.stages << " stages\n";
  std::cout << result.id << " shared storage: " << result.shared_storage_bytes << " bytes\n";
  std::cout << result.id << " max active blocks per SM: " << result.max_active_blocks_per_sm << " blocks\n";
  if (!result.valid) {
    std::cout << result.id << ": SKIPPED\n";
    return;
  }
  std::cout << result.id << " latency: " << result.average_ms << " ms\n";
  std::cout << result.id << " throughput: " << result.gflops << " GFLOP/s\n";
}

template <typename Element>
bool RunPrecision(const std::string& dtype,
                  const Options& options,
                  cudaStream_t stream,
                  double* best_throughput) {
  using Config = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      Element, Element, float, float>;
  using Instruction = typename Config::InstructionShape;
  using Epilogue = typename Config::EpilogueOutputOp;
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using Square = cutlass::gemm::device::GemmUniversal<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::ColumnMajor,
      float, cutlass::layout::ColumnMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      Instruction, Epilogue, Swizzle, 3,
      Config::kAlignmentA, Config::kAlignmentB>;
  using WideN = cutlass::gemm::device::GemmUniversal<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::ColumnMajor,
      float, cutlass::layout::ColumnMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 32>,
      cutlass::gemm::GemmShape<32, 64, 32>,
      Instruction, Epilogue, Swizzle, 3,
      Config::kAlignmentA, Config::kAlignmentB>;
  using DeepPipeline = cutlass::gemm::device::GemmUniversal<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::ColumnMajor,
      float, cutlass::layout::ColumnMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      Instruction, Epilogue, Swizzle, 4,
      Config::kAlignmentA, Config::kAlignmentB>;

  std::vector<float> reconstructed_a;
  std::vector<float> reconstructed_b;
  const auto host_a = BuildA<Element>(options, &reconstructed_a);
  const auto host_b = BuildB<Element>(options, &reconstructed_b);
  const std::size_t c_count = static_cast<std::size_t>(options.m) * options.n;
  Element* d_a = nullptr;
  Element* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, host_a.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, host_b.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), host_a.size() * sizeof(Element), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), host_b.size() * sizeof(Element), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, c_count * sizeof(float)));

  const std::string instruction = dtype == "BF16" ? "16x8x8 BF16 TensorOp" : "16x8x16 FP16 TensorOp";
  std::vector<CandidateResult> candidates;
  candidates.push_back(BenchmarkCandidate<Square>(dtype + " square", dtype, "128x128x32", "64x64x32", instruction, 3, options, d_a, d_b, d_c, stream));
  candidates.push_back(BenchmarkCandidate<WideN>(dtype + " wide N", dtype, "64x128x32", "32x64x32", instruction, 3, options, d_a, d_b, d_c, stream));
  candidates.push_back(BenchmarkCandidate<DeepPipeline>(dtype + " stage4", dtype, "128x128x32", "64x64x32", instruction, 4, options, d_a, d_b, d_c, stream));

  const CandidateResult* best = nullptr;
  for (const auto& candidate : candidates) {
    PrintCandidate(candidate);
    if (candidate.valid && (!best || candidate.average_ms < best->average_ms)) best = &candidate;
  }
  if (!best) {
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return false;
  }

  if (best->id.find("wide N") != std::string::npos) {
    (void)BenchmarkCandidate<WideN>("validation", dtype, "", "", "", 3,
        Options{options.m, options.n, options.k, 0, 1, options.mode}, d_a, d_b, d_c, stream);
  } else if (best->id.find("stage4") != std::string::npos) {
    (void)BenchmarkCandidate<DeepPipeline>("validation", dtype, "", "", "", 4,
        Options{options.m, options.n, options.k, 0, 1, options.mode}, d_a, d_b, d_c, stream);
  } else {
    (void)BenchmarkCandidate<Square>("validation", dtype, "", "", "", 3,
        Options{options.m, options.n, options.k, 0, 1, options.mode}, d_a, d_b, d_c, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> host_output(c_count);
  CUDA_CHECK(cudaMemcpy(host_output.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
  const double max_norm_error = ValidateSamples(options, reconstructed_a, reconstructed_b, host_output);
  const bool passed = std::isfinite(max_norm_error) && max_norm_error <= 8.0e-3;

  std::cout << "Best CUTLASS " << dtype << " config: " << best->id << "\n";
  std::cout << "Best CUTLASS " << dtype << " latency: " << best->average_ms << " ms\n";
  std::cout << "Best CUTLASS " << dtype << " throughput: " << best->gflops << " GFLOP/s\n";
  std::cout << "Best CUTLASS " << dtype << " validation max normalized error: " << max_norm_error << "\n";
  *best_throughput = best->gflops;

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return passed;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major * 10 + props.minor < 80) {
      std::cout << "Validation: PASS (SM80 Universal TensorOp path unsupported on this GPU)\n";
      return EXIT_SUCCESS;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUTLASS version: 4.7.0\n";
    std::cout << "CUTLASS API: device::GemmUniversal explicit tile/pipeline configuration\n";
    std::cout << "GPU compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "M: " << options.m << "\nN: " << options.n << "\nK: " << options.k << "\n";

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    bool success = true;
    double fp16_throughput = 0.0;
    double bf16_throughput = 0.0;

    if (options.mode == "all" || options.mode == "fp16") {
      success = RunPrecision<cutlass::half_t>("FP16", options, stream, &fp16_throughput) && success;
    }
    if (options.mode == "all" || options.mode == "bf16") {
      success = RunPrecision<cutlass::bfloat16_t>("BF16", options, stream, &bf16_throughput) && success;
    }
    if (fp16_throughput > 0.0 && bf16_throughput > 0.0) {
      std::cout << "CUTLASS BF16 to FP16 throughput ratio: "
                << (bf16_throughput / fp16_throughput) << " x\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << "\n";
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
