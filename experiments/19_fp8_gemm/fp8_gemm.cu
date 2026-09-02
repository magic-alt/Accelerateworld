#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {

#define CUBLAS_CHECK(call)                                                                    \
  do {                                                                                        \
    const cublasStatus_t status__ = (call);                                                    \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                                   \
      throw std::runtime_error(std::string("cuBLAS/cuBLASLt error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " status=" +                       \
                               std::to_string(static_cast<int>(status__)));                    \
    }                                                                                         \
  } while (false)

enum class Mode { kBf16, kE4m3, kE5m2 };

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int heuristics = 16;
  int warmup = 2;
  int iterations = 10;
  std::uint64_t workspace_mb = 32;
  std::string mode = "all";
};

struct CandidateResult {
  cublasLtMatmulAlgo_t algo{};
  bool valid = false;
  std::size_t workspace_bytes = 0;
  float waves_count = 0.0f;
  double average_ms = std::numeric_limits<double>::infinity();
  double gflops = 0.0;
  std::int32_t algo_id = -1;
  std::uint32_t tile_id = 0;
  std::uint32_t stages_id = 0;
  std::uint32_t splitk = 0;
};

struct QuantStats {
  double amax = 0.0;
  double max_finite = 0.0;
  double dequant_scale = 1.0;
  std::uint64_t clipped = 0;
  std::uint64_t near_limit = 0;
  double max_reconstruction_error = 0.0;
};

struct ModeResult {
  bool supported = false;
  bool passed = false;
  std::string skip_reason;
  int heuristic_candidates = 0;
  int measured_candidates = 0;
  CandidateResult best;
  double max_correctness_abs_error = 0.0;
  double max_correctness_normalized_error = 0.0;
  double max_accuracy_abs_error = 0.0;
  double max_accuracy_normalized_error = 0.0;
  QuantStats a_quant;
  QuantStats b_quant;
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
    } else if (arg == "--heuristics" && i + 1 < argc) {
      options.heuristics = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--workspace-mb" && i + 1 < argc) {
      options.workspace_mb = std::stoull(argv[++i]);
    } else if (arg == "--mode" && i + 1 < argc) {
      options.mode = argv[++i];
    } else if (arg == "--help") {
      std::cout << "Usage: aw_fp8_gemm [--m M] [--n N] [--k K] "
                   "[--heuristics N] [--warmup N] [--iterations N] [--workspace-mb N] "
                   "[--mode all|bf16|e4m3|e5m2]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.heuristics <= 0 ||
      options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument("matrix dimensions, heuristic count and iterations must be positive");
  }
  if (options.m % 16 != 0 || options.n % 16 != 0 || options.k % 16 != 0) {
    throw std::invalid_argument("M, N and K must be multiples of 16 for this aligned FP8/Tensor Core lab");
  }
  if (options.mode != "all" && options.mode != "bf16" && options.mode != "e4m3" &&
      options.mode != "e5m2") {
    throw std::invalid_argument("mode must be all, bf16, e4m3 or e5m2");
  }
  return options;
}

const char* ModeName(Mode mode) {
  switch (mode) {
    case Mode::kBf16: return "BF16 reference";
    case Mode::kE4m3: return "E4M3";
    case Mode::kE5m2: return "E5M2";
  }
  return "unknown";
}

bool WantsMode(const Options& options, Mode mode) {
  if (options.mode == "all") return true;
  if (mode == Mode::kBf16) return options.mode == "bf16";
  if (mode == Mode::kE4m3) return options.mode == "e4m3";
  return options.mode == "e5m2";
}

bool WantsFp8(const Options& options) {
  return options.mode == "all" || options.mode == "e4m3" || options.mode == "e5m2";
}

bool HardwareSupports(Mode mode, int major, int minor, std::string* reason) {
  const int cc = major * 10 + minor;
  if (mode == Mode::kBf16 && cc < 80) {
    *reason = "requires compute capability >= 8.0";
    return false;
  }
  if ((mode == Mode::kE4m3 || mode == Mode::kE5m2) && cc < 89) {
    *reason = "requires FP8 Tensor Core support (compute capability >= 8.9)";
    return false;
  }
  return true;
}

template <typename T>
T GetAlgoConfig(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  T value{};
  std::size_t written = 0;
  const cublasStatus_t status =
      cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), &written);
  return status == CUBLAS_STATUS_SUCCESS ? value : T{};
}

void PopulateAlgoMetadata(CandidateResult* result) {
  result->algo_id = GetAlgoConfig<std::int32_t>(result->algo, CUBLASLT_ALGO_CONFIG_ID);
  result->tile_id = GetAlgoConfig<std::uint32_t>(result->algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
  result->stages_id = GetAlgoConfig<std::uint32_t>(result->algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
  result->splitk = GetAlgoConfig<std::uint32_t>(result->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
}

std::vector<float> BuildInputAStorage(const Options& options) {
  std::vector<float> values(static_cast<std::size_t>(options.k) * options.m);
  for (int logical_row = 0; logical_row < options.m; ++logical_row) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (logical_row * 17 + inner * 13 + 3) % 61 - 30;
      const float modulation = 1.0f + static_cast<float>((logical_row + inner) % 7) * 0.03125f;
      values[static_cast<std::size_t>(inner) + static_cast<std::size_t>(logical_row) * options.k] =
          static_cast<float>(code) * 0.021f * modulation;
    }
  }
  return values;
}

std::vector<float> BuildInputBStorage(const Options& options) {
  std::vector<float> values(static_cast<std::size_t>(options.k) * options.n);
  for (int col = 0; col < options.n; ++col) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (inner * 11 + col * 7 + 5) % 53 - 26;
      const float modulation = 1.0f + static_cast<float>((inner + 2 * col) % 5) * 0.046875f;
      values[static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k] =
          static_cast<float>(code) * 0.023f * modulation;
    }
  }
  return values;
}

template <typename Narrow>
std::vector<Narrow> QuantizeNarrow(const std::vector<float>& source,
                                   double max_finite,
                                   QuantStats* stats,
                                   std::vector<float>* reconstructed,
                                   bool apply_scale) {
  for (float value : source) {
    stats->amax = std::max(stats->amax, static_cast<double>(std::abs(value)));
  }
  stats->max_finite = max_finite;
  stats->dequant_scale = apply_scale && stats->amax > 0.0 ? stats->amax / max_finite : 1.0;
  const double quant_multiplier = 1.0 / stats->dequant_scale;

  std::vector<Narrow> output(source.size());
  reconstructed->resize(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    const double scaled = static_cast<double>(source[i]) * quant_multiplier;
    const double magnitude = std::abs(scaled);
    if (apply_scale && magnitude > max_finite) ++stats->clipped;
    if (apply_scale && magnitude >= max_finite * 0.99) ++stats->near_limit;

    const Narrow quantized(static_cast<float>(scaled));
    output[i] = quantized;
    const double restored = static_cast<double>(static_cast<float>(quantized)) * stats->dequant_scale;
    (*reconstructed)[i] = static_cast<float>(restored);
    stats->max_reconstruction_error =
        std::max(stats->max_reconstruction_error,
                 std::abs(restored - static_cast<double>(source[i])));
  }
  return output;
}

CandidateResult BenchmarkCandidate(cublasLtHandle_t lt,
                                   cublasLtMatmulDesc_t operation,
                                   cublasLtMatrixLayout_t a_desc,
                                   cublasLtMatrixLayout_t b_desc,
                                   cublasLtMatrixLayout_t c_desc,
                                   const void* a,
                                   const void* b,
                                   float* c,
                                   const cublasLtMatmulAlgo_t& algo,
                                   std::size_t workspace_bytes,
                                   void* workspace,
                                   cudaStream_t stream,
                                   const Options& options,
                                   float waves_count) {
  CandidateResult result;
  result.algo = algo;
  result.workspace_bytes = workspace_bytes;
  result.waves_count = waves_count;
  PopulateAlgoMetadata(&result);

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  for (int i = 0; i < options.warmup; ++i) {
    const cublasStatus_t status = cublasLtMatmul(
        lt, operation, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, c, c_desc,
        &result.algo, workspace, workspace_bytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return result;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < options.iterations; ++i) {
    const cublasStatus_t status = cublasLtMatmul(
        lt, operation, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, c, c_desc,
        &result.algo, workspace, workspace_bytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      return result;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  result.average_ms = static_cast<double>(elapsed_ms) / options.iterations;
  const double flops = 2.0 * static_cast<double>(options.m) * options.n * options.k;
  result.gflops = flops / (result.average_ms / 1000.0) / 1.0e9;
  result.valid = std::isfinite(result.average_ms) && result.average_ms > 0.0;
  return result;
}

void MeasureErrors(const Options& options,
                   const std::vector<float>& original_a,
                   const std::vector<float>& original_b,
                   const std::vector<float>& reconstructed_a,
                   const std::vector<float>& reconstructed_b,
                   const std::vector<float>& output,
                   ModeResult* result) {
  constexpr int kSamples = 32;
  for (int sample = 0; sample < kSamples; ++sample) {
    const int row = (sample * 97 + 11) % options.m;
    const int col = (sample * 53 + 7) % options.n;

    double original_reference = 0.0;
    double quantized_reference = 0.0;
    for (int inner = 0; inner < options.k; ++inner) {
      const std::size_t a_index =
          static_cast<std::size_t>(inner) + static_cast<std::size_t>(row) * options.k;
      const std::size_t b_index =
          static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k;
      original_reference +=
          static_cast<double>(original_a[a_index]) * static_cast<double>(original_b[b_index]);
      quantized_reference +=
          static_cast<double>(reconstructed_a[a_index]) * static_cast<double>(reconstructed_b[b_index]);
    }

    const double actual =
        output[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * options.m];
    const double correctness_abs = std::abs(actual - quantized_reference);
    const double correctness_norm =
        correctness_abs / std::max(1.0, std::abs(quantized_reference));
    result->max_correctness_abs_error =
        std::max(result->max_correctness_abs_error, correctness_abs);
    result->max_correctness_normalized_error =
        std::max(result->max_correctness_normalized_error, correctness_norm);

    const double accuracy_abs = std::abs(actual - original_reference);
    const double accuracy_norm = accuracy_abs / std::max(1.0, std::abs(original_reference));
    result->max_accuracy_abs_error = std::max(result->max_accuracy_abs_error, accuracy_abs);
    result->max_accuracy_normalized_error =
        std::max(result->max_accuracy_normalized_error, accuracy_norm);
  }
}

ModeResult RunMode(Mode mode,
                   const Options& options,
                   int major,
                   int minor,
                   cublasLtHandle_t lt,
                   cudaStream_t stream,
                   const void* d_a,
                   const void* d_b,
                   cudaDataType_t input_type,
                   const std::vector<float>& original_a,
                   const std::vector<float>& original_b,
                   const std::vector<float>& reconstructed_a,
                   const std::vector<float>& reconstructed_b,
                   const QuantStats& a_quant,
                   const QuantStats& b_quant,
                   float* d_c,
                   void* workspace,
                   std::size_t workspace_bytes) {
  ModeResult output;
  output.a_quant = a_quant;
  output.b_quant = b_quant;
  if (!HardwareSupports(mode, major, minor, &output.skip_reason)) return output;
  output.supported = true;

  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a_desc{};
  cublasLtMatrixLayout_t b_desc{};
  cublasLtMatrixLayout_t c_desc{};
  cublasLtMatmulPreference_t preference{};

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  float* d_scale_a = nullptr;
  float* d_scale_b = nullptr;
  if (mode == Mode::kE4m3 || mode == Mode::kE5m2) {
    CUDA_CHECK(cudaMalloc(&d_scale_a, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale_b, sizeof(float)));
    const float scale_a = static_cast<float>(a_quant.dequant_scale);
    const float scale_b = static_cast<float>(b_quant.dequant_scale);
    CUDA_CHECK(cudaMemcpy(d_scale_a, &scale_a, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_b, &scale_b, sizeof(float), cudaMemcpyHostToDevice));

    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));

    const std::int8_t fast_accum = 0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));
  }

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, input_type, options.k, options.m, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, input_type, options.k, options.n, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, options.m, options.n, options.m));
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(options.heuristics);
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, operation, a_desc, b_desc, c_desc, c_desc, preference,
      options.heuristics, heuristic_results.data(), &returned));
  output.heuristic_candidates = returned;

  for (int i = 0; i < returned; ++i) {
    const auto& heuristic = heuristic_results[i];
    if (heuristic.state != CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize > workspace_bytes) continue;
    CandidateResult candidate = BenchmarkCandidate(
        lt, operation, a_desc, b_desc, c_desc, d_a, d_b, d_c, heuristic.algo,
        heuristic.workspaceSize, workspace, stream, options, heuristic.wavesCount);
    if (!candidate.valid) continue;
    ++output.measured_candidates;
    if (!output.best.valid || candidate.average_ms < output.best.average_ms) output.best = candidate;
  }

  if (output.best.valid) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        lt, operation, &alpha, d_a, a_desc, d_b, b_desc, &beta,
        d_c, c_desc, d_c, c_desc, &output.best.algo,
        workspace, output.best.workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> host_output(static_cast<std::size_t>(options.m) * options.n);
    CUDA_CHECK(cudaMemcpy(
        host_output.data(), d_c, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    MeasureErrors(options, original_a, original_b, reconstructed_a, reconstructed_b, host_output, &output);
    constexpr double kCorrectnessThreshold = 2.0e-3;
    output.passed = std::isfinite(output.max_correctness_normalized_error) &&
                    output.max_correctness_normalized_error <= kCorrectnessThreshold;
  }

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
  if (d_scale_a != nullptr) CUDA_CHECK(cudaFree(d_scale_a));
  if (d_scale_b != nullptr) CUDA_CHECK(cudaFree(d_scale_b));
  return output;
}

void PrintQuantStats(const std::string& prefix, const QuantStats& stats, std::size_t elements) {
  std::cout << prefix << " amax: " << stats.amax << "\n";
  std::cout << prefix << " dequant scale: " << stats.dequant_scale << "\n";
  std::cout << prefix << " clipped elements: " << stats.clipped << " elements\n";
  std::cout << prefix << " clipped rate: "
            << (100.0 * static_cast<double>(stats.clipped) / elements) << " %\n";
  std::cout << prefix << " near limit elements: " << stats.near_limit << " elements\n";
  std::cout << prefix << " near limit rate: "
            << (100.0 * static_cast<double>(stats.near_limit) / elements) << " %\n";
  std::cout << prefix << " max reconstruction error: " << stats.max_reconstruction_error << "\n";
}

void PrintModeResult(Mode mode,
                     const ModeResult& result,
                     std::size_t a_elements,
                     std::size_t b_elements) {
  const std::string name = ModeName(mode);
  if (!result.supported) {
    std::cout << name << ": SKIPPED (" << result.skip_reason << ")\n";
    return;
  }
  std::cout << name << " heuristic candidates: " << result.heuristic_candidates << " candidates\n";
  std::cout << name << " measured candidates: " << result.measured_candidates << " candidates\n";
  if (!result.best.valid) {
    std::cout << name << ": FAIL (no executable cuBLASLt candidate)\n";
    return;
  }
  std::cout << name << " latency: " << result.best.average_ms << " ms\n";
  std::cout << name << " throughput: " << result.best.gflops << " GFLOP/s\n";
  std::cout << name << " best algo id: " << result.best.algo_id << "\n";
  std::cout << name << " best tile id: " << result.best.tile_id << "\n";
  std::cout << name << " best stages id: " << result.best.stages_id << "\n";
  std::cout << name << " best split k: " << result.best.splitk << "\n";
  std::cout << name << " best workspace: " << result.best.workspace_bytes << " bytes\n";
  std::cout << name << " waves: " << result.best.waves_count << "\n";
  std::cout << name << " correctness max abs error: " << result.max_correctness_abs_error << "\n";
  std::cout << name << " correctness max normalized error: "
            << result.max_correctness_normalized_error << "\n";
  std::cout << name << " accuracy max abs error: " << result.max_accuracy_abs_error << "\n";
  std::cout << name << " accuracy max normalized error: "
            << result.max_accuracy_normalized_error << "\n";
  if (mode == Mode::kE4m3 || mode == Mode::kE5m2) {
    PrintQuantStats(name + " A", result.a_quant, a_elements);
    PrintQuantStats(name + " B", result.b_quant, b_elements);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const auto host_a = BuildInputAStorage(options);
    const auto host_b = BuildInputBStorage(options);

    QuantStats bf16_a_stats{};
    QuantStats bf16_b_stats{};
    std::vector<float> bf16_a_reconstructed;
    std::vector<float> bf16_b_reconstructed;
    const auto host_a_bf16 = QuantizeNarrow<__nv_bfloat16>(
        host_a, 1.0, &bf16_a_stats, &bf16_a_reconstructed, false);
    const auto host_b_bf16 = QuantizeNarrow<__nv_bfloat16>(
        host_b, 1.0, &bf16_b_stats, &bf16_b_reconstructed, false);

    QuantStats e4_a_stats{};
    QuantStats e4_b_stats{};
    std::vector<float> e4_a_reconstructed;
    std::vector<float> e4_b_reconstructed;
    const auto host_a_e4 = QuantizeNarrow<__nv_fp8_e4m3>(
        host_a, 448.0, &e4_a_stats, &e4_a_reconstructed, true);
    const auto host_b_e4 = QuantizeNarrow<__nv_fp8_e4m3>(
        host_b, 448.0, &e4_b_stats, &e4_b_reconstructed, true);

    QuantStats e5_a_stats{};
    QuantStats e5_b_stats{};
    std::vector<float> e5_a_reconstructed;
    std::vector<float> e5_b_reconstructed;
    const auto host_a_e5 = QuantizeNarrow<__nv_fp8_e5m2>(
        host_a, 57344.0, &e5_a_stats, &e5_a_reconstructed, true);
    const auto host_b_e5 = QuantizeNarrow<__nv_fp8_e5m2>(
        host_b, 57344.0, &e5_b_stats, &e5_b_reconstructed, true);

    __nv_bfloat16* d_a_bf16 = nullptr;
    __nv_bfloat16* d_b_bf16 = nullptr;
    __nv_fp8_e4m3* d_a_e4 = nullptr;
    __nv_fp8_e4m3* d_b_e4 = nullptr;
    __nv_fp8_e5m2* d_a_e5 = nullptr;
    __nv_fp8_e5m2* d_b_e5 = nullptr;
    float* d_c = nullptr;
    void* workspace = nullptr;

    const std::size_t a_elements = host_a.size();
    const std::size_t b_elements = host_b.size();
    const std::size_t c_elements = static_cast<std::size_t>(options.m) * options.n;
    const std::size_t workspace_bytes =
        static_cast<std::size_t>(options.workspace_mb) * 1024ull * 1024ull;

    CUDA_CHECK(cudaMalloc(&d_a_bf16, a_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_b_bf16, b_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_a_e4, a_elements * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_b_e4, b_elements * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_a_e5, a_elements * sizeof(__nv_fp8_e5m2)));
    CUDA_CHECK(cudaMalloc(&d_b_e5, b_elements * sizeof(__nv_fp8_e5m2)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elements * sizeof(float)));
    if (workspace_bytes > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    CUDA_CHECK(cudaMemcpy(d_a_bf16, host_a_bf16.data(),
                          a_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_bf16, host_b_bf16.data(),
                          b_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_e4, host_a_e4.data(),
                          a_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_e4, host_b_e4.data(),
                          b_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_e5, host_a_e5.data(),
                          a_elements * sizeof(__nv_fp8_e5m2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_e5, host_b_e5.data(),
                          b_elements * sizeof(__nv_fp8_e5m2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, c_elements * sizeof(float)));

    cublasLtHandle_t lt{};
    CUBLAS_CHECK(cublasLtCreate(&lt));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FP8 cuBLASLt GEMM\n";
    std::cout << "GPU compute capability: " << properties.major << "." << properties.minor << "\n";
    std::cout << "M: " << options.m << "\n";
    std::cout << "N: " << options.n << "\n";
    std::cout << "K: " << options.k << "\n";
    std::cout << "Layout: TN\n";
    std::cout << "Workspace budget: " << workspace_bytes << " bytes\n";

    ModeResult bf16_result;
    bool bf16_ran = false;
    if (WantsMode(options, Mode::kBf16) || WantsFp8(options)) {
      bf16_result = RunMode(
          Mode::kBf16, options, properties.major, properties.minor, lt, stream,
          d_a_bf16, d_b_bf16, CUDA_R_16BF, host_a, host_b,
          bf16_a_reconstructed, bf16_b_reconstructed, bf16_a_stats, bf16_b_stats,
          d_c, workspace, workspace_bytes);
      bf16_ran = true;
      PrintModeResult(Mode::kBf16, bf16_result, a_elements, b_elements);
    }

    ModeResult e4_result;
    if (WantsMode(options, Mode::kE4m3)) {
      e4_result = RunMode(
          Mode::kE4m3, options, properties.major, properties.minor, lt, stream,
          d_a_e4, d_b_e4, CUDA_R_8F_E4M3, host_a, host_b,
          e4_a_reconstructed, e4_b_reconstructed, e4_a_stats, e4_b_stats,
          d_c, workspace, workspace_bytes);
      PrintModeResult(Mode::kE4m3, e4_result, a_elements, b_elements);
      if (bf16_result.best.valid && e4_result.best.valid) {
        std::cout << "E4M3 speedup vs BF16: "
                  << (e4_result.best.gflops / bf16_result.best.gflops) << " x\n";
        std::cout << "E4M3 accuracy degradation vs BF16: "
                  << (e4_result.max_accuracy_normalized_error -
                      bf16_result.max_accuracy_normalized_error) << "\n";
      }
    }

    ModeResult e5_result;
    if (WantsMode(options, Mode::kE5m2)) {
      e5_result = RunMode(
          Mode::kE5m2, options, properties.major, properties.minor, lt, stream,
          d_a_e5, d_b_e5, CUDA_R_8F_E5M2, host_a, host_b,
          e5_a_reconstructed, e5_b_reconstructed, e5_a_stats, e5_b_stats,
          d_c, workspace, workspace_bytes);
      PrintModeResult(Mode::kE5m2, e5_result, a_elements, b_elements);
      if (bf16_result.best.valid && e5_result.best.valid) {
        std::cout << "E5M2 speedup vs BF16: "
                  << (e5_result.best.gflops / bf16_result.best.gflops) << " x\n";
        std::cout << "E5M2 accuracy degradation vs BF16: "
                  << (e5_result.max_accuracy_normalized_error -
                      bf16_result.max_accuracy_normalized_error) << "\n";
      }
    }

    bool success = true;
    bool any_supported = false;
    if (bf16_ran && bf16_result.supported) {
      any_supported = true;
      success = success && bf16_result.passed;
    }
    if (WantsMode(options, Mode::kE4m3) && e4_result.supported) {
      any_supported = true;
      success = success && e4_result.passed;
    }
    if (WantsMode(options, Mode::kE5m2) && e5_result.supported) {
      any_supported = true;
      success = success && e5_result.passed;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasLtDestroy(lt));
    if (workspace != nullptr) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_a_bf16));
    CUDA_CHECK(cudaFree(d_b_bf16));
    CUDA_CHECK(cudaFree(d_a_e4));
    CUDA_CHECK(cudaFree(d_b_e4));
    CUDA_CHECK(cudaFree(d_a_e5));
    CUDA_CHECK(cudaFree(d_b_e5));

    if (!any_supported) {
      std::cout << "Validation: PASS (all requested modes unsupported on this GPU)\n";
      return EXIT_SUCCESS;
    }
    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << "\n";
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
