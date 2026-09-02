#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

enum class Mode { kFp32, kTf32, kBf16, kFp16 };

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

struct ModeResult {
  bool supported = false;
  bool passed = false;
  std::string skip_reason;
  int heuristic_candidates = 0;
  int measured_candidates = 0;
  CandidateResult best;
  double max_abs_error = 0.0;
  double max_normalized_error = 0.0;
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
      std::cout << "Usage: aw_mixed_precision_gemm [--m M] [--n N] [--k K] "
                   "[--heuristics N] [--warmup N] [--iterations N] [--workspace-mb N] "
                   "[--mode all|fp32|tf32|bf16|fp16]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.heuristics <= 0 ||
      options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument("matrix dimensions, heuristic count and iterations must be positive");
  }
  if (options.mode != "all" && options.mode != "fp32" && options.mode != "tf32" &&
      options.mode != "bf16" && options.mode != "fp16") {
    throw std::invalid_argument("mode must be all, fp32, tf32, bf16 or fp16");
  }
  return options;
}

std::string ModeName(Mode mode) {
  switch (mode) {
    case Mode::kFp32: return "FP32";
    case Mode::kTf32: return "TF32";
    case Mode::kBf16: return "BF16";
    case Mode::kFp16: return "FP16";
  }
  return "unknown";
}

bool WantsMode(const Options& options, Mode mode) {
  if (options.mode == "all") {
    return true;
  }
  if (mode == Mode::kFp32) return options.mode == "fp32";
  if (mode == Mode::kTf32) return options.mode == "tf32";
  if (mode == Mode::kBf16) return options.mode == "bf16";
  return options.mode == "fp16";
}

bool HardwareSupports(Mode mode, int major, int minor, std::string* reason) {
  const int cc = major * 10 + minor;
  if ((mode == Mode::kTf32 || mode == Mode::kBf16) && cc < 80) {
    *reason = "requires compute capability >= 8.0";
    return false;
  }
  if (mode == Mode::kFp16 && cc < 70) {
    *reason = "requires Tensor Core capable compute capability >= 7.0";
    return false;
  }
  return true;
}

__global__ void FloatToHalfKernel(const float* input, __half* output, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = __float2half_rn(input[index]);
  }
}

__global__ void FloatToBf16Kernel(const float* input, __nv_bfloat16* output, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
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
    if (status != CUBLAS_STATUS_SUCCESS) {
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

std::vector<float> BuildInputA(const Options& options) {
  std::vector<float> values(static_cast<std::size_t>(options.m) * options.k);
  for (int col = 0; col < options.k; ++col) {
    for (int row = 0; row < options.m; ++row) {
      const int code = (row * 17 + col * 13 + 3) % 29 - 14;
      values[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * options.m] =
          static_cast<float>(code) * 0.017f;
    }
  }
  return values;
}

std::vector<float> BuildInputB(const Options& options) {
  std::vector<float> values(static_cast<std::size_t>(options.k) * options.n);
  for (int col = 0; col < options.n; ++col) {
    for (int row = 0; row < options.k; ++row) {
      const int code = (row * 11 + col * 7 + 5) % 23 - 11;
      values[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * options.k] =
          static_cast<float>(code) * 0.019f;
    }
  }
  return values;
}

void MeasureSampleError(const Options& options,
                        const std::vector<float>& a,
                        const std::vector<float>& b,
                        const std::vector<float>& c,
                        double* max_abs_error,
                        double* max_normalized_error) {
  *max_abs_error = 0.0;
  *max_normalized_error = 0.0;
  constexpr int kSamples = 32;
  for (int sample = 0; sample < kSamples; ++sample) {
    const int row = (sample * 97 + 11) % options.m;
    const int col = (sample * 53 + 7) % options.n;
    double reference = 0.0;
    for (int inner = 0; inner < options.k; ++inner) {
      const double av = a[static_cast<std::size_t>(row) + static_cast<std::size_t>(inner) * options.m];
      const double bv = b[static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k];
      reference += av * bv;
    }
    const double actual = c[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * options.m];
    const double abs_error = std::abs(actual - reference);
    const double normalized = abs_error / std::max(1.0, std::abs(reference));
    *max_abs_error = std::max(*max_abs_error, abs_error);
    *max_normalized_error = std::max(*max_normalized_error, normalized);
  }
}

double ErrorThreshold(Mode mode) {
  switch (mode) {
    case Mode::kFp32: return 1.0e-4;
    case Mode::kTf32: return 2.0e-2;
    case Mode::kBf16: return 5.0e-2;
    case Mode::kFp16: return 2.0e-2;
  }
  return 1.0;
}

ModeResult RunMode(Mode mode,
                   const Options& options,
                   int major,
                   int minor,
                   cublasLtHandle_t lt,
                   cudaStream_t stream,
                   const float* d_a32,
                   const float* d_b32,
                   const __half* d_a16,
                   const __half* d_b16,
                   const __nv_bfloat16* d_abf16,
                   const __nv_bfloat16* d_bbf16,
                   float* d_c,
                   void* workspace,
                   std::size_t workspace_bytes,
                   const std::vector<float>& host_a,
                   const std::vector<float>& host_b) {
  ModeResult output;
  if (!HardwareSupports(mode, major, minor, &output.skip_reason)) {
    return output;
  }
  output.supported = true;

  cudaDataType_t input_type = CUDA_R_32F;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
  const void* a = d_a32;
  const void* b = d_b32;

  if (mode == Mode::kTf32) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  } else if (mode == Mode::kBf16) {
    input_type = CUDA_R_16BF;
    compute_type = CUBLAS_COMPUTE_32F;
    a = d_abf16;
    b = d_bbf16;
  } else if (mode == Mode::kFp16) {
    input_type = CUDA_R_16F;
    compute_type = CUBLAS_COMPUTE_32F;
    a = d_a16;
    b = d_b16;
  }

  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a_desc{};
  cublasLtMatrixLayout_t b_desc{};
  cublasLtMatrixLayout_t c_desc{};
  cublasLtMatmulPreference_t preference{};

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation, compute_type, CUDA_R_32F));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, input_type, options.m, options.k, options.m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, input_type, options.k, options.n, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, options.m, options.n, options.m));
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(options.heuristics);
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, operation, a_desc, b_desc, c_desc, c_desc, preference, options.heuristics,
      heuristic_results.data(), &returned));
  output.heuristic_candidates = returned;

  CandidateResult best;
  for (int i = 0; i < returned; ++i) {
    const auto& heuristic = heuristic_results[i];
    if (heuristic.state != CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize > workspace_bytes) {
      continue;
    }
    CandidateResult candidate = BenchmarkCandidate(
        lt, operation, a_desc, b_desc, c_desc, a, b, d_c, heuristic.algo,
        heuristic.workspaceSize, workspace, stream, options, heuristic.wavesCount);
    if (!candidate.valid) {
      continue;
    }
    ++output.measured_candidates;
    if (!best.valid || candidate.average_ms < best.average_ms) {
      best = candidate;
    }
  }

  if (!best.valid) {
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
    throw std::runtime_error("No executable cuBLASLt candidate for mode " + ModeName(mode));
  }

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  CUBLAS_CHECK(cublasLtMatmul(
      lt, operation, &alpha, a, a_desc, b, b_desc, &beta, d_c, c_desc, d_c, c_desc,
      &best.algo, workspace, best.workspace_bytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> host_c(static_cast<std::size_t>(options.m) * options.n);
  CUDA_CHECK(cudaMemcpy(host_c.data(), d_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  MeasureSampleError(options, host_a, host_b, host_c,
                     &output.max_abs_error, &output.max_normalized_error);
  output.best = best;
  output.passed = output.max_normalized_error <= ErrorThreshold(mode);

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
  return output;
}

void PrintModeResult(Mode mode, const ModeResult& result) {
  const std::string name = ModeName(mode);
  if (!result.supported) {
    std::cout << name << " status: SKIP (" << result.skip_reason << ")\n";
    return;
  }
  std::cout << name << " status: " << (result.passed ? "PASS" : "FAIL") << '\n';
  std::cout << name << " heuristic candidates: " << result.heuristic_candidates << '\n';
  std::cout << name << " measured candidates: " << result.measured_candidates << '\n';
  std::cout << name << " average latency: " << result.best.average_ms << " ms\n";
  std::cout << name << " throughput: " << result.best.gflops << " GFLOP/s\n";
  std::cout << name << " best algo id: " << result.best.algo_id << '\n';
  std::cout << name << " best tile id: " << result.best.tile_id << '\n';
  std::cout << name << " best stages id: " << result.best.stages_id << '\n';
  std::cout << name << " best split-k: " << result.best.splitk << '\n';
  std::cout << name << " best workspace: " << result.best.workspace_bytes << " bytes\n";
  std::cout << name << " best waves: " << result.best.waves_count << '\n';
  std::cout << name << " max abs error: " << result.max_abs_error << '\n';
  std::cout << name << " max normalized error: " << result.max_normalized_error << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::vector<float> host_a = BuildInputA(options);
    const std::vector<float> host_b = BuildInputB(options);
    const std::size_t a_elements = host_a.size();
    const std::size_t b_elements = host_b.size();
    const std::size_t c_elements = static_cast<std::size_t>(options.m) * options.n;

    float* d_a32 = nullptr;
    float* d_b32 = nullptr;
    float* d_c = nullptr;
    __half* d_a16 = nullptr;
    __half* d_b16 = nullptr;
    __nv_bfloat16* d_abf16 = nullptr;
    __nv_bfloat16* d_bbf16 = nullptr;
    void* workspace = nullptr;
    const std::size_t workspace_bytes =
        static_cast<std::size_t>(options.workspace_mb * 1024ull * 1024ull);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a32), a_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b32), b_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), c_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a16), a_elements * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b16), b_elements * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_abf16), a_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbf16), b_elements * sizeof(__nv_bfloat16)));
    if (workspace_bytes > 0) {
      CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    }

    CUDA_CHECK(cudaMemcpy(d_a32, host_a.data(), a_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b32, host_b.data(), b_elements * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int kThreads = 256;
    const int a_blocks = static_cast<int>((a_elements + kThreads - 1) / kThreads);
    const int b_blocks = static_cast<int>((b_elements + kThreads - 1) / kThreads);
    FloatToHalfKernel<<<a_blocks, kThreads>>>(d_a32, d_a16, a_elements);
    FloatToHalfKernel<<<b_blocks, kThreads>>>(d_b32, d_b16, b_elements);
    FloatToBf16Kernel<<<a_blocks, kThreads>>>(d_a32, d_abf16, a_elements);
    FloatToBf16Kernel<<<b_blocks, kThreads>>>(d_b32, d_bbf16, b_elements);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasLtHandle_t lt{};
    CUBLAS_CHECK(cublasLtCreate(&lt));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "cuBLASLt Mixed Precision GEMM\n";
    std::cout << "GPU: " << properties.name << '\n';
    std::cout << "Compute capability: " << properties.major << '.' << properties.minor << '\n';
    std::cout << "Shape: " << options.m << " x " << options.n << " x " << options.k << '\n';
    std::cout << "Workspace budget: " << workspace_bytes << " bytes\n";
    std::cout << "Requested mode: " << options.mode << '\n';

    bool success = true;
    const std::vector<Mode> modes = {Mode::kFp32, Mode::kTf32, Mode::kBf16, Mode::kFp16};
    for (Mode mode : modes) {
      if (!WantsMode(options, mode)) {
        continue;
      }
      const ModeResult result = RunMode(
          mode, options, properties.major, properties.minor, lt, stream,
          d_a32, d_b32, d_a16, d_b16, d_abf16, d_bbf16, d_c,
          workspace, workspace_bytes, host_a, host_b);
      PrintModeResult(mode, result);
      if (result.supported && !result.passed) {
        success = false;
      }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasLtDestroy(lt));
    if (workspace != nullptr) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_bbf16));
    CUDA_CHECK(cudaFree(d_abf16));
    CUDA_CHECK(cudaFree(d_b16));
    CUDA_CHECK(cudaFree(d_a16));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b32));
    CUDA_CHECK(cudaFree(d_a32));

    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << '\n';
    return success ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
