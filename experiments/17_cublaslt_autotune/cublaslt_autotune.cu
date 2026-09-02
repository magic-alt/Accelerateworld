#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int heuristics = 16;
  int warmup = 2;
  int iterations = 10;
  std::uint64_t workspace_mb = 32;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  cublasLtOrder_t order = CUBLASLT_ORDER_COL;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  std::string cache_dir = ".cache/cublaslt";
  bool no_cache = false;
  bool force_autotune = false;
};

struct LayoutShape {
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  std::int64_t ld = 0;
  std::size_t elements = 0;
};

struct CandidateResult {
  cublasLtMatmulAlgo_t algo{};
  bool valid = false;
  bool from_cache = false;
  cublasStatus_t launch_status = CUBLAS_STATUS_SUCCESS;
  std::size_t workspace_bytes = 0;
  float waves_count = 0.0f;
  double average_ms = std::numeric_limits<double>::infinity();
  double gflops = 0.0;
  std::int32_t algo_id = -1;
  std::uint32_t tile_id = 0;
  std::uint32_t stages_id = 0;
  std::uint32_t splitk = 0;
  std::uint32_t reduction_scheme = 0;
  std::uint32_t cta_swizzling = 0;
  std::uint32_t custom_option = 0;
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
    } else if (arg == "--transa" && i + 1 < argc) {
      const std::string value = argv[++i];
      options.transa = (value == "T" || value == "t") ? CUBLAS_OP_T : CUBLAS_OP_N;
      if (value != "N" && value != "n" && value != "T" && value != "t") {
        throw std::invalid_argument("transa must be N or T");
      }
    } else if (arg == "--transb" && i + 1 < argc) {
      const std::string value = argv[++i];
      options.transb = (value == "T" || value == "t") ? CUBLAS_OP_T : CUBLAS_OP_N;
      if (value != "N" && value != "n" && value != "T" && value != "t") {
        throw std::invalid_argument("transb must be N or T");
      }
    } else if (arg == "--order" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "col") {
        options.order = CUBLASLT_ORDER_COL;
      } else if (value == "row") {
        options.order = CUBLASLT_ORDER_ROW;
      } else {
        throw std::invalid_argument("order must be col or row");
      }
    } else if (arg == "--epilogue" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "none") {
        options.epilogue = CUBLASLT_EPILOGUE_DEFAULT;
      } else if (value == "relu") {
        options.epilogue = CUBLASLT_EPILOGUE_RELU;
      } else {
        throw std::invalid_argument("epilogue must be none or relu");
      }
    } else if (arg == "--cache-dir" && i + 1 < argc) {
      options.cache_dir = argv[++i];
    } else if (arg == "--no-cache") {
      options.no_cache = true;
    } else if (arg == "--force-autotune") {
      options.force_autotune = true;
    } else if (arg == "--help") {
      std::cout << "Usage: aw_cublaslt_autotune [--m M] [--n N] [--k K] [--heuristics N] "
                   "[--warmup N] [--iterations N] [--workspace-mb N] [--transa N|T] "
                   "[--transb N|T] [--order col|row] [--epilogue none|relu] "
                   "[--cache-dir PATH] [--no-cache] [--force-autotune]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.heuristics <= 0 ||
      options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument("matrix dimensions, heuristic count and iterations must be positive");
  }
  if (options.order == CUBLASLT_ORDER_ROW && options.epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
    throw std::invalid_argument("CUDA 13 cuBLASLt does not support ReLU epilogue with row-major D");
  }
  return options;
}

__global__ void FillKernel(float* data, std::size_t n, float value) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = value;
  }
}

LayoutShape MakeShape(int logical_rows, int logical_cols, cublasOperation_t transpose,
                      cublasLtOrder_t order) {
  LayoutShape shape;
  shape.rows = static_cast<std::uint64_t>(transpose == CUBLAS_OP_N ? logical_rows : logical_cols);
  shape.cols = static_cast<std::uint64_t>(transpose == CUBLAS_OP_N ? logical_cols : logical_rows);
  shape.ld = static_cast<std::int64_t>(order == CUBLASLT_ORDER_COL ? shape.rows : shape.cols);
  shape.elements = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
  return shape;
}

cublasLtMatrixLayout_t CreateLayout(const LayoutShape& shape, cublasLtOrder_t order) {
  cublasLtMatrixLayout_t layout{};
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, shape.rows, shape.cols, shape.ld));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                                sizeof(order)));
  return layout;
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
  result->reduction_scheme =
      GetAlgoConfig<std::uint32_t>(result->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  result->cta_swizzling =
      GetAlgoConfig<std::uint32_t>(result->algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
  result->custom_option =
      GetAlgoConfig<std::uint32_t>(result->algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
}

std::string OpName(cublasOperation_t op) { return op == CUBLAS_OP_T ? "T" : "N"; }
std::string OrderName(cublasLtOrder_t order) { return order == CUBLASLT_ORDER_ROW ? "row" : "col"; }
std::string EpilogueName(cublasLtEpilogue_t epilogue) {
  return epilogue == CUBLASLT_EPILOGUE_RELU ? "relu" : "none";
}

std::filesystem::path CachePath(const Options& options, int sm, std::size_t cublaslt_version) {
  const std::uint64_t workspace_bytes = options.workspace_mb * 1024ull * 1024ull;
  std::string name = "sm" + std::to_string(sm) + "_v" + std::to_string(cublaslt_version) +
                     "_m" + std::to_string(options.m) + "_n" + std::to_string(options.n) +
                     "_k" + std::to_string(options.k) + "_ta" + OpName(options.transa) +
                     "_tb" + OpName(options.transb) + "_" + OrderName(options.order) +
                     "_" + EpilogueName(options.epilogue) + "_ws" +
                     std::to_string(workspace_bytes) + ".bin";
  return std::filesystem::path(options.cache_dir) / name;
}

bool LoadCachedAlgo(const std::filesystem::path& path, std::size_t cublaslt_version,
                    cublasLtMatmulAlgo_t* algo, std::size_t* workspace_bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  char magic[8]{};
  std::uint64_t version = 0;
  std::uint64_t workspace = 0;
  std::uint32_t algo_size = 0;
  input.read(magic, sizeof(magic));
  input.read(reinterpret_cast<char*>(&version), sizeof(version));
  input.read(reinterpret_cast<char*>(&workspace), sizeof(workspace));
  input.read(reinterpret_cast<char*>(&algo_size), sizeof(algo_size));
  if (!input || std::string(magic, magic + 6) != "AWLTC1" ||
      version != static_cast<std::uint64_t>(cublaslt_version) ||
      algo_size != sizeof(cublasLtMatmulAlgo_t)) {
    return false;
  }
  input.read(reinterpret_cast<char*>(algo), sizeof(*algo));
  if (!input) {
    return false;
  }
  *workspace_bytes = static_cast<std::size_t>(workspace);
  return true;
}

void SaveCachedAlgo(const std::filesystem::path& path, std::size_t cublaslt_version,
                    const CandidateResult& result) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("Unable to open autotune cache for writing: " + path.string());
  }
  char magic[8] = {'A', 'W', 'L', 'T', 'C', '1', '\0', '\0'};
  const std::uint64_t version = static_cast<std::uint64_t>(cublaslt_version);
  const std::uint64_t workspace = static_cast<std::uint64_t>(result.workspace_bytes);
  const std::uint32_t algo_size = static_cast<std::uint32_t>(sizeof(cublasLtMatmulAlgo_t));
  output.write(magic, sizeof(magic));
  output.write(reinterpret_cast<const char*>(&version), sizeof(version));
  output.write(reinterpret_cast<const char*>(&workspace), sizeof(workspace));
  output.write(reinterpret_cast<const char*>(&algo_size), sizeof(algo_size));
  output.write(reinterpret_cast<const char*>(&result.algo), sizeof(result.algo));
  if (!output) {
    throw std::runtime_error("Failed to write autotune cache: " + path.string());
  }
}

CandidateResult BenchmarkAlgo(cublasLtHandle_t lt, cublasLtMatmulDesc_t operation,
                              cublasLtMatrixLayout_t a_desc, cublasLtMatrixLayout_t b_desc,
                              cublasLtMatrixLayout_t c_desc, cublasLtMatrixLayout_t d_desc,
                              const float* a, const float* b, float* c, const cublasLtMatmulAlgo_t& algo,
                              std::size_t workspace_bytes, void* workspace, cudaStream_t stream,
                              const Options& options, float alpha, float beta, bool from_cache,
                              float waves_count) {
  CandidateResult result;
  result.algo = algo;
  result.workspace_bytes = workspace_bytes;
  result.from_cache = from_cache;
  result.waves_count = waves_count;
  PopulateAlgoMetadata(&result);

  for (int i = 0; i < options.warmup; ++i) {
    const cublasStatus_t status = cublasLtMatmul(
        lt, operation, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, c, d_desc, &result.algo,
        workspace, workspace_bytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      result.launch_status = status;
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
        lt, operation, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, c, d_desc, &result.algo,
        workspace, workspace_bytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      result.launch_status = status;
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
  result.valid = true;
  return result;
}

double BenchmarkLegacy(cublasHandle_t handle, const float* a, const float* b, float* c,
                       const LayoutShape& a_shape, const LayoutShape& b_shape, int ldc,
                       cudaStream_t stream, const Options& options, float alpha, float beta) {
  for (int i = 0; i < options.warmup; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle, options.transa, options.transb, options.m, options.n,
                              options.k, &alpha, a, CUDA_R_32F, static_cast<int>(a_shape.ld), b,
                              CUDA_R_32F, static_cast<int>(b_shape.ld), &beta, c, CUDA_R_32F, ldc,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < options.iterations; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle, options.transa, options.transb, options.m, options.n,
                              options.k, &alpha, a, CUDA_R_32F, static_cast<int>(a_shape.ld), b,
                              CUDA_R_32F, static_cast<int>(b_shape.ld), &beta, c, CUDA_R_32F, ldc,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / options.iterations;
}

double ValidateOutput(const float* d_output, std::size_t elements, double expected) {
  std::vector<float> output(elements);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, elements * sizeof(float), cudaMemcpyDeviceToHost));
  double max_abs_error = 0.0;
  for (float value : output) {
    max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(value) - expected));
  }
  return max_abs_error;
}

void PrintCandidate(int index, const CandidateResult& candidate) {
  std::cout << "Candidate " << index << " algo id: " << candidate.algo_id << '\n';
  std::cout << "Candidate " << index << " tile id: " << candidate.tile_id << '\n';
  std::cout << "Candidate " << index << " stages id: " << candidate.stages_id << '\n';
  std::cout << "Candidate " << index << " splitK: " << candidate.splitk << '\n';
  std::cout << "Candidate " << index << " workspace: " << candidate.workspace_bytes << " bytes\n";
  std::cout << "Candidate " << index << " waves count: " << candidate.waves_count << " waves\n";
  std::cout << "Candidate " << index << " average time: " << candidate.average_ms << " ms\n";
  std::cout << "Candidate " << index << " throughput: " << candidate.gflops << " GFLOP/s\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::uint64_t workspace_limit = options.workspace_mb * 1024ull * 1024ull;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int sm = properties.major * 10 + properties.minor;
    const std::size_t cublaslt_version = cublasLtGetVersion();

    const LayoutShape a_shape = MakeShape(options.m, options.k, options.transa, options.order);
    const LayoutShape b_shape = MakeShape(options.k, options.n, options.transb, options.order);
    LayoutShape c_shape;
    c_shape.rows = static_cast<std::uint64_t>(options.m);
    c_shape.cols = static_cast<std::uint64_t>(options.n);
    c_shape.ld = options.order == CUBLASLT_ORDER_COL ? options.m : options.n;
    c_shape.elements = static_cast<std::size_t>(options.m) * options.n;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), a_shape.elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), b_shape.elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), c_shape.elements * sizeof(float)));

    void* workspace = nullptr;
    if (workspace_limit > 0) {
      CUDA_CHECK(cudaMalloc(&workspace, static_cast<std::size_t>(workspace_limit)));
    }

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr int kThreads = 256;
    const float a_value = options.epilogue == CUBLASLT_EPILOGUE_RELU ? -0.5f : 0.5f;
    const float b_value = 0.25f;
    FillKernel<<<static_cast<int>((a_shape.elements + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        d_a, a_shape.elements, a_value);
    FillKernel<<<static_cast<int>((b_shape.elements + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        d_b, b_shape.elements, b_value);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cublasLtHandle_t lt{};
    CUBLAS_CHECK(cublasLtCreate(&lt));
    cublasHandle_t legacy{};
    CUBLAS_CHECK(cublasCreate(&legacy));
    CUBLAS_CHECK(cublasSetStream(legacy, stream));

    cublasLtMatmulDesc_t operation{};
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &options.transa, sizeof(options.transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &options.transb, sizeof(options.transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &options.epilogue, sizeof(options.epilogue)));

    cublasLtMatrixLayout_t a_desc = CreateLayout(a_shape, options.order);
    cublasLtMatrixLayout_t b_desc = CreateLayout(b_shape, options.order);
    cublasLtMatrixLayout_t c_desc = CreateLayout(c_shape, options.order);
    cublasLtMatrixLayout_t d_desc = CreateLayout(c_shape, options.order);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double expected_unfused = static_cast<double>(options.k) * a_value * b_value;
    const double expected = options.epilogue == CUBLASLT_EPILOGUE_RELU
                                ? std::max(0.0, expected_unfused)
                                : expected_unfused;

    double legacy_ms = 0.0;
    double legacy_gflops = 0.0;
    const bool legacy_comparable =
        options.order == CUBLASLT_ORDER_COL && options.epilogue == CUBLASLT_EPILOGUE_DEFAULT;
    if (legacy_comparable) {
      legacy_ms = BenchmarkLegacy(legacy, d_a, d_b, d_c, a_shape, b_shape,
                                  static_cast<int>(c_shape.ld), stream, options, alpha, beta);
      const double legacy_flops = 2.0 * static_cast<double>(options.m) * options.n * options.k;
      legacy_gflops = legacy_flops / (legacy_ms / 1000.0) / 1.0e9;
    }

    const std::filesystem::path cache_path = CachePath(options, sm, cublaslt_version);
    CandidateResult best;
    bool cache_hit = false;
    int returned_candidates = 0;
    int benchmarked_candidates = 0;
    double heuristic_query_us = 0.0;
    double heuristic_first_gflops = 0.0;

    if (!options.no_cache && !options.force_autotune) {
      cublasLtMatmulAlgo_t cached_algo{};
      std::size_t cached_workspace = 0;
      if (LoadCachedAlgo(cache_path, cublaslt_version, &cached_algo, &cached_workspace) &&
          cached_workspace <= workspace_limit) {
        cublasLtMatmulHeuristicResult_t check{};
        const cublasStatus_t check_status = cublasLtMatmulAlgoCheck(
            lt, operation, a_desc, b_desc, c_desc, d_desc, &cached_algo, &check);
        if (check_status == CUBLAS_STATUS_SUCCESS && check.state == CUBLAS_STATUS_SUCCESS &&
            check.workspaceSize <= workspace_limit) {
          best = BenchmarkAlgo(lt, operation, a_desc, b_desc, c_desc, d_desc, d_a, d_b, d_c,
                               cached_algo, check.workspaceSize, workspace, stream, options, alpha,
                               beta, true, check.wavesCount);
          cache_hit = best.valid;
        }
      }
    }

    if (!cache_hit) {
      cublasLtMatmulPreference_t preference{};
      CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
      CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit,
          sizeof(workspace_limit)));

      std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
          static_cast<std::size_t>(options.heuristics));
      const auto query_start = std::chrono::steady_clock::now();
      CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
          lt, operation, a_desc, b_desc, c_desc, d_desc, preference, options.heuristics,
          heuristics.data(), &returned_candidates));
      const auto query_stop = std::chrono::steady_clock::now();
      heuristic_query_us =
          std::chrono::duration<double, std::micro>(query_stop - query_start).count();

      for (int i = 0; i < returned_candidates; ++i) {
        const auto& heuristic = heuristics[static_cast<std::size_t>(i)];
        if (heuristic.state != CUBLAS_STATUS_SUCCESS || heuristic.workspaceSize > workspace_limit) {
          continue;
        }
        CandidateResult candidate = BenchmarkAlgo(
            lt, operation, a_desc, b_desc, c_desc, d_desc, d_a, d_b, d_c, heuristic.algo,
            heuristic.workspaceSize, workspace, stream, options, alpha, beta, false,
            heuristic.wavesCount);
        if (!candidate.valid) {
          continue;
        }
        ++benchmarked_candidates;
        if (heuristic_first_gflops == 0.0) {
          heuristic_first_gflops = candidate.gflops;
        }
        PrintCandidate(i, candidate);
        if (!best.valid || candidate.average_ms < best.average_ms) {
          best = candidate;
        }
      }
      CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));

      if (!best.valid) {
        throw std::runtime_error("No cuBLASLt heuristic candidate successfully executed");
      }
      if (!options.no_cache) {
        SaveCachedAlgo(cache_path, cublaslt_version, best);
      }
    }

    const double max_abs_error = ValidateOutput(d_c, c_shape.elements, expected);
    const bool success = max_abs_error <= 1.0e-3;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "cuBLASLt Heuristic / Autotune Lab\n";
    std::cout << "M: " << options.m << '\n';
    std::cout << "N: " << options.n << '\n';
    std::cout << "K: " << options.k << '\n';
    std::cout << "Workspace budget: " << workspace_limit << " bytes\n";
    std::cout << "Heuristic request count: " << options.heuristics << " candidates\n";
    std::cout << "Heuristic candidates returned: " << returned_candidates << " candidates\n";
    std::cout << "Heuristic candidates benchmarked: " << benchmarked_candidates << " candidates\n";
    std::cout << "Heuristic query latency: " << heuristic_query_us << " us\n";
    std::cout << "Cache hit: " << (cache_hit ? 1 : 0) << " bool\n";
    std::cout << "Best algo id: " << best.algo_id << '\n';
    std::cout << "Best tile id: " << best.tile_id << '\n';
    std::cout << "Best stages id: " << best.stages_id << '\n';
    std::cout << "Best splitK: " << best.splitk << '\n';
    std::cout << "Best workspace: " << best.workspace_bytes << " bytes\n";
    std::cout << "Best waves count: " << best.waves_count << " waves\n";
    std::cout << "Best cuBLASLt average time: " << best.average_ms << " ms\n";
    std::cout << "Best cuBLASLt throughput: " << best.gflops << " GFLOP/s\n";
    if (heuristic_first_gflops > 0.0) {
      std::cout << "Autotune speedup vs heuristic first: " << best.gflops / heuristic_first_gflops
                << " x\n";
    }
    if (legacy_comparable) {
      std::cout << "Legacy cuBLAS average time: " << legacy_ms << " ms\n";
      std::cout << "Legacy cuBLAS throughput: " << legacy_gflops << " GFLOP/s\n";
      std::cout << "cuBLASLt speedup vs legacy: " << best.gflops / legacy_gflops << " x\n";
    }
    std::cout << "Max abs error: " << max_abs_error << '\n';
    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << '\n';

    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
    CUBLAS_CHECK(cublasDestroy(legacy));
    CUBLAS_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (workspace != nullptr) {
      CUDA_CHECK(cudaFree(workspace));
    }
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
    return success ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
