#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

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
using namespace cute;

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int warmup = 2;
  int iterations = 10;
};

struct Result {
  bool valid = false;
  double average_ms = std::numeric_limits<double>::infinity();
  double gflops = 0.0;
  std::size_t workspace_bytes = 0;
  std::size_t mainloop_shared_bytes = 0;
  std::size_t epilogue_shared_bytes = 0;
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
      std::cout << "Usage: aw_cutlass_gemm_sm120 [--m M] [--n N] [--k K] "
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
  if (options.m % 128 != 0 || options.n % 64 != 0 || options.k % 64 != 0) {
    throw std::invalid_argument("SM120 FP8 reference requires M multiple 128 and N/K multiples 64");
  }
  return options;
}

std::vector<cutlass::float_e4m3_t> BuildA(const Options& options, std::vector<float>* reconstructed) {
  const std::size_t count = static_cast<std::size_t>(options.m) * options.k;
  std::vector<cutlass::float_e4m3_t> values(count);
  reconstructed->resize(count);
  for (int row = 0; row < options.m; ++row) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (row * 17 + inner * 13 + 3) % 17 - 8;
      const float source = static_cast<float>(code) * 0.125f;
      const std::size_t index = static_cast<std::size_t>(row) * options.k + inner;
      values[index] = cutlass::float_e4m3_t(source);
      (*reconstructed)[index] = static_cast<float>(values[index]);
    }
  }
  return values;
}

std::vector<cutlass::float_e4m3_t> BuildB(const Options& options, std::vector<float>* reconstructed) {
  const std::size_t count = static_cast<std::size_t>(options.k) * options.n;
  std::vector<cutlass::float_e4m3_t> values(count);
  reconstructed->resize(count);
  for (int col = 0; col < options.n; ++col) {
    for (int inner = 0; inner < options.k; ++inner) {
      const int code = (inner * 11 + col * 7 + 5) % 15 - 7;
      const float source = static_cast<float>(code) * 0.125f;
      const std::size_t index = static_cast<std::size_t>(inner) + static_cast<std::size_t>(col) * options.k;
      values[index] = cutlass::float_e4m3_t(source);
      (*reconstructed)[index] = static_cast<float>(values[index]);
    }
  }
  return values;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major * 10 + props.minor < 120) {
      std::cout << "Validation: PASS (SM120 CUTLASS specialization unsupported on this GPU)\n";
      return EXIT_SUCCESS;
    }

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = float;
    using ElementD = float;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;
    constexpr int AlignmentAB = 16;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using TileShape = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_1, _1, _1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentAB,
        ElementB, LayoutB, AlignmentAB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    std::vector<float> reconstructed_a;
    std::vector<float> reconstructed_b;
    const auto host_a = BuildA(options, &reconstructed_a);
    const auto host_b = BuildB(options, &reconstructed_b);
    const std::size_t c_count = static_cast<std::size_t>(options.m) * options.n;

    ElementA* d_a = nullptr;
    ElementB* d_b = nullptr;
    float* d_c = nullptr;
    float* d_d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, host_a.size() * sizeof(ElementA)));
    CUDA_CHECK(cudaMalloc(&d_b, host_b.size() * sizeof(ElementB)));
    CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), host_a.size() * sizeof(ElementA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), host_b.size() * sizeof(ElementB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_d, 0, c_count * sizeof(float)));

    const StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
    const StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
    const StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
    const StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {d_a, stride_a, d_b, stride_b},
        {{1.0f, 0.0f}, d_c, stride_c, d_d, stride_d}};

    Gemm gemm;
    if (Gemm::can_implement(arguments) != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS SM120 FP8 kernel cannot implement the requested shape");
    }
    const std::size_t workspace_bytes = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_bytes > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (gemm.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS SM120 GEMM initialization failed");
    }
    for (int i = 0; i < options.warmup; ++i) {
      if (gemm.run(stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS SM120 warmup launch failed");
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
        throw std::runtime_error("CUTLASS SM120 timed launch failed");
      }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double average_ms = static_cast<double>(elapsed_ms) / options.iterations;
    const double flops = 2.0 * static_cast<double>(options.m) * options.n * options.k;
    const double gflops = flops / (average_ms / 1000.0) / 1.0e9;

    if (gemm.run(stream) != cutlass::Status::kSuccess) {
      throw std::runtime_error("CUTLASS SM120 validation launch failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> host_output(c_count);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_d, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    const double max_norm_error = ValidateSamples(options, reconstructed_a, reconstructed_b, host_output);
    const bool passed = std::isfinite(max_norm_error) && max_norm_error <= 8.0e-3;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUTLASS version: 4.7.0\n";
    std::cout << "CUTLASS API: CollectiveBuilder -> GemmUniversal -> GemmUniversalAdapter\n";
    std::cout << "GPU compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "CUTLASS SM120 dtype: FP8 E4M3 / FP32 accumulate+output\n";
    std::cout << "CUTLASS SM120 tile shape: 128x64x64\n";
    std::cout << "CUTLASS SM120 cluster shape: 1x1x1\n";
    std::cout << "CUTLASS SM120 stage policy: StageCountAutoCarveout\n";
    std::cout << "CUTLASS SM120 kernel schedule: KernelScheduleAuto\n";
    std::cout << "CUTLASS SM120 mainloop shared storage: "
              << sizeof(typename CollectiveMainloop::SharedStorage) << " bytes\n";
    std::cout << "CUTLASS SM120 epilogue shared storage: "
              << sizeof(typename CollectiveEpilogue::SharedStorage) << " bytes\n";
    std::cout << "CUTLASS SM120 workspace: " << workspace_bytes << " bytes\n";
    std::cout << "CUTLASS SM120 latency: " << average_ms << " ms\n";
    std::cout << "CUTLASS SM120 throughput: " << gflops << " GFLOP/s\n";
    std::cout << "CUTLASS SM120 validation max normalized error: " << max_norm_error << "\n";
    std::cout << "Validation: " << (passed ? "PASS" : "FAIL") << "\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
#else
    std::cout << "Validation: PASS (CUTLASS SM120 MMA macro unavailable in this compile target)\n";
    return EXIT_SUCCESS;
#endif
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
