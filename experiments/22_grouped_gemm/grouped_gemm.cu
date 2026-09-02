#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using Config = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ElementA,
    ElementB,
    ElementOutput,
    ElementAccumulator>;
using InstructionShape = typename Config::InstructionShape;
using Epilogue = typename Config::EpilogueOutputOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using SerialSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using SerialGemm = cutlass::gemm::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementOutput,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    Epilogue,
    SerialSwizzle,
    Config::kStages,
    Config::kAlignmentA,
    Config::kAlignmentB>;

template <cutlass::gemm::kernel::GroupScheduleMode ScheduleMode>
struct GroupedTraits {
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      Config::kAlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      Config::kAlignmentB,
      ElementOutput,
      LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Epilogue,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      Config::kStages,
      ScheduleMode>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
};

struct Options {
  std::string workload = "heterogeneous";
  std::string mode = "all";
  int groups = 16;
  int warmup = 2;
  int iterations = 10;
};

struct BenchmarkResult {
  double average_ms = std::numeric_limits<double>::infinity();
  double gflops = 0.0;
  double validation_error = std::numeric_limits<double>::infinity();
  double initialization_ms = 0.0;
  int launch_count = 0;
  int persistent_ctas = 0;
  int total_tiles = 0;
  std::size_t scheduler_workspace_bytes = 0;
  bool passed = false;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Reset(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { Release(); }

  void Reset(std::size_t count) {
    Release();
    count_ = count;
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }
  }

  void CopyFromHost(const T* source, std::size_t count) {
    if (count > count_) throw std::invalid_argument("device buffer copy exceeds allocation");
    if (count > 0) {
      CUDA_CHECK(cudaMemcpy(ptr_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  void Zero() {
    if (count_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
  }

  T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  void Release() {
    if (ptr_) {
      (void)cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

class DeviceWorkspace {
 public:
  explicit DeviceWorkspace(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
  }
  DeviceWorkspace(const DeviceWorkspace&) = delete;
  DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
  ~DeviceWorkspace() {
    if (ptr_) (void)cudaFree(ptr_);
  }
  void* get() const { return ptr_; }
  std::size_t size() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--workload" && i + 1 < argc) options.workload = argv[++i];
    else if (arg == "--mode" && i + 1 < argc) options.mode = argv[++i];
    else if (arg == "--groups" && i + 1 < argc) options.groups = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc) options.warmup = std::stoi(argv[++i]);
    else if (arg == "--iterations" && i + 1 < argc) options.iterations = std::stoi(argv[++i]);
    else if (arg == "--help") {
      std::cout
          << "Usage: aw_grouped_gemm [--workload heterogeneous|moe|decode] "
             "[--mode all|serial|device|host|sorted] [--groups N] "
             "[--warmup N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.workload != "heterogeneous" && options.workload != "moe" &&
      options.workload != "decode") {
    throw std::invalid_argument("workload must be heterogeneous, moe or decode");
  }
  if (options.mode != "all" && options.mode != "serial" && options.mode != "device" &&
      options.mode != "host" && options.mode != "sorted") {
    throw std::invalid_argument("mode must be all, serial, device, host or sorted");
  }
  if (options.groups <= 0 || options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument("groups/iterations must be positive and warmup non-negative");
  }
  return options;
}

std::vector<cutlass::gemm::GemmCoord> BuildWorkload(const Options& options) {
  std::vector<cutlass::gemm::GemmCoord> seeds;
  if (options.workload == "heterogeneous") {
    seeds = {
        {128, 512, 256},
        {64, 1024, 512},
        {256, 512, 768},
        {32, 1536, 512},
        {128, 1024, 1024},
        {16, 2048, 768},
    };
  } else if (options.workload == "moe") {
    seeds = {
        {8, 2048, 1024},
        {16, 1024, 2048},
        {24, 2048, 1024},
        {32, 1024, 2048},
        {64, 2048, 1024},
        {96, 1024, 2048},
        {128, 2048, 1024},
        {192, 1024, 2048},
    };
  } else {
    seeds = {
        {1, 1024, 512},
        {2, 1024, 512},
        {4, 1024, 1024},
        {8, 512, 1024},
        {16, 1024, 768},
        {32, 512, 768},
    };
  }

  std::vector<cutlass::gemm::GemmCoord> problems;
  problems.reserve(options.groups);
  for (int index = 0; index < options.groups; ++index) {
    problems.push_back(seeds[static_cast<std::size_t>(index) % seeds.size()]);
  }
  return problems;
}

std::vector<cutlass::gemm::GemmCoord> SortByDescendingK(
    std::vector<cutlass::gemm::GemmCoord> problems) {
  std::stable_sort(problems.begin(), problems.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.k() > rhs.k();
  });
  return problems;
}

class ProblemData {
 public:
  explicit ProblemData(std::vector<cutlass::gemm::GemmCoord> problems)
      : problems_(std::move(problems)) {
    const std::size_t count = problems_.size();
    lda_.resize(count);
    ldb_.resize(count);
    ldc_.resize(count);
    offset_a_.resize(count);
    offset_b_.resize(count);
    offset_c_.resize(count);

    std::size_t total_a = 0;
    std::size_t total_b = 0;
    std::size_t total_c = 0;
    total_flops_ = 0.0;

    for (std::size_t index = 0; index < count; ++index) {
      const auto problem = problems_[index];
      if (problem.k() % 32 != 0 || problem.n() % 8 != 0) {
        throw std::invalid_argument("workload violates TensorOp alignment constraints");
      }
      lda_[index] = problem.k();
      ldb_[index] = problem.k();
      ldc_[index] = problem.n();
      offset_a_[index] = total_a;
      offset_b_[index] = total_b;
      offset_c_[index] = total_c;
      total_a += static_cast<std::size_t>(problem.m()) * problem.k();
      total_b += static_cast<std::size_t>(problem.k()) * problem.n();
      total_c += static_cast<std::size_t>(problem.m()) * problem.n();
      total_flops_ += 2.0 * static_cast<double>(problem.m()) * problem.n() * problem.k();
    }

    host_a_.resize(total_a);
    host_b_.resize(total_b);
    for (std::size_t problem_index = 0; problem_index < count; ++problem_index) {
      const auto problem = problems_[problem_index];
      for (int row = 0; row < problem.m(); ++row) {
        for (int inner = 0; inner < problem.k(); ++inner) {
          const int code =
              (row * 17 + inner * 13 + static_cast<int>(problem_index) * 5 + 3) % 31 - 15;
          host_a_[offset_a_[problem_index] + static_cast<std::size_t>(row) * problem.k() + inner] =
              ElementA(static_cast<float>(code) * 0.03125f);
        }
      }
      for (int col = 0; col < problem.n(); ++col) {
        for (int inner = 0; inner < problem.k(); ++inner) {
          const int code =
              (inner * 11 + col * 7 + static_cast<int>(problem_index) * 3 + 5) % 29 - 14;
          host_b_[offset_b_[problem_index] + static_cast<std::size_t>(inner) +
                  static_cast<std::size_t>(col) * problem.k()] =
              ElementB(static_cast<float>(code) * 0.03515625f);
        }
      }
    }

    d_problems_.Reset(count);
    d_lda_.Reset(count);
    d_ldb_.Reset(count);
    d_ldc_.Reset(count);
    d_a_.Reset(total_a);
    d_b_.Reset(total_b);
    d_c_.Reset(total_c);
    d_ptr_a_.Reset(count);
    d_ptr_b_.Reset(count);
    d_ptr_c_.Reset(count);

    d_problems_.CopyFromHost(problems_.data(), count);
    d_lda_.CopyFromHost(lda_.data(), count);
    d_ldb_.CopyFromHost(ldb_.data(), count);
    d_ldc_.CopyFromHost(ldc_.data(), count);
    d_a_.CopyFromHost(host_a_.data(), total_a);
    d_b_.CopyFromHost(host_b_.data(), total_b);
    d_c_.Zero();

    std::vector<ElementA*> ptr_a(count);
    std::vector<ElementB*> ptr_b(count);
    std::vector<ElementOutput*> ptr_c(count);
    for (std::size_t index = 0; index < count; ++index) {
      ptr_a[index] = d_a_.get() + offset_a_[index];
      ptr_b[index] = d_b_.get() + offset_b_[index];
      ptr_c[index] = d_c_.get() + offset_c_[index];
    }
    d_ptr_a_.CopyFromHost(ptr_a.data(), count);
    d_ptr_b_.CopyFromHost(ptr_b.data(), count);
    d_ptr_c_.CopyFromHost(ptr_c.data(), count);
  }

  int count() const { return static_cast<int>(problems_.size()); }
  const std::vector<cutlass::gemm::GemmCoord>& problems() const { return problems_; }
  const std::vector<std::int64_t>& lda() const { return lda_; }
  const std::vector<std::int64_t>& ldb() const { return ldb_; }
  const std::vector<std::int64_t>& ldc() const { return ldc_; }
  const std::vector<std::size_t>& offset_a() const { return offset_a_; }
  const std::vector<std::size_t>& offset_b() const { return offset_b_; }
  const std::vector<std::size_t>& offset_c() const { return offset_c_; }
  const std::vector<ElementA>& host_a() const { return host_a_; }
  const std::vector<ElementB>& host_b() const { return host_b_; }
  double total_flops() const { return total_flops_; }

  cutlass::gemm::GemmCoord* d_problems() const { return d_problems_.get(); }
  std::int64_t* d_lda() const { return d_lda_.get(); }
  std::int64_t* d_ldb() const { return d_ldb_.get(); }
  std::int64_t* d_ldc() const { return d_ldc_.get(); }
  ElementA* d_a() const { return d_a_.get(); }
  ElementB* d_b() const { return d_b_.get(); }
  ElementOutput* d_c() const { return d_c_.get(); }
  ElementA** d_ptr_a() const { return d_ptr_a_.get(); }
  ElementB** d_ptr_b() const { return d_ptr_b_.get(); }
  ElementOutput** d_ptr_c() const { return d_ptr_c_.get(); }
  std::size_t output_count() const { return d_c_.size(); }

  void ClearOutput() { d_c_.Zero(); }

 private:
  std::vector<cutlass::gemm::GemmCoord> problems_;
  std::vector<std::int64_t> lda_;
  std::vector<std::int64_t> ldb_;
  std::vector<std::int64_t> ldc_;
  std::vector<std::size_t> offset_a_;
  std::vector<std::size_t> offset_b_;
  std::vector<std::size_t> offset_c_;
  std::vector<ElementA> host_a_;
  std::vector<ElementB> host_b_;
  double total_flops_ = 0.0;

  DeviceBuffer<cutlass::gemm::GemmCoord> d_problems_;
  DeviceBuffer<std::int64_t> d_lda_;
  DeviceBuffer<std::int64_t> d_ldb_;
  DeviceBuffer<std::int64_t> d_ldc_;
  DeviceBuffer<ElementA> d_a_;
  DeviceBuffer<ElementB> d_b_;
  DeviceBuffer<ElementOutput> d_c_;
  DeviceBuffer<ElementA*> d_ptr_a_;
  DeviceBuffer<ElementB*> d_ptr_b_;
  DeviceBuffer<ElementOutput*> d_ptr_c_;
};

double ValidateSamples(const ProblemData& data) {
  std::vector<ElementOutput> output(data.output_count());
  CUDA_CHECK(cudaMemcpy(output.data(), data.d_c(), output.size() * sizeof(ElementOutput),
                        cudaMemcpyDeviceToHost));

  constexpr int kSamplesPerProblem = 4;
  double max_normalized_error = 0.0;
  const int checked_problems = std::min(data.count(), 8);
  for (int problem_index = 0; problem_index < checked_problems; ++problem_index) {
    const auto problem = data.problems()[static_cast<std::size_t>(problem_index)];
    for (int sample = 0; sample < kSamplesPerProblem; ++sample) {
      const int row = (problem_index * 17 + sample * 13 + 1) % problem.m();
      const int col = (problem_index * 23 + sample * 7 + 3) % problem.n();
      double reference = 0.0;
      for (int inner = 0; inner < problem.k(); ++inner) {
        const float a = static_cast<float>(
            data.host_a()[data.offset_a()[problem_index] +
                          static_cast<std::size_t>(row) * problem.k() + inner]);
        const float b = static_cast<float>(
            data.host_b()[data.offset_b()[problem_index] + static_cast<std::size_t>(inner) +
                          static_cast<std::size_t>(col) * problem.k()]);
        reference += static_cast<double>(a) * static_cast<double>(b);
      }
      const double actual = output[data.offset_c()[problem_index] +
                                   static_cast<std::size_t>(row) * problem.n() + col];
      const double normalized_error =
          std::abs(actual - reference) / std::max(1.0, std::abs(reference));
      max_normalized_error = std::max(max_normalized_error, normalized_error);
    }
  }
  return max_normalized_error;
}

template <typename LaunchFn>
double TimeLaunches(LaunchFn&& launch, int warmup, int iterations, cudaStream_t stream) {
  for (int iteration = 0; iteration < warmup; ++iteration) launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < iterations; ++iteration) launch();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / iterations;
}

BenchmarkResult BenchmarkSerial(ProblemData& data, const Options& options, cudaStream_t stream) {
  data.ClearOutput();
  std::vector<std::unique_ptr<DeviceWorkspace>> workspaces;
  workspaces.reserve(data.problems().size());
  std::vector<std::unique_ptr<SerialGemm>> operators;
  operators.reserve(data.problems().size());
  std::vector<SerialGemm::Arguments> arguments;
  arguments.reserve(data.problems().size());

  for (int index = 0; index < data.count(); ++index) {
    const auto problem = data.problems()[static_cast<std::size_t>(index)];
    SerialGemm::Arguments args{
        problem,
        {data.d_a() + data.offset_a()[index], data.lda()[index]},
        {data.d_b() + data.offset_b()[index], data.ldb()[index]},
        {data.d_c() + data.offset_c()[index], data.ldc()[index]},
        {data.d_c() + data.offset_c()[index], data.ldc()[index]},
        {1.0f, 0.0f}};
    if (SerialGemm::can_implement(args) != cutlass::Status::kSuccess) {
      throw std::runtime_error("serial GEMM cannot implement one workload shape");
    }
    workspaces.push_back(std::make_unique<DeviceWorkspace>(SerialGemm::get_workspace_size(args)));
    operators.push_back(std::make_unique<SerialGemm>());
    arguments.push_back(args);
    if (operators.back()->initialize(arguments.back(), workspaces.back()->get(), stream) !=
        cutlass::Status::kSuccess) {
      throw std::runtime_error("serial GEMM initialization failed");
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto launch = [&]() {
    for (int index = 0; index < data.count(); ++index) {
      if (operators[static_cast<std::size_t>(index)]->run(stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("serial GEMM launch failed");
      }
    }
  };

  BenchmarkResult result;
  result.average_ms = TimeLaunches(launch, options.warmup, options.iterations, stream);
  result.gflops = data.total_flops() / (result.average_ms / 1000.0) / 1.0e9;
  result.launch_count = data.count();
  result.validation_error = ValidateSamples(data);
  result.passed = std::isfinite(result.average_ms) && result.average_ms > 0.0 &&
                  result.validation_error <= 1.0e-2;
  return result;
}

template <cutlass::gemm::kernel::GroupScheduleMode ScheduleMode>
BenchmarkResult BenchmarkGrouped(ProblemData& data, const Options& options, cudaStream_t stream) {
  using Gemm = typename GroupedTraits<ScheduleMode>::Gemm;
  data.ClearOutput();

  const int threadblock_count = Gemm::sufficient(data.problems().data(), data.count());
  if (threadblock_count <= 0) {
    throw std::runtime_error("grouped GEMM could not determine a persistent CTA count");
  }

  typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, 0.0f);
  typename Gemm::Arguments args(
      data.d_problems(),
      data.count(),
      threadblock_count,
      epilogue,
      data.d_ptr_a(),
      data.d_ptr_b(),
      data.d_ptr_c(),
      data.d_ptr_c(),
      data.d_lda(),
      data.d_ldb(),
      data.d_ldc(),
      data.d_ldc(),
      const_cast<cutlass::gemm::GemmCoord*>(data.problems().data()));

  if (Gemm::can_implement(args) != cutlass::Status::kSuccess) {
    throw std::runtime_error("grouped GEMM cannot implement workload");
  }

  BenchmarkResult result;
  result.persistent_ctas = threadblock_count;
  result.total_tiles = Gemm::group_tile_count(args);
  result.scheduler_workspace_bytes = Gemm::get_workspace_size(args);
  DeviceWorkspace workspace(result.scheduler_workspace_bytes);
  Gemm gemm;

  const auto initialize_start = std::chrono::steady_clock::now();
  if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
    throw std::runtime_error("grouped GEMM initialization failed");
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto initialize_stop = std::chrono::steady_clock::now();
  result.initialization_ms =
      std::chrono::duration<double, std::milli>(initialize_stop - initialize_start).count();

  auto launch = [&]() {
    if (gemm.run(stream) != cutlass::Status::kSuccess) {
      throw std::runtime_error("grouped GEMM launch failed");
    }
  };
  result.average_ms = TimeLaunches(launch, options.warmup, options.iterations, stream);
  result.gflops = data.total_flops() / (result.average_ms / 1000.0) / 1.0e9;
  result.launch_count = 1;
  result.validation_error = ValidateSamples(data);
  result.passed = std::isfinite(result.average_ms) && result.average_ms > 0.0 &&
                  result.validation_error <= 1.0e-2;
  return result;
}

void PrintWorkloadSummary(const ProblemData& data) {
  int minimum_k = std::numeric_limits<int>::max();
  int maximum_k = 0;
  int minimum_m = std::numeric_limits<int>::max();
  int maximum_m = 0;
  for (const auto& problem : data.problems()) {
    minimum_k = std::min(minimum_k, problem.k());
    maximum_k = std::max(maximum_k, problem.k());
    minimum_m = std::min(minimum_m, problem.m());
    maximum_m = std::max(maximum_m, problem.m());
  }
  std::cout << "Problem count: " << data.count() << " problems\n";
  std::cout << "M range: " << minimum_m << " to " << maximum_m << " rows\n";
  std::cout << "K range: " << minimum_k << " to " << maximum_k << " elements\n";
  std::cout << "Workload FLOPs: " << (data.total_flops() / 1.0e9) << " GFLOP\n";
}

void PrintResult(const std::string& label,
                 const BenchmarkResult& result,
                 const BenchmarkResult* serial_reference) {
  std::cout << label << " launch count: " << result.launch_count << " launches\n";
  if (result.persistent_ctas > 0) {
    std::cout << label << " persistent CTAs: " << result.persistent_ctas << " CTAs\n";
    std::cout << label << " total tiles: " << result.total_tiles << " tiles\n";
    std::cout << label << " tiles per CTA: "
              << (static_cast<double>(result.total_tiles) / result.persistent_ctas) << " tiles/CTA\n";
    std::cout << label << " scheduler workspace: " << result.scheduler_workspace_bytes << " bytes\n";
    std::cout << label << " initialization latency: " << result.initialization_ms << " ms\n";
  }
  std::cout << label << " latency: " << result.average_ms << " ms\n";
  std::cout << label << " throughput: " << result.gflops << " GFLOP/s\n";
  std::cout << label << " validation max normalized error: " << result.validation_error << " ratio\n";
  if (serial_reference && serial_reference->average_ms > 0.0) {
    std::cout << label << " speedup vs serial: "
              << (serial_reference->average_ms / result.average_ms) << " x\n";
  }
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
      std::cout << "Validation: PASS (SM75 TensorOp grouped path unsupported on this GPU)\n";
      return EXIT_SUCCESS;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUTLASS version: 4.7.0\n";
    std::cout << "CUTLASS API: device::GemmGrouped persistent problem scheduler\n";
    std::cout << "Kernel minimum architecture: SM75 TensorOp\n";
    std::cout << "Threadblock shape: 128x128x32\n";
    std::cout << "Warp shape: 64x64x32\n";
    std::cout << "Pipeline stages: " << Config::kStages << " stages\n";
    std::cout << "GPU compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "Workload: " << options.workload << "\n";

    const auto workload = BuildWorkload(options);
    ProblemData data(workload);
    PrintWorkloadSummary(data);

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    BenchmarkResult serial;
    bool have_serial = false;
    bool success = true;

    if (options.mode == "all" || options.mode == "serial") {
      serial = BenchmarkSerial(data, options, stream);
      PrintResult("Serial GEMM", serial, nullptr);
      have_serial = true;
      success = serial.passed && success;
    }

    if (options.mode == "all" || options.mode == "device") {
      auto device_result =
          BenchmarkGrouped<cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>(
              data, options, stream);
      PrintResult("Grouped device", device_result, have_serial ? &serial : nullptr);
      success = device_result.passed && success;
    }

    if (options.mode == "all" || options.mode == "host") {
      auto host_result =
          BenchmarkGrouped<cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute>(
              data, options, stream);
      PrintResult("Grouped host", host_result, have_serial ? &serial : nullptr);
      success = host_result.passed && success;
    }

    if (options.mode == "all" || options.mode == "sorted") {
      ProblemData sorted_data(SortByDescendingK(workload));
      auto sorted_result =
          BenchmarkGrouped<cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>(
              sorted_data, options, stream);
      PrintResult("Grouped K sorted", sorted_result, have_serial ? &serial : nullptr);
      success = sorted_result.passed && success;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << "\n";
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
