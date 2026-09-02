#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <climits>
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

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;
constexpr unsigned int kFullWarpMask = 0xffffffffu;
constexpr int kWarpSize = 32;

enum class Distribution { kUniform, kHot, kSingle, kAll };

struct Options {
  std::size_t elements = 1u << 23;
  int bins = 256;
  int iterations = 20;
  int hot_percent = 90;
  Distribution distribution = Distribution::kAll;
};

struct Measurement {
  double milliseconds = 0.0;
  unsigned long long max_abs_error = 0;
};

Distribution ParseDistribution(const std::string& value) {
  if (value == "uniform") return Distribution::kUniform;
  if (value == "hot") return Distribution::kHot;
  if (value == "single") return Distribution::kSingle;
  if (value == "all") return Distribution::kAll;
  throw std::invalid_argument("distribution must be one of: uniform, hot, single, all");
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--elements" && i + 1 < argc) {
      options.elements = std::stoull(argv[++i]);
    } else if (arg == "--bins" && i + 1 < argc) {
      options.bins = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--hot-percent" && i + 1 < argc) {
      options.hot_percent = std::stoi(argv[++i]);
    } else if (arg == "--distribution" && i + 1 < argc) {
      options.distribution = ParseDistribution(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_histogram [--elements N] [--bins N] [--iterations N] "
                   "[--hot-percent 1..99] [--distribution uniform|hot|single|all]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.elements == 0 || options.elements > static_cast<std::size_t>(INT_MAX)) {
    throw std::invalid_argument("elements must be in [1, INT_MAX]");
  }
  if (options.bins < 2 || options.bins > 4096) {
    throw std::invalid_argument("bins must be in [2, 4096]");
  }
  if (options.iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (options.hot_percent <= 0 || options.hot_percent >= 100) {
    throw std::invalid_argument("hot-percent must be in [1, 99]");
  }
  return options;
}

std::uint32_t Mix(std::uint32_t x) {
  x += 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

std::string DistributionName(Distribution distribution) {
  switch (distribution) {
    case Distribution::kUniform:
      return "Uniform";
    case Distribution::kHot:
      return "Hot";
    case Distribution::kSingle:
      return "Single";
    case Distribution::kAll:
      break;
  }
  return "All";
}

std::vector<std::uint32_t> MakeInput(const Options& options, Distribution distribution) {
  std::vector<std::uint32_t> input(options.elements);
  for (std::size_t i = 0; i < input.size(); ++i) {
    const std::uint32_t mixed = Mix(static_cast<std::uint32_t>(i));
    if (distribution == Distribution::kUniform) {
      input[i] = mixed % static_cast<std::uint32_t>(options.bins);
    } else if (distribution == Distribution::kHot) {
      if (mixed % 100u < static_cast<std::uint32_t>(options.hot_percent)) {
        input[i] = 0u;
      } else {
        input[i] = 1u + Mix(mixed ^ 0xa5a5a5a5u) % static_cast<std::uint32_t>(options.bins - 1);
      }
    } else {
      input[i] = 0u;
    }
  }
  return input;
}

std::vector<std::uint32_t> ExpectedHistogram(const std::vector<std::uint32_t>& input,
                                             int bins) {
  std::vector<std::uint32_t> expected(static_cast<std::size_t>(bins), 0u);
  for (std::uint32_t value : input) {
    ++expected[value];
  }
  return expected;
}

__global__ void GlobalAtomicHistogramKernel(const std::uint32_t* input,
                                             std::uint32_t* histogram,
                                             std::size_t n) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  while (index < n) {
    atomicAdd(&histogram[input[index]], 1u);
    index += stride;
  }
}

__global__ void WarpAggregatedHistogramKernel(const std::uint32_t* input,
                                               std::uint32_t* histogram,
                                               std::size_t n) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  while (true) {
    const bool valid = index < n;
    const unsigned int active = __ballot_sync(kFullWarpMask, valid);
    if (active == 0u) {
      break;
    }
    if (valid) {
      const std::uint32_t bin = input[index];
      const unsigned int peers = __match_any_sync(active, bin);
      const int leader = __ffs(static_cast<int>(peers)) - 1;
      if (lane == leader) {
        atomicAdd(&histogram[bin], static_cast<unsigned int>(__popc(peers)));
      }
    }
    index += stride;
  }
}

__global__ void SharedPrivatizedHistogramKernel(const std::uint32_t* input,
                                                 std::uint32_t* histogram,
                                                 std::size_t n,
                                                 int bins) {
  extern __shared__ std::uint32_t local_histogram[];
  for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    local_histogram[bin] = 0u;
  }
  __syncthreads();

  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  while (index < n) {
    atomicAdd(&local_histogram[input[index]], 1u);
    index += stride;
  }
  __syncthreads();

  for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    const std::uint32_t count = local_histogram[bin];
    if (count != 0u) {
      atomicAdd(&histogram[bin], count);
    }
  }
}

__global__ void PerBlockHistogramKernel(const std::uint32_t* input,
                                         std::uint32_t* block_histograms,
                                         std::size_t n,
                                         int bins) {
  extern __shared__ std::uint32_t local_histogram[];
  for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    local_histogram[bin] = 0u;
  }
  __syncthreads();

  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  while (index < n) {
    atomicAdd(&local_histogram[input[index]], 1u);
    index += stride;
  }
  __syncthreads();

  std::uint32_t* block_output =
      block_histograms + static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(bins);
  for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
    block_output[bin] = local_histogram[bin];
  }
}

__global__ void MergeBlockHistogramsKernel(const std::uint32_t* block_histograms,
                                            std::uint32_t* histogram,
                                            int blocks,
                                            int bins) {
  for (int bin = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; bin < bins;
       bin += static_cast<int>(blockDim.x) * gridDim.x) {
    unsigned long long sum = 0;
    for (int block = 0; block < blocks; ++block) {
      sum += block_histograms[static_cast<std::size_t>(block) * static_cast<std::size_t>(bins) +
                              static_cast<std::size_t>(bin)];
    }
    histogram[bin] = static_cast<std::uint32_t>(sum);
  }
}

template <typename Launch>
double TimeAlgorithm(int iterations, Launch&& launch) {
  launch();
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    launch();
  }
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / iterations;
}

unsigned long long ValidateHistogram(const std::uint32_t* d_histogram,
                                     const std::vector<std::uint32_t>& expected) {
  std::vector<std::uint32_t> actual(expected.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), d_histogram, actual.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost));
  unsigned long long max_error = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const unsigned long long a = actual[i];
    const unsigned long long b = expected[i];
    const unsigned long long error = a > b ? a - b : b - a;
    max_error = std::max(max_error, error);
  }
  return max_error;
}

double ThroughputGItems(std::size_t elements, double milliseconds) {
  return static_cast<double>(elements) / (milliseconds * 1.0e6);
}

bool RunDistribution(const Options& options, Distribution distribution) {
  const std::vector<std::uint32_t> input = MakeInput(options, distribution);
  const std::vector<std::uint32_t> expected = ExpectedHistogram(input, options.bins);
  const std::size_t input_bytes = input.size() * sizeof(std::uint32_t);
  const std::size_t histogram_bytes = static_cast<std::size_t>(options.bins) * sizeof(std::uint32_t);
  const int blocks = std::min(
      static_cast<int>((options.elements + static_cast<std::size_t>(kThreads) - 1) / kThreads),
      kMaxBlocks);
  const std::size_t shared_bytes = histogram_bytes;
  const std::size_t block_histogram_bytes =
      static_cast<std::size_t>(blocks) * static_cast<std::size_t>(options.bins) * sizeof(std::uint32_t);

  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (shared_bytes > properties.sharedMemPerBlock) {
    throw std::runtime_error("requested bin count exceeds default per-block shared-memory capacity");
  }

  std::uint32_t* d_input = nullptr;
  std::uint32_t* d_histogram = nullptr;
  std::uint32_t* d_block_histograms = nullptr;
  void* d_cub_temp = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_histogram), histogram_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_histograms), block_histogram_bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

  std::size_t cub_temp_bytes = 0;
  const int num_levels = options.bins + 1;
  const std::uint32_t lower_level = 0u;
  const std::uint32_t upper_level = static_cast<std::uint32_t>(options.bins);
  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
      nullptr, cub_temp_bytes, d_input, d_histogram, num_levels, lower_level, upper_level,
      static_cast<int>(options.elements)));
  CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

  Measurement naive;
  naive.milliseconds = TimeAlgorithm(options.iterations, [&] {
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, histogram_bytes));
    GlobalAtomicHistogramKernel<<<blocks, kThreads>>>(d_input, d_histogram, options.elements);
  });
  naive.max_abs_error = ValidateHistogram(d_histogram, expected);

  Measurement warp;
  warp.milliseconds = TimeAlgorithm(options.iterations, [&] {
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, histogram_bytes));
    WarpAggregatedHistogramKernel<<<blocks, kThreads>>>(d_input, d_histogram, options.elements);
  });
  warp.max_abs_error = ValidateHistogram(d_histogram, expected);

  Measurement shared;
  shared.milliseconds = TimeAlgorithm(options.iterations, [&] {
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, histogram_bytes));
    SharedPrivatizedHistogramKernel<<<blocks, kThreads, shared_bytes>>>(
        d_input, d_histogram, options.elements, options.bins);
  });
  shared.max_abs_error = ValidateHistogram(d_histogram, expected);

  const int merge_blocks = std::max(1, std::min((options.bins + kThreads - 1) / kThreads, 32));
  Measurement two_pass;
  two_pass.milliseconds = TimeAlgorithm(options.iterations, [&] {
    PerBlockHistogramKernel<<<blocks, kThreads, shared_bytes>>>(
        d_input, d_block_histograms, options.elements, options.bins);
    MergeBlockHistogramsKernel<<<merge_blocks, kThreads>>>(
        d_block_histograms, d_histogram, blocks, options.bins);
  });
  two_pass.max_abs_error = ValidateHistogram(d_histogram, expected);

  Measurement cub;
  cub.milliseconds = TimeAlgorithm(options.iterations, [&] {
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(
        d_cub_temp, cub_temp_bytes, d_input, d_histogram, num_levels, lower_level, upper_level,
        static_cast<int>(options.elements)));
  });
  cub.max_abs_error = ValidateHistogram(d_histogram, expected);

  const unsigned long long max_error = std::max(
      {naive.max_abs_error, warp.max_abs_error, shared.max_abs_error, two_pass.max_abs_error,
       cub.max_abs_error});
  const std::string name = DistributionName(distribution);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << name << " naive global atomic: " << naive.milliseconds << " ms\n";
  std::cout << name << " warp aggregated atomic: " << warp.milliseconds << " ms\n";
  std::cout << name << " shared privatized: " << shared.milliseconds << " ms\n";
  std::cout << name << " two pass: " << two_pass.milliseconds << " ms\n";
  std::cout << name << " CUB DeviceHistogram: " << cub.milliseconds << " ms\n";
  std::cout << name << " naive throughput: " << ThroughputGItems(options.elements, naive.milliseconds)
            << " Gitems/s\n";
  std::cout << name << " warp aggregated throughput: "
            << ThroughputGItems(options.elements, warp.milliseconds) << " Gitems/s\n";
  std::cout << name << " shared privatized throughput: "
            << ThroughputGItems(options.elements, shared.milliseconds) << " Gitems/s\n";
  std::cout << name << " two pass throughput: "
            << ThroughputGItems(options.elements, two_pass.milliseconds) << " Gitems/s\n";
  std::cout << name << " CUB throughput: " << ThroughputGItems(options.elements, cub.milliseconds)
            << " Gitems/s\n";
  std::cout << name << " warp speedup vs naive: " << naive.milliseconds / warp.milliseconds
            << " x\n";
  std::cout << name << " shared speedup vs naive: " << naive.milliseconds / shared.milliseconds
            << " x\n";
  std::cout << name << " two pass speedup vs naive: " << naive.milliseconds / two_pass.milliseconds
            << " x\n";
  std::cout << name << " CUB speedup vs naive: " << naive.milliseconds / cub.milliseconds << " x\n";
  std::cout << name << " max abs error: " << max_error << " count\n";

  CUDA_CHECK(cudaFree(d_cub_temp));
  CUDA_CHECK(cudaFree(d_block_histograms));
  CUDA_CHECK(cudaFree(d_histogram));
  CUDA_CHECK(cudaFree(d_input));
  return max_error == 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    std::cout << "Histogram / Atomics Contention\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  Bins: " << options.bins << '\n';
    std::cout << "  Iterations: " << options.iterations << '\n';
    std::cout << "  Hot distribution percentage: " << options.hot_percent << "%\n";

    bool success = true;
    if (options.distribution == Distribution::kAll || options.distribution == Distribution::kUniform) {
      success = RunDistribution(options, Distribution::kUniform) && success;
    }
    if (options.distribution == Distribution::kAll || options.distribution == Distribution::kHot) {
      success = RunDistribution(options, Distribution::kHot) && success;
    }
    if (options.distribution == Distribution::kAll || options.distribution == Distribution::kSingle) {
      success = RunDistribution(options, Distribution::kSingle) && success;
    }

    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << '\n';
    return success ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
