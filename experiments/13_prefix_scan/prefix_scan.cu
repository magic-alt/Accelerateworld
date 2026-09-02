#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

struct Options {
  std::size_t elements = 1u << 22;
  int iterations = 10;
};

struct Measurement {
  double milliseconds = 0.0;
  bool valid = false;
  std::size_t first_bad_index = 0;
  std::uint32_t actual = 0;
  std::uint32_t expected = 0;
};

struct WorkspaceLevel {
  std::size_t block_count = 0;
  std::uint32_t* block_sums = nullptr;
  std::uint32_t* block_offsets = nullptr;
};

using Workspace = std::vector<WorkspaceLevel>;

enum class ScanKind {
  kShared,
  kWarp,
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--elements" && i + 1 < argc) {
      options.elements = std::stoull(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_prefix_scan [--elements N] [--iterations N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.elements == 0 || options.iterations <= 0) {
    throw std::invalid_argument("elements and iterations must be positive");
  }
  if (options.elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("elements must fit in a signed 32-bit CUB item count");
  }
  return options;
}

std::size_t DivUp(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

__global__ void FillOnesKernel(std::uint32_t* data, std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    data[index] = 1u;
  }
}

__global__ void SerialExclusiveScanKernel(const std::uint32_t* input, std::uint32_t* output,
                                          std::size_t n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  std::uint32_t running = 0;
  for (std::size_t index = 0; index < n; ++index) {
    output[index] = running;
    running += input[index];
  }
}

template <int kThreads>
__global__ void SharedBlockExclusiveScanKernel(const std::uint32_t* input,
                                                std::uint32_t* output,
                                                std::uint32_t* block_sums,
                                                std::size_t n) {
  static_assert((kThreads & (kThreads - 1)) == 0, "block size must be a power of two");
  __shared__ std::uint32_t values[kThreads];

  const int tid = threadIdx.x;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * kThreads + tid;
  const std::uint32_t x = index < n ? input[index] : 0u;
  values[tid] = x;
  __syncthreads();

  // Hillis-Steele inclusive scan. The first barrier in each iteration prevents
  // writers from racing with threads that still need the previous iteration's values.
  for (int offset = 1; offset < kThreads; offset <<= 1) {
    const std::uint32_t addend = tid >= offset ? values[tid - offset] : 0u;
    __syncthreads();
    if (tid >= offset) {
      values[tid] += addend;
    }
    __syncthreads();
  }

  if (index < n) {
    output[index] = tid == 0 ? 0u : values[tid - 1];
  }
  if (block_sums != nullptr && tid == kThreads - 1) {
    block_sums[blockIdx.x] = values[kThreads - 1];
  }
}

__device__ __forceinline__ std::uint32_t WarpInclusiveScan(std::uint32_t value, int lane) {
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const std::uint32_t from_lower = __shfl_up_sync(kFullWarpMask, value, offset);
    if (lane >= offset) {
      value += from_lower;
    }
  }
  return value;
}

template <int kThreads>
__global__ void WarpBlockExclusiveScanKernel(const std::uint32_t* input,
                                              std::uint32_t* output,
                                              std::uint32_t* block_sums,
                                              std::size_t n) {
  static_assert(kThreads % kWarpSize == 0, "block size must contain complete warps");
  constexpr int kWarpCount = kThreads / kWarpSize;
  __shared__ std::uint32_t warp_prefix[kWarpCount];

  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * kThreads + tid;
  const std::uint32_t x = index < n ? input[index] : 0u;

  const std::uint32_t warp_inclusive = WarpInclusiveScan(x, lane);
  if (lane == kWarpSize - 1) {
    warp_prefix[warp_id] = warp_inclusive;
  }
  __syncthreads();

  // The first warp scans one subtotal per warp, turning the warp primitive into
  // a complete block scan.
  if (warp_id == 0) {
    std::uint32_t subtotal = lane < kWarpCount ? warp_prefix[lane] : 0u;
    subtotal = WarpInclusiveScan(subtotal, lane);
    if (lane < kWarpCount) {
      warp_prefix[lane] = subtotal;
    }
  }
  __syncthreads();

  const std::uint32_t prior_warps = warp_id == 0 ? 0u : warp_prefix[warp_id - 1];
  if (index < n) {
    output[index] = prior_warps + warp_inclusive - x;
  }
  if (block_sums != nullptr && tid == kThreads - 1) {
    block_sums[blockIdx.x] = warp_prefix[kWarpCount - 1];
  }
}

__global__ void UniformAddKernel(std::uint32_t* output, const std::uint32_t* block_offsets,
                                 std::size_t n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] += block_offsets[blockIdx.x];
  }
}

Workspace AllocateWorkspace(std::size_t n) {
  Workspace workspace;
  std::size_t item_count = n;
  while (true) {
    const std::size_t block_count = DivUp(item_count, static_cast<std::size_t>(kBlockSize));
    if (block_count <= 1) {
      break;
    }

    WorkspaceLevel level;
    level.block_count = block_count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&level.block_sums),
                          block_count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&level.block_offsets),
                          block_count * sizeof(std::uint32_t)));
    workspace.push_back(level);
    item_count = block_count;
  }
  return workspace;
}

void FreeWorkspace(Workspace* workspace) {
  for (WorkspaceLevel& level : *workspace) {
    CUDA_CHECK(cudaFree(level.block_sums));
    CUDA_CHECK(cudaFree(level.block_offsets));
    level.block_sums = nullptr;
    level.block_offsets = nullptr;
  }
  workspace->clear();
}

void LaunchHierarchicalScan(ScanKind kind, const std::uint32_t* input, std::uint32_t* output,
                            std::size_t n, Workspace& workspace, std::size_t depth = 0) {
  const std::size_t block_count = DivUp(n, static_cast<std::size_t>(kBlockSize));
  std::uint32_t* block_sums = block_count > 1 ? workspace.at(depth).block_sums : nullptr;

  if (kind == ScanKind::kShared) {
    SharedBlockExclusiveScanKernel<kBlockSize>
        <<<static_cast<unsigned int>(block_count), kBlockSize>>>(input, output, block_sums, n);
  } else {
    WarpBlockExclusiveScanKernel<kBlockSize>
        <<<static_cast<unsigned int>(block_count), kBlockSize>>>(input, output, block_sums, n);
  }

  if (block_count > 1) {
    WorkspaceLevel& level = workspace.at(depth);
    LaunchHierarchicalScan(kind, level.block_sums, level.block_offsets, block_count, workspace,
                           depth + 1);
    UniformAddKernel<<<static_cast<unsigned int>(block_count), kBlockSize>>>(
        output, level.block_offsets, n);
  }
}

template <typename Launch>
double TimeGpuOperation(int iterations, Launch&& launch) {
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

Measurement ValidateResult(double milliseconds, const std::uint32_t* d_output, std::size_t n) {
  std::vector<std::uint32_t> host(n);
  CUDA_CHECK(cudaMemcpy(host.data(), d_output, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

  Measurement measurement;
  measurement.milliseconds = milliseconds;
  measurement.valid = true;
  for (std::size_t index = 0; index < n; ++index) {
    const std::uint32_t expected = static_cast<std::uint32_t>(index);
    if (host[index] != expected) {
      measurement.valid = false;
      measurement.first_bad_index = index;
      measurement.actual = host[index];
      measurement.expected = expected;
      break;
    }
  }
  return measurement;
}

double UsefulBandwidthGbPerSecond(std::size_t bytes, double milliseconds) {
  // One logical input read plus one logical output write. Hierarchical algorithms
  // perform additional internal traffic, so this is intentionally "useful" bandwidth.
  return (2.0 * static_cast<double>(bytes)) / (milliseconds / 1000.0) / 1.0e9;
}

void PrintValidation(const char* name, const Measurement& measurement) {
  if (measurement.valid) {
    std::cout << "  " << name << " validation: PASS\n";
    return;
  }
  std::cout << "  " << name << " validation: FAIL at index " << measurement.first_bad_index
            << " (actual " << measurement.actual << ", expected " << measurement.expected
            << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::size_t bytes = options.elements * sizeof(std::uint32_t);

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));

    const std::size_t fill_blocks = DivUp(options.elements, static_cast<std::size_t>(kBlockSize));
    FillOnesKernel<<<static_cast<unsigned int>(fill_blocks), kBlockSize>>>(d_input,
                                                                          options.elements);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    Workspace shared_workspace = AllocateWorkspace(options.elements);
    Workspace warp_workspace = AllocateWorkspace(options.elements);

    void* d_cub_temp = nullptr;
    std::size_t cub_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes, d_input, d_output,
                                             static_cast<int>(options.elements)));
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

    const double serial_ms = TimeGpuOperation(options.iterations, [&] {
      SerialExclusiveScanKernel<<<1, 1>>>(d_input, d_output, options.elements);
    });
    const Measurement serial = ValidateResult(serial_ms, d_output, options.elements);

    const double shared_ms = TimeGpuOperation(options.iterations, [&] {
      LaunchHierarchicalScan(ScanKind::kShared, d_input, d_output, options.elements,
                             shared_workspace);
    });
    const Measurement shared = ValidateResult(shared_ms, d_output, options.elements);

    const double warp_ms = TimeGpuOperation(options.iterations, [&] {
      LaunchHierarchicalScan(ScanKind::kWarp, d_input, d_output, options.elements,
                             warp_workspace);
    });
    const Measurement warp = ValidateResult(warp_ms, d_output, options.elements);

    const double cub_ms = TimeGpuOperation(options.iterations, [&] {
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_cub_temp, cub_temp_bytes, d_input, d_output,
                                               static_cast<int>(options.elements)));
    });
    const Measurement cub = ValidateResult(cub_ms, d_output, options.elements);

    const double shared_bandwidth = UsefulBandwidthGbPerSecond(bytes, shared.milliseconds);
    const double warp_bandwidth = UsefulBandwidthGbPerSecond(bytes, warp.milliseconds);
    const double cub_bandwidth = UsefulBandwidthGbPerSecond(bytes, cub.milliseconds);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Prefix Scan (exclusive uint32 sum)\n";
    std::cout << "  Elements: " << options.elements << '\n';
    std::cout << "  GPU serial scan: " << serial.milliseconds << " ms\n";
    std::cout << "  Shared hierarchical scan: " << shared.milliseconds << " ms\n";
    std::cout << "  Warp hierarchical scan: " << warp.milliseconds << " ms\n";
    std::cout << "  CUB DeviceScan: " << cub.milliseconds << " ms\n";
    std::cout << "  Shared speedup vs serial: " << serial.milliseconds / shared.milliseconds
              << " x\n";
    std::cout << "  Warp speedup vs shared: " << shared.milliseconds / warp.milliseconds
              << " x\n";
    std::cout << "  CUB speedup vs warp: " << warp.milliseconds / cub.milliseconds << " x\n";
    std::cout << "  Shared hierarchical useful bandwidth: " << shared_bandwidth << " GB/s\n";
    std::cout << "  Warp hierarchical useful bandwidth: " << warp_bandwidth << " GB/s\n";
    std::cout << "  CUB useful bandwidth: " << cub_bandwidth << " GB/s\n";
    std::cout << "  Hierarchy levels: " << warp_workspace.size() + 1 << '\n';
    std::cout << "  CUB temporary storage: " << cub_temp_bytes << " bytes\n";

    PrintValidation("GPU serial", serial);
    PrintValidation("Shared hierarchical", shared);
    PrintValidation("Warp hierarchical", warp);
    PrintValidation("CUB", cub);

    const bool valid = serial.valid && shared.valid && warp.valid && cub.valid;

    CUDA_CHECK(cudaFree(d_cub_temp));
    FreeWorkspace(&shared_workspace);
    FreeWorkspace(&warp_workspace);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    if (!valid) {
      return 3;
    }
    std::cout << "  Validation: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
