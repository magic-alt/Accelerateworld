#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {

constexpr std::size_t kMinBytes = 4096;
constexpr std::uint32_t kSeedXor = 0x9e3779b9u;

struct Options {
  std::size_t bytes = 4u << 20;
  int cycles = 64;
  int rounds = 8;
  int streams = 4;
  std::uint64_t release_threshold_mb = 256;
};

struct PoolStats {
  std::uint64_t reserved_current = 0;
  std::uint64_t reserved_high = 0;
  std::uint64_t used_current = 0;
  std::uint64_t used_high = 0;
};

struct ScenarioResult {
  std::vector<double> round_us_per_cycle;
  std::uint64_t checksum = 0;
  std::uint64_t expected_checksum = 0;
  PoolStats stats;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--bytes" && i + 1 < argc) {
      options.bytes = std::stoull(argv[++i]);
    } else if (arg == "--cycles" && i + 1 < argc) {
      options.cycles = std::stoi(argv[++i]);
    } else if (arg == "--rounds" && i + 1 < argc) {
      options.rounds = std::stoi(argv[++i]);
    } else if (arg == "--streams" && i + 1 < argc) {
      options.streams = std::stoi(argv[++i]);
    } else if (arg == "--release-threshold-mb" && i + 1 < argc) {
      options.release_threshold_mb = std::stoull(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: aw_async_memory_pool [--bytes N] [--cycles N] [--rounds N] "
                   "[--streams N] [--release-threshold-mb N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.bytes < kMinBytes || options.bytes % sizeof(std::uint32_t) != 0) {
    throw std::invalid_argument("bytes must be >= 4096 and divisible by 4");
  }
  if (options.cycles <= 0 || options.rounds < 2) {
    throw std::invalid_argument("cycles must be positive and rounds must be at least 2");
  }
  if (options.streams <= 0 || options.streams > 32) {
    throw std::invalid_argument("streams must be in [1, 32]");
  }
  return options;
}

__global__ void TouchAllocationKernel(std::uint32_t* buffer,
                                      std::size_t words,
                                      std::uint32_t seed,
                                      unsigned long long* checksum) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t first = seed;
    const std::uint32_t last = seed ^ kSeedXor;
    buffer[0] = first;
    buffer[words - 1] = last;
    atomicAdd(checksum, static_cast<unsigned long long>(first) +
                            static_cast<unsigned long long>(last));
  }
}

std::uint64_t Contribution(std::uint32_t seed) {
  return static_cast<std::uint64_t>(seed) +
         static_cast<std::uint64_t>(seed ^ kSeedXor);
}

std::size_t PatternBytes(std::size_t base, int cycle, bool mixed_sizes) {
  if (!mixed_sizes) {
    return base;
  }
  constexpr int kPattern[] = {1, 2, 4, 8};
  const std::size_t quarter = std::max(kMinBytes, base / 4);
  return quarter * static_cast<std::size_t>(kPattern[cycle % 4]);
}

std::uint32_t SeedFor(int round, int cycle, int cycles) {
  return static_cast<std::uint32_t>(round * cycles + cycle + 1);
}

void ResetChecksum(unsigned long long* d_checksum) {
  CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));
}

std::uint64_t ReadChecksum(unsigned long long* d_checksum) {
  unsigned long long value = 0;
  CUDA_CHECK(cudaMemcpy(&value, d_checksum, sizeof(value), cudaMemcpyDeviceToHost));
  return value;
}

PoolStats ReadPoolStats(cudaMemPool_t pool) {
  PoolStats stats;
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent,
                                     &stats.reserved_current));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh,
                                     &stats.reserved_high));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent,
                                     &stats.used_current));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh,
                                     &stats.used_high));
  return stats;
}

void ResetPoolHighWatermarks(cudaMemPool_t pool) {
  std::uint64_t zero = 0;
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero));
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
}

void SetReleaseThreshold(cudaMemPool_t pool, std::uint64_t bytes) {
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &bytes));
}

double SteadyStateAverage(const std::vector<double>& values) {
  if (values.size() < 2) {
    return values.empty() ? 0.0 : values.front();
  }
  return std::accumulate(values.begin() + 1, values.end(), 0.0) /
         static_cast<double>(values.size() - 1);
}

double MiB(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

ScenarioResult RunLegacy(const Options& options,
                         unsigned long long* d_checksum,
                         cudaStream_t stream,
                         bool mixed_sizes) {
  ScenarioResult result;
  ResetChecksum(d_checksum);

  for (int round = 0; round < options.rounds; ++round) {
    const auto start = std::chrono::steady_clock::now();
    for (int cycle = 0; cycle < options.cycles; ++cycle) {
      const std::size_t bytes = PatternBytes(options.bytes, cycle, mixed_sizes);
      const std::size_t words = bytes / sizeof(std::uint32_t);
      std::uint32_t* ptr = nullptr;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
      const std::uint32_t seed = SeedFor(round, cycle, options.cycles);
      TouchAllocationKernel<<<1, 1, 0, stream>>>(ptr, words, seed, d_checksum);
      CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(ptr));
      result.expected_checksum += Contribution(seed);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    result.round_us_per_cycle.push_back(elapsed_us / options.cycles);
  }

  result.checksum = ReadChecksum(d_checksum);
  return result;
}

ScenarioResult RunAsyncSingleStream(const Options& options,
                                    unsigned long long* d_checksum,
                                    cudaStream_t stream,
                                    cudaMemPool_t pool,
                                    std::uint64_t release_threshold,
                                    bool mixed_sizes) {
  ScenarioResult result;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
  SetReleaseThreshold(pool, release_threshold);
  ResetPoolHighWatermarks(pool);
  ResetChecksum(d_checksum);

  for (int round = 0; round < options.rounds; ++round) {
    const auto start = std::chrono::steady_clock::now();
    for (int cycle = 0; cycle < options.cycles; ++cycle) {
      const std::size_t bytes = PatternBytes(options.bytes, cycle, mixed_sizes);
      const std::size_t words = bytes / sizeof(std::uint32_t);
      std::uint32_t* ptr = nullptr;
      CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr), bytes, stream));
      const std::uint32_t seed = SeedFor(round, cycle, options.cycles);
      TouchAllocationKernel<<<1, 1, 0, stream>>>(ptr, words, seed, d_checksum);
      CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaFreeAsync(ptr, stream));
      result.expected_checksum += Contribution(seed);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    result.round_us_per_cycle.push_back(elapsed_us / options.cycles);
  }

  result.checksum = ReadChecksum(d_checksum);
  result.stats = ReadPoolStats(pool);
  return result;
}

ScenarioResult RunAsyncMultiStream(const Options& options,
                                   unsigned long long* d_checksum,
                                   const std::vector<cudaStream_t>& streams,
                                   cudaMemPool_t pool,
                                   std::uint64_t release_threshold) {
  ScenarioResult result;
  for (cudaStream_t stream : streams) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
  SetReleaseThreshold(pool, release_threshold);
  ResetPoolHighWatermarks(pool);
  ResetChecksum(d_checksum);

  for (int round = 0; round < options.rounds; ++round) {
    const auto start = std::chrono::steady_clock::now();
    for (int cycle = 0; cycle < options.cycles; ++cycle) {
      cudaStream_t stream = streams[static_cast<std::size_t>(cycle) % streams.size()];
      const std::size_t bytes = PatternBytes(options.bytes, cycle, true);
      const std::size_t words = bytes / sizeof(std::uint32_t);
      std::uint32_t* ptr = nullptr;
      CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr), bytes, stream));
      const std::uint32_t seed = SeedFor(round, cycle, options.cycles);
      TouchAllocationKernel<<<1, 1, 0, stream>>>(ptr, words, seed, d_checksum);
      CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaFreeAsync(ptr, stream));
      result.expected_checksum += Contribution(seed);
    }
    for (cudaStream_t stream : streams) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    result.round_us_per_cycle.push_back(elapsed_us / options.cycles);
  }

  result.checksum = ReadChecksum(d_checksum);
  result.stats = ReadPoolStats(pool);
  return result;
}

bool Valid(const ScenarioResult& result) {
  return result.checksum == result.expected_checksum;
}

void PrintScenario(const std::string& name, const ScenarioResult& result) {
  std::cout << name << " cold round cycle latency: " << result.round_us_per_cycle.front()
            << " us/cycle\n";
  std::cout << name << " steady state cycle latency: "
            << SteadyStateAverage(result.round_us_per_cycle) << " us/cycle\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int memory_pools_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device));
    if (!memory_pools_supported) {
      std::cerr << "Stream-ordered memory pools are not supported by this CUDA device/runtime\n";
      return 4;
    }

    cudaMemPool_t pool{};
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t original_release_threshold = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &original_release_threshold));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<cudaStream_t> streams(static_cast<std::size_t>(options.streams));
    for (cudaStream_t& item : streams) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&item, cudaStreamNonBlocking));
    }

    unsigned long long* d_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_checksum), sizeof(unsigned long long)));

    const std::uint64_t retained_threshold =
        options.release_threshold_mb * 1024ull * 1024ull;

    const ScenarioResult legacy = RunLegacy(options, d_checksum, stream, false);
    const ScenarioResult async_default =
        RunAsyncSingleStream(options, d_checksum, stream, pool, 0, false);
    const ScenarioResult async_retained =
        RunAsyncSingleStream(options, d_checksum, stream, pool, retained_threshold, false);
    const ScenarioResult async_mixed =
        RunAsyncSingleStream(options, d_checksum, stream, pool, retained_threshold, true);
    const ScenarioResult async_multistream =
        RunAsyncMultiStream(options, d_checksum, streams, pool, retained_threshold);

    SetReleaseThreshold(pool, original_release_threshold);
    CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));

    const bool success = Valid(legacy) && Valid(async_default) && Valid(async_retained) &&
                         Valid(async_mixed) && Valid(async_multistream);

    const double legacy_steady = SteadyStateAverage(legacy.round_us_per_cycle);
    const double default_steady = SteadyStateAverage(async_default.round_us_per_cycle);
    const double retained_steady = SteadyStateAverage(async_retained.round_us_per_cycle);
    const double mixed_steady = SteadyStateAverage(async_mixed.round_us_per_cycle);
    const double multistream_steady = SteadyStateAverage(async_multistream.round_us_per_cycle);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Asynchronous Memory Pools\n";
    std::cout << "  Bytes per fixed allocation: " << options.bytes << " bytes\n";
    std::cout << "  Cycles per round: " << options.cycles << '\n';
    std::cout << "  Rounds: " << options.rounds << '\n';
    std::cout << "  Multi-stream count: " << options.streams << '\n';
    std::cout << "  Retained release threshold: " << options.release_threshold_mb << " MiB\n";

    PrintScenario("Legacy cudaMalloc/cudaFree", legacy);
    PrintScenario("Async default threshold", async_default);
    PrintScenario("Retained pool", async_retained);
    PrintScenario("Mixed size retained pool", async_mixed);
    PrintScenario("Multi stream retained pool", async_multistream);

    std::cout << "Legacy steady state cycle latency: " << legacy_steady << " us/cycle\n";
    std::cout << "Async default steady state cycle latency: " << default_steady << " us/cycle\n";
    std::cout << "Retained pool steady state cycle latency: " << retained_steady << " us/cycle\n";
    std::cout << "Mixed size retained steady state cycle latency: " << mixed_steady << " us/cycle\n";
    std::cout << "Multi stream retained steady state cycle latency: " << multistream_steady
              << " us/cycle\n";
    std::cout << "Retained pool speedup vs legacy: " << legacy_steady / retained_steady << " x\n";
    std::cout << "Multi stream speedup vs legacy: " << legacy_steady / multistream_steady << " x\n";
    std::cout << "Default pool reserved current after sync: "
              << MiB(async_default.stats.reserved_current) << " MiB\n";
    std::cout << "Retained pool reserved current after sync: "
              << MiB(async_retained.stats.reserved_current) << " MiB\n";
    std::cout << "Mixed size reserved high: " << MiB(async_mixed.stats.reserved_high) << " MiB\n";
    std::cout << "Mixed size used high: " << MiB(async_mixed.stats.used_high) << " MiB\n";
    const double reserved_to_used = async_mixed.stats.used_high == 0
                                        ? 0.0
                                        : static_cast<double>(async_mixed.stats.reserved_high) /
                                              static_cast<double>(async_mixed.stats.used_high);
    std::cout << "Mixed size reserved to used high ratio: " << reserved_to_used << " x\n";
    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_checksum));
    for (cudaStream_t item : streams) {
      CUDA_CHECK(cudaStreamDestroy(item));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return success ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
