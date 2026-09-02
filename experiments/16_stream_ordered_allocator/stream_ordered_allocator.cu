#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "accelerateworld/cuda_check.hpp"

namespace {

constexpr std::size_t kMinBytes = 4096;
constexpr std::uint32_t kSeedXor = 0x9e3779b9u;

struct Options {
  std::size_t bytes = 4u << 20;
  int cycles = 32;
  int rounds = 6;
  unsigned long long spin_cycles = 5000;
  std::uint64_t release_threshold_mb = 256;
  std::uint64_t pool_max_mb = 64;
  bool skip_pressure_probe = false;
};

struct PoolPolicy {
  int follow_event_dependencies = 1;
  int allow_opportunistic = 1;
  int allow_internal_dependencies = 1;
  std::uint64_t release_threshold = 0;
};

struct PoolStats {
  std::uint64_t reserved_current = 0;
  std::uint64_t reserved_high = 0;
  std::uint64_t used_current = 0;
  std::uint64_t used_high = 0;
};

enum class DependencyMode {
  kSameStream,
  kNone,
  kEventWait,
  kWaitForCompletion,
};

struct ScenarioResult {
  std::vector<double> round_us_per_cycle;
  std::uint64_t checksum = 0;
  std::uint64_t expected_checksum = 0;
  std::uint64_t pointer_reuse_count = 0;
  std::uint64_t second_allocation_successes = 0;
  std::uint64_t second_allocation_attempts = 0;
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
    } else if (arg == "--spin-cycles" && i + 1 < argc) {
      options.spin_cycles = std::stoull(argv[++i]);
    } else if (arg == "--release-threshold-mb" && i + 1 < argc) {
      options.release_threshold_mb = std::stoull(argv[++i]);
    } else if (arg == "--pool-max-mb" && i + 1 < argc) {
      options.pool_max_mb = std::stoull(argv[++i]);
    } else if (arg == "--skip-pressure-probe") {
      options.skip_pressure_probe = true;
    } else if (arg == "--help") {
      std::cout << "Usage: aw_stream_ordered_allocator [--bytes N] [--cycles N] [--rounds N] "
                   "[--spin-cycles N] [--release-threshold-mb N] [--pool-max-mb N] "
                   "[--skip-pressure-probe]\n";
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
  const std::uint64_t pool_max_bytes = options.pool_max_mb * 1024ull * 1024ull;
  if (pool_max_bytes < options.bytes * 2ull) {
    throw std::invalid_argument("pool-max-mb must provide at least 2x the normal allocation size");
  }
  return options;
}

__global__ void TouchAllocationKernel(std::uint32_t* buffer,
                                      std::size_t words,
                                      std::uint32_t seed,
                                      unsigned long long spin_cycles,
                                      unsigned long long* checksum) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned long long start = clock64();
    while (clock64() - start < spin_cycles) {
    }
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

std::uint32_t SeedFor(int round, int cycle, int side, int cycles) {
  return static_cast<std::uint32_t>((round * cycles + cycle) * 2 + side + 1);
}

void ResetChecksum(unsigned long long* d_checksum) {
  CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));
}

std::uint64_t ReadChecksum(unsigned long long* d_checksum) {
  unsigned long long value = 0;
  CUDA_CHECK(cudaMemcpy(&value, d_checksum, sizeof(value), cudaMemcpyDeviceToHost));
  return value;
}

PoolPolicy ReadPoolPolicy(cudaMemPool_t pool) {
  PoolPolicy policy;
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolReuseFollowEventDependencies,
                                     &policy.follow_event_dependencies));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolReuseAllowOpportunistic,
                                     &policy.allow_opportunistic));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies,
                                     &policy.allow_internal_dependencies));
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &policy.release_threshold));
  return policy;
}

void SetPoolPolicy(cudaMemPool_t pool, const PoolPolicy& policy) {
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseFollowEventDependencies,
                                     &policy.follow_event_dependencies));
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic,
                                     &policy.allow_opportunistic));
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowInternalDependencies,
                                     &policy.allow_internal_dependencies));
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &policy.release_threshold));
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

void PreparePool(cudaMemPool_t pool,
                 cudaStream_t producer,
                 cudaStream_t consumer,
                 const PoolPolicy& policy) {
  CUDA_CHECK(cudaStreamSynchronize(producer));
  if (consumer != producer) {
    CUDA_CHECK(cudaStreamSynchronize(consumer));
  }
  CUDA_CHECK(cudaMemPoolTrimTo(pool, 0));
  SetPoolPolicy(pool, policy);
  ResetPoolHighWatermarks(pool);
}

double SteadyStateAverage(const std::vector<double>& values) {
  if (values.size() < 2) {
    return values.empty() ? 0.0 : values.front();
  }
  return std::accumulate(values.begin() + 1, values.end(), 0.0) /
         static_cast<double>(values.size() - 1);
}

double RatioPercent(std::uint64_t numerator, std::uint64_t denominator) {
  if (denominator == 0) {
    return 0.0;
  }
  return 100.0 * static_cast<double>(numerator) / static_cast<double>(denominator);
}

double MiB(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void WaitForEventCompletion(cudaEvent_t event) {
  while (true) {
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) {
      return;
    }
    if (status != cudaErrorNotReady) {
      CUDA_CHECK(status);
    }
    std::this_thread::yield();
  }
}

ScenarioResult RunPairScenario(const Options& options,
                               unsigned long long* d_checksum,
                               cudaMemPool_t pool,
                               cudaStream_t producer,
                               cudaStream_t consumer,
                               const PoolPolicy& policy,
                               DependencyMode dependency,
                               std::size_t allocation_bytes,
                               bool allow_second_allocation_failure) {
  ScenarioResult result;
  PreparePool(pool, producer, consumer, policy);
  ResetChecksum(d_checksum);

  cudaEvent_t dependency_event{};
  CUDA_CHECK(cudaEventCreateWithFlags(&dependency_event, cudaEventDisableTiming));

  for (int round = 0; round < options.rounds; ++round) {
    const auto start = std::chrono::steady_clock::now();
    for (int cycle = 0; cycle < options.cycles; ++cycle) {
      std::uint32_t* first = nullptr;
      CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&first), allocation_bytes,
                                         pool, producer));
      const std::uint32_t first_seed = SeedFor(round, cycle, 0, options.cycles);
      TouchAllocationKernel<<<1, 1, 0, producer>>>(
          first, allocation_bytes / sizeof(std::uint32_t), first_seed,
          options.spin_cycles, d_checksum);
      CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaFreeAsync(first, producer));
      result.expected_checksum += Contribution(first_seed);

      cudaStream_t second_stream = consumer;
      if (dependency == DependencyMode::kSameStream) {
        second_stream = producer;
      } else if (dependency == DependencyMode::kEventWait) {
        CUDA_CHECK(cudaEventRecord(dependency_event, producer));
        CUDA_CHECK(cudaStreamWaitEvent(consumer, dependency_event, 0));
      } else if (dependency == DependencyMode::kWaitForCompletion) {
        CUDA_CHECK(cudaEventRecord(dependency_event, producer));
        WaitForEventCompletion(dependency_event);
      }

      ++result.second_allocation_attempts;
      std::uint32_t* second = nullptr;
      const cudaError_t allocation_status = cudaMallocFromPoolAsync(
          reinterpret_cast<void**>(&second), allocation_bytes, pool, second_stream);
      if (allocation_status == cudaErrorMemoryAllocation && allow_second_allocation_failure) {
        (void)cudaGetLastError();
        continue;
      }
      CUDA_CHECK(allocation_status);
      ++result.second_allocation_successes;
      if (second == first) {
        ++result.pointer_reuse_count;
      }

      const std::uint32_t second_seed = SeedFor(round, cycle, 1, options.cycles);
      TouchAllocationKernel<<<1, 1, 0, second_stream>>>(
          second, allocation_bytes / sizeof(std::uint32_t), second_seed,
          options.spin_cycles, d_checksum);
      CUDA_KERNEL_CHECK();
      CUDA_CHECK(cudaFreeAsync(second, second_stream));
      result.expected_checksum += Contribution(second_seed);
    }

    CUDA_CHECK(cudaStreamSynchronize(producer));
    if (consumer != producer) {
      CUDA_CHECK(cudaStreamSynchronize(consumer));
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    result.round_us_per_cycle.push_back(elapsed_us / options.cycles);
  }

  result.checksum = ReadChecksum(d_checksum);
  result.stats = ReadPoolStats(pool);
  CUDA_CHECK(cudaEventDestroy(dependency_event));
  return result;
}

bool Valid(const ScenarioResult& result) {
  return result.checksum == result.expected_checksum;
}

void PrintScenario(const std::string& name, const ScenarioResult& result) {
  std::cout << name << " cold cycle latency: " << result.round_us_per_cycle.front()
            << " us/cycle\n";
  std::cout << name << " steady state cycle latency: "
            << SteadyStateAverage(result.round_us_per_cycle) << " us/cycle\n";
  std::cout << name << " pointer reuse ratio: "
            << RatioPercent(result.pointer_reuse_count, result.second_allocation_successes)
            << " %\n";
  std::cout << name << " allocation success rate: "
            << RatioPercent(result.second_allocation_successes,
                            result.second_allocation_attempts)
            << " %\n";
  std::cout << name << " reserved high: " << MiB(result.stats.reserved_high) << " MiB\n";
  std::cout << name << " used high: " << MiB(result.stats.used_high) << " MiB\n";
}

cudaMemPool_t CreateCustomPool(int device, std::uint64_t max_bytes) {
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device;
  props.maxSize = static_cast<std::size_t>(max_bytes);

  cudaMemPool_t pool{};
  CUDA_CHECK(cudaMemPoolCreate(&pool, &props));
  return pool;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int memory_pools_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_pools_supported,
                                      cudaDevAttrMemoryPoolsSupported, device));
    if (!memory_pools_supported) {
      std::cerr << "Stream-ordered memory pools are not supported by this CUDA device/runtime\n";
      return 4;
    }

    cudaMemPool_t default_pool{};
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&default_pool, device));
    const PoolPolicy original_default_policy = ReadPoolPolicy(default_pool);

    const std::uint64_t release_threshold =
        options.release_threshold_mb * 1024ull * 1024ull;
    const std::uint64_t custom_pool_max = options.pool_max_mb * 1024ull * 1024ull;
    cudaMemPool_t custom_pool = CreateCustomPool(device, custom_pool_max);

    cudaStream_t producer{};
    cudaStream_t consumer{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));

    unsigned long long* d_checksum = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_checksum), sizeof(unsigned long long)));

    const PoolPolicy strict{0, 0, 0, release_threshold};
    const PoolPolicy follow_on{1, 0, 0, release_threshold};
    const PoolPolicy opportunistic_on{0, 1, 0, release_threshold};
    const PoolPolicy internal_on{0, 0, 1, release_threshold};

    const ScenarioResult same_stream = RunPairScenario(
        options, d_checksum, custom_pool, producer, producer, strict,
        DependencyMode::kSameStream, options.bytes, false);
    const ScenarioResult no_dependency = RunPairScenario(
        options, d_checksum, custom_pool, producer, consumer, strict,
        DependencyMode::kNone, options.bytes, false);
    const ScenarioResult event_follow_off = RunPairScenario(
        options, d_checksum, custom_pool, producer, consumer, strict,
        DependencyMode::kEventWait, options.bytes, false);
    const ScenarioResult event_follow_on = RunPairScenario(
        options, d_checksum, custom_pool, producer, consumer, follow_on,
        DependencyMode::kEventWait, options.bytes, false);
    const ScenarioResult opportunistic_off = RunPairScenario(
        options, d_checksum, custom_pool, producer, consumer, strict,
        DependencyMode::kWaitForCompletion, options.bytes, false);
    const ScenarioResult opportunistic_on = RunPairScenario(
        options, d_checksum, custom_pool, producer, consumer, opportunistic_on,
        DependencyMode::kWaitForCompletion, options.bytes, false);

    ScenarioResult internal_off;
    ScenarioResult internal_enabled;
    bool ran_pressure_probe = false;
    if (!options.skip_pressure_probe) {
      const std::size_t pressure_bytes =
          static_cast<std::size_t>((custom_pool_max * 3ull) / 4ull);
      internal_off = RunPairScenario(
          options, d_checksum, custom_pool, producer, consumer, strict,
          DependencyMode::kNone, pressure_bytes, true);
      internal_enabled = RunPairScenario(
          options, d_checksum, custom_pool, producer, consumer, internal_on,
          DependencyMode::kNone, pressure_bytes, true);
      ran_pressure_probe = true;
    }

    const ScenarioResult default_event = RunPairScenario(
        options, d_checksum, default_pool, producer, consumer, follow_on,
        DependencyMode::kEventWait, options.bytes, false);
    const ScenarioResult custom_event = RunPairScenario(
        options, d_checksum, custom_pool, producer, consumer, follow_on,
        DependencyMode::kEventWait, options.bytes, false);

    SetPoolPolicy(default_pool, original_default_policy);
    CUDA_CHECK(cudaStreamSynchronize(producer));
    CUDA_CHECK(cudaStreamSynchronize(consumer));
    CUDA_CHECK(cudaMemPoolTrimTo(custom_pool, 0));
    CUDA_CHECK(cudaMemPoolTrimTo(default_pool, 0));

    bool success = Valid(same_stream) && Valid(no_dependency) &&
                   Valid(event_follow_off) && Valid(event_follow_on) &&
                   Valid(opportunistic_off) && Valid(opportunistic_on) &&
                   Valid(default_event) && Valid(custom_event);
    if (ran_pressure_probe) {
      success = success && Valid(internal_off) && Valid(internal_enabled);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Stream-ordered Allocator Policy Matrix\n";
    std::cout << "  Allocation size: " << options.bytes << " bytes\n";
    std::cout << "  Cycles per round: " << options.cycles << '\n';
    std::cout << "  Rounds: " << options.rounds << '\n';
    std::cout << "  Spin cycles per touch: " << options.spin_cycles << '\n';
    std::cout << "  Release threshold: " << options.release_threshold_mb << " MiB\n";
    std::cout << "  Custom pool max size: " << options.pool_max_mb << " MiB\n";

    PrintScenario("Same stream", same_stream);
    PrintScenario("Cross stream no dependency", no_dependency);
    PrintScenario("Event follow OFF", event_follow_off);
    PrintScenario("Event follow ON", event_follow_on);
    PrintScenario("Opportunistic OFF", opportunistic_off);
    PrintScenario("Opportunistic ON", opportunistic_on);
    if (ran_pressure_probe) {
      PrintScenario("Internal dependency OFF pressure", internal_off);
      PrintScenario("Internal dependency ON pressure", internal_enabled);
    }
    PrintScenario("Default event follow", default_event);
    PrintScenario("Custom event follow", custom_event);

    std::cout << "Event follow reuse delta: "
              << RatioPercent(event_follow_on.pointer_reuse_count,
                              event_follow_on.second_allocation_successes) -
                     RatioPercent(event_follow_off.pointer_reuse_count,
                                  event_follow_off.second_allocation_successes)
              << " percentage-points\n";
    std::cout << "Opportunistic reuse delta: "
              << RatioPercent(opportunistic_on.pointer_reuse_count,
                              opportunistic_on.second_allocation_successes) -
                     RatioPercent(opportunistic_off.pointer_reuse_count,
                                  opportunistic_off.second_allocation_successes)
              << " percentage-points\n";
    std::cout << "Default vs custom event latency ratio: "
              << SteadyStateAverage(default_event.round_us_per_cycle) /
                     SteadyStateAverage(custom_event.round_us_per_cycle)
              << " x\n";
    std::cout << "Validation: " << (success ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_checksum));
    CUDA_CHECK(cudaStreamDestroy(consumer));
    CUDA_CHECK(cudaStreamDestroy(producer));
    CUDA_CHECK(cudaMemPoolDestroy(custom_pool));
    return success ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
