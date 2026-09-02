# Asynchronous Memory Pools

This experiment studies CUDA allocation overhead at the runtime/scheduling layer rather than inside a compute kernel. The central question is how `cudaMallocAsync` / `cudaFreeAsync` and CUDA memory pools change allocation latency, reuse behavior and retained memory compared with traditional `cudaMalloc` / `cudaFree`.

## Why this is a Stage 2 experiment

Traditional device allocation is expensive and may synchronize CUDA work. The stream-ordered allocator instead places allocation/free operations into stream order and services them from a device memory pool when possible. This is directly relevant to inference runtimes that repeatedly create temporary activation buffers, attention workspaces, routing buffers or decode scratch storage.

## Scenarios

The executable compares five scenarios:

1. **Legacy `cudaMalloc` / `cudaFree`** — allocate, touch, synchronize and free each cycle.
2. **Async default threshold** — same-size `cudaMallocAsync` / `cudaFreeAsync` cycles on one non-blocking stream using the device default pool with release threshold `0`.
3. **Retained pool** — same workload with a configurable non-zero `cudaMemPoolAttrReleaseThreshold` so reserved memory can survive synchronization points and be reused by later rounds.
4. **Mixed-size retained pool** — cycles through a deterministic `{1/4x, 1/2x, 1x, 2x}` allocation-size pattern to expose pool reuse and reserved-vs-used high-water behavior.
5. **Multi-stream retained pool** — distributes mixed-size allocations across multiple non-blocking streams that share the same default device pool.

The following ROADMAP item will go deeper into cross-stream event dependencies and the allocator reuse-policy attributes. This experiment establishes the memory-pool baseline first.

## Correctness

Every allocation is used before it is freed. A tiny kernel writes deterministic values to the first and last words of the allocation and atomically accumulates a persistent 64-bit checksum. The host computes the exact expected checksum for every scenario.

The benchmark fails if any scenario produces a different checksum.

## Device support

The executable calls:

```cpp
cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
```

before using the stream-ordered allocator. GPU Baseline v2 also declares the `memory_pool` capability for the supported RTX 20/30/40/50 profiles; runtime support is still verified by CUDA itself.

## Timing model

Allocator overhead is measured with **host wall-clock time**, not CUDA Events.

That distinction is intentional: `cudaMalloc` / `cudaFree` and async allocator submission/synchronization involve host-runtime behavior that CUDA Events do not measure completely.

Each scenario is divided into rounds. Every round executes multiple allocation/use/free cycles and then reaches its required synchronization boundary.

The benchmark reports both:

- **cold round cycle latency** — first round after trimming the pool;
- **steady-state cycle latency** — average of subsequent rounds.

The difference is useful evidence: a retained pool is expected to benefit from already-reserved backing memory after the first round.

## Pool statistics

The benchmark captures CUDA memory-pool watermarks through:

- `cudaMemPoolAttrReservedMemCurrent`;
- `cudaMemPoolAttrReservedMemHigh`;
- `cudaMemPoolAttrUsedMemCurrent`;
- `cudaMemPoolAttrUsedMemHigh`.

The mixed-size scenario additionally reports `reserved_high / used_high` as a simple pool-overhead / fragmentation proxy. It is not a full allocator-fragmentation model, but it makes otherwise invisible retention behavior observable.

## Canonical GPU Baseline v2 workload

```text
bytes per fixed allocation = 4 MiB
cycles per round           = 64
rounds                     = 8
multi-stream count         = 4
retained threshold         = 256 MiB
```

The primary cross-generation metric is:

```text
Retained pool steady state cycle latency (us/cycle)
```

Lower is better.

Raw evidence also keeps legacy/default/mixed-size/multi-stream latency plus pool watermarks and speedups.

## Run

```bash
cmake --preset release
cmake --build --preset release-build
./build/release/bin/aw_async_memory_pool \
  --bytes 4194304 \
  --cycles 64 \
  --rounds 8 \
  --streams 4 \
  --release-threshold-mb 256
```

Or run the complete capability-aware baseline:

```bash
python scripts/run_gpu_baseline.py
```

## What to inspect later

With Nsight Systems and additional allocator experiments, inspect:

- CPU time spent in legacy allocation/free calls;
- synchronization gaps introduced by legacy allocation;
- first-round versus steady-state async latency;
- reserved memory retained across synchronization points;
- mixed-size pool growth and high-water marks;
- concurrency when several streams share one pool;
- allocator dependencies inserted or followed across streams;
- the effect of disabling opportunistic/internal/event-based reuse policies.

## Connection to inference infrastructure

Production inference engines rarely allocate one tensor once and exit. They repeatedly recycle temporary buffers for prefill, decode, sampling, attention workspaces, MoE routing and KV-cache metadata. Understanding CUDA's stream-ordered allocator therefore bridges low-level CUDA experiments and the memory-management layer of an actual LLM runtime.
