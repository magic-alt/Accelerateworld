# Stream-ordered Allocator Experiments

This Stage 2 experiment studies **cross-stream memory reuse semantics** in CUDA's stream-ordered allocator. The previous asynchronous-memory-pool experiment established allocation latency, release-threshold tuning, mixed-size reuse and basic multi-stream sharing. This experiment goes one level deeper: it asks *which ordering relationship allows a free in one stream to satisfy a later allocation in another stream?*

## Policy matrix

The executable exercises the same allocation/use/free pattern while changing only stream ordering and memory-pool reuse policy:

1. **Same-stream reuse baseline** — allocate, use, free and allocate again in the same stream. Basic stream ordering already makes reuse safe.
2. **Cross-stream, no dependency** — producer and consumer streams have no explicit ordering and all cross-stream reuse policies are disabled.
3. **Event dependency, policy OFF** — producer records an event after `cudaFreeAsync`; consumer waits on that event, but `cudaMemPoolReuseFollowEventDependencies` is disabled.
4. **Event dependency, policy ON** — identical event ordering with `cudaMemPoolReuseFollowEventDependencies` enabled.
5. **Opportunistic OFF/ON** — the producer free is allowed to complete and is observed with non-blocking event queries; no stream dependency is inserted. This isolates `cudaMemPoolReuseAllowOpportunistic`.
6. **Internal dependency OFF/ON under pressure** — an explicit custom pool has a finite `maxSize`; two concurrent pressure-size allocations cannot both fit. This exposes whether `cudaMemPoolReuseAllowInternalDependencies` can establish a safe dependency instead of obtaining more backing memory.
7. **Default pool vs custom pool** — the same explicit-event workload is run on the device default pool and on an explicit pool created with `cudaMemPoolCreate`.

The benchmark records outcomes rather than asserting that a particular pointer address must be reused. CUDA is allowed to change allocator internals across driver releases; pointer reuse is useful evidence, not part of correctness.

## Why these three policies matter

CUDA memory pools expose three cross-stream reuse controls:

- `cudaMemPoolReuseFollowEventDependencies` — allows an allocation to reuse a free in another stream when CUDA event/null-stream ordering establishes a dependency;
- `cudaMemPoolReuseAllowOpportunistic` — allows already-completed frees to be reused even without a stream dependency;
- `cudaMemPoolReuseAllowInternalDependencies` — allows the allocator to insert a stream dependency when reuse is needed and safe ordering is otherwise missing.

All three are enabled by default. This experiment deliberately disables them and turns them back on one at a time.

## Custom pool

The explicit pool is created with:

```cpp
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = device;
props.maxSize = ...;
cudaMemPoolCreate(&pool, &props);
```

Normal scenarios use a 4 MiB allocation in a 64 MiB pool. The internal-dependency pressure probe uses roughly 75% of the pool per allocation, so two allocations cannot coexist if the pending free cannot be reused.

## Correctness

Every successful allocation is touched by a CUDA kernel before its stream-ordered free. The kernel writes deterministic values and atomically contributes to a persistent 64-bit checksum. The host independently computes the expected checksum.

A failed second allocation in the pressure probe is treated as **observed policy behavior**, not as data corruption. All work that did execute must still match the exact checksum.

## Measurement model

Allocator/scheduler overhead is measured with synchronized host wall-clock time. Every scenario reports:

- cold first-round latency;
- steady-state latency;
- pointer-address reuse ratio;
- second-allocation success rate;
- `ReservedMemHigh`;
- `UsedMemHigh`.

The pointer reuse ratio is intentionally descriptive. A high ratio shows that the allocator reused the same virtual address, but CUDA does not promise pointer identity as an API contract.

## Canonical GPU Baseline v2 workload

```text
allocation size       = 4 MiB
cycles per round      = 32
rounds                = 6
spin cycles per touch = 5,000
release threshold     = 256 MiB
custom pool max       = 64 MiB
```

The primary cross-generation metric is:

```text
Custom event follow steady state cycle latency (us/cycle)
```

Lower is better. Raw evidence retains the complete reuse-policy matrix and pool watermarks.

## Run

```bash
cmake --preset release
cmake --build --preset release-build
./build/release/bin/aw_stream_ordered_allocator \
  --bytes 4194304 \
  --cycles 32 \
  --rounds 6 \
  --spin-cycles 5000 \
  --release-threshold-mb 256 \
  --pool-max-mb 64
```

Or run the complete capability-aware suite:

```bash
python scripts/run_gpu_baseline.py
```

## What to inspect with Nsight Systems later

- whether cross-stream allocations introduce synchronization edges;
- host API duration of `cudaMallocFromPoolAsync` under each policy;
- explicit `cudaEventRecord` / `cudaStreamWaitEvent` ordering;
- allocator-inserted dependencies when internal reuse is enabled;
- overlap lost or preserved by each policy;
- default-pool versus explicit-pool reserved memory behavior.

## Connection to LLM runtimes

Inference runtimes repeatedly recycle temporary buffers across prefill, decode, attention, sampling and MoE work scheduled on multiple CUDA streams. Reusing a buffer too early is a lifetime bug; refusing safe reuse increases reserved memory and allocator pressure. The same dependency model appears later in activation/workspace allocators, KV-cache page reuse, continuous batching and CUDA Graph decode paths.
