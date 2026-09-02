# Histogram / Atomics Contention

This experiment studies how histogram performance changes as many GPU threads contend for the same counters. The key variable is not only the number of bins but also the **input distribution**: an implementation that looks fine on a uniform distribution can collapse when most samples target one hot bin.

## Optimization ladder

The executable compares five implementations under the same input, correctness oracle and timing model:

1. **Naive global atomics** — every sample issues one `atomicAdd` directly to the global histogram.
2. **Warp-aggregated atomics** — lanes with the same bin are grouped with `__match_any_sync`; one leader issues a single global atomic for the group.
3. **Shared-memory privatization** — each block accumulates into a shared-memory histogram, then flushes non-zero bins to the global histogram.
4. **Per-block histogram + explicit merge** — each block writes a private histogram to global memory and a second kernel merges all block histograms without global histogram atomics.
5. **CUB `DeviceHistogram::HistogramEven`** — production-library device-wide reference.

The custom implementations deliberately expose the algorithmic tradeoffs instead of hiding them behind a library call.

## Contention distributions

The benchmark runs three deterministic distributions by default:

- `uniform`: hashed samples are spread approximately uniformly across all bins;
- `hot`: `--hot-percent` of samples target bin 0 (90% in the canonical baseline), with the remainder spread across other bins;
- `single`: every sample targets bin 0, creating the maximum-contention case.

These distributions make the transition from memory-bandwidth pressure to atomic serialization visible.

## Why warp aggregation helps

For a warp in which many lanes target the same bin, `__match_any_sync` returns the set of lanes with the same value. Only the group leader performs a global `atomicAdd`, using the group population count as the increment. In the single-bin case this can reduce global atomic operations by up to roughly one operation per active warp iteration instead of one per lane.

## Why shared-memory privatization helps

A block-private histogram moves most contention from global memory into shared memory. The block later flushes at most one update per non-zero bin to global memory. This is especially effective when the number of bins is small enough to fit comfortably in shared memory.

## Why the two-pass design is separate

The shared-privatized path still performs global atomics while flushing block-local bins. The explicit two-pass path instead writes one dense histogram per block and launches a merge kernel that sums those block histograms. This trades extra global-memory traffic and workspace for deterministic merge work with no global histogram atomics.

This distinction becomes useful in workloads such as routing tables, compaction metadata and token/expert counts where contention can be extreme and intermediate workspace is acceptable.

## Correctness

Input values are generated deterministically on the host. An independent CPU histogram is built before launching any GPU algorithm. Every GPU implementation is validated bin-by-bin against that oracle, and the executable fails if any count differs.

The default `uint32_t` counter type is exact for the canonical workload and avoids floating-point ambiguity.

## Timing

Each algorithm gets one untimed warm-up launch. CUDA Events then measure repeated device execution.

- allocations and host-to-device input copies are outside the measured region;
- custom atomic paths include the output reset they require;
- shared-memory workspace and per-block histogram workspace are allocated once and reused;
- CUB temporary storage is queried and allocated before timing;
- the two-pass metric includes both per-block histogram construction and the global merge kernel.

Throughput is reported in `Gitems/s`.

## GPU Baseline v2 metric

The canonical cross-generation primary metric is:

```text
Hot two pass throughput (Gitems/s)
```

The hot distribution is deliberately used because it stresses contention rather than measuring only an easy uniform case. Raw evidence also retains uniform/single-bin results and all naive/warp/shared/CUB measurements.

## Run

```bash
cmake --preset release
cmake --build --preset release-build
./build/release/bin/aw_histogram \
  --elements 8388608 \
  --bins 256 \
  --iterations 20 \
  --hot-percent 90 \
  --distribution all
```

Or run the complete capability-aware baseline:

```bash
python scripts/run_gpu_baseline.py
```

## What to inspect with Nsight Compute later

- global atomic instruction count and throughput;
- L2 atomic serialization and memory transactions;
- shared-memory atomic throughput and bank behavior;
- occupancy changes as the bin count increases shared-memory usage;
- global bytes written by the explicit per-block histogram workspace;
- merge-kernel bandwidth and instruction mix;
- how uniform, hot and single-bin distributions change bottlenecks on Turing, Ampere, Ada and Blackwell.

## Connection to AI infrastructure

The same patterns appear in:

- MoE token-to-expert routing counts;
- routing/dispatch metadata construction;
- compaction and stream compaction bookkeeping;
- sparse indexing and bucketization;
- quantization calibration histograms;
- KV-cache/page metadata accounting;
- sorting/radix preprocessing and frequency counting.

The goal is to understand when atomics are cheap, when aggregation is sufficient, and when privatization or a multi-pass algorithm is the better systems choice.
