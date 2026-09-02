# Reduction: atomics -> shared memory -> warp shuffle

This experiment compares three ways to reduce a large FP32 vector to one scalar while keeping the workload and correctness oracle identical.

## Question

How much synchronization and intermediate-memory traffic can be removed as reduction moves from a global atomic baseline to block-level shared memory and finally to warp register exchange?

## Implementations

### 1. Per-element atomic baseline

Every active thread reads one value and performs `atomicAdd` to a single global scalar. This is intentionally contention-heavy and establishes a simple correctness/performance floor.

### 2. Shared-memory block reduction

Each thread accumulates a grid-stride local sum, writes one value to shared memory, then participates in a tree reduction with `__syncthreads()` at every reduction level. Only one global atomic is issued per block.

### 3. Warp-shuffle reduction

Each thread first accumulates a grid-stride local sum. Values inside each warp are then reduced with `__shfl_down_sync`, so the intra-warp tree stays in registers rather than shared memory. Warp leaders write one subtotal each to shared memory; the first warp reduces those subtotals and one lane performs the final per-block global atomic.

For the current 256-thread block this changes the block-level reduction from a 256-value shared-memory tree to eight warp-local register reductions plus an eight-value inter-warp reduction.

## Correctness

The input is filled with `1.0f`, so the expected result is exactly the number of elements. All three implementations are executed and validated independently; the experiment fails if the maximum absolute error exceeds the configured tolerance.

## Timing

Each implementation receives one untimed warm-up launch. CUDA Events measure the repeated reduction region. Allocation and input initialization are outside the measured region, while resetting the output scalar is included consistently for all three paths.

The canonical GPU Baseline v2 metric is:

```text
Warp shuffle useful bandwidth (GB/s)
```

The output also reports atomic/shared/warp latency and the shared-to-warp speedup.

## Run

```bash
cmake --preset release
cmake --build --preset release-build
./build/release/bin/aw_reduction --elements 8388608 --iterations 20
```

Or use the capability-aware baseline runner on the installed RTX GPU:

```bash
python scripts/run_gpu_baseline.py
```

## What to inspect with Nsight Compute later

- global load throughput and DRAM bandwidth utilization;
- atomic throughput/contention for the baseline;
- shared-memory transactions and bank conflicts;
- barrier instructions and synchronization overhead;
- achieved occupancy and register usage;
- instruction count difference between shared-memory and shuffle trees.

## Portability

The implementation uses synchronized warp shuffle intrinsics and is compiled by the repository's RTX 20/30/40/50 portable CUDA targets (`sm_75`, `sm_86`, `sm_89`, `sm_120`). Future production-oriented experiments may also compare CUB warp/block collectives, but this experiment intentionally keeps the shuffle tree visible for learning and profiling.
