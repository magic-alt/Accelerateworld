# Prefix Scan: serial -> shared -> warp -> hierarchical -> CUB

This experiment implements an exclusive prefix sum over `uint32_t` values and keeps every implementation on the same device-resident input/output buffers.

## Why scan matters

Prefix scan is a building block for stream compaction, histogram offsets, radix sorting, sparse indexing, attention metadata preparation, token routing and KV-cache indexing. Unlike reduction, every output depends on all preceding inputs, so the implementation must preserve both parallelism and ordering.

For deterministic validation the input is all ones:

```text
input  = [1, 1, 1, 1, ...]
output = [0, 1, 2, 3, ...]
```

Using integer addition makes the oracle exact and prevents floating-point associativity from obscuring synchronization or indexing defects.

## Optimization ladder

### 1. GPU serial scan

One GPU thread walks the entire array and writes an exclusive running sum. It is deliberately inefficient but gives the simplest device-memory baseline.

### 2. Shared-memory block scan

Each 256-thread block performs a Hillis-Steele scan in shared memory. The block emits both per-element exclusive results and one block total.

This version exposes the synchronization cost directly: every scan level requires barriers and shared-memory traffic.

### 3. Warp scan

`WarpInclusiveScan` uses `__shfl_up_sync` so values move between lanes through registers rather than shared memory. All 32 lanes participate with the same full-warp mask.

### 4. Warp-composed block scan

Each warp scans its own values. Warp leaders publish eight subtotals for the current 256-thread block. The first warp scans those subtotals and each warp adds the prefix belonging to the preceding warps.

This turns the warp primitive into a complete block-wide exclusive scan while using only a small shared-memory array for cross-warp communication.

### 5. Recursive multi-block scan

A device-wide scan needs prefixes for block totals too. The host-side launcher recursively scans the block-total array until one block is sufficient, then applies a uniform-add kernel while unwinding the hierarchy.

Workspace allocations happen before timing. A timed iteration therefore consists only of the scan kernels and hierarchy launches, not `cudaMalloc`/`cudaFree`.

### 6. CUB DeviceScan reference

`cub::DeviceScan::ExclusiveSum` is the vendor-library reference. Accelerateworld intentionally uses CUB's explicit two-phase temporary-storage API because it is available in the repository's CUDA 13.0 toolchain. Temporary storage is queried and allocated once before timing.

CUB is not treated as an implementation to beat at all costs. It provides a production-quality reference that shows the remaining gap between an educational hierarchical scan and a tuned device-wide scan.

## Measurement boundary

All implementations receive one untimed warm-up. CUDA Events measure device execution only. Input allocation, input fill, workspace allocation, CUB temporary-storage allocation and host validation copies are outside the measured region.

Reported useful bandwidth counts one logical input read plus one logical output write:

```text
useful bytes = 2 * N * sizeof(uint32_t)
```

The hierarchical implementations perform extra traffic for block sums and uniform-add passes, so this is deliberately named **useful bandwidth**, not physical DRAM traffic.

GPU Baseline v2 uses:

```text
Warp hierarchical useful bandwidth (GB/s)
```

as the cross-generation primary metric, while CUB bandwidth is retained as the optimized library reference.

## Correctness

After each timed implementation the complete output is copied to the host and checked against `output[i] == i`. The executable fails on the first mismatch and reports the index, actual value and expected value.

GPU Baseline v2 additionally runs Compute Sanitizer memcheck and racecheck on smaller scan workloads.

## Run

```bash
cmake --preset release
cmake --build --preset release-build
./build/release/bin/aw_prefix_scan --elements 4194304 --iterations 10
```

Or run the complete capability-aware baseline:

```bash
python scripts/run_gpu_baseline.py
```

## What to inspect with Nsight Compute later

- shared-memory transactions and barrier pressure in the Hillis-Steele path;
- shuffle instruction count in the warp path;
- global memory throughput;
- achieved occupancy and register use;
- hierarchy depth and launch overhead as `N` grows;
- useful bandwidth as a fraction of measured DRAM bandwidth;
- the gap between the educational hierarchy and CUB DeviceScan.

## Portability

The kernels require only CUDA functionality available on the repository's supported RTX generations. They compile under the portable targets `sm_75`, `sm_86`, `sm_89` and `sm_120`. No architecture-specific fast path is required for the initial scan implementation.
