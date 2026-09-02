# cuBLASLt Heuristic / Autotune Lab

This experiment studies GEMM dispatch rather than implementing another handwritten matrix-multiply kernel. It starts from the existing cuBLAS SGEMM baseline and introduces the cuBLASLt descriptor, heuristic and algorithm-selection model used by modern inference runtimes.

## Progression

```text
cuBLAS FP32 GEMM reference
        ↓
cuBLASLt matmul descriptor
        ↓
layout / transpose / epilogue
        ↓
workspace preference
        ↓
cublasLtMatmulAlgoGetHeuristic
        ↓
heuristic candidate list
        ↓
warm up every executable candidate
        ↓
CUDA Event timing
        ↓
select measured best algorithm
        ↓
optional shape-dependent persistent cache
```

The experiment deliberately keeps compute precision at `CUBLAS_COMPUTE_32F`. TF32 and BF16 are separate ROADMAP items so algorithm-selection effects are not mixed with reduced-precision effects.

## Why heuristic order is not the final answer

`cublasLtMatmulAlgoGetHeuristic` returns candidates in estimated performance order, but production dispatchers commonly benchmark a shortlist because the actual winner can depend on:

- exact `M/N/K` shape;
- transpose flags;
- row- vs column-major layout;
- fused epilogue;
- workspace budget;
- GPU architecture;
- library version and driver/runtime behavior.

The benchmark therefore times every returned candidate that successfully launches and chooses the lowest measured average latency.

## Candidate evidence

For every executable candidate the program reports:

- algorithm ID;
- tile ID;
- stages ID;
- split-K count;
- workspace requirement;
- heuristic waves count;
- measured latency;
- measured GFLOP/s.

It also reports the speedup of the measured winner over the first executable heuristic candidate. A non-1.0 result demonstrates why runtime autotuning can differ from static heuristic ordering.

## Workspace budget

The CLI exposes `--workspace-mb`. This is passed through `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`, so changing the budget can change both the candidate set and the winner.

Useful sweeps:

```bash
./build/release/bin/aw_cublaslt_autotune --m 1024 --n 1024 --k 1024 --workspace-mb 0 --no-cache
./build/release/bin/aw_cublaslt_autotune --m 1024 --n 1024 --k 1024 --workspace-mb 8 --no-cache
./build/release/bin/aw_cublaslt_autotune --m 1024 --n 1024 --k 1024 --workspace-mb 32 --no-cache
./build/release/bin/aw_cublaslt_autotune --m 1024 --n 1024 --k 1024 --workspace-mb 128 --no-cache
```

## Layout, transpose and epilogue

Supported controls:

```text
--order col|row
--transa N|T
--transb N|T
--epilogue none|relu
```

CUDA 13 cuBLASLt does not support ReLU epilogues with row-major output, so that combination is rejected explicitly instead of relying on a later opaque library failure.

For `epilogue=none`, the experiment also times a strict FP32 `cublasGemmEx` reference on column-major layouts. For fused ReLU or row-major layouts the legacy comparison is skipped because it would no longer be an equivalent operation.

## Shape-dependent persistent cache

cuBLASLt documents its algorithm descriptor as serializable and reusable with the same library version. When cache is enabled, Accelerateworld stores the measured winning descriptor under a key containing:

```text
SM architecture
cuBLASLt version
M / N / K
transpose A / B
memory order
epilogue
workspace budget
```

Example:

```bash
# First invocation: heuristic query + candidate benchmark + save winner.
./build/release/bin/aw_cublaslt_autotune \
  --m 2048 --n 4096 --k 4096 \
  --workspace-mb 32 \
  --cache-dir .cache/cublaslt

# Same shape/configuration: restore and validate cached algorithm first.
./build/release/bin/aw_cublaslt_autotune \
  --m 2048 --n 4096 --k 4096 \
  --workspace-mb 32 \
  --cache-dir .cache/cublaslt
```

`--force-autotune` ignores an existing cache entry and refreshes it. `--no-cache` disables both cache reads and writes.

GPU Baseline v2 always uses `--no-cache` so cross-generation evidence captures the full autotune process instead of depending on local cache history.

## Representative inference shapes

The executable accepts arbitrary rectangular GEMMs, which is important because inference workloads are rarely only square matrices.

Suggested manual sweeps:

```text
1024 x 1024 x 1024     square baseline
2048 x 4096 x 4096     prefill-like projection
1 x 4096 x 4096        single-request decode-like GEMM
4 x 4096 x 4096        small decode batch
16 x 4096 x 4096       larger decode batch
4096 x 11008 x 4096    MLP-style projection
```

These are performance shapes, not a claim that every model uses exactly these dimensions.

## Timing and correctness

Candidate kernel time is measured with CUDA Events on one non-blocking stream. Allocation, descriptor construction, heuristic query and persistent-cache I/O are outside kernel timing.

Heuristic-query host latency is reported separately because dispatch overhead matters for dynamic-shape workloads.

Input values are deterministic and the full output matrix is copied back after the selected algorithm runs. The CPU computes the exact constant reference value; `Validation: PASS` requires the maximum absolute error to remain within tolerance.

For the ReLU path, the raw GEMM result is deliberately negative so correctness verifies that the fused epilogue actually clamps it to zero.

## Canonical GPU Baseline v2 workload

```text
M / N / K            = 1024 / 1024 / 1024
heuristic candidates = 16
warmup                = 2
iterations            = 10
workspace             = 32 MiB
layout                = column-major
transpose             = N / N
epilogue              = none
cache                  = disabled
```

Primary metric:

```text
Best cuBLASLt throughput (GFLOP/s)
```

Higher is better.

## Connection to inference runtimes

A production LLM runtime does not treat GEMM as one immutable kernel. It effectively performs a dispatch problem:

```text
(shape, dtype, layout, epilogue, architecture, workspace)
                         ↓
                   algorithm policy
                         ↓
                      kernel
```

This experiment provides the dispatch/autotune layer that later BF16, TF32, FP8, FP4, CUTLASS and grouped/persistent GEMM stages can reuse.
