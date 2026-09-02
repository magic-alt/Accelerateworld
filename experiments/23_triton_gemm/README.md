# 23 — Triton Auto-tuned GEMM

This Stage 4 experiment moves from a single Triton vector-add kernel to an explicit GEMM tuning problem:

```text
shape family
    ↓
BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K
    ↓
num_warps / num_stages
    ↓
triton.autotune
    ↓
M,N,K tuning key
    ↓
selected config per shape
    ↓
PyTorch framework baseline
    ↓
direct cuBLAS baseline
    ↓
shape-dependent winner
```

The goal is not to claim that one Triton configuration beats vendor libraries universally. The goal is to retain enough evidence to explain **why the best launch/tile configuration changes with GEMM shape**, especially when moving from token-parallel prefill to small-M autoregressive decode.

## Numerical contract

All three compared paths use the same experiment-level contract:

```text
A: row-major FP16 [M, K]
B: row-major FP16 [K, N]
accumulation: FP32
C: row-major FP16 [M, N]
```

The providers are:

1. **Triton autotuned** — handwritten blocked `tl.dot` kernel;
2. **PyTorch `torch.mm(..., out=...)`** — framework-dispatched vendor GEMM path with preallocated output;
3. **direct cuBLAS** — `cublasGemmEx` with `CUBLAS_COMPUTE_32F` and FP16 output, built as `aw_triton_cublas_fp16_gemm`.

Keeping the output type equal is important. The earlier Stage 3 mixed-precision cuBLASLt experiment intentionally writes FP32 output, so its throughput is useful architectural evidence but is not an apples-to-apples winner baseline for this FP16-output Triton experiment.

## Autotune search space

`lab_config.py` retains ten explicit candidates spanning:

- `BLOCK_SIZE_M`: 16, 32, 64, 128;
- `BLOCK_SIZE_N`: 64, 128, 256;
- `BLOCK_SIZE_K`: 32, 64;
- `num_warps`: 2, 4, 8;
- `num_stages`: 3, 4, 5;
- `GROUP_SIZE_M = 8` for grouped program ordering/L2 locality.

The large-M candidates are adapted from the official Triton matrix-multiplication tutorial. Extra 16/32-M candidates are retained deliberately for decode workloads where launching a 64/128-row tile wastes most lanes.

The tuner uses:

```python
@triton.autotune(
    configs=...,
    key=["M", "N", "K"],
    cache_results=True,
)
```

Changing any of `M/N/K` creates a distinct tuning key. `cache_results=True` allows Triton to persist tuning measurements in its cache. The benchmark reports the first invocation as **compile/autotune or disk-cache restore latency** rather than pretending it is steady-state kernel latency.

The selected `best_config` is printed and retained in optional JSON evidence.

## Workload families

The benchmark separates three performance regimes plus a small validation family.

### Balanced

```text
1024 x 1024 x 1024
2048 x 2048 x 2048
```

These are useful for understanding conventional square GEMM behavior before introducing LLM aspect ratios.

### Prefill

Representative hidden-size-4096 projections with `M=512` token parallelism:

```text
QKV:        512 x 12288 x 4096
attn out:   512 x  4096 x 4096
MLP up:     512 x 11008 x 4096
MLP down:   512 x  4096 x 11008
```

The exact dimensions are representative laboratory shapes, not a claim that every production model uses these widths.

### Decode

Small-M projections:

```text
QKV:        1 x 12288 x 4096
MLP up:     1 x 11008 x 4096
MLP down:   4 x  4096 x 11008
projection:16 x  4096 x 4096
```

This family is intentionally hostile to large `BLOCK_SIZE_M` values and exposes the difference between throughput-oriented large tiles and latency-oriented small-M dispatch.

### Validation

The physical-GPU CI uses two reduced shapes to exercise both regimes without turning every validation run into a long tuning sweep:

```text
prefill-like: 128 x 1536 x 1024
decode-like:    8 x 1024 x 1024
```

## Build the direct cuBLAS reference

Configure for the physical GPU when possible. For the RTX 5060 reference machine:

```bash
cmake --preset rtx5060
cmake --build --preset rtx5060-build --target aw_triton_cublas_fp16_gemm
```

Other examples:

```bash
cmake --preset rtx30
cmake --build --preset rtx30-build --target aw_triton_cublas_fp16_gemm

cmake --preset rtx40
cmake --build --preset rtx40-build --target aw_triton_cublas_fp16_gemm

cmake --preset rtx50
cmake --build --preset rtx50-build --target aw_triton_cublas_fp16_gemm
```

The Python benchmark automatically looks for the binary under the generation-specific build directories. It can also be supplied explicitly:

```bash
python experiments/23_triton_gemm/benchmark.py \
  --family validation \
  --cublas-exe build/rtx5060/bin/aw_triton_cublas_fp16_gemm \
  --require-cublas
```

`--require-cublas` is recommended when collecting comparison evidence. Without it, the Triton/PyTorch comparison still runs and the direct baseline is explicitly reported as unavailable.

## Run the workload matrix

Fast validation:

```bash
python experiments/23_triton_gemm/benchmark.py \
  --family validation \
  --warmup 5 \
  --iterations 20
```

Prefill:

```bash
python experiments/23_triton_gemm/benchmark.py \
  --family prefill \
  --warmup 10 \
  --iterations 30 \
  --require-cublas \
  --json results/triton-gemm-prefill.json
```

Decode:

```bash
python experiments/23_triton_gemm/benchmark.py \
  --family decode \
  --warmup 20 \
  --iterations 100 \
  --require-cublas \
  --json results/triton-gemm-decode.json
```

All canonical shapes:

```bash
python experiments/23_triton_gemm/benchmark.py \
  --family all \
  --require-cublas \
  --json results/triton-gemm-all.json
```

## Timing boundary

For steady-state provider timing:

- A/B/C allocations are outside the timed interval;
- Triton autotuning/compilation happens before steady-state timing;
- PyTorch uses a preallocated `out=` tensor;
- direct cuBLAS allocates/copies outside its CUDA Event interval;
- repeated GEMM launches are bracketed by CUDA Events;
- reported latency is average device-stream execution per GEMM.

The first Triton invocation is measured separately with host time plus `torch.cuda.synchronize()`. It is labeled `compile/autotune or disk-cache restore` because a prior `cache_results=True` run may make it a cache hit.

## Correctness

Triton is compared against `torch.mm` on the same random FP16 tensors. Inputs are scaled to avoid avoidable FP16-output overflow. The test uses `torch.testing.assert_close` plus a max normalized-error report.

The direct cuBLAS executable uses independent deterministic FP16 inputs and validates sampled outputs against a CPU reference that first respects the narrowed FP16 inputs and finally rounds the expected output to FP16.

## Reading the result

Do not reduce the experiment to one global speedup number. Inspect at least:

```text
shape
    ↓
selected BLOCK_M/N/K
    ↓
selected warps/stages
    ↓
Triton latency
    ↓
PyTorch latency
    ↓
direct cuBLAS latency
    ↓
shape winner
```

Expected questions include:

- Does prefill prefer larger M/N tiles and more warps?
- Does decode consistently select 16/32-M tiles?
- When does a deeper software pipeline help?
- Do two shapes with the same FLOP count select different configurations because their aspect ratios differ?
- When Triton loses to cuBLAS, is the gap concentrated in large regular GEMMs or small-M dispatch?
- When Triton wins, is the result stable after autotune-cache warmup?

These are hypotheses to measure on each GPU generation, not conclusions hard-coded by the repository.

## Evidence boundary

Hosted CUDA CI can prove:

- the direct cuBLAS reference compiles for RTX 20/30/40/50 targets;
- the CTest target is registered;
- Python files are syntactically valid;
- the autotune search space, LLM shape families, winner logic and cuBLAS parser pass offline tests.

Hosted CI has no physical GPU and therefore does **not** claim:

- Triton JIT compilation success on a specific GPU;
- which config wins a shape;
- Triton vs PyTorch/cuBLAS latency;
- disk-cache speedup;
- cross-generation performance.

Those results belong to `python-gpu-validation` or manually retained GPU evidence.

## References

- Triton Matrix Multiplication tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- `triton.autotune`: https://triton-lang.org/main/python-api/generated/triton.autotune.html
- `triton.Config`: https://triton-lang.org/main/python-api/generated/triton.Config.html
- NVIDIA cuBLAS documentation: https://docs.nvidia.com/cuda/cublas/
