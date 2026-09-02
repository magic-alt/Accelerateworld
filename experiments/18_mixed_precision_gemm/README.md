# Mixed-precision cuBLASLt GEMM

This experiment compares GEMM precision modes while holding the problem shape, input values, autotune procedure, workspace budget, output type, timing boundary, and correctness oracle constant.

## Modes

| Mode | A/B storage | Compute / accumulate | RTX 20 | RTX 30 | RTX 40 | RTX 50 |
|---|---|---|---:|---:|---:|---:|
| Strict FP32 | FP32 | `CUBLAS_COMPUTE_32F_PEDANTIC` | run | run | run | run |
| TF32 | FP32 | `CUBLAS_COMPUTE_32F_FAST_TF32` | skip | run | run | run |
| BF16 | BF16 | FP32 | skip | run | run | run |
| FP16 | FP16 | FP32 | run | run | run | run |

CUDA documents TF32 and BF16 Tensor Core support for compute capability 8.0 and later. The executable also checks the physical device compute capability at runtime; GPU Baseline v2 separately gates TF32 and BF16 records from `hardware/rtx_capabilities.json`.

## Autotune path

Each supported mode independently performs:

```text
cuBLASLt descriptors
        ↓
workspace budget
        ↓
cublasLtMatmulAlgoGetHeuristic
        ↓
heuristic shortlist
        ↓
warm up every executable candidate
        ↓
CUDA Event timing
        ↓
lowest measured latency
        ↓
selected algorithm
```

This intentionally does not reuse the winner from another precision mode. Different input types and compute types can produce different valid algorithms, workspace requirements, tiles, stages, split-K decisions, and performance rankings.

## Correctness and accuracy

A single deterministic FP32 master input pair is generated on the host. FP16 and BF16 inputs are created by explicit round-to-nearest GPU conversion from those same values.

After the selected algorithm runs, the FP32 output matrix is copied to the host. Thirty-two deterministic output coordinates are checked against FP64 CPU dot products using the original FP32 master inputs. The benchmark reports:

- max absolute error;
- max normalized error, `abs(error) / max(1, abs(reference))`;
- a mode-specific validation threshold.

The goal is to make the speed/accuracy trade-off visible without adding an O(MNK) CPU reference GEMM to every benchmark run.

## Why Strict FP32 uses PEDANTIC

The baseline deliberately uses `CUBLAS_COMPUTE_32F_PEDANTIC` rather than a fast 32-bit mode. TF32 is tested separately with `CUBLAS_COMPUTE_32F_FAST_TF32`, so reduced-mantissa Tensor Core behavior cannot silently contaminate the FP32 reference.

## GPU Baseline v2 records

The same executable is registered as four canonical benchmark records:

```text
mixed_precision_fp32   requires: cuda
mixed_precision_fp16   requires: tensor_core + wmma_fp16
mixed_precision_tf32   requires: tf32
mixed_precision_bf16   requires: bf16
```

This is important for Turing. RTX 20 should still produce FP32 and FP16 evidence while its TF32 and BF16 records are explicitly marked `skipped`, rather than skipping the entire precision experiment.

Canonical shape: `1024 x 1024 x 1024`, 16 heuristic candidates, 2 warmups, 10 timed iterations, 32 MiB workspace.

## Example

```bash
./build/release/bin/aw_mixed_precision_gemm \
  --m 1024 --n 1024 --k 1024 \
  --heuristics 16 --warmup 2 --iterations 10 \
  --workspace-mb 32 --mode all
```

Representative inference-oriented shapes can then be studied separately, for example large-prefill GEMMs versus M=1/4/16 decode GEMMs.

## Evidence boundary

Hosted CI proves CUDA 13 compile/link portability and CTest registration. It cannot prove Tensor Core execution, algorithm winners, throughput, or numerical error on a physical GPU. Those results belong in GPU Baseline v2 evidence from real RTX hardware.
