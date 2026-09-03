# Accelerateworld

Accelerateworld is a reproducible **GPU / AI Infrastructure learning and benchmark lab** for NVIDIA RTX GPUs. It starts with CUDA execution and memory behavior, progresses through streams, graphs, vendor libraries and Tensor Cores, and then connects handwritten kernels to PyTorch, Triton and LLM workloads.

The repository is structured as an engineering project rather than a folder of `.cu` snippets: every performance experiment needs a hypothesis, a correctness oracle, an explicit timing boundary, a benchmark metric and a repeatable validation path.

## Supported RTX generations

GPU Baseline v2 supports the main CUDA compute capabilities used by RTX 20/30/40/50:

| RTX generation | Architecture | Compute capability | CUDA target |
|---|---|---:|---:|
| RTX 20 | Turing | 7.5 | `sm_75` |
| RTX 30 | Ampere | 8.6 | `sm_86` |
| RTX 40 | Ada | 8.9 | `sm_89` |
| RTX 50 | Blackwell | 12.0 | `sm_120` |

RTX 5060 remains the project's first **reference GPU**, but it no longer has a special baseline implementation. Hardware is detected at runtime and resolved through `hardware/rtx_capabilities.json`.

See [docs/GPU_BASELINE_V2.md](docs/GPU_BASELINE_V2.md).

## Learning path

| ID | Experiment | Main question |
|---|---|---|
| 00 | Device query | What GPU/runtime capabilities are actually available? |
| 01 | Vector add | How do grid/block mapping and bandwidth-limited kernels behave? |
| 02 | Matrix multiply | What does shared-memory tiling change? |
| 03 | Reduction | Why does aggregation strategy matter for atomics and synchronization? |
| 04 | Memory coalescing | How expensive are scattered global-memory accesses? |
| 05 | Matrix transpose | How do coalescing, shared memory and bank-conflict padding interact? |
| 06 | Streams + pinned memory | When can transfer and compute overlap? |
| 07 | CUDA Graph | How much host launch overhead can graph replay remove? |
| 08 | cuBLAS GEMM | What does the vendor-optimized baseline look like? |
| 09 | Tensor Core WMMA | How are FP16 Tensor Core matrix operations expressed directly? |
| 10 | PyTorch CUDA Extension | How does a custom CUDA kernel become a framework operator? |
| 11 | Triton | How does a GPU compiler DSL compare with handwritten CUDA/framework ops? |
| 12 | LLM kernels | How do RMSNorm and later transformer kernels map to the GPU? |
| 13 | Prefix scan | How do warp scans compose into block- and device-wide parallel primitives, and how close can an educational hierarchy get to CUB? |
| 14 | Histogram / atomics | How do contention, warp aggregation, shared privatization and multi-pass merging change atomic-heavy workloads? |
| 15 | Async memory pool | How do stream-ordered allocation, pool retention, mixed sizes and multi-stream sharing change allocator overhead? |
| 16 | Stream-ordered allocator | How do explicit events and memory-pool reuse policies control safe cross-stream buffer reuse? |
| 17 | cuBLASLt autotune | How do shape, layout, workspace and heuristic candidates determine GEMM dispatch? |
| 18 | Mixed-precision GEMM | How do Strict FP32, TF32, BF16 and FP16 trade accuracy for throughput across RTX generations? |
| 19 | FP8 GEMM | How do E4M3/E5M2 scaling, saturation, autotuning and accuracy differ between Ada and Blackwell? |
| 20 | FP4 GEMM | How do E2M1 payload packing, 16-value UE4M3 block scales and two-level scaling change Blackwell GEMM accuracy and throughput? |
| 21 | CUTLASS GEMM | How do SIMT/TensorOp, threadblock/warp/MMA shapes, pipeline stages and Blackwell collective scheduling compose production GEMM kernels? |
| 22 | Persistent / Grouped GEMM | How do persistent CTAs and grouped problem schedulers amortize heterogeneous GEMM dispatch for MoE- and decode-like workloads? |
| 23 | Triton Auto-tuned GEMM | How do M/N/K shape families change the best Triton tile, warp and pipeline configuration relative to PyTorch and direct cuBLAS? |
| 24 | Nsight Systems Framework Trace | Where do Python/framework overhead, compilation, CUDA launches, cuBLAS, Triton and custom CUDA kernels appear on one timeline? |
| 25 | SwiGLU Mixed Precision | How do FP16/BF16 storage, FP32 projection/post-op arithmetic, Inductor fusion, Triton and vectorized custom CUDA behave across decode/prefill MLP shapes? |
| 26 | RoPE | How do interleaved/half-split rotary layouts, cached FP32 angles, FP16/BF16 Q/K tensors and long-context positions change positional-kernel behavior? |
| 27 | Online Softmax | How do stable two-pass and online max/sum reductions compose across warps/blocks for causal decode/prefill attention rows? |
| 28 | FlashAttention-style Attention | How can QK, online normalization and PV accumulation stream without materializing the score/probability matrix? |
| 29 | KV-cache Update / Read | How do contiguous token/head-major layouts behave for prefill writes, decode append and attention-compatible reads? |
| 30 | Paged KV Cache | How do block tables, non-contiguous physical pages, fragmentation and page reuse change KV-cache memory/runtime behavior? |

See [experiments/README.md](experiments/README.md) for the tutorial index and [docs/ROADMAP.md](docs/ROADMAP.md) for the progression toward inference-runtime integration and performance engineering.

## GPU Baseline v2

### Detect the current RTX GPU

```bash
python scripts/detect_gpu.py
```

The detector records the exact model, VRAM, driver, compute capability, native SM target, architecture generation and supported feature flags.

### Run a native baseline

Linux / WSL2:

```bash
python3 scripts/run_gpu_baseline.py
```

Windows:

```powershell
python scripts/run_gpu_baseline.py
```

The runner automatically builds for the physical GPU's native SM, runs CTest, applies capability-aware Compute Sanitizer checks, executes the canonical benchmark manifest and writes a unified `baseline.json` under:

```text
results/gpu-baselines/<gpu>/<timestamp>-<commit>/
```

The mixed-precision GEMM records use separate feature gates, so RTX 20 still records Strict FP32/FP16 evidence while TF32/BF16 are explicitly marked `skipped`; RTX 30/40/50 execute all four modes. FP8 records are separately gated: RTX 20/30 record explicit skips, while RTX 40/Ada and RTX 50/Blackwell execute E4M3/E5M2 tensorwide-scaled cuBLASLt workloads. FP4 is narrower again: only RTX 50/Blackwell executes the E2M1 + 16-element UE4M3 block-scaled record; RTX 20/30/40 record an explicit skip.

CUTLASS 4.7.0 is pinned for the native GEMM lab. RTX 20+ can retain the SIMT and SM75 TensorOp references; RTX 30/40/50 additionally run the SM80 Universal FP16/BF16 configurations; RTX 50 adds the SM120 `CollectiveBuilder -> GemmUniversalAdapter` FP8 specialization. This lets the same baseline retain explicit kernel-configuration evidence alongside the cuBLASLt autotuner.

The persistent/grouped GEMM record deliberately returns to one common SM75 TensorOp kernel contract across RTX 20/30/40/50. It compares serial per-problem launches with CUTLASS `device::GemmGrouped` device scheduling, host-precomputed scheduling and descending-K problem ordering, while retaining persistent CTA/tile counts and scheduler initialization evidence in the raw log.

### Compare multiple RTX GPUs

```bash
python scripts/compare_gpu_baselines.py \
  results/gpu-baselines/geforce-rtx-3090/*/baseline.json \
  results/gpu-baselines/geforce-rtx-4090/*/baseline.json \
  results/gpu-baselines/geforce-rtx-5060/*/baseline.json \
  --markdown results/rtx-comparison.md \
  --json results/rtx-comparison.json
```

Benchmark execution is **feature gated** from `benchmarks/manifest.json`. Unsupported experiments are recorded as `skipped`; a supported experiment that fails is recorded as `failed`.

## CMake presets

Portable compile coverage:

```bash
cmake --preset release
cmake --build --preset release-build
```

Generation-specific presets:

```bash
cmake --preset rtx20   # Turing / sm_75
cmake --preset rtx30   # Ampere / sm_86
cmake --preset rtx40   # Ada / sm_89
cmake --preset rtx50   # Blackwell / sm_120
```

`rtx5060` remains as a compatibility/reference alias for the RTX 50 preset.

## Native CUDA quick start

### Linux / WSL2

```bash
./scripts/check_environment.sh
cmake --preset release
cmake --build --preset release-build
ctest --preset release-test
```

### Windows PowerShell

```powershell
.\scripts\check_environment.ps1
python scripts/detect_gpu.py
.\scripts\run_gpu_validation.ps1
```

The legacy `run_gpu_validation.sh/.ps1` entry points now delegate to GPU Baseline v2.

## PyTorch / Triton / profiling / LLM layer

Create an isolated environment on a CUDA-capable Linux machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r python/requirements-gpu.txt
bash scripts/run_ai_gpu_validation.sh
```

The PyTorch extension registers a dispatcher-backed fused `silu(gate) * up` CUDA operator and integrates it with Autograd, FakeTensor, Dynamo, AOTAutograd and Inductor. The Triton layer starts with vector add and includes an FP16 GEMM autotuner that searches tile/warp/pipeline configurations across balanced, prefill and small-M decode shapes against PyTorch and a direct cuBLAS reference. Experiment 24 then places PyTorch eager, the custom CUDA op, `torch.compile`, the PyTorch/cuBLAS GEMM path and the Triton GEMM onto one NVTX-annotated Nsight Systems timeline.

On a GPU machine with the Nsight Systems CLI installed:

```bash
bash scripts/run_nsys_framework_trace.sh
```

This retains both steady-state and first-use `.nsys-rep` reports plus `nsys stats` summaries under `results/nsys-framework-trace/`.

Stage 5 is the production-oriented LLM kernel path. Experiments 25–27 cover mixed-precision SwiGLU, RoPE and online softmax. Experiment 28 combines the online state with score-matrix-free FlashAttention-style QK/PV streaming. Experiment 29 introduces stateful contiguous K/V update/read with token-major/head-major layouts. Experiment 30 then decouples logical token positions from physical storage with a block table, page pool, non-contiguous allocation, free-list reuse and fragmentation accounting, while benchmarking paged PyTorch/Triton/custom-CUDA paths against experiment 29's contiguous CUDA baseline.

BF16 follows the common capability registry and is skipped on RTX 20 rather than treated as a failure. Hosted multi-architecture compilation is evidence of source portability, not physical GPU performance.

## CI model

CI is split by what it can prove:

- `.github/workflows/build.yml`: CUDA 13/NVCC portable compile validation for `sm_75;sm_86;sm_89;sm_120`, CTest discovery, Python syntax checks, tutorial coverage, offline experiment-logic tests and an Nsight command-construction dry run;
- `.github/workflows/python-extension-build.yml`: PyTorch CUDA Extension compile validation for RTX 20/30/40/50 targets, including SwiGLU, RoPE, online softmax, FlashAttention, contiguous KV cache and paged KV cache;
- `.github/workflows/gpu-validation.yml`: runtime auto-detection, correctness, sanitizer and benchmark evidence on a physical GPU runner;
- `.github/workflows/python-gpu-validation.yml`: PyTorch compiler/runtime, Triton autotuning, mixed-precision LLM kernels, stateful KV-cache benchmarks, Nsight Systems framework traces and physical-GPU runtime validation.

A hosted runner passing `nvcc` compilation or an Nsight dry run is **not** recorded as physical GPU trace or benchmark evidence.

## Repository layout

```text
Accelerateworld/
├── .devcontainer/
├── .github/workflows/
├── benchmarks/
│   ├── manifest.json
│   └── gpu-baseline-v2.schema.json
├── cmake/
├── docs/
├── experiments/
├── hardware/
│   └── rtx_capabilities.json
├── include/accelerateworld/
├── python/
├── results/
└── scripts/
    ├── baseline_lib.py
    ├── detect_gpu.py
    ├── run_gpu_baseline.py
    ├── run_benchmark_suite.py
    └── compare_gpu_baselines.py
```

## Benchmark rules

Read [docs/BENCHMARKING.md](docs/BENCHMARKING.md). In short:

1. establish an independent correctness oracle;
2. warm up before timing;
3. use CUDA Events for stream/kernel latency and synchronized host timing for end-to-end measurements;
4. report workload shape/dtype plus GPU, driver, CUDA and framework/compiler versions;
5. retain raw logs/CI artifacts;
6. explain the expected bottleneck before claiming an optimization;
7. compare only equivalent workload definitions across GPUs.

## Primary references

The project follows concepts and APIs from NVIDIA CUDA Programming/Best Practices documentation, CUDA GPU Compute Capability tables, CUDA Samples, Compute Sanitizer, CUDA stream-ordered memory allocator, cuBLAS/cuBLASLt/CUB, CUTLASS, Nsight Systems, PyTorch custom operator/C++ extension documentation, Triton tutorials and production LLM inference-runtime designs such as block-table/paged KV-cache memory management.

## License

No project license has been selected yet. Add one explicitly before accepting third-party contributions.
