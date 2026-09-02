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

See [docs/ROADMAP.md](docs/ROADMAP.md) for the progression toward asynchronous allocators, cuBLASLt, CUTLASS, RoPE, online softmax, FlashAttention-style kernels, KV cache, quantization and a minimal inference runtime.

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

## PyTorch / Triton / LLM layer

Create an isolated environment on a CUDA-capable Linux machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r python/requirements-gpu.txt
bash scripts/run_ai_gpu_validation.sh
```

The PyTorch extension registers a dispatcher-backed fused `silu(gate) * up` CUDA operator. The Triton stage implements vector add, followed by an introductory Triton RMSNorm kernel for transformer workloads.

The same capability registry is intended to gate future `RoPE -> Online Softmax -> FlashAttention-style -> KV Cache -> Minimal Decoder Runtime` benchmarks across RTX generations.

## CI model

CI is split by what it can prove:

- `.github/workflows/build.yml`: CUDA 13/NVCC portable compile validation for `sm_75;sm_86;sm_89;sm_120`, CTest discovery and Python syntax checks;
- `.github/workflows/python-extension-build.yml`: PyTorch CUDA Extension compile validation for RTX 20/30/40/50 targets;
- `.github/workflows/gpu-validation.yml`: runtime auto-detection, correctness, sanitizer and benchmark evidence on a physical GPU runner;
- `.github/workflows/python-gpu-validation.yml`: PyTorch, Triton and LLM-kernel runtime validation on a physical GPU runner.

A hosted runner passing `nvcc` compilation is **not** recorded as GPU runtime validation.

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

The project follows concepts and APIs from NVIDIA CUDA Programming/Best Practices documentation, CUDA GPU Compute Capability tables, CUDA Samples, Compute Sanitizer, cuBLAS/CUB, PyTorch custom operator/C++ extension documentation and Triton tutorials.

## License

No project license has been selected yet. Add one explicitly before accepting third-party contributions.
