# Accelerateworld

Accelerateworld is a reproducible **GPU / AI Infrastructure learning and benchmark lab**. It starts with CUDA execution and memory behavior, progresses through streams, graphs, vendor libraries and Tensor Cores, and then connects handwritten kernels to PyTorch, Triton and LLM workloads.

The repository is intentionally structured as an engineering project rather than a folder of `.cu` snippets: every performance experiment needs a hypothesis, a correctness oracle, an explicit timing boundary, a benchmark metric and a repeatable validation path.

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

See [docs/ROADMAP.md](docs/ROADMAP.md) for the progression toward cuBLASLt, CUTLASS, FP8, FlashAttention-style kernels, KV cache, quantization and a minimal inference runtime.

## RTX 5060 target

RTX 5060 is treated as a first-class Blackwell `sm_120` development target. The CMake preset is:

```bash
cmake --preset rtx5060
cmake --build --preset rtx5060-build
ctest --preset rtx5060-test
```

Do not publish benchmark numbers from compilation alone. A GPU result is valid only after runtime execution on the named GPU.

## Native CUDA quick start

### Linux / WSL2

```bash
./scripts/check_environment.sh
cmake --preset release
cmake --build --preset release-build
ctest --preset release-test
```

### RTX 5060 Windows PowerShell

```powershell
.\scripts\check_environment.ps1
cmake --preset rtx5060
cmake --build --preset rtx5060-build
ctest --preset rtx5060-test
$env:ACCELERATEWORLD_CUDA_ARCH = "120"
.\scripts\run_gpu_validation.ps1
```

### Full Linux GPU validation

```bash
ACCELERATEWORLD_CUDA_ARCH=120 bash scripts/run_gpu_validation.sh
```

This builds the native experiments, runs CTest, executes selected Compute Sanitizer tools, then captures the benchmark log under `results/`.

## PyTorch / Triton / LLM layer

Create an isolated environment on a CUDA-capable Linux machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r python/requirements-gpu.txt
bash scripts/run_ai_gpu_validation.sh
```

The first PyTorch extension registers a dispatcher-backed fused `silu(gate) * up` CUDA operator. The Triton stage implements vector add, followed by an introductory Triton RMSNorm kernel for transformer workloads.

## CI model

CI is deliberately split by what it can prove:

- `.github/workflows/build.yml`: GPU-independent CUDA 13/NVCC compile validation, CTest discovery and Python syntax checks;
- `.github/workflows/gpu-validation.yml`: native runtime correctness, sanitizer and benchmark evidence on a GPU runner;
- `.github/workflows/python-gpu-validation.yml`: PyTorch extension build, Triton execution and LLM-kernel benchmarks on a GPU runner.

A standard hosted runner passing `nvcc` compilation is **not** recorded as GPU runtime validation.

## Repository layout

```text
Accelerateworld/
├── .devcontainer/
├── .github/workflows/
├── cmake/
├── docs/
│   ├── BENCHMARKING.md
│   ├── GPU_VALIDATION.md
│   ├── PLATFORM.md
│   └── ROADMAP.md
├── experiments/
│   ├── 00_device_query/
│   ├── 01_vector_add/
│   ├── ...
│   ├── 10_pytorch_extension/
│   ├── 11_triton/
│   └── 12_llm_kernels/
├── include/accelerateworld/
├── python/
├── results/
└── scripts/
```

## Benchmark rules

Read [docs/BENCHMARKING.md](docs/BENCHMARKING.md). In short:

1. establish an independent correctness oracle;
2. warm up before timing;
3. use CUDA Events for stream/kernel latency and synchronized host timing for end-to-end measurements;
4. report workload shape/dtype plus GPU, driver, CUDA and framework/compiler versions;
5. retain raw logs/CI artifacts;
6. explain the expected bottleneck before claiming an optimization.

## Primary references

The project follows concepts and APIs from NVIDIA CUDA Programming/Best Practices documentation, CUDA Samples, Compute Sanitizer, cuBLAS, PyTorch custom operator/C++ extension documentation, and Triton tutorials.

## License

No project license has been selected yet. Add one explicitly before accepting third-party contributions.
