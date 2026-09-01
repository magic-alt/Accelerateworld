# Accelerateworld

Accelerateworld is a reproducible CUDA experiments lab for learning GPU programming, validating kernel correctness, measuring performance, and documenting optimization results.

The repository is intentionally structured like a small engineering project rather than a collection of loose `.cu` files: every experiment has a question, a correctness oracle, a benchmark, CTest coverage, and a repeatable GPU-validation path.

## Scope

The first milestone contains three experiments:

| ID | Experiment | What it teaches | Validation |
|---|---|---|---|
| 00 | Device query | CUDA runtime/driver/device capabilities | Device discovery + capability report |
| 01 | Vector add | Grid/block mapping, memory transfers, kernel timing | CPU oracle + max error |
| 02 | Matrix multiply | Naive vs shared-memory tiling | CPU oracle (small N) / cross-kernel check + GFLOP/s |

The next milestones can add reductions, transpose/coalescing, streams, pinned memory, CUDA Graphs, cuBLAS, Tensor Cores, Triton, and PyTorch custom CUDA operators.

## RTX 5060 target

NVIDIA lists GeForce RTX 5060 as compute capability **12.0**. Use CUDA Toolkit 13.x (or another toolkit version that supports Blackwell `sm_120`) and the dedicated preset:

```bash
cmake --preset rtx5060
cmake --build --preset rtx5060-build
ctest --preset rtx5060-test
```

On Windows, run these commands from a Visual Studio Developer PowerShell after installing the NVIDIA driver, CUDA Toolkit, CMake, and Ninja.

## Quick start

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

cmake --preset rtx5060
cmake --build --preset rtx5060-build
ctest --preset rtx5060-test
```

### Docker

A CUDA development image is provided for reproducible compilation:

```bash
docker build -t accelerateworld-cuda .
```

To run CUDA code, the host must have an NVIDIA GPU, a compatible driver, and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all accelerateworld-cuda \
  bash -lc "cmake --preset rtx5060 && cmake --build --preset rtx5060-build && ctest --preset rtx5060-test"
```

## GPU validation

Standard GitHub-hosted runners do not expose a GPU, so the normal CI workflow performs **compile-time validation only**.

Real GPU validation is separated into `.github/workflows/gpu-validation.yml` and is intended for:

- a self-hosted RTX 5060 runner (`self-hosted`, `linux`, `x64`, `gpu`), or
- an organization/enterprise GPU larger runner, configured with the appropriate runner name and CUDA architecture.

The validation script runs:

1. `nvidia-smi` and `nvcc --version`
2. CMake configure/build
3. CTest correctness tests
4. Compute Sanitizer `memcheck`
5. `racecheck`, `initcheck`, and `synccheck` where relevant
6. vector-add and matrix-multiply benchmarks
7. result capture under `results/`

See [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md).

## Repository layout

```text
Accelerateworld/
├── .devcontainer/
├── .github/workflows/
├── cmake/
├── docs/
├── experiments/
│   ├── 00_device_query/
│   ├── 01_vector_add/
│   └── 02_matmul/
├── include/accelerateworld/
├── scripts/
├── CMakeLists.txt
├── CMakePresets.json
└── Dockerfile
```

## Design rules

Every new experiment should include:

- a clear hypothesis or learning objective;
- deterministic correctness validation;
- GPU-side timing with CUDA Events for kernels;
- a documented performance metric;
- an automated CTest entry;
- Compute Sanitizer compatibility;
- reproducible commands and expected behavior.

Read [docs/PLATFORM.md](docs/PLATFORM.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before adding experiments.

## References

The structure and validation philosophy are informed by:

- NVIDIA CUDA Samples
- NVIDIA CUDA C++ Programming Guide
- NVIDIA CUDA C++ Best Practices Guide
- NVIDIA Compute Sanitizer documentation
- Modern CMake target-based practices

## License

No project license has been selected yet. Add one explicitly before accepting third-party contributions.
