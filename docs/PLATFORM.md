# CUDA Experiment Platform

## 1. Purpose

Accelerateworld is an experiment-oriented CUDA repository. The unit of work is not a source file; it is a reproducible experiment with a hypothesis, implementation, correctness check, performance metric, hardware capability contract and captured result.

This design borrows the category-oriented organization of NVIDIA CUDA Samples while adding project-level CI, CTest, sanitizer, benchmark, feature-gating and contribution rules.

## 2. Experiment lifecycle

Each experiment should follow:

```text
Question
  -> Hypothesis
  -> Capability requirements
  -> CPU/reference baseline
  -> CUDA implementation
  -> Correctness validation
  -> GPU-side measurement
  -> Profiling
  -> Optimization
  -> Re-measurement
  -> Documented conclusion
```

A performance claim is not accepted unless the correctness test still passes.

## 3. Directory convention

```text
experiments/
  NN_experiment_name/
    CMakeLists.txt
    *.cu
```

Use ascending numeric prefixes to keep the learning path obvious.

Future experiments should add a local `README.md` once they become more than a single concept.

## 4. Build and architecture model

The project uses target-based CMake and C++17/CUDA C++17.

Important cache variables:

- `ACCELERATEWORLD_CUDA_ARCHITECTURES`
- `ACCELERATEWORLD_ENABLE_TESTING`
- `ACCELERATEWORLD_ENABLE_FAST_MATH`

Portable CI currently targets the primary RTX compute capabilities:

- `sm_75` — RTX 20 / Turing
- `sm_86` — RTX 30 / Ampere
- `sm_89` — RTX 40 / Ada
- `sm_120` — RTX 50 / Blackwell

Local benchmark execution should normally build only for the detected physical GPU. `scripts/run_gpu_baseline.py` performs this automatically.

Generation-specific CMake presets (`rtx20`, `rtx30`, `rtx40`, `rtx50`) exist for explicit compile validation. `rtx5060` is retained only as a compatibility/reference alias for RTX 50.

## 5. Hardware capability model

`hardware/rtx_capabilities.json` is the capability registry used by GPU Baseline v2.

The runtime flow is:

```text
nvidia-smi
  -> exact GPU + compute capability
  -> SM profile
  -> RTX generation
  -> feature flags
  -> benchmark feature gates
```

Feature flags are deliberately conservative. Architecture-sensitive benchmarks must declare requirements rather than assume that every RTX generation implements the same numerical formats or specialized instructions.

When a new RTX generation appears, update the registry only after verifying the official NVIDIA compute capability and selected CUDA Toolkit support.

## 6. Correctness policy

Every kernel experiment needs a deterministic oracle.

Preferred order:

1. CPU reference implementation for small input sizes.
2. Trusted CUDA/library implementation.
3. Cross-implementation comparison only when a CPU/library oracle is impractical.

Tests should report a numeric error metric, not only a boolean.

## 7. Measurement policy

Kernel timing should use CUDA Events rather than host wall-clock timers.

Each benchmark should document:

- problem size;
- dtype;
- required hardware features;
- warmup behavior;
- number of measured iterations;
- latency;
- throughput metric (GB/s, GFLOP/s, items/s, etc.);
- GPU model and compute capability;
- CUDA toolkit and driver versions.

Host-to-device/device-to-host copies must be clearly separated from kernel-only measurements unless end-to-end latency is the explicit metric.

## 8. Benchmark manifest and feature gating

`benchmarks/manifest.json` defines canonical workload shapes, required features and the primary metric used for cross-GPU comparison.

A benchmark result has one of three states:

- `passed`: supported and completed successfully;
- `failed`: supported but execution/correctness failed;
- `skipped`: the detected GPU lacks one or more declared features.

A skip is not silently converted into a pass.

## 9. Debugging and sanitization

Real GPU validation runs NVIDIA Compute Sanitizer on representative kernels. Capability-specific checks are selected only when the target experiment is valid for the detected GPU.

A sanitizer failure is considered a correctness failure even when numerical output appears correct.

## 10. Profiling roadmap

After correctness and basic benchmarking, use:

- Nsight Systems for timeline/CPU-GPU interaction;
- Nsight Compute for kernel-level metrics;
- occupancy and achieved memory bandwidth;
- global-memory load/store efficiency;
- shared-memory bank conflicts;
- warp execution efficiency;
- Tensor Core / math-pipe utilization where applicable.

Profiler output is intentionally not committed by default because it is machine-specific and can be large.

## 11. CI split

### Build CI

Runs on a normal GitHub-hosted CPU runner inside an NVIDIA CUDA development container.

It verifies:

- JSON/configuration validity;
- Baseline v2 offline unit tests;
- CMake configuration;
- NVCC compilation for RTX 20/30/40/50 SM targets;
- test registration.

It **does not** claim runtime correctness.

### GPU validation

Runs only on a GPU-capable runner.

It verifies:

- exact GPU discovery;
- native SM resolution;
- runtime correctness;
- sanitizer cleanliness;
- feature-gated benchmark execution;
- evidence generation.

This split avoids the anti-pattern of treating “NVCC compiled” as “CUDA code was tested.”

## 12. Result provenance

GPU Baseline v2 emits `baseline.json` following `benchmarks/gpu-baseline-v2.schema.json`.

The evidence records:

```text
commit SHA
GPU model
RTX generation
compute capability / SM
feature flags
driver version
CUDA toolkit
OS
benchmark workload
iterations
command line
parsed metrics
raw stdout/stderr
validation state
```

Cross-generation comparison should use `scripts/compare_gpu_baselines.py` and equivalent workload definitions. Never compare performance numbers from different environments without labeling the hardware/software differences.
