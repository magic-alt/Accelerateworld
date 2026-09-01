# CUDA Experiment Platform

## 1. Purpose

Accelerateworld is an experiment-oriented CUDA repository. The unit of work is not a source file; it is a reproducible experiment with a hypothesis, implementation, correctness check, performance metric, and captured result.

This design borrows the category-oriented organization of NVIDIA CUDA Samples while adding project-level CI, CTest, sanitizer, benchmark, and contribution rules.

## 2. Experiment lifecycle

Each experiment should follow:

```text
Question
  -> Hypothesis
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

## 4. Build model

The project uses target-based CMake and C++17/CUDA C++17.

Important cache variables:

- `ACCELERATEWORLD_CUDA_ARCHITECTURES`
- `ACCELERATEWORLD_ENABLE_TESTING`
- `ACCELERATEWORLD_ENABLE_FAST_MATH`

The default portable build currently targets:

- `sm_75` — Turing/T4 cloud validation
- `sm_86` — Ampere
- `sm_89` — Ada
- `sm_120` — Blackwell consumer/workstation GPUs including RTX 5060

For local RTX 5060 iteration, prefer only `120` to reduce compile time.

## 5. Correctness policy

Every kernel experiment needs a deterministic oracle.

Preferred order:

1. CPU reference implementation for small input sizes.
2. Trusted CUDA/library implementation.
3. Cross-implementation comparison only when a CPU/library oracle is impractical.

Tests should report a numeric error metric, not only a boolean.

## 6. Measurement policy

Kernel timing should use CUDA Events rather than host wall-clock timers.

Each benchmark should document:

- problem size;
- warmup behavior;
- number of measured iterations;
- latency;
- throughput metric (GB/s, GFLOP/s, items/s, etc.);
- GPU model and compute capability;
- CUDA toolkit and driver versions.

Host-to-device/device-to-host copies must be clearly separated from kernel-only measurements unless end-to-end latency is the explicit metric.

## 7. Debugging and sanitization

Real GPU validation runs NVIDIA Compute Sanitizer:

- `memcheck`
- `racecheck`
- `initcheck`
- `synccheck`

A sanitizer failure is considered a correctness failure even when numerical output appears correct.

## 8. Profiling roadmap

After correctness and basic benchmarking, use:

- Nsight Systems for timeline/CPU-GPU interaction;
- Nsight Compute for kernel-level metrics;
- occupancy and achieved memory bandwidth;
- global-memory load/store efficiency;
- shared-memory bank conflicts;
- warp execution efficiency.

Profiler output is intentionally not committed by default because it is machine-specific and can be large.

## 9. CI split

### Build CI

Runs on a normal GitHub-hosted CPU runner inside an NVIDIA CUDA development container.

It verifies:

- CMake configuration;
- NVCC compilation;
- all configured GPU architectures;
- test registration.

It **does not** claim runtime correctness.

### GPU validation

Runs only on a GPU-capable runner.

It verifies:

- CUDA device visibility;
- runtime correctness;
- sanitizer cleanliness;
- benchmark execution.

This split avoids the common anti-pattern of treating “NVCC compiled” as “CUDA code was tested.”

## 10. Result provenance

When recording benchmark numbers, include:

```text
commit SHA
GPU
compute capability
driver version
CUDA toolkit
OS
power limit (if changed)
clock policy (if changed)
problem size
iterations
command line
```

Never compare performance numbers from different environments without labeling the hardware/software differences.
