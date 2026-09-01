# Benchmarking Standard

Accelerateworld treats performance numbers as experimental evidence, not decoration. A benchmark is valid only when the workload is correct, the environment is recorded, warm-up is performed, and the measurement boundary is explicit.

## Measurement layers

### Kernel latency

Use CUDA Events when measuring work executed on a CUDA stream. Report milliseconds or microseconds per iteration after warm-up.

### End-to-end latency

Use host wall-clock timing plus an explicit device/stream synchronization when the experiment includes host/device transfer, launch overhead, Python/framework overhead, graph launch, or orchestration cost.

### Throughput

Choose a workload-specific metric:

- memory kernels: useful GB/s;
- GEMM: GFLOP/s or TFLOP/s;
- transfer pipelines: end-to-end GB/s;
- LLM kernels: latency plus effective tokens/elements/bytes per second where meaningful.

Do not compare metrics with different measurement boundaries.

## Correctness first

Every optimized implementation needs an independent oracle whenever practical:

1. CPU/reference implementation for small workloads;
2. framework/vendor implementation for PyTorch/Triton/cuBLAS experiments;
3. explicit numeric tolerances appropriate to dtype;
4. Compute Sanitizer for CUDA memory/synchronization defects.

An optimized kernel that is faster but fails the oracle is a failed experiment.

## Required environment record

A published result should include:

- GPU model and VRAM;
- compute capability;
- driver version;
- CUDA toolkit/runtime version;
- compiler version;
- OS;
- PyTorch/Triton versions when applicable;
- shape, dtype, batch size and iteration count;
- relevant clocks/power limits if manually changed.

## Benchmark hygiene

- warm up kernels and libraries before timing;
- keep initialization/allocation outside kernel-only timing unless allocation is the subject;
- synchronize at the end of the measured region;
- use enough iterations to suppress launch/timer noise;
- prefer median/percentiles for noisy framework measurements;
- avoid claiming cross-machine speedups without identical workload definitions;
- retain raw logs under `results/` or CI artifacts.

## Performance model

Each experiment should identify its likely bottleneck before optimizing:

- memory bandwidth / coalescing;
- shared-memory bank conflicts;
- occupancy / register pressure;
- instruction throughput;
- launch overhead;
- PCIe transfer;
- Tensor Core utilization;
- framework/compiler overhead.

Later milestones will add Nsight Compute counters and a roofline-style report so hypotheses can be checked against hardware counters instead of inferred from wall time alone.
