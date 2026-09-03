# Experiment 01 — Vector Add

Vector addition is the smallest useful CUDA kernel in the repository and the right place to learn launch geometry, bounds checks, event timing and bandwidth accounting without arithmetic complexity hiding the memory behavior.

## Goal and kernel

The kernel computes:

```text
C[i] = A[i] + B[i]
```

One CUDA thread owns one element. `index = blockIdx.x * blockDim.x + threadIdx.x`, followed by `if (index < n)`, is the canonical 1-D grid-stride boundary pattern for a problem that does not necessarily divide evenly by the block size.

The program launches 256 threads per block and rounds the block count upward.

## Memory and performance model

Each useful element causes two float loads and one float store. The benchmark therefore reports useful traffic as:

```text
bytes = 3 * N * sizeof(float)
effective_bandwidth = bytes / kernel_time
```

This kernel has very low arithmetic intensity, so performance should primarily track global-memory throughput once N is large enough. The reported GB/s is a useful-traffic metric, not a direct DRAM transaction counter.

## Timing methodology

The program performs an untimed warm-up launch and device synchronization, then records CUDA events around repeated kernel launches. Host-to-device setup and the final device-to-host correctness copy are outside the timed kernel interval.

Run it with:

```bash
cmake --preset release
cmake --build --preset release-build --target aw_vector_add
./build/release/bin/aw_vector_add --elements 16777216 --iterations 50
```

Small N is useful for debugging; large N is required before interpreting bandwidth.

## Correctness

Host vectors are initialized deterministically. After timing, C is copied back and compared element-by-element with a double-precision host expression. The maximum absolute error must remain below `1e-6`.

If this validation fails, investigate indexing, launch size, stale asynchronous errors and memory lifetime before looking at performance.

## What to study

Use Nsight Compute later to connect this simple source to memory transactions, achieved bandwidth and occupancy. Change block size experimentally, but do not conclude that the fastest block size here is universally best. The next memory experiments deliberately break coalescing and add shared-memory reuse so you can see why access pattern matters more than memorizing a single launch configuration.
