# Experiment 00 — CUDA Device Query

This is the repository's hardware ground truth. Before optimizing a kernel, establish what CUDA runtime is actually visible and which architectural features the selected GPU can support.

## Goal

`device_query.cu` asks the CUDA runtime for the driver/runtime versions, visible device count and a focused set of `cudaDeviceProp` fields. The output anchors later capability gating; a binary compiled for several architectures still executes on one concrete device with one compute capability.

## What the program reads

For each visible GPU the executable prints:

```text
GPU name
compute capability major.minor
global memory GiB
SM count
warp size
max threads per block / per SM
shared memory per block
memory bus width
```

It also prints the numeric CUDA driver and runtime versions. If `cudaGetDeviceCount` returns zero, the program exits with code 2 instead of pretending the environment is usable.

## Build and run

From the repository root:

```bash
cmake --preset release
cmake --build --preset release-build --target aw_device_query
./build/release/bin/aw_device_query
```

On Windows use the corresponding preset/output path documented by the root README.

## How to interpret the fields

Compute capability is the most important dispatch key in this repository. For example, BF16 runtime paths are gated to Ampere-class capability or newer, while the common FP16 path remains available on the supported Turing target. SM count affects available parallelism; warp size explains why later reductions are designed around groups of 32 threads; shared-memory limits constrain tile sizes.

Memory bus width alone is not a bandwidth benchmark. It is a hardware descriptor. Measured bandwidth comes from later experiments and depends on clocking, memory type, access pattern and system state.

## Correctness and failure modes

This experiment does not contain a numerical kernel. Correctness means every CUDA API call succeeds through `CUDA_CHECK` and at least one device is visible. Common failures are container/WSL GPU passthrough problems, an incompatible driver/runtime combination, or running on a host where CUDA libraries exist but no NVIDIA device is exposed.

## Connection to the roadmap

Experiment 00 establishes the facts used by `hardware/rtx_capabilities.json`, GPU Baseline v2 and every later feature gate. Keep its output with benchmark evidence when validating a new RTX generation; otherwise performance numbers lose the architectural context needed to compare them responsibly.
