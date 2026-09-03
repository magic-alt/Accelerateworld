# Experiment 06 — Pinned Memory, Async Copies and Multiple Streams

Kernel speed is only part of application performance. This lab compares a pageable, sequential host-device pipeline with a pinned-memory pipeline split across multiple non-blocking CUDA streams.

## Workload

Each iteration performs:

```text
H2D copy -> ScaleKernel -> D2H copy
```

The baseline uses pageable `std::vector` memory and synchronous `cudaMemcpy`. The optimized path allocates host buffers with `cudaMallocHost`, partitions the array into chunks, and submits `cudaMemcpyAsync -> kernel -> cudaMemcpyAsync` on several streams.

## Why pinned memory matters

Asynchronous DMA between host and device requires page-locked host memory for reliable overlap. Pageable transfers can require hidden staging and generally cannot express the same direct asynchronous pipeline. Pinned memory is therefore a resource to use deliberately, not a universal replacement for normal host allocation.

## Build and run

```bash
cmake --preset release
cmake --build --preset release-build --target aw_streams_pinned
./build/release/bin/aw_streams_pinned --elements 16777216 --streams 4 --iterations 10
```

The benchmark uses wall-clock end-to-end timing because copy scheduling and host submission overhead are part of the question. Each iteration synchronizes after all streams have been submitted.

## Interpreting overlap

Multiple streams create the **opportunity** for overlap; they do not guarantee it. Whether H2D, compute and D2H overlap depends on copy engines, kernel resource usage, transfer sizes and GPU architecture. A speedup below 1 is a valid result for some shapes and systems.

## Correctness and profiling

The scale result is checked on the pinned output buffer against the pageable input. Nsight Systems is the preferred tool for this lab: inspect whether copy-engine activity and kernels actually overlap, whether chunks are balanced, and whether submission gaps dominate. These scheduling ideas later reappear in data loading, KV-cache movement and multi-stage inference pipelines.
