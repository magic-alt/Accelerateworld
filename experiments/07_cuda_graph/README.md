# Experiment 07 — CUDA Graph Launch Overhead

This experiment targets CPU/runtime submission overhead rather than GPU arithmetic throughput. It is most relevant when individual kernels are small and a fixed sequence is replayed many times, as in decode loops.

## Workload

One iteration launches two kernels:

```text
AffineKernel: temp = alpha * input + beta
ReluKernel:   output = max(temp, 0)
```

The direct path submits those kernels normally for every iteration. The graph path captures the two launches once on a non-blocking stream, instantiates a `cudaGraphExec_t`, and repeatedly calls `cudaGraphLaunch`.

## Build and run

```bash
cmake --preset release
cmake --build --preset release-build --target aw_cuda_graph
./build/release/bin/aw_cuda_graph --elements 65536 --iterations 1000
```

Default N is deliberately modest so launch overhead is visible. If you make each kernel very expensive, both paths become GPU-work dominated and the graph advantage should shrink as a fraction of total time.

## Timing boundary

The benchmark uses host steady-clock time and one synchronization after each repeated sequence, because the question is end-to-end submission/replay overhead. Graph capture and instantiation are outside the replay timing; they are setup costs amortized across repeated iterations.

## Correctness

After the graph run, output is copied to host and checked against the scalar affine+ReLU expression. Capture/replay optimization is never allowed to change numerical semantics.

## Production connection

CUDA Graphs require stable execution structure and stable memory addresses. That constraint becomes a runtime design issue for LLM decode: buffers, batch shapes and KV-cache addresses must be organized so replay remains valid. Stage 6 later revisits this idea after the repository has stateful cache management and a decoder loop.
