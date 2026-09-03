# Experiment 04 — Global-memory Coalescing

This experiment isolates one of the highest-leverage CUDA rules: threads in a warp should access nearby addresses whenever the algorithm permits it.

## Goal

`CoalescedReadKernel` maps logical element i to `input[i]`. `StridedReadKernel` maps the same output i to `input[i*stride]`. Both write the same number of output values and execute almost no arithmetic; only the physical input-address pattern changes.

```text
coalesced warp: input[0], input[1], input[2], ...
strided warp:   input[0], input[S], input[2S], ...
```

The default stride is 32, deliberately spreading neighboring threads across distant locations.

## Fair comparison

The program allocates `elements * stride` physical input entries so every strided read is valid. It reports **useful** bandwidth using the logical input+output bytes, not the larger physical address span. This makes the number easy to interpret while reminding you that actual DRAM transactions can be much larger for the strided case.

## Build and run

```bash
cmake --preset release
cmake --build --preset release-build --target aw_memory_coalescing
./build/release/bin/aw_memory_coalescing --elements 1048576 --stride 32 --iterations 30
```

Sweep stride through 1, 2, 4, 8, 16 and 32. `stride=1` should collapse the two access patterns into essentially the same experiment.

## Correctness and timing

An initialization kernel writes a deterministic pattern over the full physical allocation. Each provider is warmed before CUDA-event timing. The final strided output is copied to host and checked against the exact source index used by each logical thread.

## What to learn

Coalescing is about memory transactions, not source-code aesthetics. A kernel can launch many threads and still waste bandwidth if a warp touches many cache lines for little useful data. Use Nsight Compute to compare requested/useful bytes with actual memory-sector traffic. This concept reappears in transpose, RoPE layout choices, KV-cache layouts and quantized weight packing.
