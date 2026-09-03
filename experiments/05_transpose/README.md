# Experiment 05 — Matrix Transpose and Shared-memory Padding

Transpose is a compact demonstration of two memory problems at once: making global-memory loads/stores coalesced and avoiding shared-memory bank conflicts while changing data layout.

## Goal

The naive kernel reads rows contiguously but writes the transposed matrix with a large stride. The tiled kernel uses a 32×32 shared-memory tile so both global reads and global writes can be coalesced.

```text
global input -> shared tile -> synchronization -> transposed indexing -> global output
```

Only 32×8 threads are used; each thread processes four rows of the 32×32 tile.

## Why the tile is 32x33

The declaration is:

```cpp
__shared__ float tile[32][33];
```

The extra column changes the shared-memory row stride. Without padding, reading the tile by transposed indices can make many threads hit the same bank. Padding by one breaks that pathological alignment while adding only a small amount of shared storage.

## Build and run

```bash
cmake --preset release
cmake --build --preset release-build --target aw_transpose
./build/release/bin/aw_transpose --size 2048 --iterations 30
```

The program reports latency, effective read+write GB/s and naive/tiled speedup. CUDA events time the kernels after warm-up.

## Correctness

The final tiled result is copied to the host and every element is checked against `input[col*N + row]`. Edge guards allow matrix sizes that do not exactly fill the final tile.

## Profiling questions

Inspect global load/store efficiency and shared-memory bank-conflict metrics. Then remove the `+1` padding locally and compare. This experiment is a template for reasoning about layout transformations later in Tensor Core input staging, RoPE variants and KV-cache page layouts: a mathematically trivial permutation can be performance-critical because of how addresses map to memory hardware.
