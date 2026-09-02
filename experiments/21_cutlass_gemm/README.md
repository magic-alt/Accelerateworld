# CUTLASS GEMM Lab

This stage moves below cuBLASLt's algorithm-dispatch interface and makes the GEMM kernel configuration visible.

The repository pins **NVIDIA CUTLASS 4.7.0**. The source archive is consumed header-only by the Accelerateworld targets; CUTLASS examples, tests and profiler are not built as part of the normal project build.

## Questions

1. How do threadblock, warp and MMA instruction shapes affect the same GEMM shape?
2. How does pipeline depth trade shared-memory residency against latency?
3. How large is the Tensor Core advantage over a CUTLASS SIMT reference?
4. Why does a shape-dependent winner change for square, wide-M and wide-N GEMMs?
5. How does the legacy CUTLASS 2.x kernel configuration model relate to the modern `CollectiveBuilder -> GemmUniversal -> GemmUniversalAdapter` hierarchy?
6. How does a hand-selected/measured CUTLASS configuration compare with the cuBLASLt autotuner from experiment 17?
7. Which parts of FP8/FP4 are kernel-configuration concerns versus quantization/scaling concerns already isolated in experiments 19 and 20?

## Targets

### `aw_cutlass_gemm_simt`

A FP32 CUTLASS `OpClassSimt` reference. It exists so Tensor Core acceleration is measured against another CUTLASS kernel rather than against a handwritten educational matrix multiply.

### `aw_cutlass_gemm_sm75`

FP16 input / FP32 accumulation using explicit `cutlass::gemm::device::Gemm` TensorOp templates. Three candidates expose the traditional CUTLASS hierarchy directly:

| Candidate | Threadblock | Warp | MMA instruction | Pipeline |
|---|---:|---:|---:|---:|
| square | 128x128x32 | 64x64x32 | 16x8x8 | default SM75 stages |
| wide-N | 64x128x32 | 32x64x32 | 16x8x8 | default SM75 stages |
| wide-M | 128x64x32 | 64x32x32 | 16x8x8 | default SM75 stages |

Every candidate is warmed up and timed. The measured lowest-latency configuration is selected for a second correctness launch.

The architecture tag is a **minimum architecture**, so this target remains useful on later GPUs as a controlled legacy-style configuration reference.

### `aw_cutlass_gemm_sm80`

Uses CUTLASS `device::GemmUniversal` on the Ampere TensorOp programming model. FP16 and BF16 run independent candidate sets:

- 128x128x32 / 64x64x32 / 3 stages;
- 64x128x32 / 32x64x32 / 3 stages;
- 128x128x32 / 64x64x32 / 4 stages.

Besides latency and throughput, the experiment records CUTLASS kernel shared-storage size and `maximum_active_blocks()` as simple resource-pressure evidence. These are not substitutes for Nsight Compute register/occupancy counters; those belong in Stage 7.

### `aw_cutlass_gemm_sm120`

A Blackwell GeForce specialization using the modern CUTLASS hierarchy:

```text
CollectiveBuilder (mainloop)
        +
CollectiveBuilder (epilogue)
        ↓
GemmUniversal kernel
        ↓
GemmUniversalAdapter device API
```

The reference uses FP8 E4M3 input / FP32 accumulation and output with:

```text
TileShape      = 128 x 64 x 64
ClusterShape   = 1 x 1 x 1
Stage policy   = StageCountAutoCarveout
Kernel schedule= KernelScheduleAuto
```

The input values are deliberately chosen from exactly representable E4M3 values. This target is about **CUTLASS architecture specialization**, not a second FP8 quantization experiment. Tensorwide scaling, saturation and accuracy studies stay in experiment 19.

CUTLASS 4.7.0's own SM120 TensorOp unit tests gate this specialization on the **feature-specific `sm_120a`/`sm_120f` architecture family**, rather than ordinary `sm_120` code generation. Accelerateworld therefore keeps its generic RTX 50 project target at compute capability `120`, but compiles this one native CUTLASS target as `120a-real;120a-virtual`. That distinction is important: compute capability identifies the GPU generation, while the `a` target enables architecture-specific Blackwell instructions required by this CUTLASS kernel family.

## Architecture-aware build

The CMake file intersects each target with `ACCELERATEWORLD_CUDA_ARCHITECTURES`:

```text
native RTX 20 / sm75
    simt + sm75 targets

native RTX 30 / sm86
    simt + sm75 + sm80 targets

native RTX 40 / sm89
    simt + sm75 + sm80 targets

native RTX 50 / sm120
    simt + sm75 + sm80 + sm120 target
                         └─ native CUTLASS codegen: sm_120a
```

The hosted portable build still compiles all three architecture tiers, but a native Turing build is not forced to compile a Blackwell-only kernel. The SM120 target intentionally mirrors NVIDIA CUTLASS 4.7.0's feature-specific Blackwell code-generation contract instead of treating plain `sm_120` as interchangeable with `sm_120a`.

## Correctness boundary

All native targets use deterministic matrices. FP16/BF16/FP8 CPU references are built from the **actual narrowed values**, then selected output coordinates are recomputed with FP64 accumulation. This separates kernel/layout mistakes from expected input quantization.

Timing contains repeated GEMM launches only. Allocation, H2D copies, candidate setup and validation D2H copies stay outside the CUDA Event interval.

## Shape-dependent dispatch

Try representative inference shapes rather than only square GEMMs:

```bash
# square / prefill-like
./build/release/bin/aw_cutlass_gemm_sm80 --m 4096 --n 4096 --k 4096 --mode fp16

# small-M / decode-like
./build/release/bin/aw_cutlass_gemm_sm80 --m 64 --n 4096 --k 4096 --mode fp16

# wide output projection
./build/release/bin/aw_cutlass_gemm_sm80 --m 1024 --n 8192 --k 4096 --mode bf16
```

The point is not that three hard-coded candidates form a production autotuner. They make the dispatch dimensions explicit enough to understand why production runtimes maintain architecture/shape/dtype-specific kernel tables.

## cuBLASLt comparison

Experiment 17 already runs a fresh cuBLASLt heuristic shortlist and measured autotune. Use the same `M/N/K`, precision and timing policy, then compare:

```text
cuBLASLt measured winner
        vs
CUTLASS measured winner
```

GPU Baseline v2 keeps both records so the comparison can be performed from retained evidence rather than from copied console numbers.

## CUTLASS profiler reference

CUTLASS 4.7.0's profiler is intentionally optional because building the full operation library is much heavier than compiling this repository's selected templates.

From a checked-out CUTLASS 4.7.0 tree:

```bash
cmake -S . -B build-profiler \
  -DCUTLASS_NVCC_ARCHS=80 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_CUBLAS=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
cmake --build build-profiler --target cutlass_profiler --parallel

./build-profiler/tools/profiler/cutlass_profiler \
  --operation=Gemm \
  --m=1024 --n=1024 --k=1024
```

Profiler output includes operation class, CTA shape, warp count, instruction shape, stages and minimum compute capability. It is the broad-search reference; the native Accelerateworld targets are the small, readable configuration lab.

For Ada/Blackwell FP8 and Blackwell NVFP4, use the profiler to enumerate architecture-native kernels and compare them with the already isolated scaling experiments in `19_fp8_gemm` and `20_fp4_gemm`. This stage deliberately does not duplicate those quantizers.

## Evidence boundary

Hosted CI can prove:

- CUTLASS 4.7.0 source pinning and header availability;
- architecture-selective target configuration;
- CUDA/NVCC compilation of the explicit template configurations, including feature-specific `sm_120a` code generation for the Blackwell target;
- CTest discovery and Baseline v2 feature gating.

Only a physical GPU can establish:

- actual tile winner for each shape;
- SIMT/TensorOp speedup;
- shared-memory/occupancy behavior at runtime;
- CUTLASS vs cuBLASLt GFLOP/s;
- Blackwell SM120 Tensor Core execution;
- profiler winners and Compute Sanitizer results.
