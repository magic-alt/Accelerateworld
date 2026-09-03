# Experiments — CUDA to LLM Kernel Engineering

Each numbered directory is an executable tutorial, not an isolated code snippet. Read them in order if you want the repository's intended progression from CUDA execution basics to framework integration and production-oriented LLM kernels.

## How to use the tutorials

For native CUDA experiments, configure and build from the repository root:

```bash
cmake --preset release
cmake --build --preset release-build
```

Python/Triton/PyTorch experiments use `python/requirements-gpu.txt` and require a physical CUDA GPU for runtime evidence. Hosted CI may compile CUDA for multiple architectures without possessing a GPU; the individual README files call out that evidence boundary.

## Tutorial index

| # | Directory | Main lesson |
|---:|---|---|
| 00 | `00_device_query` | CUDA driver/runtime/device properties and compute capability |
| 01 | `01_vector_add` | grid/block indexing, bounds checks, event timing, bandwidth |
| 02 | `02_matmul` | naive GEMM versus shared-memory tiled GEMM |
| 03 | `03_reduction` | hierarchical reductions and warp shuffles |
| 04 | `04_memory_coalescing` | coalesced versus strided global-memory access |
| 05 | `05_transpose` | shared-memory transpose, coalescing and bank-conflict padding |
| 06 | `06_streams_pinned` | pinned memory, async copies, multiple streams and overlap |
| 07 | `07_cuda_graph` | launch overhead and CUDA Graph capture/replay |
| 08 | `08_cublas_gemm` | vendor-library SGEMM baseline and FLOP accounting |
| 09 | `09_tensor_core_wmma` | WMMA Tensor Core GEMM with FP16 inputs/FP32 accumulation |
| 10 | `10_pytorch_extension` | dispatcher-backed CUDA op, autograd, FakeTensor and compile integration |
| 11 | `11_triton` | Triton program model through vector add |
| 12 | `12_llm_kernels` | RMSNorm as the first LLM-specific fused reduction kernel |
| 13 | `13_prefix_scan` | scan algorithms and cross-block composition |
| 14 | `14_histogram` | atomics, contention and privatization |
| 15 | `15_async_memory_pool` | `cudaMallocAsync` and memory-pool behavior |
| 16 | `16_stream_ordered_allocator` | stream-ordered allocation semantics |
| 17 | `17_cublaslt_autotune` | cuBLASLt heuristics, workspace and algorithm selection |
| 18 | `18_mixed_precision_gemm` | BF16/TF32 mixed-precision GEMM and capability gates |
| 19 | `19_fp8_gemm` | FP8 experiments on supported architectures |
| 20 | `20_fp4_gemm` | FP4/Blackwell-oriented low-precision experiments |
| 21 | `21_cutlass_gemm` | CUTLASS GEMM across architecture-specific paths |
| 22 | `22_grouped_gemm` | grouped/persistent GEMM scheduling |
| 23 | `23_triton_gemm` | Triton auto-tuned GEMM versus PyTorch/direct cuBLAS |
| 24 | `24_nsys_trace` | Nsight Systems framework/compiler/runtime timeline tracing |
| 25 | `25_swiglu_mixed_precision` | mixed-precision SwiGLU post-op fusion |
| 26 | `26_rope` | RoPE layouts, GQA-style Q/K shapes and long positions |
| 27 | `27_online_softmax` | stable and online softmax with warp/block reductions |
| 28 | `28_flash_attention` | score-matrix-free streaming attention with online normalization |
| 29 | `29_kv_cache` | stateful K/V prefill writes, decode append, layout and attention-compatible reads |

## Evidence model

Every tutorial should distinguish three classes of evidence:

```text
static/source reasoning
        ↓
hosted compile/offline correctness
        ↓
physical GPU runtime/profile evidence
```

A successful `nvcc` build for `sm_75;sm_86;sm_89;sm_120` proves portability of compilation, not runtime performance on four GPUs. Likewise, a Triton file passing Python syntax validation does not prove the kernel JITs or runs. Physical results belong in `results/` and should include GPU identity, software versions and workload parameters.

## Documentation contract

`scripts/check_experiment_docs.py` is run by hosted CI. Each numbered experiment must have a substantive `README.md`, multiple sections, at least one command/code example, and an entry in this index. New experiments should explain the question being tested, algorithm/dataflow, file structure, build/run commands, correctness gates, performance interpretation, hardware requirements and how the experiment connects to the ROADMAP.
