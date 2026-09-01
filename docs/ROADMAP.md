# GPU / AI Infrastructure Roadmap

The repository is organized as a progression from CUDA execution fundamentals to kernels used in LLM systems.

## Stage 0 — Reproducible CUDA platform

- [x] device query
- [x] CMake/CMakePresets
- [x] CUDA 13 compile CI
- [x] RTX 5060 `sm_120` target
- [x] CTest + Compute Sanitizer path
- [x] Docker/devcontainer

## Stage 1 — Memory and parallel primitives

- [x] vector add
- [x] reduction
- [x] coalesced vs strided access
- [x] naive vs tiled/padded transpose
- [x] shared-memory tiled matrix multiply
- [ ] warp shuffle reduction
- [ ] prefix scan
- [ ] histogram / atomics contention

## Stage 2 — Scheduling and execution overhead

- [x] pinned memory
- [x] multiple CUDA streams
- [x] overlap copy and compute
- [x] CUDA Graph capture/replay
- [ ] asynchronous memory pools (`cudaMallocAsync`)
- [ ] stream-ordered allocator experiments

## Stage 3 — Math libraries and Tensor Cores

- [x] cuBLAS SGEMM baseline
- [x] WMMA FP16/FP32 Tensor Core GEMM
- [ ] cuBLASLt heuristic/autotune experiments
- [ ] BF16/TF32/FP8 comparisons
- [ ] CUTLASS GEMM
- [ ] persistent/grouped GEMM

## Stage 4 — Framework and compiler integration

- [x] PyTorch CUDA Extension build path
- [x] dispatcher-registered fused CUDA op
- [x] Triton vector-add baseline
- [ ] PyTorch autograd registration
- [ ] FakeTensor/meta kernel and `torch.compile`
- [ ] Triton auto-tuned GEMM
- [ ] Nsight Systems framework trace

## Stage 5 — LLM kernel lab

- [x] RMSNorm v0
- [x] fused SiLU*mul custom CUDA op
- [ ] SwiGLU fusion with mixed precision
- [ ] RoPE
- [ ] online softmax
- [ ] FlashAttention-style attention
- [ ] KV-cache update/read
- [ ] paged KV cache
- [ ] quantize/dequantize kernels
- [ ] INT8/INT4 weight-only GEMM
- [ ] fused dequantize + GEMM

## Stage 6 — Inference runtime

- [ ] minimal transformer decoder using repository kernels
- [ ] continuous batching
- [ ] KV-cache allocator
- [ ] CUDA Graph decode path
- [ ] speculative decoding experiment
- [ ] benchmark against PyTorch eager/compile and a production inference engine

## Stage 7 — Performance engineering

- [ ] Nsight Compute automated profiles
- [ ] Nsight Systems timelines
- [ ] roofline analysis
- [ ] occupancy/register/shared-memory reports
- [ ] benchmark history and regression thresholds
- [ ] multi-GPU NCCL experiments

The target outcome is not to reimplement an entire production framework. It is to build enough of each layer to understand where performance comes from, how correctness is maintained, and how CUDA kernels integrate into an AI runtime.
