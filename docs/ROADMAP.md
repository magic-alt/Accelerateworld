# GPU / AI Infrastructure Roadmap

The repository is organized as a progression from CUDA execution fundamentals to kernels used in LLM systems. GPU Baseline v2 makes the roadmap cross-generation: RTX 20/Turing, RTX 30/Ampere, RTX 40/Ada and RTX 50/Blackwell share one capability-aware benchmark/evidence model.

## Stage 0 — Reproducible CUDA platform

- [x] device query
- [x] CMake/CMakePresets
- [x] CUDA 13 portable compile CI for `sm_75;sm_86;sm_89;sm_120`
- [x] RTX 20/30/40/50 generation presets
- [x] runtime GPU detection + capability registry
- [x] benchmark feature gating
- [x] unified GPU Baseline v2 JSON schema
- [x] cross-GPU baseline comparison tool
- [x] RTX 5060 as first reference GPU (not a special execution path)
- [x] CTest + Compute Sanitizer path
- [x] Docker/devcontainer
- [ ] collect real baseline evidence from at least one GPU in each supported RTX generation

## Stage 1 — Memory and parallel primitives

- [x] vector add
- [x] reduction
- [x] coalesced vs strided access
- [x] naive vs tiled/padded transpose
- [x] shared-memory tiled matrix multiply
- [x] warp shuffle reduction
- [x] prefix scan
- [x] histogram / atomics contention

## Stage 2 — Scheduling and execution overhead

- [x] pinned memory
- [x] multiple CUDA streams
- [x] overlap copy and compute
- [x] CUDA Graph capture/replay
- [x] asynchronous memory pools (`cudaMallocAsync`)
- [x] stream-ordered allocator experiments

## Stage 3 — Math libraries and Tensor Cores

- [x] cuBLAS SGEMM baseline
- [x] WMMA FP16/FP32 Tensor Core GEMM
- [x] cuBLASLt heuristic/autotune experiments
- [x] BF16/TF32 comparisons with capability gates
- [x] FP8 comparisons on supported architectures
- [x] FP4 experiments on supported Blackwell targets
- [x] CUTLASS GEMM
- [x] persistent/grouped GEMM

## Stage 4 — Framework and compiler integration

- [x] PyTorch CUDA Extension build path
- [x] dispatcher-registered fused CUDA op
- [x] portable PyTorch extension compile targets for RTX 20/30/40/50
- [x] Triton vector-add baseline
- [x] PyTorch autograd registration
- [x] FakeTensor/meta kernel and `torch.compile`
- [x] Triton auto-tuned GEMM
- [x] Nsight Systems framework trace

## Stage 5 — LLM kernel lab

Every new LLM benchmark must declare its hardware requirements in the same feature-gating model used by GPU Baseline v2. Common FP16 kernels should run across supported RTX generations; BF16/FP8/FP4 paths should be skipped when the detected GPU does not support the required feature.

- [x] RMSNorm v0
- [x] fused SiLU*mul custom CUDA op
- [x] SwiGLU fusion with mixed precision
- [x] RoPE
- [x] online softmax
- [x] FlashAttention-style attention
- [x] KV-cache update/read
- [x] paged KV cache
- [ ] quantize/dequantize kernels
- [ ] INT8/INT4 weight-only GEMM
- [ ] fused dequantize + GEMM
- [ ] cross-generation benchmark matrix for representative prefill/decode shapes

## Stage 6 — Inference runtime

- [ ] minimal transformer decoder using repository kernels
- [ ] prefill benchmark with TTFT / tokens-per-second evidence
- [ ] decode benchmark with TPOT / tokens-per-second evidence
- [ ] continuous batching
- [ ] KV-cache allocator
- [ ] CUDA Graph decode path
- [ ] speculative decoding experiment
- [ ] benchmark against PyTorch eager/compile and a production inference engine
- [ ] compare identical decoder workloads across RTX generations using GPU Baseline v2 metadata

## Stage 7 — Performance engineering

- [ ] Nsight Compute automated profiles
- [ ] Nsight Systems timelines
- [ ] roofline analysis
- [ ] occupancy/register/shared-memory reports
- [ ] benchmark history and regression thresholds
- [ ] architecture-normalized comparison (bandwidth utilization, Tensor Core utilization, launch overhead)
- [ ] multi-GPU NCCL experiments

The target outcome is not to reimplement an entire production framework. It is to build enough of each layer to understand where performance comes from, how correctness is maintained, how architectural capabilities change optimization choices, and how CUDA kernels integrate into an AI runtime.
