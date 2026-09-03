# Experiment 28 — FlashAttention-style Attention

This lab is the point where the Stage 5 exercises stop treating GEMM and softmax as separate operators and begin reasoning about the **dataflow of an attention kernel**. The objective is not to claim parity with production FlashAttention libraries. It is to implement and measure the core idea correctly: generate score tiles, normalize them online, accumulate `P·V`, and never materialize the full attention-score/probability matrix in global memory.

## Goal

A conventional pedagogical attention pipeline is:

```text
Q @ K^T -> [Q,K] scores -> scale/mask -> softmax -> [Q,K] probabilities -> P @ V -> O
```

That form is easy to understand but writes and rereads an `O(QK)` intermediate. This experiment instead studies:

```text
Q row/registers
  -> K/V tile
  -> QK score tile
  -> causal mask + scale
  -> online (m,l)
  -> rescale previous O accumulator
  -> add probability-weighted V tile
  -> next K/V tile
  -> final O / l
```

The full `[Q,K]` matrix never becomes a persistent CUDA tensor in the Triton or custom-CUDA provider.

## Tensor and GQA contract

Inputs are contiguous:

```text
Q: [B, QH, Q, D]
K: [B, KVH, K, D]
V: [B, KVH, K, D]
O: [B, QH, Q, D]
```

`QH % KVH == 0`, so grouped-query attention is represented directly. Query heads are partitioned into contiguous groups that reuse one KV head. The first kernel version supports `D=64` and `D=128`, which keeps one query/output row within a 128-thread custom CUDA block while covering common decoder head dimensions.

For causal execution:

```text
absolute_query = query_start + query_index
valid key      = key_index <= absolute_query
```

This is important for decode and chunked prefill. Using only a local triangular mask would be wrong when `Q < K` and the query chunk belongs near the end of an existing KV prefix.

## FP32 oracle

`reference.py` intentionally materializes the score tensor because the oracle should be simple and auditable rather than memory optimal:

```text
scores = FP32(Q) @ FP32(K)^T / sqrt(D)
masked = scores.masked_fill(invalid, -inf)
P      = stable_softmax(masked)
O      = P @ FP32(V)
```

GQA is expanded explicitly for the oracle/SDPA baselines. `reference_test.py` runs on CPU in hosted CI and compares the manual FP32 implementation with PyTorch scaled-dot-product attention for causal and non-causal cases. A `V=1` invariant additionally proves that normalized attention rows return exactly the all-ones vector within FP32 tolerance.

## Online attention recurrence

Experiment 27 established the composable softmax state `(m,l)`. Here the same state is coupled to an output numerator accumulator `O_acc`.

For a newly produced score `s`:

```text
m_new = max(m_old, s)
alpha = exp(m_old - m_new)
beta  = exp(s - m_new)

l_new   = l_old * alpha + beta
O_acc   = O_acc * alpha + beta * V
m_old   = m_new
```

On the first valid key the implementation uses `alpha=0`, which avoids an undefined `exp(-inf - -inf)` path. At the end:

```text
O = O_acc / l
```

All score, normalization-state and output-accumulator arithmetic is FP32 even when Q/K/V/O storage is FP16 or BF16.

## Custom CUDA provider

`attention_kernel.cu` assigns one CUDA block to one `(batch, q_head, query)` row. With `D<=128`, each active thread owns one head-dimension component:

```text
thread d keeps q[d] in a register
thread d keeps output_acc[d] in FP32

for every valid key:
    each thread computes q[d] * k[key,d]
    warp shuffle + block reduction -> scalar QK score
    thread 0 updates running m/l and alpha/beta
    every active thread updates output_acc[d] using v[key,d]
```

The Q vector is therefore not reread for each key. The output numerator also remains in registers for the complete key scan. Only a few scalar reduction/state values use shared memory.

This is deliberately a correctness/dataflow implementation rather than a claim of production throughput: processing keys sequentially leaves substantial parallelism untapped. Later optimization can tile multiple query rows and key rows, pipeline loads, and use Tensor Core MMA while retaining the same recurrence.

BF16 conversion code is compile-guarded for `__CUDA_ARCH__ >= 800` and runtime-gated to compute capability 8.0+, preserving portable extension builds for `sm_75;sm_86;sm_89;sm_120`.

## Triton provider

`attention_triton.py` maps one program to one query row and scans K/V in `BLOCK_K=16` tiles. The Q vector is loaded once, each K tile produces a vector of scaled scores, and the tile is merged into the running state before its V contribution updates the FP32 output accumulator.

Unlike the custom CUDA v0, the Triton program expresses a complete key tile at once:

```text
K_tile [BK,D]
Q      [D]
score  [BK]
P_tile [BK]
V_tile [BK,D]
O_acc  [D]
```

No `[Q,K]` tensor is allocated. This provider is the clearer stepping stone toward a production tiled implementation, while the handwritten CUDA provider exposes the synchronization and reduction mechanics explicitly.

## PyTorch baselines

The benchmark compares four providers:

```text
pytorch_sdpa
  PyTorch scaled_dot_product_attention

torch_compile_sdpa
  the same SDPA boundary through torch.compile

triton_flash
  score-matrix-free tiled streaming implementation

cuda_flash
  score-matrix-free handwritten CUDA online implementation
```

For GQA, K/V expansion for the PyTorch baseline is prepared outside the timed call. Causal masks are also prepared before timing. This makes the comparison focus on the attention operator boundary; the JSON records the original compact GQA shape so the memory contract remains explicit.

## Shape families

Reduced physical-GPU validation uses small workloads so the self-hosted job remains practical:

```text
validation_decode_gqa:   B1 QH8 KVH2 Q1  K128 D64
validation_prefill:      B1 QH8 KVH8 Q32 K32  D64
validation_chunked_gqa:  B1 QH8 KVH2 Q16 K256 D64, query_start=240
```

Representative families add:

```text
decode:  K=2048 and K=32768, D=128, GQA
prefill: Q/K=128 and 512, D=128
chunked: Q=128, K=4096, query_start=3968
control: non-causal Q=64, K=1024
```

The 32K decode case is especially useful for comparing the theoretical score-matrix footprint with a score-matrix-free implementation, but it is not part of hosted CI and no runtime number is fabricated.

## Build and run

Build the PyTorch CUDA extension:

```bash
cd experiments/28_flash_attention
python setup.py build_ext --inplace
python reference_test.py
```

On a physical GPU with the repository GPU requirements installed:

```bash
python experiments/28_flash_attention/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --warmup 5 \
  --iterations 10 \
  --json results/flash-attention-validation.json
```

The full family is intentionally opt-in:

```bash
python experiments/28_flash_attention/benchmark.py --family all --dtypes fp16
```

## Evidence and metrics

For each shape/dtype/provider the JSON stores latency, logical input/output bandwidth, logical QK+PV TFLOP/s, normalized error against the FP32 oracle, and the provider winner. Each shape also records:

```text
materialized_score_bytes_fp32 = B * QH * Q * K * 4
```

That value is not claimed as the exact workspace of PyTorch SDPA; it is the explicit memory cost of a naive FP32 score matrix that the streaming providers avoid.

Hosted CI proves CPU oracle behavior, static shape/GQA/causal contracts, Python syntax, and multi-architecture CUDA-extension compilation/import. Only a physical GPU run can establish Triton JIT correctness, CUDA runtime numerical error, latency, bandwidth, TFLOP/s, or a provider winner.

## What to inspect next

When profiling on a GPU, answer these questions rather than only recording one latency number:

- Is the v0 CUDA kernel dominated by serial key iteration and block reductions?
- How does decode (`Q=1`) differ from prefill in available parallelism?
- Does GQA reduce K/V traffic enough to be visible at long K?
- At what K does avoiding a materialized score matrix become operationally important?
- Does `torch.compile` change the SDPA boundary or simply preserve a library call?
- Which registers/shared-memory resources limit occupancy in the custom provider?

The next strict ROADMAP item after this experiment is **KV-cache update/read**, where attention stops receiving static K/V tensors and begins interacting with the stateful memory layout of a decoder runtime.
