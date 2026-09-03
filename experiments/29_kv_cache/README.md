# Experiment 29 — KV-cache Update / Read

This lab is the first explicitly **stateful LLM inference memory** experiment in Accelerateworld. Experiments 26–28 operate on Q/K/V tensors presented to an attention call; this experiment asks how K/V state is laid out, appended during decode, written during prefill, and gathered back into the attention-compatible form needed by later runtime work.

## Question and contract

The logical update tensor is always:

```text
new K/V [batch, kv_heads, tokens, head_dim]
positions [batch, tokens]
```

The read API always returns:

```text
K/V [batch, kv_heads, tokens, head_dim]
```

Only KV heads are stored. For GQA or MQA, `q_heads` is metadata used to describe the consumer relationship; the cache does **not** duplicate a KV head once per query head. That distinction matters because cache capacity is often one of the dominant memory costs of long-context inference.

Two physical layouts are compared:

```text
token_major: [B, capacity, KVH, D]
head_major:  [B, KVH, capacity, D]
```

Both keep `D` contiguous. Token-major groups all KV heads belonging to one token, while head-major makes the time axis for one KV head adjacent at the outer tensor level. The experiment measures the consequence rather than assuming one layout is universally superior.

## Position-to-slot mapping and bounds safety

`positions[b,t]` is the cache slot used by token `t` in batch item `b`. Prefill normally uses a contiguous sequence such as `0..T-1`; decode appends one token at `position=T`; the reference implementation also tests non-contiguous positions.

Public reference/Triton/CUDA APIs default to a bounds check:

```text
0 <= min(position)
max(position) < capacity
```

For CUDA tensors that safety check requires reading min/max back to the host and therefore synchronizes. The benchmark validates its immutable positions once before timing, then calls the low-level providers with `check_bounds=False`. The CUDA kernels still include an in-kernel range guard, but performance evidence is only valid because the benchmark has already established the host-side contract.

This separation is deliberate: production runtimes commonly validate allocator/page-table state outside the innermost decode copy kernel.

## Prefill bulk write and decode append

The same update API covers two very different regimes:

```text
prefill:
[B, KVH, many_tokens, D]
        ↓
bulk cache write

decode:
[B, KVH, 1, D]
        ↓
single-slot append
```

Prefill is a bandwidth/parallelism workload; decode exposes launch and small-transfer overhead. The benchmark reports them separately instead of averaging them into one number.

## Custom CUDA mapping

`kv_cache_kernel.cu` moves two 16-bit elements per thread using a raw `uint32_t` pair:

```text
FP16/BF16 pair (2 × 16 bits)
        ↓
32-bit vector load
        ↓
layout-dependent cache offset
        ↓
32-bit vector store
```

No arithmetic is performed on K/V values, so raw pair movement preserves the exact FP16/BF16 bit pattern. `head_dim` is restricted to 64 or 128 in v0, both naturally even and aligned for pair access.

Flat pair offsets are:

```text
token_major:
(((b * capacity + position) * KVH + head) * (D/2) + pair)

head_major:
(((b * KVH + head) * capacity + position) * (D/2) + pair)
```

The update kernel reads `[B,KVH,T,D]` and writes the selected cache slots. The read kernel performs the reverse gather and always produces the attention-compatible `[B,KVH,T,D]` output.

BF16 runtime follows the repository policy: RTX20/Turing records BF16 as skipped; Ampere/Ada/Blackwell execute BF16 when PyTorch reports support. The kernel itself is a bitwise copy, which keeps portable compilation simple across `sm_75;sm_86;sm_89;sm_120`.

## Triton mapping

`kv_cache_triton.py` assigns one Triton program to each `(batch, kv_head, token)` row and vectorizes across `D`:

```text
program_id
   ↓
(b, kv_head, token)
   ↓
load position
   ↓
load contiguous D-vector from new K/V or cache
   ↓
compute token-major/head-major base offset
   ↓
store contiguous D-vector
```

This is intentionally simpler than paged attention. Experiment 29 establishes contiguous-cache semantics first; the next ROADMAP item will replace the direct `position -> slot` mapping with a page/block table.

## PyTorch oracle

`reference.py` is both the CPU oracle and a GPU framework baseline. It uses `scatter_` for update and `gather` for read. CPU `reference_test.py` verifies:

- token-major and head-major layouts;
- bulk prefill writes;
- single-token decode append;
- non-contiguous slot mapping;
- attention-compatible reads;
- out-of-capacity rejection.

Because KV-cache movement is bit-preserving, physical-GPU correctness uses exact tensor equality rather than a floating-point tolerance.

## Shape families

Reduced physical validation includes an MQA decode shape and a GQA prefill shape:

```text
validation_mqa_decode:
B=1, QH=8, KVH=1, capacity=128, prefill=32, D=64

validation_gqa_prefill:
B=1, QH=8, KVH=2, capacity=256, prefill=64, D=128
```

Representative families additionally include:

```text
MQA decode: capacity 2048
GQA decode: capacity 4096
GQA prefill: 512 / 2048 tokens
long decode: capacity 32768, KVH=8, D=128
```

Each shape is run for both layouts and every supported dtype.

## Benchmark phases and metrics

For each provider (`pytorch`, `triton`, `cuda`) the benchmark records:

```text
prefill_write_ms
prefill_write_logical_gbps

decode_append_ms
decode_append_us
decode_append_logical_gbps

context_read_ms
context_read_logical_gbps
```

Logical traffic counts K+V source/destination movement. It is useful for comparing equivalent providers in this lab, but it is not a hardware DRAM counter. Nsight Compute is required later for actual memory-transaction and cache-hit evidence.

Read provider APIs allocate their returned K/V tensors, so the read timing is an API-level latency rather than a pure preallocated copy-kernel number. The custom extension also exposes `read_out` for future allocator/runtime integration.

## Build and run

Install the GPU Python environment, build the extension, and run reduced validation:

```bash
python -m pip install -r python/requirements-gpu.txt
cd experiments/29_kv_cache
python setup.py build_ext --inplace
python reference_test.py
cd ../..
python experiments/29_kv_cache/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --layouts token_major,head_major \
  --warmup 5 \
  --iterations 20 \
  --json results/kv-cache-validation.json
```

Use `--family decode`, `--family prefill`, or `--family long` for larger physical-GPU studies.

## Hosted CI versus physical evidence

Hosted CI proves:

- Python/offline shape and layout logic;
- CPU reference semantics and bounds handling;
- extension source can compile for RTX20/30/40/50 targets;
- no-GPU import exposes `update`, `read`, and `read_out`;
- the 30 numbered experiment tutorials remain complete.

Hosted CI does **not** prove Triton JIT execution, physical-GPU exact-copy correctness, decode append latency, context-read bandwidth, layout winners, or long-context behavior. Those claims require the self-hosted GPU workflow and its retained JSON artifact.

## What to inspect next

Useful profiling questions include:

- Does token-major improve decode append locality when all KV heads for one token arrive together?
- Does head-major improve long context reads for a single KV head?
- At `T=1`, is launch/API overhead larger than the payload copy time?
- Does 32-bit pair access compile into the expected vectorized global load/store instructions?
- How much of a 32K read is served from L2 versus DRAM?
- How does GQA reduce cache footprint relative to storing one K/V stream per query head?

The next strict Stage 5 item is **paged KV cache**. Experiment 29 intentionally uses direct contiguous storage so that page-table lookup, fragmentation and non-contiguous block gathering can be isolated in that next lab instead of being mixed into this baseline.
