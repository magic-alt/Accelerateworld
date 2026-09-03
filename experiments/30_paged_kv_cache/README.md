# Experiment 30 — Paged KV Cache

This lab moves from experiment 29's **contiguous stateful KV cache** into the page-table model used by modern LLM inference runtimes. The performance question is no longer only “how fast can K/V bytes be copied?” It is also:

> Can logical token positions be decoupled from physical memory placement without making decode append and long-context attention reads prohibitively expensive?

The experiment deliberately separates **runtime metadata management** from **GPU data movement**. A deterministic CPU page allocator owns the free list, page reuse and fragmentation accounting. PyTorch, Triton and handwritten CUDA kernels consume an already-built block table and do only logical-to-physical address translation plus K/V movement.

## 1. From contiguous cache to paged cache

Experiment 29 stores each sequence in one logical contiguous capacity. The paged model instead uses fixed-size physical pages:

```text
logical tokens
0 ........ 15 | 16 ....... 31 | 32 ....... 47
      block 0  |      block 1  |      block 2
           \             |              /
            \            |             /
             block table lookup
                    ↓
physical page 8 | physical page 2 | physical page 14
```

For a logical token position `p`:

```text
logical_block = p // page_size
slot          = p % page_size
physical_page = block_table[batch, logical_block]
```

K/V is then addressed inside that physical page.

## 2. Tensor and metadata contract

Physical K and V page pools use one canonical layout:

```text
page_k/page_v: [physical_pages, KVH, page_size, head_dim]
block_table:   [batch, max_blocks] int64
positions:     [batch, tokens] int64

update input:  [batch, KVH, tokens, head_dim]
read output:   [batch, KVH, tokens, head_dim]
```

`head_dim` stays contiguous. The cache stores only **KV heads**, so MQA and GQA reduce cache footprint naturally; K/V is never expanded to query-head count.

Two page sizes are represented in the shape matrix: 16 and 32 tokens/page. Smaller pages reduce internal fragmentation but enlarge block tables and increase mapping frequency; larger pages do the opposite.

## 3. Block table lifecycle

`allocator.py` owns the CPU-side runtime model. It creates a deterministic fragmented physical-page order, then allocates blocks round-robin across sequences. This makes each sequence's logical pages intentionally non-contiguous while preserving a stable test case.

The GPU kernels do **not** allocate or free pages:

```text
PageAllocator / free list
        ↓
block_table creation or update
        ↓
immutable metadata for timed region
        ↓
PyTorch / Triton / CUDA update/read kernel
```

This boundary matters. Timing Python allocator work inside a decode kernel benchmark would measure host bookkeeping rather than the GPU memory path.

## 4. Decode page append

A decode append writes one new token:

```text
new K/V [B, KVH, 1, D]
position [B, 1]
        ↓
position // page_size
        ↓
logical block
        ↓
block_table lookup
        ↓
physical page + slot
        ↓
K/V write
```

Some shapes intentionally set `prefill_tokens % page_size == 0`. The first decode token therefore starts a **new page**, exercising the page-table transition rather than repeatedly writing into an already-open page.

The page allocator obtains that page before the timed kernel region. This experiment measures the append data path; Stage 6 will integrate allocator scheduling with the runtime.

## 5. Cross-page attention-compatible read

The read API accepts arbitrary logical positions and returns `[B, KVH, T, D]`. A context read may cross many non-contiguous physical pages. The returned tensor is contiguous and directly compatible with an attention consumer, making the benchmark useful for comparing paged read overhead against experiment 29's contiguous head-major CUDA path.

## 6. Fragmentation model

### Internal fragmentation

The last allocated page for a sequence is often only partially used:

```text
allocated_slots = ceil(tokens / page_size) * page_size
internal_unused = allocated_slots - tokens
```

The benchmark records both the unused slot count and ratio.

### External fragmentation

Free physical pages may be scattered. The allocator records the largest contiguous free run and computes a simple external-fragmentation indicator:

```text
1 - largest_free_run / free_pages
```

A page-based runtime does not require a large contiguous free region to extend a sequence, which is precisely the flexibility this model is intended to expose.

## 7. Free list and page reuse

`allocator_reuse_demo()` proves that pages released by one sequence return to the free list and are reused by a later sequence:

```text
sequence A allocates pages
        ↓
sequence A finishes
        ↓
release(A)
        ↓
pages return to free list
        ↓
sequence C reuses released pages
```

This is an offline/runtime-metadata test, not a GPU timing result.

## 8. PyTorch reference

`reference.py` is the correctness oracle. It performs the same two-level address translation:

```python
logical_blocks = torch.div(positions, page_size, rounding_mode="floor")
slots = positions.remainder(page_size)
physical_pages = block_table.gather(1, logical_blocks)
```

It validates logical bounds, verifies that requested block-table entries are allocated, and checks that physical page IDs stay inside the page pool.

CPU validation covers page sizes 16/32, cross-page reads/writes, intentionally non-contiguous mappings, attention-compatible output layout, rejection of an unallocated logical block, and free-list release/reuse. Because KV-cache update/read is pure movement, correctness is **exact**, not tolerance-based.

## 9. Triton mapping

`paged_kv_triton.py` launches one program for each `(batch, KV head, token)`. The program loads the logical position, performs block-table translation, then vectorizes over `head_dim`:

```text
program id
   ↓
(b, head, token)
   ↓
position
   ↓
logical block + slot
   ↓
physical page
   ↓
contiguous D load/store
```

## 10. Handwritten CUDA mapping

`paged_kv_kernel.cu` preserves experiment 29's pair-copy strategy. Two 16-bit FP16/BF16 values are moved as one raw `uint32_t`, so no half or BF16 arithmetic/rounding is introduced.

Each thread owns one pair and performs:

```text
linear pair index
        ↓
(batch, head, token, pair)
        ↓
position
        ↓
logical_block = position / page_size
slot          = position % page_size
        ↓
physical_page = block_table[batch, logical_block]
        ↓
page pair offset
```

The kernel retains in-device guards. Public APIs default to host-visible validation; benchmarks disable repeated checks only after immutable positions and block tables have been validated once.

## 11. Contiguous-cache baseline

Experiment 29's **head-major custom CUDA cache** is reused as the contiguous baseline:

```text
cuda_contiguous
vs
pytorch_paged
vs
triton_paged
vs
cuda_paged
```

This directly measures the latency added by page-table indirection relative to an equivalent contiguous custom CUDA movement path.

## 12. Benchmark phases and metrics

`benchmark.py` separates prefill bulk write, decode single-token append, and cross-page context read. It records latency, logical GB/s, physical page-pool bytes, block-table bytes, contiguous reserved bytes, internal/external fragmentation, whether physical pages are non-contiguous, paged-CUDA/contiguous-CUDA latency ratios and phase winners.

“Logical GB/s” counts equivalent K+V movement; it is not a measured DRAM-transaction claim.

## 13. Shape families

Reduced validation includes MQA and GQA:

```text
validation_mqa_cross_page
B=2, QH=8, KVH=1, page=16, prefill=16, D=64

validation_gqa_fragmented
B=2, QH=8, KVH=2, page=16, prefill=31, D=128
```

Representative families include MQA decode to 2K, GQA decode to 4K, a page-size-32 prefill case, 2K prefill, and GQA long decode with 32K logical capacity.

## 14. Build and run

Build both the contiguous baseline and paged extension:

```bash
pushd experiments/29_kv_cache
python setup.py build_ext --inplace
popd

pushd experiments/30_paged_kv_cache
python setup.py build_ext --inplace
python reference_test.py
popd
```

Run reduced physical-GPU validation:

```bash
python experiments/30_paged_kv_cache/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --warmup 5 \
  --iterations 20 \
  --json results/paged-kv-cache-validation.json
```

## 15. Hosted CI versus physical GPU evidence

Hosted CI can prove allocator/block-table algebra, CPU correctness, unallocated-page rejection, free-list reuse, source-level mapping, Python syntax, portable `sm_75/sm_86/sm_89/sm_120` extension compilation and import/export.

Hosted CI cannot prove Triton JIT/runtime correctness, physical GPU exact-copy behavior, decode latency, block-table cache behavior, L2/DRAM traffic, a page-size winner, a paged-vs-contiguous winner or 32K performance. Those claims require the physical-GPU JSON artifact.

## 16. Profiling questions

1. Does decode append remain launch-latency dominated after adding one block-table load?
2. Does page-table indirection materially change long-context read throughput?
3. Is the block table resident in cache for steady-state decode?
4. How do 16-token and 32-token pages trade metadata traffic against fragmentation?
5. Does GQA make address-translation overhead proportionally more visible?
6. Does non-contiguous page order change L2 behavior relative to contiguous cache?

## 17. Deliberate limitations

This is an educational runtime/kernel boundary, not a complete inference scheduler: there is no concurrent scheduler, prefix sharing/copy-on-write, CUDA-side page allocator, eviction/offload policy or fused paged-attention kernel. Allocator time is intentionally excluded from GPU timings.

The next ROADMAP item moves into **quantize/dequantize kernels**. Stage 6 later returns to allocator integration, continuous batching and an end-to-end decoder runtime.
