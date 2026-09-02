# Experiment 26 — RoPE kernel lab

This experiment moves Stage 5 from MLP activation fusion into attention positional-kernel engineering. It compares a common Q/K RoPE contract across PyTorch eager, `torch.compile`/Inductor, Triton and handwritten CUDA while preserving the repository's cross-generation capability rules.

## Workload contract

```text
Q [batch, sequence, q_heads, head_dim]
K [batch, sequence, kv_heads, head_dim]
position_ids [batch, sequence]
FP32 cos/sin cache [max_position, rotary_dim / 2]
        ↓
rotate first rotary_dim channels
        ↓
copy tail channels unchanged
        ↓
Q_rot / K_rot in FP16 or BF16
```

Q and K may have different head counts, which makes the validation representative of grouped-query attention. They must share batch, sequence, head dimension and dtype.

## Two RoPE layouts

### Interleaved

Pairs are adjacent:

```text
(x0, x1), (x2, x3), ...
```

For each pair and position-dependent angle:

```text
y0 = x0 * cos - x1 * sin
y1 = x0 * sin + x1 * cos
```

The custom CUDA path maps this naturally to `half2` / `__nv_bfloat162` vector loads and stores.

### Half-split

Pairs span the two halves of the rotary prefix:

```text
(x[0], x[R/2]), (x[1], x[R/2+1]), ...
```

The same rotation is applied, but the partner values are not adjacent in memory. The custom CUDA kernel therefore keeps one logical pair per thread but uses scalar loads/stores. This layout difference is intentionally visible in the benchmark rather than hidden behind a synthetic transpose.

## Numerical contract

The cache is always FP32. The reference path casts Q/K to FP32 before rotation:

```text
FP16/BF16 input
      ↓
FP32 x0/x1
      +
FP32 cos/sin cache
      ↓
FP32 multiply/add/subtract
      ↓
FP16/BF16 store
```

The FP32 oracle additionally checks pair-norm preservation, which follows from RoPE being a 2D rotation. When `rotary_dim < head_dim`, every provider must copy the unrotated tail exactly.

## Cache model

`build_rope_cache()` precomputes:

```text
inv_freq[i] = theta ^ (-2i / rotary_dim)
angle[p, i] = p * inv_freq[i]
cos[p, i], sin[p, i]
```

Cache construction is deliberately outside the timing boundary. This experiment studies the runtime kernel used by a decoder with an existing positional cache, not the one-time cost of constructing trigonometric tables.

Long-context shapes use explicit high position IDs (32K and 128K) so correctness is tested beyond short prompt positions without changing the kernel contract.

## Providers

### PyTorch eager

The eager oracle uses regular tensor indexing and FP32 arithmetic, then casts once to the input dtype.

### `torch.compile`

Separate interleaved and half-split functions are compiled with:

```python
torch.compile(..., fullgraph=True, dynamic=True)
```

This lets Inductor fuse indexing/rotation/cast operations while retaining dynamic batch/sequence/head dimensions.

### Triton

The Triton kernel uses one logical work item for either:

- one rotary pair, or
- one tail element.

Q and K are launched independently because GQA permits different head counts.

### Custom CUDA

For interleaved layout:

```text
FP16 → half2
BF16 → __nv_bfloat162
```

For half-split layout, one thread still owns both members of the logical pair, but reads the two separated addresses directly.

BF16 device code is protected with `__CUDA_ARCH__ >= 800`; runtime also rejects BF16 below compute capability 8.0. This keeps the same extension source portable across RTX 20/30/40/50 compile targets while recording BF16 as unsupported on Turing.

## Shape families

Reduced self-hosted validation:

```text
validation_decode
  B=1 S=1 QH=32 KVH=8 D=128 R=128 pos=127

validation_decode_long
  B=1 S=1 QH=32 KVH=8 D=128 R=128 pos=32767

validation_prefill_partial
  B=1 S=64 QH=32 KVH=8 D=128 R=64 pos=4096..4159
```

Representative decode:

```text
S=1, QH=32, KVH=8, D=128
position = 0 / 32767 / 131071
```

Representative prefill:

```text
S=128 / 512, QH=32, KVH=8, D=128
full rotary and partial rotary cases
```

## Metrics

Each shape × dtype × layout record retains:

- latency for eager / compile / Triton / custom CUDA;
- logical GB/s based on Q/K input + output traffic;
- million tensor elements/s;
- normalized error against the FP32 oracle;
- FP32 pair-norm preservation error;
- exact-tail-copy validation for partial rotary dimensions;
- provider winner;
- cache dimensions/bytes and position range.

`logical_gbps` intentionally excludes cache traffic because cache reuse depends on head count and hardware cache behavior. It is a normalization metric, not a claim of measured DRAM bandwidth.

## Run

```bash
cd experiments/26_rope
python setup.py build_ext --inplace
python reference_test.py
python benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --layouts interleaved,half_split \
  --json ../../results/rope-validation.json
```

On RTX 20, BF16 is recorded as skipped. RTX 30/40/50 run BF16 when PyTorch reports CUDA BF16 support.

## Evidence boundary

Hosted CI can prove:

- Python source and offline configuration logic;
- CPU FP32 oracle, norm-preservation and partial-tail correctness;
- CUDA extension compilation for `7.5;8.6;8.9;12.0`;
- BF16 guards do not break the RTX 20 compile target;
- extension import/export on a no-GPU runner.

Only the self-hosted physical GPU workflow can prove actual FP16/BF16 runtime correctness, Inductor/Triton/custom-CUDA kernel execution and latency.

## Next ROADMAP item

After this experiment merges, the next strict Stage 5 item is **online softmax**.
