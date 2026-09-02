# Experiment 27 — Online Softmax for Attention Scores

This lab is the reduction/numerical-stability bridge between the earlier standalone LLM kernels and the next FlashAttention-style attention experiment.

## Question

How should a production-oriented row softmax over attention scores be implemented when:

- logits are stored in FP16/BF16;
- max, exponentials, sums and online state use FP32;
- rows range from short decode contexts to 32K keys;
- causal masking must be fused into the kernel;
- a traditional stable two-pass reduction and a streaming online reduction must agree;
- PyTorch eager/Inductor, Triton and handwritten CUDA must share one correctness contract?

## Tensor contract

Scores are contiguous:

```text
scores [batch, heads, query_length, key_length]
```

Each `(batch, head, query)` row is independently normalized across `key_length`.

For causal shapes:

```text
absolute_query_position = query_start + query_index
valid key               = key_index <= absolute_query_position
masked output           = exactly 0
```

The shape table includes ordinary prefill, decode, long decode and chunked prefill where `query_length < key_length`.

## Stable softmax baseline

The auditable reference is:

```text
m = max(x)
y = exp(x - m)
l = sum(y)
p = y / l
```

Subtracting the row maximum is mandatory. The synthetic benchmark intentionally adds row shifts between roughly `-80` and `+80` so an unstable low-precision `exp(x)` implementation is easy to detect.

The FP32 oracle computes max/exp/sum/normalization in FP32 and applies the same causal mask before the reduction.

## Online max/sum recurrence

For a state `(m, l)` where:

```text
m = maximum value observed so far
l = sum(exp(x_i - m)) over values observed so far
```

and another state `(m_b, l_b)`, the merge is:

```text
m_new = max(m, m_b)
l_new = l   * exp(m   - m_new)
      + l_b * exp(m_b - m_new)
```

A single new scalar `x` is the state `(x, 1)`.

This merge is associative up to floating-point roundoff. That property is the key reason the state works for thread, warp, block and later FlashAttention tile composition.

`online_math.py` supplies a dependency-free implementation, and hosted offline tests compare it with stable softmax using extreme logits.

## Custom CUDA variants

### `cuda_two_pass`

One CUDA block owns one softmax row:

```text
thread-strided local max
        ↓
warp max reduction
        ↓
block max reduction
        ↓
thread-strided exp/sum
        ↓
warp sum reduction
        ↓
block sum reduction
        ↓
normalize + store
```

Warp reductions use `__shfl_down_sync`; only one value per warp is staged through shared memory for the block-level merge.

### `cuda_online`

Each thread scans its strided subset of the row and forms a local online `(m,l)` state:

```text
thread online state
        ↓
warp online-state merge
        ↓
shared state per warp
        ↓
warp-0 block-state merge
        ↓
row (m,l)
        ↓
second input scan + normalize
```

The kernel still makes a second input pass to emit probabilities. The important change is that max and denominator are one composable state rather than two independent global reduction phases. The next FlashAttention experiment can carry this same state while score tiles are produced and consumed on chip.

## Triton online kernel

The Triton implementation deliberately does not require an entire 32K row to fit in one program-local vector.

It scans fixed-size key tiles:

```text
for key tile:
    tile_max
    tile_sum
    merge with running (m,l)

for key tile:
    exp(x - m) / l
    store
```

This permits the same kernel contract to cover short rows and long-context rows. Causal masking is applied at load time and masked probabilities are stored as zero.

The official Triton fused-softmax tutorial provides the fused stable-softmax baseline, while the Triton fused-attention tutorial demonstrates the running-max/running-denominator structure that this experiment isolates before attention fusion.

## Dtypes

```text
FP16 input/output
    FP32 max/exp/sum/state

BF16 input/output
    FP32 max/exp/sum/state
```

Capability gating follows the repository model:

```text
RTX20 / sm_75  → FP16, BF16 skipped
RTX30 / sm_86  → FP16 + BF16
RTX40 / sm_89  → FP16 + BF16
RTX50 / sm_120 → FP16 + BF16
```

BF16 CUDA device conversion code is guarded with `__CUDA_ARCH__ >= 800`, and runtime dispatch also rejects BF16 below compute capability 8.0. This preserves portable `sm_75;sm_86;sm_89;sm_120` extension builds.

## Providers

The physical-GPU benchmark compares:

```text
pytorch_eager
    stable FP32 math + low-precision output

torch_compile
    same graph under Dynamo/Inductor

triton_online
    tiled online (m,l) recurrence

cuda_two_pass
    explicit max reduction then sum reduction

cuda_online
    composable online state with warp/block merge
```

For causal PyTorch baselines the boolean mask is constructed before timing. Triton/CUDA derive the causal predicate from row/query indices inside the kernel and never materialize a full mask tensor.

Provider calls return a fresh output tensor, so the measured boundary includes normal framework output allocation for every provider rather than timing only preallocated custom outputs.

## Shape families

### Reduced physical-GPU validation

```text
validation_decode_128
B=1 H=8 Q=1  K=128
causal, query_start=127

validation_prefill_32
B=1 H=8 Q=32 K=32
causal, query_start=0

validation_chunked_1024
B=1 H=8 Q=32 K=1024
causal, query_start=992
```

### Decode

```text
K=128
K=2048
K=32768
```

All have `Q=1`; the absolute query position is the final key position, so the row can attend to the entire cache.

### Prefill

```text
Q=128 K=128
Q=512 K=512
Q=128 K=4096, query_start=3968  # chunked prefill
```

### Non-causal control

A non-causal `Q=64, K=1024` case separates mask overhead from pure normalization behavior.

## Correctness gates

Each physical-GPU provider must satisfy:

- finite output;
- normalized error versus FP32 oracle within dtype-specific tolerance;
- row sums close to 1;
- causal masked positions exactly zero.

The CPU-hosted FP32 reference additionally checks large positive/negative logit shifts and causal masking without requiring a GPU.

## Metrics

For each shape/dtype/provider the JSON evidence retains:

```text
latency_ms
logical_gbps
logical elements/s
normalized_error_vs_fp32
row_sum_error
masked_max_abs
winner
```

`logical_gbps` counts one low-precision input read plus one low-precision output write. It is a workload-normalized comparison metric, not a claim about the actual internal memory traffic of eager or two-pass implementations.

## Build

```bash
cd experiments/27_online_softmax
python setup.py build_ext --inplace
```

Hosted extension CI builds with:

```text
TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0"
```

## Run

Reduced validation:

```bash
python experiments/27_online_softmax/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --warmup 5 \
  --iterations 20 \
  --json results/online-softmax-validation.json
```

Long decode:

```bash
python experiments/27_online_softmax/benchmark.py --family decode --dtypes fp16,bf16
```

Prefill/chunked prefill:

```bash
python experiments/27_online_softmax/benchmark.py --family prefill --dtypes fp16,bf16
```

## Evidence boundary

Hosted CI proves only:

- Python syntax;
- dependency-free online-state algebra tests;
- attention-shape/causal contracts;
- CPU FP32 oracle behavior;
- CUDA extension compile/link for RTX20/30/40/50 targets;
- extension import/export on a no-GPU runner;
- whole-repository regression build.

Hosted CI does **not** prove:

- physical-GPU FP16/BF16 correctness;
- Triton JIT success;
- Inductor runtime behavior;
- measured warp/block reduction latency;
- decode/prefill winners;
- 32K-row throughput.

Those results are retained only by the self-hosted physical-GPU workflow as `results/online-softmax-validation.json`.

## Why this precedes FlashAttention

FlashAttention does not materialize the complete score matrix and then call a separate softmax. It streams score tiles and carries the same running normalization state while updating the output accumulator.

This experiment isolates the numerical/reduction invariant first:

```text
stable softmax
      ↓
online (m,l) state
      ↓
warp/block composition
      ↓
causal tiled rows
      ↓
FlashAttention-style tiled QK + online softmax + PV
```

The next strict ROADMAP item after this PR is **FlashAttention-style attention**.
