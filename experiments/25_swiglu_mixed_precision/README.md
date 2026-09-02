# 25 — SwiGLU Fusion with Mixed Precision

This experiment starts Stage 5 by turning the existing float32 `silu(gate) * up` teaching kernel into an inference-oriented **SwiGLU mixed-precision lab**.

The workload boundary is deliberately explicit:

```text
X [tokens, hidden]
        ↓
packed gate+up projection
X @ W_gate_up [hidden, 2 * intermediate]
        ↓
packed [tokens, 2 * intermediate]
        ↓
FP32 SiLU(gate) + multiply(up)
        ↓
FP16/BF16 output [tokens, intermediate]
```

This PR does **not** claim to fuse the projection GEMM itself into the pointwise CUDA kernel. The packed projection stays on PyTorch/cuBLAS so the experiment can isolate the production boundary that inference engines commonly fuse after a gate-up projection. A later GEMM-epilogue exercise can move the activation into a Tensor Core GEMM epilogue without conflating those two learning goals.

## Numerical contract

### FP32 oracle

For a packed projection tensor:

```python
 gate, up = packed.float().chunk(2, dim=-1)
 reference = silu(gate) * up
```

The fused Triton and CUDA kernels therefore use:

```text
FP16/BF16 load
      ↓
convert to FP32
      ↓
sigmoid / SiLU in FP32
      ↓
FP32 multiply
      ↓
round once to FP16/BF16 output
```

The gate-up projection is also configured to avoid reduced-precision FP16/BF16 GEMM reductions. This keeps the benchmark contract centered on low-precision storage with FP32 reduction rather than allowing backend-specific reduced-precision split-K behavior.

BF16 is capability gated through the same RTX registry used by GPU Baseline v2:

```text
RTX 20 / sm_75   → FP16 only
RTX 30 / sm_86   → FP16 + BF16
RTX 40 / sm_89   → FP16 + BF16
RTX 50 / sm_120  → FP16 + BF16
```

## Providers

The benchmark compares four post-op implementations.

### 1. PyTorch eager

```text
chunk gate/up
    ↓
FP32 casts
    ↓
SiLU
    ↓
multiply
    ↓
cast to low precision
```

This is the correctness-oriented unfused framework baseline.

### 2. `torch.compile`

The same function is wrapped with:

```python
torch.compile(..., fullgraph=True, dynamic=True)
```

Inductor can fuse the pointwise graph while preserving the FP32-intermediate contract.

### 3. Triton

`swiglu_triton.py` uses one program grid over `tokens * intermediate` elements. Each program maps the logical output index back to the packed gate/up locations, performs the SiLU and multiply in FP32 and stores directly to the low-precision output.

### 4. Custom CUDA

`swiglu_kernel.cu` provides vector and scalar paths:

```text
FP16 even intermediate
    → half2 load / half2 store

BF16 even intermediate
    → __nv_bfloat162 load / store

odd intermediate
    → scalar correctness fallback
```

Each vector lane is widened to FP32 before the nonlinear arithmetic. BF16 device arithmetic is compiled only for `__CUDA_ARCH__ >= 800` and the host launcher also rejects BF16 on pre-Ampere GPUs.

## Why packed gate+up projection?

A naïve SwiGLU block performs two projections:

```text
X @ W_gate
X @ W_up
```

Many LLM implementations concatenate those weights so one GEMM produces:

```text
X @ W_gate_up
        ↓
[gate | up]
```

This experiment uses that packed representation. It separates two questions cleanly:

1. how expensive is the Tensor Core projection itself?;
2. after projection, what does pointwise fusion save in kernels, temporary traffic and launch overhead?

## LLM shape families

Reduced physical-GPU CI shapes preserve the classic 4096 → 11008 MLP expansion ratio:

```text
validation_decode   tokens=4   hidden=1024 intermediate=2816
validation_prefill  tokens=64  hidden=1024 intermediate=2816
```

The full LLM family adds representative 7B-class shapes:

```text
llama2_7b_decode_1       tokens=1   hidden=4096 intermediate=11008
llama2_7b_decode_16      tokens=16  hidden=4096 intermediate=11008
llama2_7b_prefill_128    tokens=128 hidden=4096 intermediate=11008
llama2_7b_prefill_512    tokens=512 hidden=4096 intermediate=11008
```

This lets one kernel be studied in both regimes:

```text
decode
small token dimension
projection latency / launch overhead sensitive

prefill
large token dimension
projection throughput dominant
```

## Build

From this directory:

```bash
python setup.py build_ext --inplace
```

Hosted extension CI uses:

```text
TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0"
```

so the custom CUDA extension must compile for the complete RTX 20/30/40/50 portability matrix.

## Run reduced validation

```bash
python benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --warmup 5 \
  --iterations 20 \
  --json ../../results/swiglu-mixed-precision.json
```

On RTX 20, BF16 is recorded as skipped rather than failed.

Run representative LLM shapes explicitly:

```bash
python benchmark.py --family decode --dtypes fp16,bf16
python benchmark.py --family prefill --dtypes fp16,bf16
python benchmark.py --family llm --dtypes fp16,bf16
```

## Timing boundaries

The benchmark reports three related measurements.

### Projection only

```text
preallocated packed output
CUDA Event start
    ↓
torch.mm(X, W_gate_up, out=packed)
    ↓
CUDA Event stop
```

### Post-op only

The packed tensor already exists before timing. This isolates the eager/compile/Triton/custom-CUDA activation and multiply paths.

### Full block

```text
X @ W_gate_up
    ↓
provider-specific SwiGLU
```

`torch.compile` compilation and Triton JIT happen in warmup and therefore are outside steady-state CUDA Event timing. Functional full-block providers allocate their normal outputs; the post-op fused CUDA/Triton path additionally exposes `*_into` APIs so the raw fused-kernel path can reuse output storage.

## Metrics

Post-op reports:

```text
latency_ms
effective_GB/s
winner per shape/dtype
```

The effective-byte model is:

```text
2 low-precision input values + 1 low-precision output value
```

Full block reports:

```text
latency_ms
projection-equivalent TFLOP/s
winner per shape/dtype
```

Projection FLOPs are:

```text
2 * M * K * (2N) = 4 * M * K * N
```

The pointwise operations are intentionally not inflated into the GEMM FLOP metric.

## Correctness

Every post-op provider is checked against the FP32 oracle before timing. Full-block Triton/CUDA/compile paths are checked against the mixed-precision eager block. Reduced validation shapes additionally compute a true FP32 full-block oracle so the numerical effect of low-precision projection and output rounding is retained in JSON evidence.

The JSON record contains:

- GPU / compute capability;
- PyTorch and CUDA versions;
- dtype requirements;
- shape and role;
- FP32/post-op error;
- full-block provider parity;
- mixed-full-block error vs FP32 on validation shapes;
- projection latency;
- post-op provider latency/bandwidth/winner;
- full-block provider latency/TFLOP/s/winner;
- explicit BF16 skips.

## Evidence boundary

Hosted CI can prove:

- Python syntax and offline shape/capability tests;
- FP16/BF16 extension source portability across RTX 20/30/40/50 compile targets;
- vectorized `half2` / `__nv_bfloat162` paths are present alongside scalar fallback;
- BF16 is guarded by the repository capability model.

Only a physical GPU runner can prove:

- real FP16/BF16 correctness;
- Inductor/Triton JIT execution;
- actual vectorized CUDA runtime behavior;
- post-op bandwidth;
- projection/full-block latency and shape-dependent winners.

No hosted CI result is presented as a physical-GPU benchmark.

## References

- PyTorch CUDA semantics and reduced-precision GEMM controls: https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch numerical accuracy: https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- CUDA BFloat16 precision intrinsics: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html
- Triton language documentation: https://triton-lang.org/main/python-api/triton.language.html
