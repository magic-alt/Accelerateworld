# Experiment 32 — INT8 / INT4 Weight-Only GEMM

This experiment is the next strict Stage 5 LLM-kernel step after standalone quantize/dequantize kernels.

The question is no longer only:

> Can a tensor be converted to INT8 or packed INT4 correctly?

It is now:

> Can an FP16/BF16 activation multiply a low-bit weight matrix while the kernel reads scales/zero-points, unpacks the weight fragment, performs dequantization close to use, and accumulates the matrix product in FP32?

## 1. Why weight-only GEMM matters

Decoder inference repeatedly evaluates linear layers:

```text
A [M,K] @ W^T [K,N] -> C [M,N]
```

For many decode workloads `M` is very small. The same large weight matrix must be read for only one or a few activation rows. That makes weight traffic a first-order cost.

Weight-only quantization keeps the activation in FP16/BF16 while compressing the stored weight:

```text
FP16 weight  -> 2 bytes / element
INT8 weight  -> 1 byte  / element
INT4 weight  -> 0.5 byte / element
```

Ignoring qparam metadata, the storage reduction relative to FP16 is therefore approximately:

```text
INT8: 2x
INT4: 4x
```

That does **not** automatically mean a 2x/4x GEMM speedup. The kernel must also pay for unpacking, scale lookup, optional zero-point subtraction, address arithmetic and the actual multiply-accumulate work.

## 2. Exact matrix contract

The experiment uses:

```text
activation A: [M,K], FP16 or BF16

quantized weight Wq:
  INT8 -> [N,K], torch.int8
  INT4 -> [N,ceil(K/2)], torch.uint8

scales:
  FP32

zero_points:
  INT32

output C:
  [M,N], same storage dtype as activation
```

Mathematically:

```text
W[n,k] = (Wq[n,k] - zero_point[n,k]) * scale[n,k]

C[m,n] = sum_k FP32(A[m,k]) * FP32(W[n,k])
```

The output is rounded once to FP16/BF16.

## 3. Reused experiment-31 storage format

Experiment 31 established these formats:

```text
int8_sym
  q range [-127,127]
  zero-point = 0

int8_asym
  q range [-128,127]
  learned zero-point

int4_sym
  q range [-8,7]
  signed two's-complement nibble
```

Packed INT4 keeps exactly the same byte layout:

```text
byte b

low nibble  = Wq[n, 2*j]
high nibble = Wq[n, 2*j+1]
```

Sign extension is:

```text
nibble 0..7  ->  0..7
nibble 8..15 -> -8..-1
```

An odd `K` leaves the final high nibble unused.

## 4. Qparam granularity

Three qparam layouts are supported.

### Per tensor

```text
one scale / zero-point
for all N*K weights
```

### Per output channel

```text
one scale / zero-point
for each row W[n,:]
```

This is the natural channel interpretation for a linear layer because each `n` is one output channel.

### Group-wise

```text
W row n
  ├─ K[0:group_size]       -> qparam[n,0]
  ├─ K[group_size:2*group] -> qparam[n,1]
  └─ ...
```

Flattened qparam index:

```text
pidx = n * groups_per_row + k / group_size
```

Supported educational group sizes are 32, 64 and 128.

## 5. Important boundary: reference vs weight-only kernel

The FP32 PyTorch correctness oracle is allowed to materialize the dequantized matrix:

```text
low-bit Wq
    ↓
full FP32 W
    ↓
A.float() @ W.T
```

That is deliberately simple and auditable.

The Triton and handwritten-CUDA providers are **not** allowed to materialize a complete floating-point weight matrix in global memory.

Their dataflow is:

```text
quantized K tile
      ↓
unpack / int conversion
      ↓
scale + zero-point lookup
      ↓
floating weight fragment
      ↓
immediate GEMM consumption
```

This distinction is checked by offline source-contract tests.

## 6. Custom CUDA kernel

`weight_only_kernel.cu` uses a transparent 16x16 tiled kernel.

For every K tile:

```text
global activation
      ↓
FP32 shared A tile

quantized global weight
      ↓
INT8 load
or packed INT4 byte+nibble decode
      ↓
qparam lookup
      ↓
FP32 shared W tile

shared A × shared W
      ↓
FP32 fmaf accumulator
```

Pseudo-code:

```cpp
for (int k0 = 0; k0 < K; k0 += TILE) {
    A_shared = float(A[m, k0 + ...]);

    q = load_low_bit_weight(...);
    pidx = qparam_index(n, k);
    W_shared = (q - zero[pidx]) * scale[pidx];

    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk)
        acc = fmaf(A_shared[...][kk], W_shared[kk][...], acc);

    __syncthreads();
}
```

This is an educational kernel, not a claim of production-optimal Tensor Core scheduling.

## 7. FP16 and BF16

Activation and output storage support:

```text
FP16 -> all supported RTX generations

BF16 -> Ampere / Ada / Blackwell runtime
```

The BF16 device conversion code is guarded by:

```cpp
__CUDA_ARCH__ >= 800
```

and the runtime also rejects a BF16 launch on compute capability below 8.0.

The extension still compiles a portable fatbin for:

```text
sm_75
sm_86
sm_89
sm_120
```

Compile portability is not the same thing as physical BF16 runtime support on Turing.

## 8. Triton provider

The Triton kernel maps one program to an output tile:

```text
BLOCK_M x BLOCK_N
```

and loops over `BLOCK_K`.

For INT8:

```text
qweight[n,k]
```

is loaded directly.

For INT4:

```text
byte = packed[n, k//2]

even k -> byte & 0xF
odd  k -> (byte >> 4) & 0xF
```

followed by signed-nibble conversion.

The weight fragment is then scaled and fed directly into:

```python
tl.dot(...)
```

with FP32 accumulation.

## 9. Why decode and prefill behave differently

For `M=1`:

```text
one activation row
      ↓
entire weight matrix read
      ↓
very little weight reuse
```

This tends to be highly weight-bandwidth-sensitive.

As `M` grows:

```text
same weight tile
      ↓
reused across more activation rows
      ↓
more FLOPs per weight byte
```

The experiment records a logical arithmetic-intensity estimate:

```text
2*M*N*K FLOPs
/
(
  activation bytes
  + quantized weight bytes
  + qparam bytes
  + output bytes
)
```

It is a workload descriptor, not a substitute for Nsight Compute hardware counters.

## 10. Shape families

Validation:

```text
M=3 N=5  K=65   # odd packed-INT4 tail
M=8 N=16 K=128  # group-wise qparam validation
```

Decode:

```text
M=1  N=4096  K=4096
M=16 N=4096  K=4096
M=8  N=11008 K=4096
```

Prefill:

```text
M=128 N=4096 K=4096
M=512 N=4096 K=4096
```

The larger cases are physical-GPU benchmark workloads, not hosted-CI runtime tests.

## 11. Providers

The physical benchmark exposes four useful baselines:

```text
pytorch_materialized_dense
```

Dequantized weight is prepared before timing. This isolates dense GEMM after compression has already been removed.

```text
pytorch_dequant_then_gemm
```

A full floating weight matrix is reconstructed inside each timed call and then multiplied. This is the intentionally unfused end-to-end reference.

```text
triton_weight_only
```

Low-bit weight is consumed in the Triton K loop.

```text
cuda_weight_only
```

Low-bit weight is consumed in the handwritten CUDA K loop.

## 12. Correctness

The independent CPU/GPU PyTorch oracle performs:

```python
weight = dequantize_weight_reference(...)
reference = activation.float() @ weight.float().T
```

Every runtime provider is compared against this FP32 result.

The experiment records:

```text
RMSE
max absolute error
normalized max error
```

Different weight formats are not expected to have the same quantization error. INT4 is intentionally lower precision than INT8.

## 13. Benchmark metrics

Each record includes:

```text
latency_ms
logical_GB/s
logical_TFLOP/s
quantized_weight_bytes
FP16_weight_bytes
weight_compression_ratio
qparam_count
logical_arithmetic_intensity
expected_regime
provider winner
```

The byte count is a logical workload model. It does not claim actual DRAM bytes observed by the hardware.

## 14. Build

On a CUDA-capable Linux host:

```bash
cd experiments/32_weight_only_gemm
python setup.py build_ext --inplace
```

The hosted extension workflow compiles for all repository targets:

```text
TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0"
```

## 15. CPU reference validation

No physical GPU is required for:

```bash
cd experiments/32_weight_only_gemm
python reference_test.py
```

This checks:

```text
INT8 symmetric
INT8 asymmetric
packed INT4
group-wise qparams
odd K
GEMM algebra
```

## 16. Physical GPU benchmark

Reduced evidence family:

```bash
python experiments/32_weight_only_gemm/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --formats int8_sym,int4_sym \
  --granularity group \
  --group-size 64 \
  --json results/weight-only-gemm-validation.json
```

For decode:

```bash
python experiments/32_weight_only_gemm/benchmark.py \
  --family decode \
  --dtypes fp16,bf16 \
  --formats int8_sym,int4_sym
```

For prefill:

```bash
python experiments/32_weight_only_gemm/benchmark.py \
  --family prefill \
  --dtypes fp16,bf16 \
  --formats int8_sym,int4_sym
```

## 17. Evidence boundary

Hosted CI can prove:

```text
shape/qparam/storage algebra
INT4 codec contract
CPU PyTorch reference semantics
Python syntax
tutorial completeness
CUDA extension compilation for sm_75/sm_86/sm_89/sm_120
extension import/export
```

Hosted CI cannot prove:

```text
Triton JIT success
physical CUDA correctness
RTX 5060 latency
memory bandwidth
L2 behavior
Tensor Core utilization
INT4 vs INT8 winner
decode vs prefill crossover
```

Those require a physical GPU result artifact.

## 18. What this experiment intentionally does not claim

The handwritten CUDA v0 uses FP32 shared-memory fragments and scalar `fmaf`.

It does **not** claim:

```text
INT8 Tensor Core MMA
INT4 Tensor Core MMA
CUTLASS production parity
Marlin/AWQ/GPTQ kernel parity
optimal register tiling
optimal cp.async/TMA pipeline
```

The purpose is to make low-bit storage consumption, qparam indexing and small-M versus large-M economics inspectable before adding more opaque high-performance machinery.

## 19. Relationship to the next ROADMAP item

The next strict Stage 5 item remains:

```text
fused dequantize + GEMM
```

This experiment establishes the reusable weight-only numerical/storage contract first.

The next item can therefore focus on making the dequantization/GEMM pipeline more explicitly fused and production-oriented—for example, eliminating avoidable intermediate fragment conversions, improving tile scheduling and mapping low-bit dequantization into Tensor-Core-friendly fragments—without redefining INT8/INT4 storage.
