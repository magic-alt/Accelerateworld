# 24 — Nsight Systems Framework Trace

This experiment closes Stage 4 by tracing the framework/compiler/runtime path that earlier experiments built independently:

```text
Python / PyTorch eager
        ↓
custom accelerateworld::silu_mul CUDA op
        ↓
torch.compile / Dynamo / AOTAutograd / Inductor
        ↓
PyTorch GEMM vendor path (cuBLAS/cuBLASLt)
        ↓
Triton auto-tuned GEMM
        ↓
CUDA Runtime / kernel launches / synchronization
```

The goal is not another latency benchmark. The goal is to retain a **timeline artifact** that lets you answer where host time, launch time, synchronization, vendor-library calls and GPU kernels appear relative to one another.

## Trace workload

`trace_workload.py` labels the important boundaries with NVTX:

```text
aw/setup/torch_compile_first_call
aw/setup/triton_autotune_first_call

aw/trace/framework_runtime
    ├── aw/trace/pytorch_eager_silu_mul
    ├── aw/trace/custom_cuda_silu_mul
    ├── aw/trace/torch_compile_custom_op
    ├── aw/trace/pytorch_mm_cublas
    └── aw/trace/triton_autotuned_gemm
```

The provider ranges deliberately execute repeated calls and end with `torch.cuda.synchronize()` so each host NVTX range has a clear runtime boundary. The synchronization API is part of the trace rather than hidden from it; this makes launch-vs-wait behavior visible.

### Providers

1. **PyTorch eager SiLU*mul** — ordinary `F.silu(gate) * up`.
2. **Custom CUDA op** — dispatcher-backed `accelerateworld::silu_mul`.
3. **Compiled custom op** — `torch.compile(fullgraph=True)` around the same opaque custom CUDA operator plus surrounding PyTorch operations.
4. **PyTorch GEMM** — `torch.mm(..., out=...)`; Nsight's cuBLAS trace shows the framework/vendor-library path when applicable.
5. **Triton GEMM** — the Stage 4 auto-tuned FP16 GEMM from experiment 23.

Correctness is checked after capture against native PyTorch references.

## Two capture modes

### `steady`

The default mode compiles Inductor and autotunes Triton **before** calling `cudaProfilerStart()`.

```text
compile / autotune / warmup
        ↓
cudaProfilerStart
        ↓
NVTX runtime ranges
        ↓
CUDA APIs / cuBLAS / kernels
        ↓
cudaProfilerStop
```

Use this mode to study steady-state framework dispatch, GPU launch cadence and synchronization without first-use compilation dominating the report.

### `full`

Full mode calls `cudaProfilerStart()` before the first compiled/Triton calls and wraps those calls with setup NVTX ranges.

```text
cudaProfilerStart
        ↓
aw/setup/torch_compile_first_call
        ↓
Dynamo / AOTAutograd / Inductor first use
        ↓
aw/setup/triton_autotune_first_call
        ↓
Triton JIT + config search/cache restore
        ↓
steady runtime provider ranges
```

It is intentionally a first-use trace and can be much larger than `steady`.

## Collection command

`run_trace.py` constructs the low-overhead Nsight Systems command:

```bash
nsys profile \
  --trace=cuda,nvtx,cublas,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output=results/nsys-framework-trace/steady/framework-trace-steady \
  python experiments/24_nsys_trace/trace_workload.py --mode steady ...
```

When the installed Nsight Systems CLI advertises the current PyTorch integration switch, the runner automatically adds:

```text
--pytorch=functions-trace-shapes,autograd-nvtx
```

Older Nsight Systems versions remain usable because the flag is feature-detected rather than assumed.

The CUDA profiler capture range is important: Nsight Systems launches the Python process immediately but does not start collection until `torch.cuda.profiler.start()` reaches `cudaProfilerStart`. This gives the workload explicit control over whether compile/autotune is inside or outside the report.

## Post-collection reports

The raw `.nsys-rep` file is the authoritative timeline artifact. `run_trace.py` also executes `nsys stats` for:

- `cuda_api_sum` — host CUDA Runtime/Driver API time and call counts;
- `cuda_gpu_kern_sum` — GPU kernel time and instances;
- `cuda_kern_exec_sum` — launch-to-execution relationship;
- `nvtx_sum` — NVTX range duration/count summary;
- `nvtx_gpu_proj_sum` — NVTX host ranges projected onto their enclosed GPU operations;
- `nvtx_kern_sum` — kernel instances associated with NVTX ranges.

Each report is retained as text next to `trace-metadata.json`, plus profiler stdout/stderr. Nsight Systems may also create an exported SQLite database during `nsys stats`; the `.nsys-rep` remains the source report to open in the GUI.

## What to inspect in the GUI

Start with these questions:

1. **Python/framework launch gaps** — is there visible CPU time between consecutive GPU launches?
2. **Custom op boundary** — does `aw/trace/custom_cuda_silu_mul` map cleanly to the handwritten CUDA kernel?
3. **Inductor boundary** — inside `aw/trace/torch_compile_custom_op`, which surrounding operations become generated kernels and where does the opaque custom CUDA op remain?
4. **Vendor GEMM** — under `aw/trace/pytorch_mm_cublas`, which cuBLAS/cuBLASLt calls and Tensor Core kernels appear?
5. **Triton GEMM** — how many generated kernels appear and what is the launch cadence relative to the PyTorch vendor path?
6. **Synchronization** — how much host time is spent in synchronization APIs compared with asynchronous launch APIs?
7. **First-use cost** — in a `full` capture, how large are the compile/autotune ranges relative to steady runtime ranges?

Do not infer Tensor Core utilization from kernel names alone. Stage 7 Nsight Compute/roofline work is the appropriate place for hardware-counter utilization evidence.

## Run locally or on a self-hosted GPU runner

Prerequisites:

- CUDA-capable Linux environment;
- PyTorch/Triton dependencies from `python/requirements-gpu.txt`;
- Nsight Systems CLI (`nsys`) on `PATH`;
- the custom PyTorch CUDA extension built in place.

Run both canonical captures:

```bash
bash scripts/run_nsys_framework_trace.sh
```

Or run one capture directly:

```bash
python experiments/24_nsys_trace/run_trace.py \
  --mode steady \
  --m 16 --n 4096 --k 4096 \
  --iterations 8
```

Inspect command construction without requiring CUDA or Nsight Systems:

```bash
python experiments/24_nsys_trace/run_trace.py --dry-run
```

## Evidence boundary

Hosted CI can prove:

- Python syntax;
- the expected NVTX range model;
- stable `nsys profile` command construction;
- stable `nsys stats` report selection;
- the dry-run path does not require a GPU or Nsight Systems installation.

Only a physical GPU runner with Nsight Systems can prove:

- a real `.nsys-rep` timeline;
- PyTorch/Inductor/custom-op/Triton/cuBLAS CUDA activity;
- launch/synchronization timing;
- first-use compiler/autotuner behavior;
- actual kernel identities and durations.

No runtime trace numbers are fabricated from hosted CI.

## References

- NVIDIA Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA Nsight Systems Post-Collection Analysis Guide: https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
- PyTorch CUDA/NVTX APIs: https://docs.pytorch.org/docs/stable/cuda
- PyTorch CUDA profiler start/stop: https://docs.pytorch.org/docs/stable/generated/torch.cuda.profiler.start.html
