from __future__ import annotations

from pathlib import Path

TRACE_APIS = ("cuda", "nvtx", "cublas", "osrt")
STATS_REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_kern_exec_sum",
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
    "nvtx_kern_sum",
)
EXPECTED_NVTX_RANGES = (
    "aw/setup/torch_compile_first_call",
    "aw/setup/triton_autotune_first_call",
    "aw/trace/pytorch_eager_silu_mul",
    "aw/trace/custom_cuda_silu_mul",
    "aw/trace/torch_compile_custom_op",
    "aw/trace/pytorch_mm_cublas",
    "aw/trace/triton_autotuned_gemm",
)


def build_profile_command(
    *,
    nsys: str,
    python_exe: str,
    workload: Path,
    output_base: Path,
    mode: str,
    elements: int,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iterations: int,
    enable_pytorch_annotations: bool,
) -> list[str]:
    if mode not in {"steady", "full"}:
        raise ValueError("mode must be steady or full")
    if min(elements, m, n, k, iterations) <= 0 or warmup < 0:
        raise ValueError("sizes/iterations must be positive and warmup must be non-negative")

    command = [
        nsys,
        "profile",
        f"--trace={','.join(TRACE_APIS)}",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        f"--output={output_base}",
    ]
    if enable_pytorch_annotations:
        command.append("--pytorch=functions-trace-shapes,autograd-nvtx")

    command.extend(
        [
            python_exe,
            str(workload),
            "--mode",
            mode,
            "--elements",
            str(elements),
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
        ]
    )
    return command


def build_stats_command(*, nsys: str, report: str, nsys_report: Path) -> list[str]:
    if report not in STATS_REPORTS:
        raise ValueError(f"unsupported stats report: {report}")
    return [nsys, "stats", "--report", report, str(nsys_report)]
