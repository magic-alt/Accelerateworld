from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PYTORCH_EXTENSION = ROOT / "experiments" / "10_pytorch_extension"
TRITON_GEMM = ROOT / "experiments" / "23_triton_gemm"
sys.path.insert(0, str(PYTORCH_EXTENSION))
sys.path.insert(0, str(TRITON_GEMM))

from accelerateworld_ops import silu_mul  # noqa: E402
from triton_gemm import best_config_dict, matmul_into  # noqa: E402


def _compiled_body(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.tanh(silu_mul(gate, up)) + gate * 0.125


def _native_compiled_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.tanh(F.silu(gate) * up) + gate * 0.125


def _run_iterations(name: str, iterations: int, fn) -> torch.Tensor:
    output = None
    with torch.cuda.nvtx.range(name):
        for _ in range(iterations):
            output = fn()
        torch.cuda.synchronize()
    if output is None:
        raise RuntimeError("trace workload executed zero iterations")
    return output


def _start_capture() -> None:
    torch.cuda.profiler.start()


def _stop_capture() -> None:
    torch.cuda.profiler.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("steady", "full"), default="steady")
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    if min(args.elements, args.m, args.n, args.k, args.iterations) <= 0 or args.warmup < 0:
        raise ValueError("sizes/iterations must be positive and warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.manual_seed(2026)
    device = torch.device("cuda")

    gate = (torch.randn(args.elements, device=device, dtype=torch.float32) * 0.75).contiguous()
    up = torch.randn_like(gate)
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float16)
    mm_out = torch.empty((args.m, args.n), device=device, dtype=torch.float16)
    triton_out = torch.empty_like(mm_out)

    native_reference = F.silu(gate) * up
    compiled_reference = _native_compiled_reference(gate, up)
    mm_reference = torch.mm(a, b)
    torch.cuda.synchronize()

    compiled = torch.compile(_compiled_body, fullgraph=True)
    compiled_first = None
    triton_first = None

    if args.mode == "full":
        _start_capture()
        torch.cuda.nvtx.mark("aw/capture/full_begin")

        with torch.cuda.nvtx.range("aw/setup/torch_compile_first_call"):
            compiled_first = compiled(gate, up)
            torch.cuda.synchronize()

        with torch.cuda.nvtx.range("aw/setup/triton_autotune_first_call"):
            triton_first = matmul_into(a, b, triton_out)
            torch.cuda.synchronize()
    else:
        for _ in range(max(1, args.warmup)):
            compiled_first = compiled(gate, up)
            triton_first = matmul_into(a, b, triton_out)
            torch.mm(a, b, out=mm_out)
            silu_mul(gate, up)
            F.silu(gate) * up
        torch.cuda.synchronize()
        _start_capture()
        torch.cuda.nvtx.mark("aw/capture/steady_begin")

    with torch.cuda.nvtx.range("aw/trace/framework_runtime"):
        eager_output = _run_iterations(
            "aw/trace/pytorch_eager_silu_mul",
            args.iterations,
            lambda: F.silu(gate) * up,
        )
        custom_output = _run_iterations(
            "aw/trace/custom_cuda_silu_mul",
            args.iterations,
            lambda: silu_mul(gate, up),
        )
        compiled_output = _run_iterations(
            "aw/trace/torch_compile_custom_op",
            args.iterations,
            lambda: compiled(gate, up),
        )
        _run_iterations(
            "aw/trace/pytorch_mm_cublas",
            args.iterations,
            lambda: torch.mm(a, b, out=mm_out),
        )
        _run_iterations(
            "aw/trace/triton_autotuned_gemm",
            args.iterations,
            lambda: matmul_into(a, b, triton_out),
        )

    torch.cuda.nvtx.mark("aw/capture/end")
    _stop_capture()

    if compiled_first is None or triton_first is None:
        raise RuntimeError("compile/autotune warmup did not execute")

    torch.testing.assert_close(eager_output, native_reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(custom_output, native_reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(compiled_first, compiled_reference, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(compiled_output, compiled_reference, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(mm_out, mm_reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(triton_out, mm_reference, rtol=1e-2, atol=1e-2)

    print("Nsight Systems framework trace workload")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Capture mode: {args.mode}")
    print(f"  Elementwise elements: {args.elements}")
    print(f"  GEMM shape: {args.m} x {args.n} x {args.k}")
    print(f"  Iterations per runtime range: {args.iterations}")
    print(f"  Triton best config: {best_config_dict()}")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()
