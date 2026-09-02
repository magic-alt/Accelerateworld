from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import triton

from lab_config import (
    AUTOTUNE_CONFIG_SPECS,
    SHAPE_FAMILIES,
    GemmShape,
    parse_direct_cublas_output,
    pick_winner,
    shapes_for_family,
)
from triton_gemm import best_config_dict, matmul_into


def _bench_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def _throughput_gflops(shape: GemmShape, latency_ms: float) -> float:
    return shape.flops / (latency_ms / 1000.0) / 1.0e9


def _normalized_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denominator = 1.0 + reference.abs()
    return ((actual - reference).abs() / denominator).max().item()


def _discover_cublas_executable(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None

    repo_root = Path(__file__).resolve().parents[2]
    names = ["aw_triton_cublas_fp16_gemm", "aw_triton_cublas_fp16_gemm.exe"]
    build_dirs = ["rtx20", "rtx30", "rtx40", "rtx50", "rtx5060", "release", "ci"]
    for build_dir in build_dirs:
        for name in names:
            candidate = repo_root / "build" / build_dir / "bin" / name
            if candidate.is_file():
                return candidate
    return None


def _run_direct_cublas(
    executable: Path,
    shape: GemmShape,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], str]:
    command = [
        str(executable),
        "--m",
        str(shape.m),
        "--n",
        str(shape.n),
        "--k",
        str(shape.k),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    parsed = parse_direct_cublas_output(completed.stdout)
    if "latency_ms" not in parsed or "throughput_gflops" not in parsed:
        raise RuntimeError(f"failed to parse direct cuBLAS output:\n{completed.stdout}")
    return parsed, completed.stdout


def _make_inputs(shape: GemmShape, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    # Small amplitudes keep the FP16 output comparison focused on GEMM reduction order
    # instead of avoidable overflow in the correctness oracle.
    a = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.float16, generator=generator) * 0.125
    b = torch.randn((shape.k, shape.n), device="cuda", dtype=torch.float16, generator=generator) * 0.125
    return a.contiguous(), b.contiguous()


def _benchmark_shape(
    shape: GemmShape,
    warmup: int,
    iterations: int,
    seed: int,
    cublas_executable: Path | None,
) -> dict[str, object]:
    a, b = _make_inputs(shape, seed)
    triton_out = torch.empty((shape.m, shape.n), device="cuda", dtype=torch.float16)
    torch_out = torch.empty_like(triton_out)

    # First invocation performs compilation/autotuning or restores a disk-cached autotune result.
    torch.cuda.synchronize()
    first_start = time.perf_counter()
    matmul_into(a, b, triton_out)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - first_start) * 1000.0
    selected_config = best_config_dict()
    if selected_config is None:
        raise RuntimeError("Triton autotuner did not expose a selected config")

    reference = torch.mm(a, b)
    torch.testing.assert_close(triton_out, reference, rtol=2e-2, atol=2e-2)
    max_normalized_error = _normalized_error(reference, triton_out)

    triton_ms = _bench_cuda(
        lambda: matmul_into(a, b, triton_out),
        warmup,
        iterations,
    )
    torch_ms = _bench_cuda(
        lambda: torch.mm(a, b, out=torch_out),
        warmup,
        iterations,
    )

    cublas_result: dict[str, float] | None = None
    cublas_raw: str | None = None
    if cublas_executable is not None:
        cublas_result, cublas_raw = _run_direct_cublas(
            cublas_executable,
            shape,
            warmup,
            iterations,
        )

    latencies = {
        "triton": triton_ms,
        "pytorch": torch_ms,
        "direct_cublas": None if cublas_result is None else cublas_result["latency_ms"],
    }
    winner = pick_winner(latencies)

    result: dict[str, object] = {
        "name": shape.name,
        "family": shape.family,
        "role": shape.role,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "first_call_autotune_or_cache_ms": first_call_ms,
        "selected_config": selected_config,
        "max_normalized_error": max_normalized_error,
        "triton": {
            "latency_ms": triton_ms,
            "throughput_gflops": _throughput_gflops(shape, triton_ms),
        },
        "pytorch": {
            "latency_ms": torch_ms,
            "throughput_gflops": _throughput_gflops(shape, torch_ms),
        },
        "direct_cublas": cublas_result,
        "winner": winner,
    }
    if cublas_raw is not None:
        result["direct_cublas_raw"] = cublas_raw
    return result


def _print_result(result: dict[str, object]) -> None:
    print()
    print(f"Shape: {result['name']} ({result['role']})")
    print(f"  M/N/K: {result['m']} / {result['n']} / {result['k']}")
    print(
        "  First call (compile/autotune or disk-cache restore): "
        f"{result['first_call_autotune_or_cache_ms']:.3f} ms"
    )
    print(f"  Best Triton config: {json.dumps(result['selected_config'], sort_keys=True)}")
    print(f"  Max normalized error: {result['max_normalized_error']:.6g}")

    triton_result = result["triton"]
    pytorch_result = result["pytorch"]
    assert isinstance(triton_result, dict) and isinstance(pytorch_result, dict)
    print(
        f"  Triton autotuned: {triton_result['latency_ms']:.6f} ms, "
        f"{triton_result['throughput_gflops']:.3f} GFLOP/s"
    )
    print(
        f"  PyTorch torch.mm: {pytorch_result['latency_ms']:.6f} ms, "
        f"{pytorch_result['throughput_gflops']:.3f} GFLOP/s"
    )

    cublas_result = result["direct_cublas"]
    if isinstance(cublas_result, dict):
        print(
            f"  Direct cuBLAS: {cublas_result['latency_ms']:.6f} ms, "
            f"{cublas_result['throughput_gflops']:.3f} GFLOP/s"
        )
    else:
        print("  Direct cuBLAS: unavailable (build aw_triton_cublas_fp16_gemm or pass --cublas-exe)")
    print(f"  Shape winner: {result['winner']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=[*SHAPE_FAMILIES.keys(), "all"],
        default="validation",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--cublas-exe")
    parser.add_argument("--require-cublas", action="store_true")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0 or args.max_shapes < 0:
        raise ValueError("warmup/max-shapes must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    shapes = shapes_for_family(args.family)
    if args.max_shapes:
        shapes = shapes[: args.max_shapes]
    cublas_executable = _discover_cublas_executable(args.cublas_exe)
    if args.require_cublas and cublas_executable is None:
        raise RuntimeError("direct cuBLAS baseline required but aw_triton_cublas_fp16_gemm was not found")

    print("Triton Auto-tuned GEMM Lab")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Compute capability: {torch.cuda.get_device_capability()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Triton: {triton.__version__}")
    print("  Contract: FP16 input / FP32 accumulate / FP16 output")
    print(f"  Shape family: {args.family}")
    print(f"  Autotune configs: {len(AUTOTUNE_CONFIG_SPECS)}")
    print("  Autotune key: M,N,K")
    print("  Disk autotune cache: enabled (cache_results=True)")
    print(f"  Direct cuBLAS executable: {cublas_executable or 'NOT FOUND'}")

    results = [
        _benchmark_shape(
            shape,
            args.warmup,
            args.iterations,
            args.seed + index,
            cublas_executable,
        )
        for index, shape in enumerate(shapes)
    ]
    for result in results:
        _print_result(result)

    winner_counts: dict[str, int] = {}
    for result in results:
        winner = str(result["winner"])
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    print()
    print("Shape-dependent winner summary")
    for provider, count in sorted(winner_counts.items()):
        print(f"  {provider}: {count}")
    print("Validation: PASS")

    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "family": args.family,
            "autotune_config_count": len(AUTOTUNE_CONFIG_SPECS),
            "direct_cublas_executable": None if cublas_executable is None else str(cublas_executable),
            "winner_counts": winner_counts,
            "results": results,
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON evidence: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise
