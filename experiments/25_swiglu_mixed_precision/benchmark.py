from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import accelerateworld_swiglu_cuda
from lab_config import DTYPE_REQUIREMENTS, SHAPE_FAMILIES, SwiGLUShape, shapes_for_family
from swiglu_triton import swiglu as triton_swiglu
from swiglu_triton import swiglu_into as triton_swiglu_into


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

ERROR_LIMITS = {
    "fp16": 4.0e-3,
    "bf16": 3.0e-2,
}


def _configure_matmul_contract() -> None:
    # Keep the explicit FP32 full-block oracle on IEEE FP32 rather than TF32.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    # Keep the low-precision gate/up projection GEMM on an FP32 reduction
    # contract instead of backend-specific reduced-precision split-K paths.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def swiglu_fp32_reference(packed: torch.Tensor) -> torch.Tensor:
    gate, up = packed.float().chunk(2, dim=-1)
    return F.silu(gate) * up


def swiglu_eager_mixed(packed: torch.Tensor) -> torch.Tensor:
    return swiglu_fp32_reference(packed).to(packed.dtype)


def swiglu_full_eager(x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    packed = torch.mm(x, packed_weight)
    return swiglu_eager_mixed(packed)


def swiglu_full_triton(x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    packed = torch.mm(x, packed_weight)
    return triton_swiglu(packed)


def swiglu_full_cuda(x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    packed = torch.mm(x, packed_weight)
    return accelerateworld_swiglu_cuda.swiglu(packed)


COMPILED_POSTOP = torch.compile(swiglu_eager_mixed, fullgraph=True, dynamic=True)
COMPILED_FULL = torch.compile(swiglu_full_eager, fullgraph=True, dynamic=True)


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
    return float(start.elapsed_time(stop)) / iterations


def _normalized_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference32 = reference.float()
    actual32 = actual.float()
    return float(((actual32 - reference32).abs() / (1.0 + reference32.abs())).max().item())


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and bool(torch.cuda.is_bf16_supported())


def _provider_record(
    shape: SwiGLUShape, latency_ms: float, kind: str, dtype: torch.dtype
) -> dict[str, float | str]:
    if kind == "postop":
        bytes_moved = (
            3
            * shape.tokens
            * shape.intermediate
            * torch.tensor([], dtype=dtype).element_size()
        )
        throughput = bytes_moved / (latency_ms / 1000.0) / 1.0e9
        return {"latency_ms": latency_ms, "effective_gbps": throughput}
    tflops = shape.projection_flops / (latency_ms / 1000.0) / 1.0e12
    return {"latency_ms": latency_ms, "projection_equivalent_tflops": tflops}


def _validate_outputs(
    dtype_name: str,
    fp32_reference: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    limit = ERROR_LIMITS[dtype_name]
    errors: dict[str, float] = {}
    for name, output in outputs.items():
        error = _normalized_error(fp32_reference, output)
        errors[name] = error
        if error > limit:
            raise AssertionError(
                f"{dtype_name} provider {name} normalized error {error:.6g} exceeds {limit:.6g}"
            )
    return errors


def _run_shape(
    shape: SwiGLUShape,
    dtype_name: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    device = torch.device("cuda")

    # Post-op inputs are deliberately bounded so SiLU accuracy, rather than
    # saturation from an enormous synthetic projection, is what is tested.
    packed = (
        torch.randn(
            (shape.tokens, shape.packed_columns), device=device, dtype=dtype
        )
        * 0.75
    ).contiguous()
    fp32_postop = swiglu_fp32_reference(packed)
    eager_postop = swiglu_eager_mixed(packed)
    compiled_postop = COMPILED_POSTOP(packed)
    triton_postop = torch.empty(
        (shape.tokens, shape.intermediate), device=device, dtype=dtype
    )
    cuda_postop = torch.empty_like(triton_postop)
    triton_swiglu_into(packed, triton_postop)
    accelerateworld_swiglu_cuda.swiglu_out(packed, cuda_postop)
    torch.cuda.synchronize()

    postop_errors = _validate_outputs(
        dtype_name,
        fp32_postop,
        {
            "eager": eager_postop,
            "compile": compiled_postop,
            "triton": triton_postop,
            "cuda": cuda_postop,
        },
    )

    postop_latencies = {
        "eager": _bench_cuda(
            lambda: swiglu_eager_mixed(packed), warmup, iterations
        ),
        "compile": _bench_cuda(
            lambda: COMPILED_POSTOP(packed), warmup, iterations
        ),
        "triton": _bench_cuda(
            lambda: triton_swiglu_into(packed, triton_postop), warmup, iterations
        ),
        "cuda": _bench_cuda(
            lambda: accelerateworld_swiglu_cuda.swiglu_out(packed, cuda_postop),
            warmup,
            iterations,
        ),
    }

    # Full SwiGLU block: one packed gate+up projection [M,K]@[K,2N], followed
    # by the provider-specific activation/multiply implementation.
    x = (
        torch.randn((shape.tokens, shape.hidden), device=device, dtype=dtype) * 0.25
    ).contiguous()
    packed_weight = (
        torch.randn(
            (shape.hidden, shape.packed_columns), device=device, dtype=dtype
        )
        * 0.02
    ).contiguous()
    projection_out = torch.empty(
        (shape.tokens, shape.packed_columns), device=device, dtype=dtype
    )

    full_eager = swiglu_full_eager(x, packed_weight)
    full_compile = COMPILED_FULL(x, packed_weight)
    full_triton = swiglu_full_triton(x, packed_weight)
    full_cuda = swiglu_full_cuda(x, packed_weight)
    torch.cuda.synchronize()

    # Provider parity is checked against the mixed-precision eager contract.
    parity_limit = ERROR_LIMITS[dtype_name]
    full_errors: dict[str, float] = {}
    for name, output in {
        "compile": full_compile,
        "triton": full_triton,
        "cuda": full_cuda,
    }.items():
        error = _normalized_error(full_eager, output)
        full_errors[name] = error
        if error > parity_limit:
            raise AssertionError(
                f"{dtype_name} full provider {name} normalized error {error:.6g} exceeds {parity_limit:.6g}"
            )

    # Reduced validation shapes also retain an end-to-end IEEE-FP32 oracle so
    # the numerical cost of low-precision projection + output rounding is visible.
    full_vs_fp32_error: float | None = None
    if shape.hidden <= 1024:
        full_fp32 = swiglu_fp32_reference(
            torch.mm(x.float(), packed_weight.float())
        )
        full_vs_fp32_error = _normalized_error(full_fp32, full_eager)

    projection_latency = _bench_cuda(
        lambda: torch.mm(x, packed_weight, out=projection_out), warmup, iterations
    )
    full_latencies = {
        "eager": _bench_cuda(
            lambda: swiglu_full_eager(x, packed_weight), warmup, iterations
        ),
        "compile": _bench_cuda(
            lambda: COMPILED_FULL(x, packed_weight), warmup, iterations
        ),
        "triton": _bench_cuda(
            lambda: swiglu_full_triton(x, packed_weight), warmup, iterations
        ),
        "cuda": _bench_cuda(
            lambda: swiglu_full_cuda(x, packed_weight), warmup, iterations
        ),
    }

    postop_winner = min(postop_latencies, key=postop_latencies.get)
    full_winner = min(full_latencies, key=full_latencies.get)

    return {
        "shape": shape.to_dict(),
        "dtype": dtype_name,
        "requirements": list(DTYPE_REQUIREMENTS[dtype_name]),
        "numerics": {
            "postop_normalized_error_vs_fp32": postop_errors,
            "full_normalized_error_vs_mixed_eager": full_errors,
            "full_mixed_eager_vs_fp32": full_vs_fp32_error,
        },
        "projection": {
            "latency_ms": projection_latency,
            "equivalent_tflops": shape.projection_flops
            / (projection_latency / 1000.0)
            / 1.0e12,
        },
        "postop": {
            "providers": {
                name: _provider_record(shape, latency, "postop", dtype)
                for name, latency in postop_latencies.items()
            },
            "winner": postop_winner,
        },
        "full_block": {
            "providers": {
                name: _provider_record(shape, latency, "full", dtype)
                for name, latency in full_latencies.items()
            },
            "winner": full_winner,
        },
    }


def _print_result(result: dict[str, object]) -> None:
    shape = result["shape"]
    print(
        f"\n[{result['dtype']}] {shape['name']}: tokens={shape['tokens']} "
        f"hidden={shape['hidden']} intermediate={shape['intermediate']} ({shape['role']})"
    )
    print(f"  projection: {result['projection']['latency_ms']:.6f} ms")
    print("  post-op:")
    for name, record in result["postop"]["providers"].items():
        print(
            f"    {name:8s} {record['latency_ms']:.6f} ms  "
            f"{record['effective_gbps']:.3f} GB/s"
        )
    print(f"    winner: {result['postop']['winner']}")
    print("  full block:")
    for name, record in result["full_block"]["providers"].items():
        print(
            f"    {name:8s} {record['latency_ms']:.6f} ms  "
            f"{record['projection_equivalent_tflops']:.3f} TFLOP/s"
        )
    print(f"    winner: {result['full_block']['winner']}")
    print(f"  numerics: {result['numerics']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=tuple(SHAPE_FAMILIES), default="validation"
    )
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    requested_dtypes = tuple(
        part.strip() for part in args.dtypes.split(",") if part.strip()
    )
    unknown = [name for name in requested_dtypes if name not in DTYPE_MAP]
    if unknown:
        raise ValueError(f"unsupported dtypes: {unknown}")

    _configure_matmul_contract()
    torch.manual_seed(2026)

    payload: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "family": args.family,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "matmul_contract": (
            "IEEE FP32 oracle; FP16/BF16 projection inputs/outputs with "
            "reduced-precision GEMM reductions disabled"
        ),
        "results": [],
        "skipped": [],
    }

    for dtype_name in requested_dtypes:
        if dtype_name == "bf16" and not _bf16_supported():
            print(
                "[bf16] SKIP: requires compute capability >= 8.0 and PyTorch BF16 CUDA support"
            )
            payload["skipped"].append(
                {"dtype": "bf16", "requires": list(DTYPE_REQUIREMENTS["bf16"])}
            )
            continue
        for shape in shapes_for_family(args.family):
            result = _run_shape(shape, dtype_name, args.warmup, args.iterations)
            payload["results"].append(result)
            _print_result(result)
            torch.cuda.empty_cache()

    if not payload["results"]:
        raise RuntimeError("no supported SwiGLU workloads executed")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON evidence: {args.json}")

    print("\nSwiGLU mixed-precision validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
