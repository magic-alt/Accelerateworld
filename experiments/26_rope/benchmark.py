from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

import accelerateworld_rope_cuda
from lab_config import DTYPE_REQUIREMENTS, LAYOUTS, SHAPE_FAMILIES, RoPEShape, shapes_for_family
from reference import build_position_ids, build_rope_cache, rope_qk_fp32, rope_qk_mixed
from rope_triton import rope_qk as triton_rope_qk


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
ERROR_LIMITS = {"fp16": 3.0e-3, "bf16": 2.5e-2}
LAYOUT_IDS = {"interleaved": 0, "half_split": 1}


def _eager_interleaved(q, k, cos, sin, positions, rotary_dim):
    return rope_qk_mixed(q, k, cos, sin, positions, rotary_dim, "interleaved")


def _eager_half_split(q, k, cos, sin, positions, rotary_dim):
    return rope_qk_mixed(q, k, cos, sin, positions, rotary_dim, "half_split")


COMPILED = {
    "interleaved": torch.compile(_eager_interleaved, fullgraph=True, dynamic=True),
    "half_split": torch.compile(_eager_half_split, fullgraph=True, dynamic=True),
}


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and bool(torch.cuda.is_bf16_supported())


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
    ref = reference.float()
    got = actual.float()
    return float(((got - ref).abs() / (1.0 + ref.abs())).max().item())


def _pair_norm_error(before: torch.Tensor, after: torch.Tensor, rotary_dim: int, layout: str) -> float:
    pair_count = rotary_dim // 2
    x = before[..., :rotary_dim].float()
    y = after[..., :rotary_dim].float()
    if layout == "interleaved":
        xp = x.reshape(*x.shape[:-1], pair_count, 2)
        yp = y.reshape(*y.shape[:-1], pair_count, 2)
        xnorm = xp.square().sum(-1)
        ynorm = yp.square().sum(-1)
    else:
        xnorm = x[..., :pair_count].square() + x[..., pair_count:].square()
        ynorm = y[..., :pair_count].square() + y[..., pair_count:].square()
    return float(((ynorm - xnorm).abs() / (1.0 + xnorm)).max().item())


def _provider_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    layout: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    eager = rope_qk_mixed(q, k, cos, sin, positions, rotary_dim, layout)
    compiled = COMPILED[layout](q, k, cos, sin, positions, rotary_dim)
    triton = triton_rope_qk(q, k, cos, sin, positions, rotary_dim, layout)
    cuda = accelerateworld_rope_cuda.rope(
        q, k, cos, sin, positions, rotary_dim, LAYOUT_IDS[layout]
    )
    return {"eager": eager, "compile": compiled, "triton": triton, "cuda": cuda}


def _run_case(
    shape: RoPEShape,
    dtype_name: str,
    layout: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    device = torch.device("cuda")
    positions = build_position_ids(
        shape.batch, shape.sequence, shape.position_start, device=device
    )
    cos, sin = build_rope_cache(shape.max_position, shape.rotary_dim, device=device)

    q = (
        torch.randn(
            (shape.batch, shape.sequence, shape.q_heads, shape.head_dim),
            device=device,
            dtype=dtype,
        )
        * 0.5
    ).contiguous()
    k = (
        torch.randn(
            (shape.batch, shape.sequence, shape.kv_heads, shape.head_dim),
            device=device,
            dtype=dtype,
        )
        * 0.5
    ).contiguous()

    q_ref, k_ref = rope_qk_fp32(q, k, cos, sin, positions, shape.rotary_dim, layout)
    outputs = _provider_outputs(q, k, cos, sin, positions, shape.rotary_dim, layout)
    torch.cuda.synchronize()

    errors: dict[str, dict[str, float]] = {}
    limit = ERROR_LIMITS[dtype_name]
    for name, (q_out, k_out) in outputs.items():
        q_error = _normalized_error(q_ref, q_out)
        k_error = _normalized_error(k_ref, k_out)
        if max(q_error, k_error) > limit:
            raise AssertionError(
                f"{shape.name}/{dtype_name}/{layout}/{name}: error exceeds {limit}: "
                f"q={q_error}, k={k_error}"
            )
        if shape.rotary_dim < shape.head_dim:
            if not torch.equal(q_out[..., shape.rotary_dim :], q[..., shape.rotary_dim :]):
                raise AssertionError(f"{name}: Q tail changed outside rotary_dim")
            if not torch.equal(k_out[..., shape.rotary_dim :], k[..., shape.rotary_dim :]):
                raise AssertionError(f"{name}: K tail changed outside rotary_dim")
        errors[name] = {"q": q_error, "k": k_error}

    fp32_norm_error = max(
        _pair_norm_error(q, q_ref, shape.rotary_dim, layout),
        _pair_norm_error(k, k_ref, shape.rotary_dim, layout),
    )

    fns: dict[str, Callable[[], object]] = {
        "eager": lambda: rope_qk_mixed(q, k, cos, sin, positions, shape.rotary_dim, layout),
        "compile": lambda: COMPILED[layout](q, k, cos, sin, positions, shape.rotary_dim),
        "triton": lambda: triton_rope_qk(q, k, cos, sin, positions, shape.rotary_dim, layout),
        "cuda": lambda: accelerateworld_rope_cuda.rope(
            q, k, cos, sin, positions, shape.rotary_dim, LAYOUT_IDS[layout]
        ),
    }
    latencies = {name: _bench_cuda(fn, warmup, iterations) for name, fn in fns.items()}
    element_size = q.element_size()
    logical_bytes = 2 * shape.total_elements * element_size
    providers = {
        name: {
            "latency_ms": latency,
            "logical_gbps": logical_bytes / (latency / 1000.0) / 1.0e9,
            "million_elements_per_s": shape.total_elements / (latency / 1000.0) / 1.0e6,
        }
        for name, latency in latencies.items()
    }

    return {
        "shape": shape.to_dict(),
        "dtype": dtype_name,
        "layout": layout,
        "requirements": list(DTYPE_REQUIREMENTS[dtype_name]),
        "cache": {
            "dtype": "float32",
            "rows": cos.size(0),
            "pair_columns": cos.size(1),
            "bytes": (cos.numel() + sin.numel()) * cos.element_size(),
            "construction_in_timing": False,
        },
        "numerics": {
            "provider_normalized_error_vs_fp32": errors,
            "fp32_pair_norm_relative_error": fp32_norm_error,
            "tail_exact_copy_required": shape.rotary_dim < shape.head_dim,
        },
        "providers": providers,
        "winner": min(latencies, key=latencies.get),
    }


def _print_case(result: dict[str, object]) -> None:
    shape = result["shape"]
    print(
        f"\n[{result['dtype']}/{result['layout']}] {shape['name']}: "
        f"B={shape['batch']} S={shape['sequence']} QH={shape['q_heads']} "
        f"KVH={shape['kv_heads']} D={shape['head_dim']} R={shape['rotary_dim']} "
        f"pos={shape['position_start']}..{shape['position_start'] + shape['sequence'] - 1}"
    )
    for name, record in result["providers"].items():
        print(
            f"  {name:8s} {record['latency_ms']:.6f} ms  "
            f"{record['logical_gbps']:.3f} logical GB/s"
        )
    print(f"  winner: {result['winner']}")
    print(f"  numerics: {result['numerics']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=SHAPE_FAMILIES, default="validation")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    requested_dtypes = tuple(x.strip() for x in args.dtypes.split(",") if x.strip())
    requested_layouts = tuple(x.strip() for x in args.layouts.split(",") if x.strip())
    if any(x not in DTYPE_MAP for x in requested_dtypes):
        raise ValueError(f"unknown dtypes: {requested_dtypes}")
    if any(x not in LAYOUTS for x in requested_layouts):
        raise ValueError(f"unknown layouts: {requested_layouts}")

    torch.manual_seed(2026)
    payload: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "family": args.family,
        "dtypes": requested_dtypes,
        "layouts": requested_layouts,
        "cache_contract": "precomputed FP32 cos/sin cache; cache construction excluded from timing",
        "results": [],
        "skipped": [],
    }

    for dtype_name in requested_dtypes:
        if dtype_name == "bf16" and not _bf16_supported():
            print("[bf16] SKIP: requires compute capability >= 8.0 and PyTorch BF16 CUDA support")
            payload["skipped"].append(
                {"dtype": "bf16", "requires": list(DTYPE_REQUIREMENTS["bf16"])}
            )
            continue
        for layout in requested_layouts:
            for shape in shapes_for_family(args.family):
                result = _run_case(shape, dtype_name, layout, args.warmup, args.iterations)
                payload["results"].append(result)
                _print_case(result)
                torch.cuda.empty_cache()

    if not payload["results"]:
        raise RuntimeError("no supported RoPE workloads executed")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON evidence: {args.json}")
    print("\nRoPE validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
