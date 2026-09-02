from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import triton

import accelerateworld_online_softmax_cuda as cuda_softmax
from lab_config import DTYPE_REQUIREMENTS, SHAPE_FAMILIES, SoftmaxShape, shapes_for_family
from reference import causal_mask, masked_max_abs, row_sum_error, stable_softmax_fp32
from softmax_triton import online_softmax as triton_online_softmax


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
ERROR_LIMITS = {"fp16": 5.0e-3, "bf16": 3.5e-2}
ROW_SUM_LIMITS = {"fp16": 3.0e-3, "bf16": 1.5e-2}


def eager_noncausal(scores: torch.Tensor) -> torch.Tensor:
    x = scores.float()
    row_max = x.amax(dim=-1, keepdim=True)
    numerator = torch.exp(x - row_max)
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(scores.dtype)


def eager_causal(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = scores.float().masked_fill(~mask, float("-inf"))
    row_max = x.amax(dim=-1, keepdim=True)
    numerator = torch.exp(x - row_max)
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(scores.dtype)


COMPILED_NONCAUSAL = torch.compile(eager_noncausal, fullgraph=True, dynamic=True)
COMPILED_CAUSAL = torch.compile(eager_causal, fullgraph=True, dynamic=True)


def bench_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> float:
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


def normalized_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.float()
    got = actual.float()
    return float(((got - ref).abs() / (1.0 + ref.abs())).max().item())


def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and bool(torch.cuda.is_bf16_supported())


def make_scores(shape: SoftmaxShape, dtype: torch.dtype) -> torch.Tensor:
    base = torch.randn(
        (shape.batch, shape.heads, shape.query_length, shape.key_length),
        device="cuda",
        dtype=torch.float32,
    ) * 4.0
    # Large row shifts exercise overflow/underflow stability while preserving
    # the mathematical softmax distribution.
    row_shift = torch.linspace(-80.0, 80.0, shape.query_length, device="cuda").view(
        1, 1, shape.query_length, 1
    )
    return (base + row_shift).to(dtype).contiguous()


def provider_record(shape: SoftmaxShape, latency_ms: float) -> dict[str, float]:
    logical_bytes = shape.elements * 2 * 2
    gbps = logical_bytes / (latency_ms / 1000.0) / 1.0e9
    return {
        "latency_ms": latency_ms,
        "logical_gbps": gbps,
        "elements_per_second": shape.elements / (latency_ms / 1000.0),
    }


def run_shape(shape: SoftmaxShape, dtype_name: str, warmup: int, iterations: int) -> dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    scores = make_scores(shape, dtype)
    fp32_reference = stable_softmax_fp32(
        scores, causal=shape.causal, query_start=shape.query_start
    )

    mask = None
    if shape.causal:
        mask = causal_mask(
            shape.query_length, shape.key_length, shape.query_start, scores.device
        ).view(1, 1, shape.query_length, shape.key_length)

    def eager() -> torch.Tensor:
        return eager_causal(scores, mask) if shape.causal else eager_noncausal(scores)

    def compiled() -> torch.Tensor:
        return COMPILED_CAUSAL(scores, mask) if shape.causal else COMPILED_NONCAUSAL(scores)

    providers: dict[str, Callable[[], torch.Tensor]] = {
        "pytorch_eager": eager,
        "torch_compile": compiled,
        "triton_online": lambda: triton_online_softmax(
            scores, causal=shape.causal, query_start=shape.query_start
        ),
        "cuda_two_pass": lambda: cuda_softmax.softmax_two_pass(
            scores, shape.causal, shape.query_start
        ),
        "cuda_online": lambda: cuda_softmax.softmax_online(
            scores, shape.causal, shape.query_start
        ),
    }

    outputs = {name: fn() for name, fn in providers.items()}
    torch.cuda.synchronize()

    errors: dict[str, dict[str, float]] = {}
    for name, output in outputs.items():
        error = normalized_error(fp32_reference, output)
        sum_error = row_sum_error(output)
        masked_error = masked_max_abs(
            output, causal=shape.causal, query_start=shape.query_start
        )
        if not bool(torch.isfinite(output).all()):
            raise AssertionError(f"{name} produced non-finite output")
        if error > ERROR_LIMITS[dtype_name]:
            raise AssertionError(
                f"{dtype_name} {name} normalized error {error:.6g} exceeds {ERROR_LIMITS[dtype_name]:.6g}"
            )
        if sum_error > ROW_SUM_LIMITS[dtype_name]:
            raise AssertionError(
                f"{dtype_name} {name} row-sum error {sum_error:.6g} exceeds {ROW_SUM_LIMITS[dtype_name]:.6g}"
            )
        if masked_error != 0.0:
            raise AssertionError(f"{name} causal masked outputs are not exactly zero")
        errors[name] = {
            "normalized_error_vs_fp32": error,
            "row_sum_error": sum_error,
            "masked_max_abs": masked_error,
        }

    latencies = {
        name: bench_cuda(fn, warmup, iterations) for name, fn in providers.items()
    }
    winner = min(latencies, key=latencies.get)

    return {
        "shape": {
            "name": shape.name,
            "role": shape.role,
            "batch": shape.batch,
            "heads": shape.heads,
            "query_length": shape.query_length,
            "key_length": shape.key_length,
            "causal": shape.causal,
            "query_start": shape.query_start,
            "rows": shape.rows,
            "elements": shape.elements,
        },
        "dtype": dtype_name,
        "requirements": list(DTYPE_REQUIREMENTS[dtype_name]),
        "numerics": errors,
        "providers": {
            name: provider_record(shape, latency) for name, latency in latencies.items()
        },
        "winner": winner,
    }


def print_result(result: dict[str, object]) -> None:
    shape = result["shape"]
    print(
        f"\n[{result['dtype']}] {shape['name']}: B={shape['batch']} H={shape['heads']} "
        f"Q={shape['query_length']} K={shape['key_length']} causal={shape['causal']} "
        f"query_start={shape['query_start']}"
    )
    for name, record in result["providers"].items():
        print(
            f"  {name:16s} {record['latency_ms']:.6f} ms  "
            f"{record['logical_gbps']:.3f} logical GB/s"
        )
    print(f"  winner: {result['winner']}")
    print(f"  numerics: {result['numerics']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=SHAPE_FAMILIES, default="validation")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    requested = tuple(part.strip() for part in args.dtypes.split(",") if part.strip())
    unknown = [name for name in requested if name not in DTYPE_MAP]
    if unknown:
        raise ValueError(f"unsupported dtypes: {unknown}")

    torch.manual_seed(2026)
    payload: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "family": args.family,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "contract": "FP16/BF16 scores; FP32 max/exp/sum/online state; low-precision output; causal mask fused into Triton/CUDA",
        "results": [],
        "skipped": [],
    }

    for dtype_name in requested:
        if dtype_name == "bf16" and not bf16_supported():
            print("[bf16] SKIP: requires compute capability >= 8.0 and PyTorch BF16 CUDA support")
            payload["skipped"].append(
                {"dtype": "bf16", "requires": list(DTYPE_REQUIREMENTS["bf16"])}
            )
            continue
        for shape in shapes_for_family(args.family):
            result = run_shape(shape, dtype_name, args.warmup, args.iterations)
            payload["results"].append(result)
            print_result(result)

    if not payload["results"]:
        raise RuntimeError("no supported online-softmax workloads executed")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON evidence: {args.json}")

    print("\nOnline softmax validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
