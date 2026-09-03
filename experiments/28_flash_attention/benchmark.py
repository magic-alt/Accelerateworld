from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton

import accelerateworld_flash_attention_cuda as cuda_flash
from attention_triton import flash_attention as triton_flash
from lab_config import DTYPE_REQUIREMENTS, SHAPE_FAMILIES, AttentionShape, shapes_for_family
from reference import causal_mask, expand_kv_for_gqa, manual_attention_fp32, normalized_error


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
ERROR_LIMITS = {"fp16": 2.5e-2, "bf16": 6.0e-2}


def sdpa_noncausal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def sdpa_masked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


COMPILED_NONCAUSAL = torch.compile(sdpa_noncausal, fullgraph=True, dynamic=True)
COMPILED_MASKED = torch.compile(sdpa_masked, fullgraph=True, dynamic=True)


def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and bool(torch.cuda.is_bf16_supported())


def bench_cuda(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
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


def make_qkv(shape: AttentionShape, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = (torch.randn((shape.batch, shape.q_heads, shape.query_length, shape.head_dim), device="cuda") * 0.5).to(dtype)
    k = (torch.randn((shape.batch, shape.kv_heads, shape.key_length, shape.head_dim), device="cuda") * 0.5).to(dtype)
    v = (torch.randn((shape.batch, shape.kv_heads, shape.key_length, shape.head_dim), device="cuda") * 0.5).to(dtype)
    return q.contiguous(), k.contiguous(), v.contiguous()


def provider_record(shape: AttentionShape, dtype: torch.dtype, latency_ms: float) -> dict[str, float]:
    element_size = torch.tensor([], dtype=dtype).element_size()
    logical_bytes = (shape.q_elements + 2 * shape.kv_elements + shape.output_elements) * element_size
    seconds = latency_ms / 1000.0
    return {
        "latency_ms": latency_ms,
        "logical_gbps": logical_bytes / seconds / 1.0e9,
        "logical_tflops_qk_pv": shape.logical_flops / seconds / 1.0e12,
    }


def run_shape(shape: AttentionShape, dtype_name: str, warmup: int, iterations: int) -> dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    q, k, v = make_qkv(shape, dtype)
    oracle = manual_attention_fp32(q, k, v, causal=shape.causal, query_start=shape.query_start)

    k_exp, v_exp = expand_kv_for_gqa(k, v, shape.q_heads)
    mask = None
    if shape.causal:
        mask = causal_mask(shape.query_length, shape.key_length, shape.query_start, q.device).view(
            1, 1, shape.query_length, shape.key_length
        )

    def eager_sdpa() -> torch.Tensor:
        return sdpa_masked(q, k_exp, v_exp, mask) if shape.causal else sdpa_noncausal(q, k_exp, v_exp)

    def compiled_sdpa() -> torch.Tensor:
        return COMPILED_MASKED(q, k_exp, v_exp, mask) if shape.causal else COMPILED_NONCAUSAL(q, k_exp, v_exp)

    providers: dict[str, Callable[[], torch.Tensor]] = {
        "pytorch_sdpa": eager_sdpa,
        "torch_compile_sdpa": compiled_sdpa,
        "triton_flash": lambda: triton_flash(q, k, v, causal=shape.causal, query_start=shape.query_start),
        "cuda_flash": lambda: cuda_flash.flash_attention(q, k, v, shape.causal, shape.query_start),
    }

    outputs = {name: fn() for name, fn in providers.items()}
    torch.cuda.synchronize()
    numerics: dict[str, dict[str, float]] = {}
    for name, output in outputs.items():
        if not bool(torch.isfinite(output).all()):
            raise AssertionError(f"{name} produced non-finite output")
        error = normalized_error(oracle, output)
        if error > ERROR_LIMITS[dtype_name]:
            raise AssertionError(
                f"{dtype_name} {name} normalized error {error:.6g} exceeds {ERROR_LIMITS[dtype_name]:.6g}"
            )
        numerics[name] = {"normalized_error_vs_fp32": error}

    latencies = {name: bench_cuda(fn, warmup, iterations) for name, fn in providers.items()}
    winner = min(latencies, key=latencies.get)
    return {
        "shape": {
            "name": shape.name,
            "role": shape.role,
            "batch": shape.batch,
            "q_heads": shape.q_heads,
            "kv_heads": shape.kv_heads,
            "gqa_group_size": shape.gqa_group_size,
            "query_length": shape.query_length,
            "key_length": shape.key_length,
            "head_dim": shape.head_dim,
            "causal": shape.causal,
            "query_start": shape.query_start,
            "materialized_score_bytes_fp32": shape.materialized_score_bytes_fp32,
        },
        "dtype": dtype_name,
        "requirements": list(DTYPE_REQUIREMENTS[dtype_name]),
        "numerics": numerics,
        "providers": {
            name: provider_record(shape, dtype, latency) for name, latency in latencies.items()
        },
        "winner": winner,
    }


def print_result(result: dict[str, object]) -> None:
    shape = result["shape"]
    print(
        f"\n[{result['dtype']}] {shape['name']}: B={shape['batch']} QH={shape['q_heads']} KVH={shape['kv_heads']} "
        f"Q={shape['query_length']} K={shape['key_length']} D={shape['head_dim']} causal={shape['causal']}"
    )
    print(f"  avoided FP32 score matrix: {shape['materialized_score_bytes_fp32'] / (1024**2):.3f} MiB")
    for name, record in result["providers"].items():
        print(
            f"  {name:20s} {record['latency_ms']:.6f} ms  "
            f"{record['logical_tflops_qk_pv']:.3f} logical TFLOP/s"
        )
    print(f"  winner: {result['winner']}")
    print(f"  numerics: {result['numerics']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=SHAPE_FAMILIES, default="validation")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
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
        "contract": "score-matrix-free forward attention; FP16/BF16 storage; FP32 score/online-state/output accumulation; causal and GQA aware",
        "results": [],
        "skipped": [],
    }

    for dtype_name in requested:
        if dtype_name == "bf16" and not bf16_supported():
            print("[bf16] SKIP: requires compute capability >= 8.0 and PyTorch BF16 CUDA support")
            payload["skipped"].append({"dtype": "bf16", "requires": list(DTYPE_REQUIREMENTS["bf16"])})
            continue
        for shape in shapes_for_family(args.family):
            result = run_shape(shape, dtype_name, args.warmup, args.iterations)
            payload["results"].append(result)
            print_result(result)

    if not payload["results"]:
        raise RuntimeError("no supported FlashAttention workloads executed")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON evidence: {args.json}")
    print("\nFlashAttention-style validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
