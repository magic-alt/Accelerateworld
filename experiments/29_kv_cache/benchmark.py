from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import triton

import accelerateworld_kv_cache_cuda as cuda_kv
from kv_cache_triton import kv_cache_read as triton_read
from kv_cache_triton import kv_cache_update as triton_update
from lab_config import DTYPE_REQUIREMENTS, LAYOUTS, SHAPE_FAMILIES, KVCacheShape, layout_id, shapes_for_family
from reference import allocate_cache, kv_cache_read_reference, kv_cache_update_reference, validate_positions


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


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


def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and bool(torch.cuda.is_bf16_supported())


def positions(batch: int, tokens: int, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + tokens, device="cuda", dtype=torch.int64).view(1, tokens).expand(batch, tokens).contiguous()


def random_kv(shape: KVCacheShape, tokens: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_shape = (shape.batch, shape.kv_heads, tokens, shape.head_dim)
    k = torch.randn(tensor_shape, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(tensor_shape, device="cuda", dtype=dtype).contiguous()
    return k, v


def logical_gbps(shape: KVCacheShape, tokens: int, element_size: int, latency_ms: float) -> float:
    # K+V source/cache traffic: two tensors, one read and one write per logical element.
    logical_bytes = shape.transfer_elements(tokens) * element_size * 4
    return logical_bytes / (latency_ms / 1000.0) / 1.0e9


def provider_metrics(shape: KVCacheShape, element_size: int, prefill_ms: float, decode_ms: float, read_ms: float) -> dict[str, float]:
    return {
        "prefill_write_ms": prefill_ms,
        "prefill_write_logical_gbps": logical_gbps(shape, shape.prefill_tokens, element_size, prefill_ms),
        "decode_append_ms": decode_ms,
        "decode_append_us": decode_ms * 1000.0,
        "decode_append_logical_gbps": logical_gbps(shape, 1, element_size, decode_ms),
        "context_read_ms": read_ms,
        "context_read_logical_gbps": logical_gbps(shape, shape.context_tokens, element_size, read_ms),
    }


def make_provider(name: str, layout: str):
    lid = layout_id(layout)
    if name == "pytorch":
        return (
            lambda ck, cv, nk, nv, pos: kv_cache_update_reference(ck, cv, nk, nv, pos, layout=layout, check_bounds=False),
            lambda ck, cv, pos: kv_cache_read_reference(ck, cv, pos, layout=layout, check_bounds=False),
        )
    if name == "triton":
        return (
            lambda ck, cv, nk, nv, pos: triton_update(ck, cv, nk, nv, pos, layout=layout, check_bounds=False),
            lambda ck, cv, pos: triton_read(ck, cv, pos, layout=layout, check_bounds=False),
        )
    if name == "cuda":
        return (
            lambda ck, cv, nk, nv, pos: cuda_kv.update(ck, cv, nk, nv, pos, lid, False),
            lambda ck, cv, pos: tuple(cuda_kv.read(ck, cv, pos, lid, False)),
        )
    raise KeyError(name)


def run_shape(shape: KVCacheShape, dtype_name: str, layout: str, warmup: int, iterations: int) -> dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    prefill_pos = positions(shape.batch, shape.prefill_tokens)
    decode_pos = positions(shape.batch, 1, shape.decode_position)
    context_pos = positions(shape.batch, shape.context_tokens)
    for pos in (prefill_pos, decode_pos, context_pos):
        validate_positions(pos, shape.capacity)

    prefill_k, prefill_v = random_kv(shape, shape.prefill_tokens, dtype)
    decode_k, decode_v = random_kv(shape, 1, dtype)
    expected_k = torch.cat((prefill_k, decode_k), dim=2)
    expected_v = torch.cat((prefill_v, decode_v), dim=2)

    records: dict[str, object] = {}
    for provider_name in ("pytorch", "triton", "cuda"):
        update, read = make_provider(provider_name, layout)
        cache_k, cache_v = allocate_cache(
            shape.batch, shape.kv_heads, shape.capacity, shape.head_dim,
            dtype=dtype, device="cuda", layout=layout,
        )

        update(cache_k, cache_v, prefill_k, prefill_v, prefill_pos)
        update(cache_k, cache_v, decode_k, decode_v, decode_pos)
        got_k, got_v = read(cache_k, cache_v, context_pos)
        torch.cuda.synchronize()
        if not torch.equal(got_k, expected_k) or not torch.equal(got_v, expected_v):
            max_error = max(
                float((got_k.float() - expected_k.float()).abs().max().item()),
                float((got_v.float() - expected_v.float()).abs().max().item()),
            )
            raise AssertionError(f"{provider_name}/{layout}/{dtype_name} KV read mismatch: {max_error}")

        prefill_ms = bench_cuda(
            lambda: update(cache_k, cache_v, prefill_k, prefill_v, prefill_pos), warmup, iterations
        )
        decode_ms = bench_cuda(
            lambda: update(cache_k, cache_v, decode_k, decode_v, decode_pos), warmup, iterations
        )
        read_ms = bench_cuda(lambda: read(cache_k, cache_v, context_pos), warmup, iterations)
        records[provider_name] = provider_metrics(shape, prefill_k.element_size(), prefill_ms, decode_ms, read_ms)

    winners = {
        phase: min(records, key=lambda name: records[name][phase])
        for phase in ("prefill_write_ms", "decode_append_ms", "context_read_ms")
    }
    cache_bytes = shape.cache_elements_per_tensor * prefill_k.element_size() * 2
    return {
        "shape": {
            "name": shape.name,
            "role": shape.role,
            "batch": shape.batch,
            "q_heads": shape.q_heads,
            "kv_heads": shape.kv_heads,
            "group_size": shape.group_size,
            "capacity": shape.capacity,
            "prefill_tokens": shape.prefill_tokens,
            "decode_position": shape.decode_position,
            "context_tokens": shape.context_tokens,
            "head_dim": shape.head_dim,
        },
        "dtype": dtype_name,
        "layout": layout,
        "requirements": list(DTYPE_REQUIREMENTS[dtype_name]),
        "cache_bytes_k_plus_v": cache_bytes,
        "providers": records,
        "winners": winners,
    }


def print_result(result: dict[str, object]) -> None:
    s = result["shape"]
    print(
        f"\n[{result['dtype']}/{result['layout']}] {s['name']}: "
        f"QH={s['q_heads']} KVH={s['kv_heads']} capacity={s['capacity']} "
        f"prefill={s['prefill_tokens']} D={s['head_dim']}"
    )
    for name, metrics in result["providers"].items():
        print(
            f"  {name:8s} prefill={metrics['prefill_write_ms']:.6f} ms "
            f"decode={metrics['decode_append_us']:.3f} us "
            f"read={metrics['context_read_ms']:.6f} ms"
        )
    print(f"  winners: {result['winners']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=SHAPE_FAMILIES, default="validation")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    requested_dtypes = tuple(part.strip() for part in args.dtypes.split(",") if part.strip())
    requested_layouts = tuple(part.strip() for part in args.layouts.split(",") if part.strip())
    if any(name not in DTYPE_MAP for name in requested_dtypes):
        raise ValueError(f"unsupported dtypes: {requested_dtypes}")
    if any(name not in LAYOUTS for name in requested_layouts):
        raise ValueError(f"unsupported layouts: {requested_layouts}")

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
        "contract": "K/V cache stores only KV heads; update input and read output are [B,KVH,T,D]; token_major/head_major storage; FP16/BF16 bit-preserving movement",
        "results": [],
        "skipped": [],
    }

    for dtype_name in requested_dtypes:
        if dtype_name == "bf16" and not bf16_supported():
            print("[bf16] SKIP: requires compute capability >= 8.0 and PyTorch BF16 CUDA support")
            payload["skipped"].append({"dtype": "bf16", "requires": list(DTYPE_REQUIREMENTS["bf16"])})
            continue
        for layout in requested_layouts:
            for shape in shapes_for_family(args.family):
                result = run_shape(shape, dtype_name, layout, args.warmup, args.iterations)
                payload["results"].append(result)
                print_result(result)

    if not payload["results"]:
        raise RuntimeError("no supported KV-cache workloads executed")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON evidence: {args.json}")
    print("\nKV-cache update/read validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
