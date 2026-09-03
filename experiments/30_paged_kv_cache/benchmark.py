from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch
import triton

import accelerateworld_paged_kv_cache_cuda as cuda_paged
from allocator import allocator_reuse_demo, build_block_table
from lab_config import (
    DTYPE_REQUIREMENTS,
    SHAPE_FAMILIES,
    PagedKVShape,
    shapes_for_family,
)
from paged_kv_triton import paged_kv_read as triton_read
from paged_kv_triton import paged_kv_update as triton_update
from reference import (
    allocate_page_cache,
    block_table_tensor,
    paged_kv_read_reference,
    paged_kv_update_reference,
    validate_block_table,
    validate_positions,
)


ROOT = Path(__file__).resolve().parents[2]
EXP29 = ROOT / "experiments" / "29_kv_cache"
sys.path.insert(0, str(EXP29))
import accelerateworld_kv_cache_cuda as cuda_contiguous  # noqa: E402


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
HEAD_MAJOR_LAYOUT_ID = 1


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
    return (
        torch.arange(start, start + tokens, device="cuda", dtype=torch.int64)
        .view(1, tokens)
        .expand(batch, tokens)
        .contiguous()
    )


def random_kv(
    shape: PagedKVShape, tokens: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_shape = (shape.batch, shape.kv_heads, tokens, shape.head_dim)
    return (
        torch.randn(tensor_shape, device="cuda", dtype=dtype).contiguous(),
        torch.randn(tensor_shape, device="cuda", dtype=dtype).contiguous(),
    )


def logical_gbps(
    shape: PagedKVShape,
    tokens: int,
    element_size: int,
    latency_ms: float,
) -> float:
    logical_bytes = shape.transfer_elements(tokens) * element_size * 4
    return logical_bytes / (latency_ms / 1000.0) / 1.0e9


def provider_metrics(
    shape: PagedKVShape,
    element_size: int,
    prefill_ms: float,
    decode_ms: float,
    read_ms: float,
) -> dict[str, float]:
    return {
        "prefill_write_ms": prefill_ms,
        "prefill_write_logical_gbps": logical_gbps(
            shape, shape.prefill_tokens, element_size, prefill_ms
        ),
        "decode_append_ms": decode_ms,
        "decode_append_us": decode_ms * 1000.0,
        "decode_append_logical_gbps": logical_gbps(
            shape, 1, element_size, decode_ms
        ),
        "context_read_ms": read_ms,
        "context_read_logical_gbps": logical_gbps(
            shape, shape.context_tokens, element_size, read_ms
        ),
    }


def paged_provider(name: str, page_size: int):
    if name == "pytorch_paged":
        return (
            lambda pk, pv, nk, nv, pos, table: paged_kv_update_reference(
                pk, pv, nk, nv, pos, table,
                page_size=page_size, check_bounds=False,
            ),
            lambda pk, pv, pos, table: paged_kv_read_reference(
                pk, pv, pos, table,
                page_size=page_size, check_bounds=False,
            ),
        )
    if name == "triton_paged":
        return (
            lambda pk, pv, nk, nv, pos, table: triton_update(
                pk, pv, nk, nv, pos, table,
                page_size=page_size, check_bounds=False,
            ),
            lambda pk, pv, pos, table: triton_read(
                pk, pv, pos, table,
                page_size=page_size, check_bounds=False,
            ),
        )
    if name == "cuda_paged":
        return (
            lambda pk, pv, nk, nv, pos, table: cuda_paged.update(
                pk, pv, nk, nv, pos, table, page_size, False
            ),
            lambda pk, pv, pos, table: tuple(
                cuda_paged.read(pk, pv, pos, table, page_size, False)
            ),
        )
    raise KeyError(name)


def run_shape(
    shape: PagedKVShape,
    dtype_name: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    dtype = DTYPE_MAP[dtype_name]
    prefill_pos = positions(shape.batch, shape.prefill_tokens)
    decode_pos = positions(shape.batch, 1, shape.decode_position)
    context_pos = positions(shape.batch, shape.context_tokens)
    max_logical = shape.max_blocks * shape.page_size
    for pos in (prefill_pos, decode_pos, context_pos):
        validate_positions(pos, max_logical)

    lengths = [shape.context_tokens] * shape.batch
    table_list, allocator = build_block_table(
        lengths,
        max_tokens=shape.max_tokens,
        page_size=shape.page_size,
        total_pages=shape.physical_pages,
    )
    block_table = block_table_tensor(table_list, device="cuda").contiguous()
    validate_block_table(block_table, shape.physical_pages)

    prefill_k, prefill_v = random_kv(shape, shape.prefill_tokens, dtype)
    decode_k, decode_v = random_kv(shape, 1, dtype)
    expected_k = torch.cat((prefill_k, decode_k), dim=2)
    expected_v = torch.cat((prefill_v, decode_v), dim=2)

    records: dict[str, dict[str, float]] = {}
    for provider_name in ("pytorch_paged", "triton_paged", "cuda_paged"):
        update, read = paged_provider(provider_name, shape.page_size)
        page_k, page_v = allocate_page_cache(
            shape.physical_pages,
            shape.kv_heads,
            shape.page_size,
            shape.head_dim,
            dtype=dtype,
            device="cuda",
        )

        update(page_k, page_v, prefill_k, prefill_v, prefill_pos, block_table)
        update(page_k, page_v, decode_k, decode_v, decode_pos, block_table)
        got_k, got_v = read(page_k, page_v, context_pos, block_table)
        torch.cuda.synchronize()
        if not torch.equal(got_k, expected_k) or not torch.equal(got_v, expected_v):
            max_error = max(
                float((got_k.float() - expected_k.float()).abs().max().item()),
                float((got_v.float() - expected_v.float()).abs().max().item()),
            )
            raise AssertionError(
                f"{provider_name}/{dtype_name}/{shape.name} paged read mismatch: {max_error}"
            )

        prefill_ms = bench_cuda(
            lambda: update(
                page_k, page_v, prefill_k, prefill_v, prefill_pos, block_table
            ),
            warmup,
            iterations,
        )
        decode_ms = bench_cuda(
            lambda: update(
                page_k, page_v, decode_k, decode_v, decode_pos, block_table
            ),
            warmup,
            iterations,
        )
        read_ms = bench_cuda(
            lambda: read(page_k, page_v, context_pos, block_table),
            warmup,
            iterations,
        )
        records[provider_name] = provider_metrics(
            shape, prefill_k.element_size(), prefill_ms, decode_ms, read_ms
        )

    contiguous_k = torch.zeros(
        (shape.batch, shape.kv_heads, shape.max_tokens, shape.head_dim),
        device="cuda",
        dtype=dtype,
    )
    contiguous_v = torch.zeros_like(contiguous_k)

    def contiguous_update(nk, nv, pos):
        cuda_contiguous.update(
            contiguous_k,
            contiguous_v,
            nk,
            nv,
            pos,
            HEAD_MAJOR_LAYOUT_ID,
            False,
        )

    def contiguous_read(pos):
        return tuple(
            cuda_contiguous.read(
                contiguous_k,
                contiguous_v,
                pos,
                HEAD_MAJOR_LAYOUT_ID,
                False,
            )
        )

    contiguous_update(prefill_k, prefill_v, prefill_pos)
    contiguous_update(decode_k, decode_v, decode_pos)
    got_k, got_v = contiguous_read(context_pos)
    torch.cuda.synchronize()
    if not torch.equal(got_k, expected_k) or not torch.equal(got_v, expected_v):
        raise AssertionError("contiguous CUDA baseline read mismatch")

    records["cuda_contiguous"] = provider_metrics(
        shape,
        prefill_k.element_size(),
        bench_cuda(
            lambda: contiguous_update(prefill_k, prefill_v, prefill_pos),
            warmup,
            iterations,
        ),
        bench_cuda(
            lambda: contiguous_update(decode_k, decode_v, decode_pos),
            warmup,
            iterations,
        ),
        bench_cuda(lambda: contiguous_read(context_pos), warmup, iterations),
    )

    winners = {
        phase: min(records, key=lambda name: records[name][phase])
        for phase in ("prefill_write_ms", "decode_append_ms", "context_read_ms")
    }
    paged = records["cuda_paged"]
    contiguous = records["cuda_contiguous"]

    snapshot = allocator.snapshot(lengths, shape.page_size)
    element_size = prefill_k.element_size()
    paged_pool_bytes = (
        shape.physical_pages
        * shape.kv_heads
        * shape.page_size
        * shape.head_dim
        * element_size
        * 2
    )
    contiguous_reserved_bytes = (
        shape.batch
        * shape.kv_heads
        * shape.max_tokens
        * shape.head_dim
        * element_size
        * 2
    )
    block_table_bytes = block_table.numel() * block_table.element_size()

    active_pages_per_sequence = [
        row[: shape.active_pages_per_sequence] for row in table_list
    ]
    non_contiguous = any(
        any(next_page - page != 1 for page, next_page in zip(pages, pages[1:]))
        for pages in active_pages_per_sequence
        if len(pages) > 1
    )

    return {
        "shape": {
            "name": shape.name,
            "role": shape.role,
            "batch": shape.batch,
            "q_heads": shape.q_heads,
            "kv_heads": shape.kv_heads,
            "group_size": shape.group_size,
            "max_tokens": shape.max_tokens,
            "prefill_tokens": shape.prefill_tokens,
            "decode_position": shape.decode_position,
            "context_tokens": shape.context_tokens,
            "head_dim": shape.head_dim,
            "page_size": shape.page_size,
            "max_blocks": shape.max_blocks,
            "physical_pages": shape.physical_pages,
            "decode_starts_new_page": shape.decode_starts_new_page,
        },
        "dtype": dtype_name,
        "requirements": list(DTYPE_REQUIREMENTS[dtype_name]),
        "block_table_non_contiguous": non_contiguous,
        "allocator": {
            "allocated_pages": snapshot.allocated_pages,
            "free_pages": snapshot.free_pages,
            "live_tokens": snapshot.live_tokens,
            "allocated_slots": snapshot.allocated_slots,
            "internal_fragmentation_slots": snapshot.internal_fragmentation_slots,
            "internal_fragmentation_ratio": snapshot.internal_fragmentation_ratio,
            "largest_free_run": snapshot.largest_free_run,
            "external_fragmentation_ratio": snapshot.external_fragmentation_ratio,
        },
        "memory": {
            "paged_pool_bytes_k_plus_v": paged_pool_bytes,
            "block_table_bytes": block_table_bytes,
            "contiguous_reserved_bytes_k_plus_v": contiguous_reserved_bytes,
            "paged_plus_table_over_contiguous": (
                (paged_pool_bytes + block_table_bytes) / contiguous_reserved_bytes
            ),
        },
        "providers": records,
        "winners": winners,
        "cuda_paged_vs_contiguous": {
            "prefill_write_latency_ratio": (
                paged["prefill_write_ms"] / contiguous["prefill_write_ms"]
            ),
            "decode_append_latency_ratio": (
                paged["decode_append_ms"] / contiguous["decode_append_ms"]
            ),
            "context_read_latency_ratio": (
                paged["context_read_ms"] / contiguous["context_read_ms"]
            ),
        },
    }


def print_result(result: dict[str, object]) -> None:
    s = result["shape"]
    print(
        f"\n[{result['dtype']}] {s['name']}: B={s['batch']} "
        f"QH={s['q_heads']} KVH={s['kv_heads']} max={s['max_tokens']} "
        f"page={s['page_size']} prefill={s['prefill_tokens']} D={s['head_dim']}"
    )
    for name, metrics in result["providers"].items():
        print(
            f"  {name:16s} prefill={metrics['prefill_write_ms']:.6f} ms "
            f"decode={metrics['decode_append_us']:.3f} us "
            f"read={metrics['context_read_ms']:.6f} ms"
        )
    print(f"  allocator: {result['allocator']}")
    print(f"  winners: {result['winners']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=SHAPE_FAMILIES, default="validation")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    requested_dtypes = tuple(
        part.strip() for part in args.dtypes.split(",") if part.strip()
    )
    if any(name not in DTYPE_MAP for name in requested_dtypes):
        raise ValueError(f"unsupported dtypes: {requested_dtypes}")

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
        "contract": (
            "logical position -> logical block -> block_table -> physical page + slot; "
            "physical pages [P,KVH,page_size,D]; read output [B,KVH,T,D]; "
            "allocator is CPU/runtime metadata and excluded from GPU kernel timing"
        ),
        "allocator_reuse_demo": allocator_reuse_demo(),
        "results": [],
        "skipped": [],
    }

    for dtype_name in requested_dtypes:
        if dtype_name == "bf16" and not bf16_supported():
            print(
                "[bf16] SKIP: requires compute capability >= 8.0 "
                "and PyTorch BF16 CUDA support"
            )
            payload["skipped"].append(
                {"dtype": "bf16", "requires": list(DTYPE_REQUIREMENTS["bf16"])}
            )
            continue
        for shape in shapes_for_family(args.family):
            result = run_shape(
                shape, dtype_name, args.warmup, args.iterations
            )
            payload["results"].append(result)
            print_result(result)

    if not payload["results"]:
        raise RuntimeError("no supported paged KV-cache workloads executed")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON evidence: {args.json}")
    print("\nPaged KV-cache validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
