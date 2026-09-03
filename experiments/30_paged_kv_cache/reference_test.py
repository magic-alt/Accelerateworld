from __future__ import annotations

import torch

from allocator import PageAllocator, allocator_reuse_demo, build_block_table
from reference import (
    allocate_page_cache,
    block_table_tensor,
    paged_kv_read_reference,
    paged_kv_update_reference,
)


def run_layout_case(page_size: int) -> None:
    lengths = [page_size + 3, page_size * 2 + 1]
    max_tokens = page_size * 4
    total_pages = 12
    table, allocator = build_block_table(
        lengths, max_tokens=max_tokens, page_size=page_size, total_pages=total_pages
    )
    block_table = block_table_tensor(table, device="cpu")
    page_k, page_v = allocate_page_cache(
        total_pages, 2, page_size, 64, dtype=torch.float32, device="cpu"
    )

    positions = torch.tensor(
        [
            [0, page_size - 1, page_size, page_size + 2],
            [1, page_size, page_size + 1, page_size * 2],
        ],
        dtype=torch.int64,
    )
    tokens = positions.shape[1]
    values = torch.arange(2 * 2 * tokens * 64, dtype=torch.float32).reshape(2, 2, tokens, 64)
    new_k = values.contiguous()
    new_v = (values + 10000).contiguous()

    paged_kv_update_reference(
        page_k, page_v, new_k, new_v, positions, block_table,
        page_size=page_size,
    )
    got_k, got_v = paged_kv_read_reference(
        page_k, page_v, positions, block_table, page_size=page_size
    )
    torch.testing.assert_close(got_k, new_k, rtol=0, atol=0)
    torch.testing.assert_close(got_v, new_v, rtol=0, atol=0)

    for sequence, length in enumerate(lengths):
        used_blocks = (length + page_size - 1) // page_size
        pages = table[sequence][:used_blocks]
        if len(pages) > 1:
            assert any(b - a != 1 for a, b in zip(pages, pages[1:])), pages

    snapshot = allocator.snapshot(lengths, page_size)
    assert snapshot.internal_fragmentation_slots >= 0
    assert 0.0 <= snapshot.internal_fragmentation_ratio < 1.0
    assert 0.0 <= snapshot.external_fragmentation_ratio <= 1.0


def run_unallocated_rejection() -> None:
    table, _ = build_block_table(
        [8], max_tokens=64, page_size=16, total_pages=4
    )
    block_table = block_table_tensor(table, device="cpu")
    page_k, page_v = allocate_page_cache(4, 1, 16, 64, dtype=torch.float32, device="cpu")
    positions = torch.tensor([[31]], dtype=torch.int64)
    try:
        paged_kv_read_reference(
            page_k, page_v, positions, block_table, page_size=16
        )
    except IndexError:
        pass
    else:
        raise AssertionError("unallocated logical block must be rejected")


def run_allocator_reuse() -> None:
    demo = allocator_reuse_demo()
    assert demo["reuse_observed"], demo

    allocator = PageAllocator(8)
    first = [allocator.allocate_one("seq:0") for _ in range(2)]
    released = allocator.release("seq:0")
    second = [allocator.allocate_one("seq:1") for _ in range(2)]
    assert set(second).issubset(set(released))
    assert set(first) == set(released)


def main() -> int:
    run_layout_case(16)
    run_layout_case(32)
    run_unallocated_rejection()
    run_allocator_reuse()
    print("Paged KV-cache CPU reference validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
