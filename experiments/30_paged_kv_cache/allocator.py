from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class AllocatorSnapshot:
    total_pages: int
    allocated_pages: int
    free_pages: int
    active_sequences: int
    live_tokens: int
    allocated_slots: int
    internal_fragmentation_slots: int
    internal_fragmentation_ratio: float
    largest_free_run: int
    external_fragmentation_ratio: float


class PageAllocator:
    """Small deterministic CPU page allocator used to model runtime metadata.

    The GPU kernels never allocate pages. They consume a block table produced by
    this allocator, mirroring the separation between runtime memory management
    and attention/cache data movement in production inference systems.
    """

    def __init__(self, total_pages: int) -> None:
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        self.total_pages = total_pages
        self._free = list(range(0, total_pages, 2)) + list(range(1, total_pages, 2))
        self._owners: dict[str, list[int]] = {}

    @property
    def free_pages(self) -> tuple[int, ...]:
        return tuple(self._free)

    @property
    def owners(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(pages) for name, pages in self._owners.items()}

    def allocate_one(self, owner: str) -> int:
        if not self._free:
            raise MemoryError("paged KV page pool exhausted")
        page = self._free.pop(0)
        self._owners.setdefault(owner, []).append(page)
        return page

    def release(self, owner: str) -> tuple[int, ...]:
        pages = self._owners.pop(owner, [])
        self._free = list(pages) + self._free
        return tuple(pages)

    def largest_free_run(self) -> int:
        if not self._free:
            return 0
        ordered = sorted(self._free)
        best = run = 1
        for previous, current in zip(ordered, ordered[1:]):
            if current == previous + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return best

    def snapshot(self, lengths: list[int], page_size: int) -> AllocatorSnapshot:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if any(length < 0 for length in lengths):
            raise ValueError("sequence lengths must be non-negative")
        allocated_pages = sum(len(pages) for pages in self._owners.values())
        free_pages = len(self._free)
        live_tokens = sum(lengths)
        allocated_slots = sum(ceil(length / page_size) * page_size for length in lengths if length > 0)
        internal_slots = allocated_slots - live_tokens
        internal_ratio = internal_slots / allocated_slots if allocated_slots else 0.0
        largest_run = self.largest_free_run()
        external_ratio = 0.0 if free_pages <= 1 else 1.0 - largest_run / free_pages
        return AllocatorSnapshot(
            total_pages=self.total_pages,
            allocated_pages=allocated_pages,
            free_pages=free_pages,
            active_sequences=sum(1 for length in lengths if length > 0),
            live_tokens=live_tokens,
            allocated_slots=allocated_slots,
            internal_fragmentation_slots=internal_slots,
            internal_fragmentation_ratio=internal_ratio,
            largest_free_run=largest_run,
            external_fragmentation_ratio=external_ratio,
        )


def blocks_for_tokens(tokens: int, page_size: int) -> int:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return (tokens + page_size - 1) // page_size if tokens else 0


def build_block_table(
    lengths: list[int],
    *,
    max_tokens: int,
    page_size: int,
    total_pages: int,
) -> tuple[list[list[int]], PageAllocator]:
    if not lengths:
        raise ValueError("at least one sequence length is required")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if any(length < 0 or length > max_tokens for length in lengths):
        raise ValueError("sequence length exceeds logical capacity")

    max_blocks = blocks_for_tokens(max_tokens, page_size)
    needed = [blocks_for_tokens(length, page_size) for length in lengths]
    if sum(needed) > total_pages:
        raise MemoryError("not enough physical pages for requested sequences")

    allocator = PageAllocator(total_pages)
    table = [[-1] * max_blocks for _ in lengths]
    for logical_block in range(max(needed, default=0)):
        for sequence, count in enumerate(needed):
            if logical_block < count:
                table[sequence][logical_block] = allocator.allocate_one(f"seq:{sequence}")
    return table, allocator


def allocator_reuse_demo(total_pages: int = 12) -> dict[str, object]:
    allocator = PageAllocator(total_pages)
    first = [allocator.allocate_one("seq:a") for _ in range(3)]
    _ = [allocator.allocate_one("seq:b") for _ in range(2)]
    released = allocator.release("seq:a")
    reused = [allocator.allocate_one("seq:c") for _ in range(2)]
    return {
        "first_pages": first,
        "released_pages": list(released),
        "reused_pages": reused,
        "reuse_observed": all(page in released for page in reused),
    }
