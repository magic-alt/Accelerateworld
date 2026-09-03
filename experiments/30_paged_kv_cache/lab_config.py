from __future__ import annotations

from dataclasses import dataclass


DTYPE_REQUIREMENTS = {
    "fp16": ("cuda",),
    "bf16": ("cuda", "bf16"),
}

PAGE_SIZES = (16, 32)
SHAPE_FAMILIES = ("validation", "decode", "prefill", "long", "all")


@dataclass(frozen=True)
class PagedKVShape:
    name: str
    family: str
    role: str
    batch: int
    q_heads: int
    kv_heads: int
    max_tokens: int
    prefill_tokens: int
    head_dim: int
    page_size: int
    spare_pages: int = 4

    @property
    def decode_position(self) -> int:
        return self.prefill_tokens

    @property
    def context_tokens(self) -> int:
        return self.prefill_tokens + 1

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def max_blocks(self) -> int:
        return (self.max_tokens + self.page_size - 1) // self.page_size

    def pages_for_tokens(self, tokens: int) -> int:
        if tokens <= 0:
            return 0
        return (tokens + self.page_size - 1) // self.page_size

    @property
    def active_pages_per_sequence(self) -> int:
        return self.pages_for_tokens(self.context_tokens)

    @property
    def physical_pages(self) -> int:
        return self.batch * self.active_pages_per_sequence + self.spare_pages

    @property
    def decode_starts_new_page(self) -> bool:
        return self.decode_position % self.page_size == 0

    def transfer_elements(self, tokens: int) -> int:
        return self.batch * self.kv_heads * tokens * self.head_dim


SHAPES = (
    PagedKVShape(
        "validation_mqa_cross_page", "validation", "decode",
        2, 8, 1, 64, 16, 64, 16, 4,
    ),
    PagedKVShape(
        "validation_gqa_fragmented", "validation", "prefill",
        2, 8, 2, 128, 31, 128, 16, 6,
    ),
    PagedKVShape(
        "mqa_decode_2048", "decode", "decode",
        4, 32, 1, 2048, 2032, 128, 16, 8,
    ),
    PagedKVShape(
        "gqa_decode_4096", "decode", "decode",
        2, 32, 8, 4096, 4080, 128, 16, 16,
    ),
    PagedKVShape(
        "gqa_prefill_512_page32", "prefill", "prefill",
        2, 32, 8, 1024, 512, 128, 32, 8,
    ),
    PagedKVShape(
        "gqa_prefill_2048", "prefill", "prefill",
        1, 32, 8, 4096, 2048, 128, 16, 8,
    ),
    PagedKVShape(
        "gqa_long_decode_32768", "long", "long_decode",
        1, 32, 8, 32768, 32752, 128, 16, 16,
    ),
)


def validate_shape(shape: PagedKVShape) -> None:
    values = (
        shape.batch,
        shape.q_heads,
        shape.kv_heads,
        shape.max_tokens,
        shape.prefill_tokens,
        shape.head_dim,
        shape.page_size,
        shape.spare_pages,
    )
    if min(values) <= 0:
        raise ValueError(f"shape dimensions must be positive: {shape}")
    if shape.q_heads % shape.kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads: {shape}")
    if shape.prefill_tokens >= shape.max_tokens:
        raise ValueError(f"prefill must leave at least one decode token: {shape}")
    if shape.head_dim not in (64, 128):
        raise ValueError(f"v0 paged KV cache supports head_dim 64 or 128: {shape}")
    if shape.head_dim % 2 != 0:
        raise ValueError(f"32-bit pair copy requires even head_dim: {shape}")
    if shape.page_size not in PAGE_SIZES:
        raise ValueError(f"page_size must be one of {PAGE_SIZES}: {shape}")
    if shape.context_tokens > shape.max_tokens:
        raise ValueError(f"context exceeds max_tokens: {shape}")


def validate_shapes() -> None:
    names = set()
    for shape in SHAPES:
        validate_shape(shape)
        if shape.name in names:
            raise ValueError(f"duplicate shape name: {shape.name}")
        names.add(shape.name)
    if not any(shape.kv_heads == 1 for shape in SHAPES):
        raise ValueError("shape matrix must include MQA")
    if not any(shape.q_heads > shape.kv_heads > 1 for shape in SHAPES):
        raise ValueError("shape matrix must include GQA")
    if not any(shape.max_tokens >= 32768 for shape in SHAPES):
        raise ValueError("shape matrix must include a long-context workload")
    if not any(shape.decode_starts_new_page for shape in SHAPES):
        raise ValueError("shape matrix must include decode crossing into a new page")
    if set(shape.page_size for shape in SHAPES) != set(PAGE_SIZES):
        raise ValueError("shape matrix must cover every supported page size")


def shapes_for_family(family: str) -> tuple[PagedKVShape, ...]:
    if family not in SHAPE_FAMILIES:
        raise KeyError(f"unknown shape family: {family}")
    if family == "all":
        return SHAPES
    return tuple(shape for shape in SHAPES if shape.family == family)


def requirements_for_dtype(dtype: str) -> tuple[str, ...]:
    try:
        return DTYPE_REQUIREMENTS[dtype]
    except KeyError as exc:
        raise KeyError(f"unsupported dtype: {dtype}") from exc


validate_shapes()
