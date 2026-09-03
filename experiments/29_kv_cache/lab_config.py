from __future__ import annotations

from dataclasses import dataclass


DTYPE_REQUIREMENTS = {
    "fp16": ("cuda",),
    "bf16": ("cuda", "bf16"),
}

LAYOUTS = ("token_major", "head_major")
LAYOUT_IDS = {"token_major": 0, "head_major": 1}


@dataclass(frozen=True)
class KVCacheShape:
    name: str
    family: str
    role: str
    batch: int
    q_heads: int
    kv_heads: int
    capacity: int
    prefill_tokens: int
    head_dim: int

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
    def cache_elements_per_tensor(self) -> int:
        return self.batch * self.kv_heads * self.capacity * self.head_dim

    def transfer_elements(self, tokens: int) -> int:
        return self.batch * self.kv_heads * tokens * self.head_dim


SHAPES = (
    KVCacheShape("validation_mqa_decode", "validation", "decode", 1, 8, 1, 128, 32, 64),
    KVCacheShape("validation_gqa_prefill", "validation", "prefill", 1, 8, 2, 256, 64, 128),
    KVCacheShape("mqa_decode_2048", "decode", "decode", 1, 32, 1, 2048, 2047, 128),
    KVCacheShape("gqa_decode_4096", "decode", "decode", 1, 32, 8, 4096, 4095, 128),
    KVCacheShape("gqa_prefill_512", "prefill", "prefill", 1, 32, 8, 1024, 512, 128),
    KVCacheShape("gqa_prefill_2048", "prefill", "prefill", 1, 32, 8, 4096, 2048, 128),
    KVCacheShape("gqa_long_decode_32768", "long", "long_decode", 1, 32, 8, 32768, 32767, 128),
)

SHAPE_FAMILIES = ("validation", "decode", "prefill", "long", "all")


def validate_shape(shape: KVCacheShape) -> None:
    values = (
        shape.batch,
        shape.q_heads,
        shape.kv_heads,
        shape.capacity,
        shape.prefill_tokens,
        shape.head_dim,
    )
    if min(values) <= 0:
        raise ValueError(f"shape dimensions must be positive: {shape}")
    if shape.q_heads % shape.kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads: {shape}")
    if shape.prefill_tokens >= shape.capacity:
        raise ValueError(f"prefill must leave at least one decode slot: {shape}")
    if shape.head_dim not in (64, 128):
        raise ValueError(f"v0 KV cache supports head_dim 64 or 128: {shape}")
    if shape.head_dim % 2 != 0:
        raise ValueError(f"vectorized pair copy requires an even head_dim: {shape}")


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
    if not any(shape.capacity >= 32768 for shape in SHAPES):
        raise ValueError("shape matrix must include long-context cache capacity")


def shapes_for_family(family: str) -> tuple[KVCacheShape, ...]:
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


def layout_id(layout: str) -> int:
    try:
        return LAYOUT_IDS[layout]
    except KeyError as exc:
        raise KeyError(f"unsupported layout: {layout}") from exc


validate_shapes()
