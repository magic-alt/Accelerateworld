from __future__ import annotations

from dataclasses import dataclass


DTYPE_REQUIREMENTS = {
    "fp16": ("cuda",),
    "bf16": ("cuda", "bf16"),
}


@dataclass(frozen=True)
class AttentionShape:
    name: str
    family: str
    role: str
    batch: int
    q_heads: int
    kv_heads: int
    query_length: int
    key_length: int
    head_dim: int
    causal: bool
    query_start: int = 0

    @property
    def rows(self) -> int:
        return self.batch * self.q_heads * self.query_length

    @property
    def q_elements(self) -> int:
        return self.batch * self.q_heads * self.query_length * self.head_dim

    @property
    def kv_elements(self) -> int:
        return self.batch * self.kv_heads * self.key_length * self.head_dim

    @property
    def output_elements(self) -> int:
        return self.q_elements

    @property
    def score_elements(self) -> int:
        return self.batch * self.q_heads * self.query_length * self.key_length

    @property
    def max_query_position(self) -> int:
        return self.query_start + self.query_length - 1

    @property
    def gqa_group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def logical_flops(self) -> int:
        # QK^T plus PV; softmax transcendental/reduction work is intentionally excluded.
        return 4 * self.score_elements * self.head_dim

    @property
    def materialized_score_bytes_fp32(self) -> int:
        return self.score_elements * 4


def validate_shape(shape: AttentionShape) -> None:
    if min(
        shape.batch,
        shape.q_heads,
        shape.kv_heads,
        shape.query_length,
        shape.key_length,
        shape.head_dim,
    ) <= 0:
        raise ValueError(f"shape dimensions must be positive: {shape}")
    if shape.q_heads % shape.kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads for GQA: {shape}")
    if shape.head_dim not in (64, 128):
        raise ValueError(f"v0 FlashAttention lab supports head_dim 64 or 128: {shape}")
    if shape.query_start < 0:
        raise ValueError(f"query_start must be non-negative: {shape}")
    if shape.causal and shape.max_query_position >= shape.key_length:
        raise ValueError(
            f"causal query positions must fit key_length: {shape.max_query_position} >= {shape.key_length}"
        )


SHAPES = (
    AttentionShape("validation_decode_gqa", "validation", "decode", 1, 8, 2, 1, 128, 64, True, 127),
    AttentionShape("validation_prefill", "validation", "prefill", 1, 8, 8, 32, 32, 64, True, 0),
    AttentionShape("validation_chunked_gqa", "validation", "chunked_prefill", 1, 8, 2, 16, 256, 64, True, 240),
    AttentionShape("decode_2k_gqa", "decode", "decode", 1, 32, 8, 1, 2048, 128, True, 2047),
    AttentionShape("decode_32k_gqa", "decode", "long_decode", 1, 32, 8, 1, 32768, 128, True, 32767),
    AttentionShape("prefill_128", "prefill", "prefill", 1, 32, 8, 128, 128, 128, True, 0),
    AttentionShape("prefill_512", "prefill", "prefill", 1, 32, 8, 512, 512, 128, True, 0),
    AttentionShape("chunked_4k", "prefill", "chunked_prefill", 1, 32, 8, 128, 4096, 128, True, 3968),
    AttentionShape("noncausal_1k", "noncausal", "noncausal", 1, 16, 4, 64, 1024, 64, False, 0),
)

SHAPE_FAMILIES = ("validation", "decode", "prefill", "noncausal", "all")


def validate_shapes() -> None:
    names: set[str] = set()
    for shape in SHAPES:
        validate_shape(shape)
        if shape.name in names:
            raise ValueError(f"duplicate shape name: {shape.name}")
        names.add(shape.name)
    if not any(shape.key_length >= 32768 for shape in SHAPES):
        raise ValueError("shape matrix must include a 32K-key long-context case")
    if not any(shape.q_heads > shape.kv_heads for shape in SHAPES):
        raise ValueError("shape matrix must include grouped-query attention")
    if not any(shape.role == "chunked_prefill" for shape in SHAPES):
        raise ValueError("shape matrix must include chunked prefill")
    if not any(not shape.causal for shape in SHAPES):
        raise ValueError("shape matrix must include a non-causal control")


def shapes_for_family(family: str) -> tuple[AttentionShape, ...]:
    if family not in SHAPE_FAMILIES:
        raise KeyError(f"unknown family: {family}")
    if family == "all":
        return SHAPES
    return tuple(shape for shape in SHAPES if shape.family == family)


def requirements_for_dtype(dtype: str) -> tuple[str, ...]:
    try:
        return DTYPE_REQUIREMENTS[dtype]
    except KeyError as exc:
        raise KeyError(f"unsupported dtype: {dtype}") from exc


validate_shapes()
