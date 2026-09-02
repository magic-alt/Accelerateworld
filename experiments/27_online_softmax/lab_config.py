from __future__ import annotations

from dataclasses import dataclass


DTYPE_REQUIREMENTS = {
    "fp16": ("cuda",),
    "bf16": ("cuda", "bf16"),
}


@dataclass(frozen=True)
class SoftmaxShape:
    name: str
    family: str
    role: str
    batch: int
    heads: int
    query_length: int
    key_length: int
    causal: bool
    query_start: int = 0

    @property
    def rows(self) -> int:
        return self.batch * self.heads * self.query_length

    @property
    def elements(self) -> int:
        return self.rows * self.key_length

    @property
    def logical_bytes_fp16(self) -> int:
        return self.elements * 2 * 2

    @property
    def max_query_position(self) -> int:
        return self.query_start + self.query_length - 1


SHAPES = (
    SoftmaxShape("validation_decode_128", "validation", "decode", 1, 8, 1, 128, True, 127),
    SoftmaxShape("validation_prefill_32", "validation", "prefill", 1, 8, 32, 32, True, 0),
    SoftmaxShape("validation_chunked_1024", "validation", "chunked_prefill", 1, 8, 32, 1024, True, 992),
    SoftmaxShape("decode_128", "decode", "decode", 1, 32, 1, 128, True, 127),
    SoftmaxShape("decode_2048", "decode", "decode", 1, 32, 1, 2048, True, 2047),
    SoftmaxShape("decode_32768", "decode", "long_decode", 1, 32, 1, 32768, True, 32767),
    SoftmaxShape("prefill_128", "prefill", "prefill", 1, 32, 128, 128, True, 0),
    SoftmaxShape("prefill_512", "prefill", "prefill", 1, 32, 512, 512, True, 0),
    SoftmaxShape("chunked_prefill_4096", "prefill", "chunked_prefill", 1, 32, 128, 4096, True, 3968),
    SoftmaxShape("noncausal_1024", "noncausal", "noncausal", 1, 16, 64, 1024, False, 0),
)

SHAPE_FAMILIES = ("validation", "decode", "prefill", "noncausal", "all")


def validate_shape(shape: SoftmaxShape) -> None:
    if min(shape.batch, shape.heads, shape.query_length, shape.key_length) <= 0:
        raise ValueError(f"shape dimensions must be positive: {shape}")
    if shape.query_start < 0:
        raise ValueError(f"query_start must be non-negative: {shape}")
    if shape.causal and shape.max_query_position >= shape.key_length:
        raise ValueError(
            f"causal query positions must fit the key dimension: {shape.max_query_position} >= {shape.key_length}"
        )


def validate_shapes() -> None:
    names = set()
    for shape in SHAPES:
        validate_shape(shape)
        if shape.name in names:
            raise ValueError(f"duplicate shape name: {shape.name}")
        names.add(shape.name)
    if not any(shape.key_length >= 32768 for shape in SHAPES):
        raise ValueError("shape matrix must include a long-context attention row")
    if not any(shape.role == "chunked_prefill" for shape in SHAPES):
        raise ValueError("shape matrix must include chunked causal prefill")
    if not any(not shape.causal for shape in SHAPES):
        raise ValueError("shape matrix must include a non-causal control")


def shapes_for_family(family: str) -> tuple[SoftmaxShape, ...]:
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
