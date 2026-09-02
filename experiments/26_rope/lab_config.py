from __future__ import annotations

from dataclasses import asdict, dataclass


LAYOUTS = ("interleaved", "half_split")
DTYPE_REQUIREMENTS = {
    "fp16": ("cuda",),
    "bf16": ("cuda", "bf16"),
}


@dataclass(frozen=True)
class RoPEShape:
    name: str
    family: str
    role: str
    batch: int
    sequence: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    position_start: int

    def __post_init__(self) -> None:
        values = (
            self.batch,
            self.sequence,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("RoPE dimensions must be positive")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even and <= head_dim")
        if self.position_start < 0:
            raise ValueError("position_start must be non-negative")

    @property
    def max_position(self) -> int:
        return self.position_start + self.sequence - 1

    @property
    def rotated_pairs(self) -> int:
        return self.rotary_dim // 2

    @property
    def q_elements(self) -> int:
        return self.batch * self.sequence * self.q_heads * self.head_dim

    @property
    def k_elements(self) -> int:
        return self.batch * self.sequence * self.kv_heads * self.head_dim

    @property
    def total_elements(self) -> int:
        return self.q_elements + self.k_elements

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


SHAPES = (
    RoPEShape("validation_decode", "validation", "decode", 1, 1, 32, 8, 128, 128, 127),
    RoPEShape("validation_decode_long", "validation", "decode-long-context", 1, 1, 32, 8, 128, 128, 32767),
    RoPEShape("validation_prefill_partial", "validation", "prefill-partial-rotary", 1, 64, 32, 8, 128, 64, 4096),
    RoPEShape("llama_decode_1", "decode", "decode", 1, 1, 32, 8, 128, 128, 0),
    RoPEShape("llama_decode_1_32k", "decode", "decode-long-context", 1, 1, 32, 8, 128, 128, 32767),
    RoPEShape("llama_decode_1_128k", "decode", "decode-long-context", 1, 1, 32, 8, 128, 128, 131071),
    RoPEShape("llama_prefill_128", "prefill", "prefill", 1, 128, 32, 8, 128, 128, 0),
    RoPEShape("llama_prefill_512", "prefill", "prefill", 1, 512, 32, 8, 128, 128, 0),
    RoPEShape("llama_prefill_128_offset_32k", "prefill", "prefill-long-context", 1, 128, 32, 8, 128, 128, 32768),
    RoPEShape("partial_rotary_prefill", "prefill", "prefill-partial-rotary", 1, 256, 32, 8, 128, 64, 8192),
)

SHAPE_FAMILIES = ("validation", "decode", "prefill", "all")


def shapes_for_family(family: str) -> tuple[RoPEShape, ...]:
    if family not in SHAPE_FAMILIES:
        raise ValueError(f"unknown shape family: {family}")
    if family == "all":
        return SHAPES
    return tuple(shape for shape in SHAPES if shape.family == family)
