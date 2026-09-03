from __future__ import annotations

from dataclasses import dataclass

FORMATS = ("int8_sym", "int8_asym", "int4_sym")
GRANULARITIES = ("per_tensor", "per_channel", "group")
GROUP_SIZES = (32, 64, 128)
DTYPE_REQUIREMENTS = {"fp16": ("cuda",), "bf16": ("cuda", "bf16")}

@dataclass(frozen=True)
class GemmShape:
    name: str
    family: str
    role: str
    m: int
    n: int
    k: int

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def activation_bytes(self, element_size: int = 2) -> int:
        return self.m * self.k * element_size

    def output_bytes(self, element_size: int = 2) -> int:
        return self.m * self.n * element_size

SHAPES = (
    GemmShape("validation_odd_k", "validation", "correctness", 3, 5, 65),
    GemmShape("validation_grouped", "validation", "correctness", 8, 16, 128),
    GemmShape("decode_m1", "decode", "single_token", 1, 4096, 4096),
    GemmShape("decode_m16", "decode", "batched_decode", 16, 4096, 4096),
    GemmShape("prefill_m128", "prefill", "attention_projection", 128, 4096, 4096),
    GemmShape("prefill_m512", "prefill", "attention_projection", 512, 4096, 4096),
    GemmShape("mlp_decode", "decode", "mlp_projection", 8, 11008, 4096),
)
FAMILIES = ("validation", "decode", "prefill", "all")

def shapes_for_family(family: str) -> tuple[GemmShape, ...]:
    if family not in FAMILIES:
        raise KeyError(f"unknown family: {family}")
    return SHAPES if family == "all" else tuple(s for s in SHAPES if s.family == family)

def validate_format(format_name: str) -> None:
    if format_name not in FORMATS:
        raise KeyError(f"unsupported format: {format_name}")

def validate_granularity(granularity: str, group_size: int) -> None:
    if granularity not in GRANULARITIES:
        raise KeyError(f"unsupported granularity: {granularity}")
    if granularity == "group":
        if group_size not in GROUP_SIZES:
            raise ValueError(f"group_size must be one of {GROUP_SIZES}")
    elif group_size not in (0, 1):
        raise ValueError("group_size must be 0 or 1 outside group mode")

def qparam_count(n: int, k: int, granularity: str, group_size: int) -> int:
    validate_granularity(granularity, group_size)
    if granularity == "per_tensor":
        return 1
    if granularity == "per_channel":
        return n
    return n * ((k + group_size - 1) // group_size)

def quantized_weight_bytes(format_name: str, n: int, k: int) -> int:
    validate_format(format_name)
    return n * k if format_name.startswith("int8") else n * ((k + 1) // 2)

def fp16_weight_bytes(n: int, k: int) -> int:
    return n * k * 2

def compression_ratio(format_name: str, n: int, k: int) -> float:
    return fp16_weight_bytes(n, k) / quantized_weight_bytes(format_name, n, k)

def arithmetic_intensity(shape: GemmShape, format_name: str) -> float:
    bytes_moved = shape.activation_bytes() + quantized_weight_bytes(format_name, shape.n, shape.k) + shape.output_bytes()
    return shape.flops / bytes_moved

def expected_regime(shape: GemmShape) -> str:
    if shape.m <= 16:
        return "bandwidth_sensitive"
    if shape.m >= 128:
        return "compute_reuse_sensitive"
    return "mixed"

def validate_shapes() -> None:
    seen=set()
    for s in SHAPES:
        if min(s.m,s.n,s.k) <= 0:
            raise ValueError(f"invalid shape: {s}")
        if s.name in seen:
            raise ValueError(f"duplicate shape: {s.name}")
        seen.add(s.name)
    if not any(s.k % 2 for s in SHAPES):
        raise ValueError("must retain an odd-K INT4 tail shape")
    if not any(s.m == 1 for s in SHAPES):
        raise ValueError("must retain single-token decode")
    if not any(s.m >= 128 for s in SHAPES):
        raise ValueError("must retain prefill")

validate_shapes()
