from __future__ import annotations

from dataclasses import asdict, dataclass


DTYPE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "fp16": ("cuda",),
    "bf16": ("cuda", "bf16"),
}


@dataclass(frozen=True)
class SwiGLUShape:
    name: str
    role: str
    tokens: int
    hidden: int
    intermediate: int

    @property
    def packed_columns(self) -> int:
        return 2 * self.intermediate

    @property
    def projection_flops(self) -> int:
        # X[M,K] @ W[K,2N] => 2*M*K*(2N)
        return 4 * self.tokens * self.hidden * self.intermediate

    def to_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["packed_columns"] = self.packed_columns
        payload["projection_flops"] = self.projection_flops
        return payload


SHAPES: tuple[SwiGLUShape, ...] = (
    # Reduced shapes used by physical-GPU CI.  They keep the same 2.6875x
    # hidden-to-intermediate ratio as the classic 4096 -> 11008 LLaMA MLP.
    SwiGLUShape("validation_decode", "validation/decode", 4, 1024, 2816),
    SwiGLUShape("validation_prefill", "validation/prefill", 64, 1024, 2816),
    # Representative LLM inference shapes.  The lab intentionally keeps both
    # single/few-token decode and token-parallel prefill regimes.
    SwiGLUShape("llama2_7b_decode_1", "decode", 1, 4096, 11008),
    SwiGLUShape("llama2_7b_decode_16", "decode", 16, 4096, 11008),
    SwiGLUShape("llama2_7b_prefill_128", "prefill", 128, 4096, 11008),
    SwiGLUShape("llama2_7b_prefill_512", "prefill", 512, 4096, 11008),
)

SHAPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "validation": ("validation_decode", "validation_prefill"),
    "decode": ("llama2_7b_decode_1", "llama2_7b_decode_16"),
    "prefill": ("llama2_7b_prefill_128", "llama2_7b_prefill_512"),
    "llm": (
        "llama2_7b_decode_1",
        "llama2_7b_decode_16",
        "llama2_7b_prefill_128",
        "llama2_7b_prefill_512",
    ),
}


def shape_by_name(name: str) -> SwiGLUShape:
    for shape in SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"unknown SwiGLU shape: {name}")


def shapes_for_family(family: str) -> tuple[SwiGLUShape, ...]:
    try:
        names = SHAPE_FAMILIES[family]
    except KeyError as error:
        raise KeyError(f"unknown SwiGLU shape family: {family}") from error
    return tuple(shape_by_name(name) for name in names)


def requirements_for_dtype(dtype_name: str) -> tuple[str, ...]:
    try:
        return DTYPE_REQUIREMENTS[dtype_name]
    except KeyError as error:
        raise KeyError(f"unsupported SwiGLU dtype: {dtype_name}") from error
