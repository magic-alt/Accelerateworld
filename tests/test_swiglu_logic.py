from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "experiments" / "25_swiglu_mixed_precision"
sys.path.insert(0, str(LAB))

from lab_config import (  # noqa: E402
    DTYPE_REQUIREMENTS,
    SHAPES,
    SHAPE_FAMILIES,
    requirements_for_dtype,
    shape_by_name,
    shapes_for_family,
)


class SwiGLULogicTests(unittest.TestCase):
    def test_dtype_requirements_use_existing_capability_registry(self) -> None:
        registry = json.loads(
            (ROOT / "hardware" / "rtx_capabilities.json").read_text(encoding="utf-8")
        )
        known = set()
        for architecture in registry["architectures"].values():
            known.update(architecture["features"])

        for requirements in DTYPE_REQUIREMENTS.values():
            self.assertTrue(set(requirements).issubset(known))
        self.assertEqual(requirements_for_dtype("fp16"), ("cuda",))
        self.assertEqual(requirements_for_dtype("bf16"), ("cuda", "bf16"))

    def test_bf16_registry_gate_starts_at_ampere(self) -> None:
        registry = json.loads(
            (ROOT / "hardware" / "rtx_capabilities.json").read_text(encoding="utf-8")
        )["architectures"]
        self.assertFalse(registry["sm_75"]["features"]["bf16"])
        self.assertTrue(registry["sm_86"]["features"]["bf16"])
        self.assertTrue(registry["sm_89"]["features"]["bf16"])
        self.assertTrue(registry["sm_120"]["features"]["bf16"])

    def test_shape_contract_is_packed_gate_up_projection(self) -> None:
        for shape in SHAPES:
            self.assertGreater(shape.tokens, 0)
            self.assertGreater(shape.hidden, 0)
            self.assertGreater(shape.intermediate, 0)
            self.assertEqual(shape.packed_columns, 2 * shape.intermediate)
            self.assertEqual(
                shape.projection_flops,
                4 * shape.tokens * shape.hidden * shape.intermediate,
            )

    def test_families_cover_validation_decode_and_prefill(self) -> None:
        self.assertEqual(
            tuple(shape.name for shape in shapes_for_family("validation")),
            ("validation_decode", "validation_prefill"),
        )
        llm_roles = {shape.role for shape in shapes_for_family("llm")}
        self.assertEqual(llm_roles, {"decode", "prefill"})
        self.assertIn("decode", SHAPE_FAMILIES)
        self.assertIn("prefill", SHAPE_FAMILIES)
        self.assertEqual(shape_by_name("llama2_7b_decode_1").tokens, 1)
        self.assertEqual(shape_by_name("llama2_7b_prefill_512").tokens, 512)

    def test_cuda_source_keeps_vectorized_and_scalar_paths(self) -> None:
        source = (LAB / "swiglu_kernel.cu").read_text(encoding="utf-8")
        self.assertIn("__half2", source)
        self.assertIn("__nv_bfloat162", source)
        self.assertIn("intermediate % 2 == 0", source)
        self.assertIn("SwiGLUHalfScalarKernel", source)
        self.assertIn("SwiGLUBFloat16ScalarKernel", source)
        self.assertIn("__CUDA_ARCH__ >= 800", source)

    def test_unknown_shape_dtype_and_family_are_rejected(self) -> None:
        with self.assertRaises(KeyError):
            shape_by_name("missing")
        with self.assertRaises(KeyError):
            shapes_for_family("missing")
        with self.assertRaises(KeyError):
            requirements_for_dtype("fp8")


if __name__ == "__main__":
    unittest.main()
