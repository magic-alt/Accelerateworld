from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "26_rope"


def _load_config():
    spec = importlib.util.spec_from_file_location("rope_lab_config", EXP / "lab_config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RoPELogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _load_config()

    def test_layouts_and_dtype_requirements(self):
        self.assertEqual(self.config.LAYOUTS, ("interleaved", "half_split"))
        self.assertEqual(self.config.DTYPE_REQUIREMENTS["fp16"], ("cuda",))
        self.assertEqual(self.config.DTYPE_REQUIREMENTS["bf16"], ("cuda", "bf16"))

    def test_shapes_cover_decode_prefill_long_context_and_partial_rotary(self):
        shapes = self.config.SHAPES
        self.assertTrue(any(shape.role == "decode" for shape in shapes))
        self.assertTrue(any("prefill" in shape.role for shape in shapes))
        self.assertTrue(any(shape.max_position >= 131071 for shape in shapes))
        self.assertTrue(any(shape.rotary_dim < shape.head_dim for shape in shapes))
        self.assertTrue(any(shape.q_heads != shape.kv_heads for shape in shapes))
        for shape in shapes:
            self.assertEqual(shape.rotary_dim % 2, 0)
            self.assertLessEqual(shape.rotary_dim, shape.head_dim)
            self.assertEqual(shape.rotated_pairs, shape.rotary_dim // 2)

    def test_validation_family_is_bounded(self):
        shapes = self.config.shapes_for_family("validation")
        self.assertGreaterEqual(len(shapes), 3)
        self.assertTrue(all(shape.sequence <= 64 for shape in shapes))
        self.assertTrue(any(shape.max_position >= 32767 for shape in shapes))

    def test_cuda_source_has_vectorized_and_bf16_gated_paths(self):
        text = (EXP / "rope_kernel.cu").read_text(encoding="utf-8")
        self.assertIn("__half2", text)
        self.assertIn("__nv_bfloat162", text)
        self.assertIn("__CUDA_ARCH__ >= 800", text)
        self.assertIn("properties.major >= 8", text)
        self.assertIn("kInterleaved", text)
        self.assertIn("kHalfSplit", text)
        self.assertIn("tail_dim", text)

    def test_triton_source_uses_pair_and_tail_work_items(self):
        text = (EXP / "rope_triton.py").read_text(encoding="utf-8")
        self.assertIn("pair_mask", text)
        self.assertIn("tail_mask", text)
        self.assertIn("layout == 0", text)
        self.assertIn("tl.float32", text)


if __name__ == "__main__":
    unittest.main()
