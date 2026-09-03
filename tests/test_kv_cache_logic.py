from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "29_kv_cache"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


config = load_module("kv_cache_lab_config", EXP / "lab_config.py")


class KVCacheLogicTest(unittest.TestCase):
    def test_layouts_and_shape_families(self) -> None:
        self.assertEqual(config.LAYOUTS, ("token_major", "head_major"))
        self.assertEqual(config.layout_id("token_major"), 0)
        self.assertEqual(config.layout_id("head_major"), 1)
        self.assertTrue(any(shape.kv_heads == 1 for shape in config.SHAPES))
        self.assertTrue(any(shape.q_heads > shape.kv_heads > 1 for shape in config.SHAPES))
        self.assertTrue(any(shape.capacity >= 32768 for shape in config.SHAPES))

    def test_capacity_and_head_contract(self) -> None:
        for shape in config.SHAPES:
            self.assertLess(shape.prefill_tokens, shape.capacity)
            self.assertEqual(shape.q_heads % shape.kv_heads, 0)
            self.assertIn(shape.head_dim, (64, 128))
            self.assertEqual(shape.head_dim % 2, 0)

    def test_dtype_capability_contract(self) -> None:
        self.assertEqual(config.requirements_for_dtype("fp16"), ("cuda",))
        self.assertEqual(config.requirements_for_dtype("bf16"), ("cuda", "bf16"))

    def test_cuda_source_has_vectorized_pairs_and_both_layouts(self) -> None:
        source = (EXP / "kv_cache_kernel.cu").read_text(encoding="utf-8")
        self.assertIn("uint32_t", source)
        self.assertIn("kTokenMajor", source)
        self.assertIn("kHeadMajor", source)
        self.assertIn("CachePairOffset", source)
        self.assertIn("check_bounds", source)
        self.assertIn("positions.min().item<int64_t>()", source)
        self.assertIn("positions.max().item<int64_t>()", source)

    def test_triton_source_has_update_and_attention_compatible_read(self) -> None:
        source = (EXP / "kv_cache_triton.py").read_text(encoding="utf-8")
        self.assertIn("_kv_update_kernel", source)
        self.assertIn("_kv_read_kernel", source)
        self.assertIn("[batch, kv_heads, tokens, head_dim]", source)
        self.assertIn("capacity", source)


if __name__ == "__main__":
    unittest.main()
