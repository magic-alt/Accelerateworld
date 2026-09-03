from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "28_flash_attention"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load_module("aw_flash_attention_lab_config", EXP / "lab_config.py")


class FlashAttentionLogicTests(unittest.TestCase):
    def test_shape_matrix_covers_production_roles(self):
        roles = {shape.role for shape in config.SHAPES}
        self.assertIn("decode", roles)
        self.assertIn("long_decode", roles)
        self.assertIn("prefill", roles)
        self.assertIn("chunked_prefill", roles)
        self.assertIn("noncausal", roles)
        self.assertTrue(any(shape.q_heads > shape.kv_heads for shape in config.SHAPES))
        self.assertTrue(any(shape.key_length >= 32768 for shape in config.SHAPES))

    def test_attention_shape_contract(self):
        for shape in config.SHAPES:
            config.validate_shape(shape)
            self.assertEqual(shape.q_heads % shape.kv_heads, 0)
            self.assertIn(shape.head_dim, (64, 128))
            if shape.causal:
                self.assertLess(shape.max_query_position, shape.key_length)

    def test_dtype_requirements(self):
        self.assertEqual(config.requirements_for_dtype("fp16"), ("cuda",))
        self.assertEqual(config.requirements_for_dtype("bf16"), ("cuda", "bf16"))

    def test_cuda_source_is_score_matrix_free_online_attention(self):
        source = (EXP / "attention_kernel.cu").read_text(encoding="utf-8")
        for token in (
            "BlockReduceSum",
            "running_m",
            "running_l",
            "output_acc",
            "__shfl_down_sync",
            "__CUDA_ARCH__",
            "CUDART_INF_F",
        ):
            self.assertIn(token, source)
        self.assertNotIn("at::matmul", source)

    def test_triton_source_scans_key_tiles(self):
        source = (EXP / "attention_triton.py").read_text(encoding="utf-8")
        for token in ("tl.range", "running_m", "running_l", "output_acc", "BLOCK_K"):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
