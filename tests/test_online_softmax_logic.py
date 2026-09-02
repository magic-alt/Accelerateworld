from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "27_online_softmax"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OnlineSoftmaxLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_module("online_softmax_lab_config", "lab_config.py")
        cls.math = load_module("online_softmax_math", "online_math.py")

    def test_online_recurrence_matches_stable_softmax(self):
        values = [1000.0, 998.0, 991.0, -1000.0, 999.5]
        online = self.math.online_softmax(values)
        stable = self.math.stable_softmax(values)
        self.assertAlmostEqual(sum(online), 1.0, places=12)
        for got, expected in zip(online, stable):
            self.assertAlmostEqual(got, expected, places=12)

    def test_online_state_merge_is_associative_up_to_roundoff(self):
        left = self.math.online_normalizer([3.0, 1.0, -5.0])
        middle = self.math.online_normalizer([8.0, 2.0])
        right = self.math.online_normalizer([-9.0, 7.0, 6.0])
        a = self.math.combine_states(*self.math.combine_states(*left, *middle), *right)
        b = self.math.combine_states(*left, *self.math.combine_states(*middle, *right))
        self.assertAlmostEqual(a[0], b[0], places=12)
        self.assertAlmostEqual(a[1], b[1], places=12)

    def test_shape_matrix_covers_attention_regimes(self):
        shapes = self.config.SHAPES
        self.assertTrue(any(shape.role == "decode" for shape in shapes))
        self.assertTrue(any(shape.role == "prefill" for shape in shapes))
        self.assertTrue(any(shape.role == "chunked_prefill" for shape in shapes))
        self.assertTrue(any(shape.key_length >= 32768 for shape in shapes))
        self.assertTrue(any(not shape.causal for shape in shapes))
        for shape in shapes:
            self.assertGreater(shape.rows, 0)
            self.assertGreater(shape.elements, 0)
            if shape.causal:
                self.assertLess(shape.max_query_position, shape.key_length)

    def test_dtype_requirements_match_capability_model(self):
        self.assertEqual(self.config.DTYPE_REQUIREMENTS["fp16"], ("cuda",))
        self.assertEqual(self.config.DTYPE_REQUIREMENTS["bf16"], ("cuda", "bf16"))

    def test_cuda_source_has_two_pass_online_warp_and_block_reductions(self):
        source = (EXP / "softmax_kernel.cu").read_text(encoding="utf-8")
        self.assertIn("TwoPassSoftmaxKernel", source)
        self.assertIn("OnlineSoftmaxKernel", source)
        self.assertIn("CombineState", source)
        self.assertIn("WarpReduceOnline", source)
        self.assertIn("BlockReduceOnline", source)
        self.assertIn("__shfl_down_sync", source)
        self.assertIn("BlockReduceMax", source)
        self.assertIn("BlockReduceSum", source)
        self.assertIn("__CUDA_ARCH__ >= 800", source)

    def test_triton_source_keeps_tiled_online_state(self):
        source = (EXP / "softmax_triton.py").read_text(encoding="utf-8")
        self.assertIn("running_max", source)
        self.assertIn("running_sum", source)
        self.assertIn("tile_max", source)
        self.assertIn("tl.range(0, n_cols, BLOCK_SIZE)", source)
        self.assertIn("tl.exp(running_max - new_max)", source)
        self.assertIn("CAUSAL", source)


if __name__ == "__main__":
    unittest.main()
