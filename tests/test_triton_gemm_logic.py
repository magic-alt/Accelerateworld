from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "23_triton_gemm" / "lab_config.py"
SPEC = importlib.util.spec_from_file_location("accelerateworld_triton_gemm_lab_config", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load {MODULE_PATH}")
LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAB
SPEC.loader.exec_module(LAB)


class TritonGemmLogicTests(unittest.TestCase):
    def test_autotune_search_space_is_nontrivial_and_unique(self) -> None:
        LAB.validate_config_specs()
        specs = LAB.AUTOTUNE_CONFIG_SPECS
        self.assertGreaterEqual(len(specs), 8)
        signatures = {
            tuple(sorted(spec.items()))
            for spec in specs
        }
        self.assertEqual(len(signatures), len(specs))
        self.assertTrue(any(spec["BLOCK_SIZE_M"] == 16 for spec in specs))
        self.assertTrue(any(spec["BLOCK_SIZE_M"] >= 128 for spec in specs))
        self.assertEqual({spec["num_warps"] for spec in specs}, {2, 4, 8})
        self.assertGreaterEqual(len({spec["num_stages"] for spec in specs}), 3)

    def test_llm_shape_families_cover_prefill_and_small_m_decode(self) -> None:
        LAB.validate_shape_families()
        prefill = LAB.shapes_for_family("prefill")
        decode = LAB.shapes_for_family("decode")
        self.assertTrue(all(shape.m >= 128 for shape in prefill))
        self.assertTrue(all(shape.m <= 16 for shape in decode))
        self.assertTrue(any("qkv" in shape.role for shape in prefill))
        self.assertTrue(any("mlp-up" in shape.role for shape in decode))
        self.assertEqual(
            len(LAB.shapes_for_family("all")),
            len(LAB.SHAPE_FAMILIES["balanced"]) + len(prefill) + len(decode),
        )

    def test_shape_flop_count(self) -> None:
        shape = LAB.GemmShape("unit", "test", "unit", 2, 3, 4)
        self.assertEqual(shape.flops, 48)

    def test_shape_dependent_winner_ignores_unavailable_provider(self) -> None:
        self.assertEqual(
            LAB.pick_winner({"triton": 0.8, "pytorch": 1.0, "direct_cublas": None}),
            "triton",
        )
        self.assertEqual(
            LAB.pick_winner({"triton": 0.8, "pytorch": 0.7, "direct_cublas": 0.6}),
            "direct_cublas",
        )
        self.assertEqual(
            LAB.pick_winner({"triton": None, "pytorch": None, "direct_cublas": None}),
            "unavailable",
        )

    def test_direct_cublas_output_parser(self) -> None:
        parsed = LAB.parse_direct_cublas_output(
            """Direct cuBLAS FP16 GEMM
Latency: 0.125000 ms
Throughput: 17179.869184 GFLOP/s
Max normalized error: 0.001250
Validation: PASS
"""
        )
        self.assertEqual(parsed["latency_ms"], 0.125)
        self.assertAlmostEqual(parsed["throughput_gflops"], 17179.869184)
        self.assertEqual(parsed["max_normalized_error"], 0.00125)


if __name__ == "__main__":
    unittest.main()
