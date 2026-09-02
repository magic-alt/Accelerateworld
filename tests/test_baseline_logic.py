from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_lib import load_registry, missing_features, parse_metrics  # noqa: E402


class BaselineLogicTests(unittest.TestCase):
    def test_registry_covers_rtx_20_through_50(self) -> None:
        registry = load_registry()
        architectures = registry["architectures"]
        self.assertEqual(set(architectures), {"sm_75", "sm_86", "sm_89", "sm_120"})
        self.assertEqual(architectures["sm_75"]["rtx_series"], "RTX 20")
        self.assertEqual(architectures["sm_86"]["rtx_series"], "RTX 30")
        self.assertEqual(architectures["sm_89"]["rtx_series"], "RTX 40")
        self.assertEqual(architectures["sm_120"]["rtx_series"], "RTX 50")

    def test_feature_gating_is_conservative(self) -> None:
        registry = load_registry()["architectures"]
        turing = {"features": registry["sm_75"]["features"]}
        ampere = {"features": registry["sm_86"]["features"]}
        ada = {"features": registry["sm_89"]["features"]}
        blackwell = {"features": registry["sm_120"]["features"]}

        self.assertEqual(missing_features(turing, ["cuda", "memory_pool", "tensor_core", "wmma_fp16"]), [])
        self.assertEqual(missing_features(turing, ["tf32", "bf16"]), ["tf32", "bf16"])
        self.assertEqual(missing_features(turing, ["fp8", "fp4", "blackwell"]), ["fp8", "fp4", "blackwell"])
        self.assertEqual(missing_features(ampere, ["tf32", "bf16", "tensor_core"]), [])
        self.assertEqual(missing_features(ampere, ["fp8", "fp4", "blackwell"]), ["fp8", "fp4", "blackwell"])
        self.assertEqual(missing_features(ada, ["bf16", "tf32", "fp8", "memory_pool"]), [])
        self.assertEqual(missing_features(ada, ["fp4", "blackwell"]), ["fp4", "blackwell"])
        self.assertEqual(missing_features(blackwell, ["tf32", "bf16", "fp8", "fp4", "blackwell", "memory_pool"]), [])

    def test_metric_parser_handles_single_and_dual_metrics(self) -> None:
        output = """
  Effective bandwidth: 321.5000 GB/s
  Tiled: 0.2500 ms, 8796.0000 GFLOP/s
  Coalesced: 0.1000 ms, 400.0000 useful GB/s
  Speedup: 2.5000x
"""
        metrics = parse_metrics(output)
        self.assertEqual(metrics["effective_bandwidth"]["value"], 321.5)
        self.assertEqual(metrics["effective_bandwidth"]["unit"], "GB/s")
        self.assertEqual(metrics["tiled_secondary"]["value"], 8796.0)
        self.assertEqual(metrics["tiled_secondary"]["unit"], "GFLOP/s")
        self.assertEqual(metrics["coalesced_secondary"]["value"], 400.0)
        self.assertEqual(metrics["coalesced_secondary"]["unit"], "useful GB/s")
        self.assertEqual(metrics["speedup"]["unit"], "x")

    def test_manifest_requires_only_registered_features(self) -> None:
        registry = load_registry()["architectures"]
        known_features = set()
        for profile in registry.values():
            known_features.update(profile["features"].keys())

        manifest = json.loads((ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        for benchmark in manifest["benchmarks"]:
            self.assertTrue(set(benchmark.get("requires", [])).issubset(known_features))
            self.assertIn("primary_metric", benchmark)

        by_id = {benchmark["id"]: benchmark for benchmark in manifest["benchmarks"]}
        self.assertEqual(by_id["mixed_precision_tf32"]["requires"], ["cuda", "tf32"])
        self.assertEqual(by_id["mixed_precision_bf16"]["requires"], ["cuda", "bf16"])
        self.assertIn("wmma_fp16", by_id["mixed_precision_fp16"]["requires"])
        self.assertEqual(by_id["fp8_e4m3"]["requires"], ["cuda", "fp8"])
        self.assertEqual(by_id["fp8_e5m2"]["requires"], ["cuda", "fp8"])
        self.assertEqual(by_id["fp8_e4m3"]["primary_metric"]["key"], "e4m3_throughput")
        self.assertEqual(by_id["fp8_e5m2"]["primary_metric"]["key"], "e5m2_throughput")
        self.assertEqual(by_id["fp4_e2m1"]["requires"], ["cuda", "fp4", "blackwell"])
        self.assertEqual(by_id["fp4_e2m1"]["primary_metric"]["key"], "fp4_e2m1_throughput")

        self.assertEqual(by_id["cutlass_simt"]["requires"], ["cuda"])
        self.assertEqual(by_id["cutlass_sm75_fp16"]["requires"], ["cuda", "tensor_core", "wmma_fp16"])
        self.assertEqual(by_id["cutlass_sm80_bf16"]["requires"], ["cuda", "bf16"])
        self.assertEqual(by_id["cutlass_sm120_fp8"]["requires"], ["cuda", "fp8", "blackwell"])
        self.assertEqual(by_id["cutlass_sm75_fp16"]["primary_metric"]["key"], "best_cutlass_throughput")
        self.assertEqual(by_id["cutlass_sm80_bf16"]["primary_metric"]["key"], "best_cutlass_bf16_throughput")
        self.assertEqual(by_id["cutlass_sm120_fp8"]["primary_metric"]["key"], "cutlass_sm120_throughput")

        self.assertEqual(by_id["grouped_gemm"]["requires"], ["cuda", "tensor_core", "wmma_fp16"])
        self.assertEqual(by_id["grouped_gemm"]["primary_metric"]["key"], "grouped_device_throughput")
        self.assertIn("--mode", by_id["grouped_gemm"]["command"])
        self.assertIn("all", by_id["grouped_gemm"]["command"])


if __name__ == "__main__":
    unittest.main()
