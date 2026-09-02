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
        ada = {"features": registry["sm_89"]["features"]}
        blackwell = {"features": registry["sm_120"]["features"]}

        self.assertEqual(missing_features(turing, ["cuda", "tensor_core"]), [])
        self.assertEqual(missing_features(turing, ["fp8"]), ["fp8"])
        self.assertEqual(missing_features(ada, ["bf16", "fp8"]), [])
        self.assertEqual(missing_features(ada, ["fp4"]), ["fp4"])
        self.assertEqual(missing_features(blackwell, ["fp4", "blackwell"]), [])

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


if __name__ == "__main__":
    unittest.main()
