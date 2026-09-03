from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "31_quantize_dequantize"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


config = load_module("quant_lab_config", EXP / "lab_config.py")
codec = load_module("quant_codec", EXP / "codec.py")


class QuantizeDequantizeLogicTest(unittest.TestCase):
    def test_formats(self) -> None:
        self.assertEqual(config.FORMATS, ("int8_sym", "int8_asym", "int4_sym"))
        self.assertEqual(config.format_info("int8_sym")["qmin"], -127)
        self.assertEqual(config.format_info("int8_asym")["qmin"], -128)
        self.assertTrue(config.format_info("int4_sym")["packed"])

    def test_granularity_parameter_counts(self) -> None:
        self.assertEqual(config.parameter_count(4, 65, "per_tensor", 0), 1)
        self.assertEqual(config.parameter_count(4, 65, "per_channel", 0), 4)
        self.assertEqual(config.parameter_count(4, 65, "group", 32), 12)

    def test_int4_signed_nibble_codec_and_odd_tail(self) -> None:
        values = [-8, -7, -1, 0, 1, 6, 7]
        payload = codec.pack_int4_values(values)
        self.assertEqual(len(payload), 4)
        self.assertEqual(codec.unpack_int4_values(payload, len(values)), values)

    def test_rounding_contract(self) -> None:
        self.assertEqual(codec.round_half_away_from_zero(1.5), 2)
        self.assertEqual(codec.round_half_away_from_zero(-1.5), -2)
        self.assertEqual(codec.round_half_away_from_zero(1.49), 1)
        self.assertEqual(codec.round_half_away_from_zero(-1.49), -1)

    def test_shape_matrix_has_odd_tail_and_llm_weights(self) -> None:
        self.assertTrue(any(shape.cols % 2 for shape in config.SHAPES))
        self.assertTrue(any(shape.rows >= 4096 and shape.cols >= 4096 for shape in config.SHAPES))

    def test_cuda_source_contract(self) -> None:
        source = (EXP / "quant_kernel.cu").read_text(encoding="utf-8")
        self.assertIn("RoundHalfAwayFromZero", source)
        self.assertIn("QuantizeInt8Kernel", source)
        self.assertIn("QuantizeInt4Kernel", source)
        self.assertIn("nibble >= 8 ? nibble - 16", source)
        self.assertIn("__CUDA_ARCH__", source)

    def test_triton_source_contract(self) -> None:
        source = (EXP / "quant_triton.py").read_text(encoding="utf-8")
        self.assertIn("_quantize_int8_kernel", source)
        self.assertIn("_quantize_int4_kernel", source)
        self.assertIn("tl.floor", source)
        self.assertIn("tl.ceil", source)
        self.assertIn("nibble >= 8", source)


if __name__ == "__main__":
    unittest.main()
