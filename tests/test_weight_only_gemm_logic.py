from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/"experiments"/"32_weight_only_gemm"

def load_module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

config=load_module("weight_only_gemm_lab_config",EXP/"lab_config.py")

class WeightOnlyGemmLogicTests(unittest.TestCase):
    def test_shape_matrix_covers_decode_prefill_and_odd_int4_tail(self):
        self.assertTrue(any(s.m==1 for s in config.SHAPES))
        self.assertTrue(any(s.m>=128 for s in config.SHAPES))
        self.assertTrue(any(s.k%2 for s in config.SHAPES))

    def test_storage_contract_matches_int8_and_packed_int4(self):
        self.assertEqual(config.quantized_weight_bytes("int8_sym",7,65),455)
        self.assertEqual(config.quantized_weight_bytes("int4_sym",7,65),7*33)
        self.assertAlmostEqual(config.compression_ratio("int8_sym",64,128),2.0)
        self.assertAlmostEqual(config.compression_ratio("int4_sym",64,128),4.0)

    def test_group_qparams_are_output_channel_by_k_group(self):
        self.assertEqual(config.qparam_count(7,65,"group",32),21)
        self.assertEqual(config.qparam_count(7,65,"per_channel",0),7)

    def test_regime_and_intensity_expose_small_m_vs_prefill(self):
        decode=next(s for s in config.SHAPES if s.name=="decode_m1")
        prefill=next(s for s in config.SHAPES if s.name=="prefill_m128")
        self.assertEqual(config.expected_regime(decode),"bandwidth_sensitive")
        self.assertEqual(config.expected_regime(prefill),"compute_reuse_sensitive")
        self.assertGreater(config.arithmetic_intensity(prefill,"int4_sym"),config.arithmetic_intensity(decode,"int4_sym"))

    def test_cuda_source_dequantizes_weight_fragment_inside_k_loop(self):
        source=(EXP/"weight_only_kernel.cu").read_text(encoding="utf-8")
        for token in ("WeightOnlyGemmKernel","DecodeInt4","ParamIndex","__shared__ float a_tile","__shared__ float w_tile","fmaf","__CUDA_ARCH__"):
            self.assertIn(token,source)
        self.assertNotIn("dequantized_weight",source)

    def test_triton_source_uses_low_bit_weight_and_dot(self):
        source="".join((EXP/"weight_only_triton.py").read_text(encoding="utf-8").split())
        for token in ("nibble>=8","tl.dot","scales_ptr+pidx","zero_ptr+pidx","PACKED_K"):
            self.assertIn(token,source)

if __name__=="__main__":
    unittest.main()
