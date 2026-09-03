from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "30_paged_kv_cache"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


config = load_module("paged_kv_lab_config", EXP / "lab_config.py")
allocator_mod = load_module("paged_kv_allocator", EXP / "allocator.py")


class PagedKVCacheLogicTest(unittest.TestCase):
    def test_page_sizes_and_shape_matrix(self) -> None:
        self.assertEqual(config.PAGE_SIZES, (16, 32))
        self.assertTrue(any(shape.kv_heads == 1 for shape in config.SHAPES))
        self.assertTrue(any(shape.q_heads > shape.kv_heads > 1 for shape in config.SHAPES))
        self.assertTrue(any(shape.max_tokens >= 32768 for shape in config.SHAPES))
        self.assertTrue(any(shape.decode_starts_new_page for shape in config.SHAPES))
        self.assertEqual(
            {shape.page_size for shape in config.SHAPES},
            set(config.PAGE_SIZES),
        )

    def test_allocator_builds_non_contiguous_block_tables(self) -> None:
        table, allocator = allocator_mod.build_block_table(
            [33, 33], max_tokens=64, page_size=16, total_pages=12
        )
        for row in table:
            pages = [page for page in row if page >= 0]
            self.assertGreaterEqual(len(pages), 3)
            self.assertTrue(any(b - a != 1 for a, b in zip(pages, pages[1:])))
        snapshot = allocator.snapshot([33, 33], 16)
        self.assertEqual(snapshot.live_tokens, 66)
        self.assertGreater(snapshot.internal_fragmentation_slots, 0)
        self.assertGreaterEqual(snapshot.external_fragmentation_ratio, 0.0)

    def test_free_list_reuses_released_pages(self) -> None:
        demo = allocator_mod.allocator_reuse_demo()
        self.assertTrue(demo["reuse_observed"])
        self.assertTrue(
            set(demo["reused_pages"]).issubset(set(demo["released_pages"]))
        )

    def test_dtype_capability_contract(self) -> None:
        self.assertEqual(config.requirements_for_dtype("fp16"), ("cuda",))
        self.assertEqual(config.requirements_for_dtype("bf16"), ("cuda", "bf16"))

    def test_cuda_source_has_two_level_mapping_and_pair_copy(self) -> None:
        source = (EXP / "paged_kv_kernel.cu").read_text(encoding="utf-8")
        self.assertIn("uint32_t", source)
        self.assertIn("logical_block = position / page_size", source)
        self.assertIn("slot = position % page_size", source)
        self.assertIn("block_table", source)
        self.assertIn("physical_page", source)
        self.assertIn("PagePairOffset", source)
        self.assertIn("check_bounds", source)

    def test_triton_source_has_block_table_update_and_read(self) -> None:
        source = (EXP / "paged_kv_triton.py").read_text(encoding="utf-8")
        self.assertIn("_paged_kv_update_kernel", source)
        self.assertIn("_paged_kv_read_kernel", source)
        self.assertIn("logical_block = position // page_size", source)
        self.assertIn("physical_page = tl.load(block_table", source)
        self.assertIn("[batch, kv_heads, tokens, head_dim]", source)


if __name__ == "__main__":
    unittest.main()
