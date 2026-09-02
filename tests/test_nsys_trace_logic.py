from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "experiments" / "24_nsys_trace"
sys.path.insert(0, str(TRACE_DIR))

from trace_config import (  # noqa: E402
    EXPECTED_NVTX_RANGES,
    STATS_REPORTS,
    TRACE_APIS,
    build_profile_command,
    build_stats_command,
)


class NsightTraceLogicTests(unittest.TestCase):
    def test_profile_command_uses_bounded_cuda_profiler_capture(self) -> None:
        command = build_profile_command(
            nsys="nsys",
            python_exe="python3",
            workload=Path("trace_workload.py"),
            output_base=Path("results/trace"),
            mode="steady",
            elements=1024,
            m=16,
            n=512,
            k=512,
            warmup=2,
            iterations=4,
            enable_pytorch_annotations=False,
        )
        self.assertEqual(command[:2], ["nsys", "profile"])
        self.assertIn(f"--trace={','.join(TRACE_APIS)}", command)
        self.assertIn("--sample=none", command)
        self.assertIn("--cpuctxsw=none", command)
        self.assertIn("--capture-range=cudaProfilerApi", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertIn("--force-overwrite=true", command)
        self.assertNotIn("--pytorch=functions-trace-shapes,autograd-nvtx", command)

    def test_profile_command_can_enable_pytorch_annotations(self) -> None:
        command = build_profile_command(
            nsys="nsys",
            python_exe="python3",
            workload=Path("trace_workload.py"),
            output_base=Path("results/trace"),
            mode="full",
            elements=1024,
            m=16,
            n=512,
            k=512,
            warmup=1,
            iterations=2,
            enable_pytorch_annotations=True,
        )
        self.assertIn("--pytorch=functions-trace-shapes,autograd-nvtx", command)
        self.assertIn("full", command)

    def test_stats_reports_cover_api_kernel_launch_and_nvtx_views(self) -> None:
        self.assertEqual(
            STATS_REPORTS,
            (
                "cuda_api_sum",
                "cuda_gpu_kern_sum",
                "cuda_kern_exec_sum",
                "nvtx_sum",
                "nvtx_gpu_proj_sum",
                "nvtx_kern_sum",
            ),
        )
        for report in STATS_REPORTS:
            command = build_stats_command(
                nsys="nsys", report=report, nsys_report=Path("trace.nsys-rep")
            )
            self.assertEqual(command[:3], ["nsys", "stats", "--report"])
            self.assertEqual(command[3], report)

    def test_expected_ranges_cover_framework_providers(self) -> None:
        ranges = set(EXPECTED_NVTX_RANGES)
        self.assertIn("aw/setup/torch_compile_first_call", ranges)
        self.assertIn("aw/setup/triton_autotune_first_call", ranges)
        self.assertIn("aw/trace/pytorch_eager_silu_mul", ranges)
        self.assertIn("aw/trace/custom_cuda_silu_mul", ranges)
        self.assertIn("aw/trace/torch_compile_custom_op", ranges)
        self.assertIn("aw/trace/pytorch_mm_cublas", ranges)
        self.assertIn("aw/trace/triton_autotuned_gemm", ranges)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_profile_command(
                nsys="nsys",
                python_exe="python3",
                workload=Path("trace_workload.py"),
                output_base=Path("results/trace"),
                mode="invalid",
                elements=1024,
                m=16,
                n=512,
                k=512,
                warmup=1,
                iterations=2,
                enable_pytorch_annotations=False,
            )


if __name__ == "__main__":
    unittest.main()
