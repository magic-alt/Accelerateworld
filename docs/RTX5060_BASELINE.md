# RTX 5060 Baseline Procedure

This document defines the acceptance procedure for Issue #3. A baseline is valid only when it is executed on a physical NVIDIA GeForce RTX 5060 and the raw evidence is retained.

## What the baseline proves

A valid run establishes:

- the repository builds for `sm_120` on the target machine;
- all native CUDA correctness tests pass;
- selected kernels pass Compute Sanitizer checks;
- the canonical benchmark manifest executes successfully;
- device identity is NVIDIA GeForce RTX 5060 with Compute Capability 12.0;
- optional PyTorch/Triton/LLM-kernel validation executes on the same GPU;
- environment and benchmark evidence is preserved as artifacts.

Compile-only CI is not runtime evidence.

## Windows native CUDA baseline

Run from a PowerShell where the NVIDIA driver, CUDA Toolkit, CMake, Ninja and Python are available:

```powershell
git checkout feat/rtx5060-baseline-readiness
.\scripts\run_rtx5060_baseline.ps1
```

Expected evidence under `results/`:

- `rtx5060-environment.txt`
- `rtx5060-device-query.txt`
- `gpu-validation.log`
- `rtx5060-native.json`
- `rtx5060-baseline-summary.json`

The script exits non-zero if the visible GPU is not an RTX 5060, if Compute Capability is not 12.0, if correctness/sanitizer validation fails, or if any canonical native benchmark exits non-zero.

## Linux / WSL2 native + AI baseline

Triton is validated on Linux. From a CUDA-capable Linux/WSL2 environment:

```bash
git checkout feat/rtx5060-baseline-readiness
python -m pip install -r python/requirements-gpu.txt
bash scripts/run_rtx5060_baseline.sh --with-ai
```

This adds:

- `ai-gpu-validation.log`

and verifies the PyTorch CUDA extension, Triton vector add and RMSNorm experiment.

## Self-hosted GitHub runner

A dedicated workflow is provided:

```text
.github/workflows/rtx5060-baseline.yml
```

The runner must have labels:

```text
self-hosted, linux, x64, gpu
```

The workflow intentionally rejects a non-RTX-5060 machine in the baseline script. When `with_ai=true`, Python GPU dependencies are installed and the native + AI suites run together.

## Evidence verification

Evidence can be re-checked without rerunning the benchmarks:

```bash
python scripts/verify_baseline_evidence.py --result-dir results
```

For a native + AI run:

```bash
python scripts/verify_baseline_evidence.py --result-dir results --require-ai
```

## Acceptance criteria

Issue #3 may be closed only when all of the following are attached or otherwise preserved from the same target-GPU run:

1. `rtx5060-environment.txt` identifies RTX 5060;
2. `rtx5060-device-query.txt` reports Compute Capability 12.0;
3. `gpu-validation.log` ends with `GPU validation: PASS`;
4. `rtx5060-native.json` contains successful records for every benchmark in `benchmarks/manifest.json`;
5. Compute Sanitizer checks complete without relevant errors;
6. if AI validation is claimed, `ai-gpu-validation.log` ends with `AI/GPU validation: PASS`;
7. `rtx5060-baseline-summary.json` records the exact Git commit used.

Do not copy benchmark numbers manually into documentation without retaining the raw JSON/log artifact that produced them.
