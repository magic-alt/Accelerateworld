# GPU Validation Guide

Accelerateworld uses GPU Baseline v2 for runtime validation across supported NVIDIA RTX generations.

## What is validated

`run_gpu_validation.sh` and `run_gpu_validation.ps1` delegate to `scripts/run_gpu_baseline.py` and perform:

1. RTX GPU discovery and compute-capability resolution;
2. native-SM configure/build;
3. CTest numerical correctness;
4. capability-appropriate Compute Sanitizer checks;
5. feature-gated benchmark execution;
6. unified evidence capture under `results/gpu-baselines/`.

## Supported RTX architecture mapping

| RTX generation | Architecture | Compute capability | SM |
|---|---|---:|---:|
| RTX 20 | Turing | 7.5 | 75 |
| RTX 30 | Ampere | 8.6 | 86 |
| RTX 40 | Ada | 8.9 | 89 |
| RTX 50 | Blackwell | 12.0 | 120 |

Do not manually set the architecture for normal baseline execution. The runner queries `nvidia-smi` and resolves the correct native SM through `hardware/rtx_capabilities.json`.

RTX 5060 is the first reference GPU, but follows the same RTX 50 / `sm_120` path as other Blackwell RTX cards.

## Local execution

Linux / WSL2:

```bash
bash scripts/run_gpu_validation.sh
```

Windows PowerShell:

```powershell
.\scripts\run_gpu_validation.ps1
```

Select another GPU in a multi-GPU machine with:

```bash
ACCELERATEWORLD_GPU_INDEX=1 bash scripts/run_gpu_validation.sh
```

or:

```powershell
$env:ACCELERATEWORLD_GPU_INDEX = "1"
.\scripts\run_gpu_validation.ps1
```

## GitHub self-hosted GPU runner

Prerequisites:

- NVIDIA driver;
- CUDA Toolkit compatible with the target architecture;
- CMake >= 3.24;
- Ninja;
- Git;
- Python 3;
- Compute Sanitizer.

Register the runner with a `gpu` label. The workflow expects:

```yaml
runs-on: [self-hosted, linux, x64, gpu]
```

The workflow does not require an architecture input; the physical GPU is detected at runtime.

For security, dedicate the runner to repositories you trust and avoid running unreviewed pull-request code from arbitrary forks on a persistent machine.

## Feature gating

Each entry in `benchmarks/manifest.json` declares `requires` capabilities. For example:

```json
{
  "id": "tensor_core_wmma",
  "requires": ["cuda", "tensor_core", "wmma_fp16"]
}
```

Future BF16, FP8, FP4 and architecture-specific LLM kernels should use the same mechanism. Unsupported functionality is recorded as `skipped`, not `failed`.

## Evidence

A successful run produces a directory similar to:

```text
results/gpu-baselines/
  nvidia-geforce-rtx-4090/
    20260901T090000Z-abcdef12/
      gpu-profile.json
      native-benchmark.json
      baseline.json
```

`baseline.json` follows `benchmarks/gpu-baseline-v2.schema.json` and is the canonical artifact for cross-GPU comparisons.

Do not hard-code expected performance numbers: clocks, driver, power limits, thermal state, operating system and background load materially affect results.

## Cross-GPU comparison

```bash
python scripts/compare_gpu_baselines.py \
  path/to/rtx3090/baseline.json \
  path/to/rtx4090/baseline.json \
  path/to/rtx5060/baseline.json \
  --markdown results/rtx-comparison.md
```

Only compare equivalent benchmark workload definitions and record the source commit for every result.

## Troubleshooting

### `no kernel image is available for execution on the device`

Confirm detection first:

```bash
python scripts/detect_gpu.py
```

Then check that the reported SM is one of the project's supported RTX profiles (`75`, `86`, `89`, `120`) and that the installed CUDA Toolkit supports it.

### `CUDA driver version is insufficient for CUDA runtime version`

Update the NVIDIA driver or use a CUDA Toolkit/runtime compatible with the installed driver.

### Docker cannot see the GPU

Verify the host NVIDIA driver and NVIDIA Container Toolkit, then test:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```
