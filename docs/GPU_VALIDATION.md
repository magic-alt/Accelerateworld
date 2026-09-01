# GPU Validation Guide

## What is validated

`run_gpu_validation.sh` and `run_gpu_validation.ps1` perform four layers of validation:

1. environment discovery;
2. configure/build;
3. CTest numerical correctness;
4. Compute Sanitizer + benchmark execution.

## RTX 5060

NVIDIA reports the RTX 5060 as compute capability 12.0, so use:

```bash
export ACCELERATEWORLD_CUDA_ARCH=120
bash scripts/run_gpu_validation.sh
```

PowerShell:

```powershell
$env:ACCELERATEWORLD_CUDA_ARCH = "120"
.\scripts\run_gpu_validation.ps1
```

CUDA Toolkit 13.x is the recommended baseline for this repository.

## GitHub self-hosted GPU runner

A self-hosted runner is the simplest way to turn a local RTX 5060 into an auditable CI worker.

Prerequisites on the runner:

- NVIDIA driver;
- CUDA Toolkit;
- CMake >= 3.24;
- Ninja;
- Git;
- Compute Sanitizer.

Register the runner with a custom `gpu` label. The workflow expects:

```yaml
runs-on: [self-hosted, linux, x64, gpu]
```

For security, dedicate the runner to repositories you trust and avoid running unreviewed pull-request code from arbitrary forks on a persistent machine.

## GitHub GPU larger runner

GitHub also offers GPU larger runners with Tesla T4 hardware for eligible organization/enterprise plans. A T4 is compute capability 7.5, so use architecture `75` and replace the workflow `runs-on` value with the configured larger-runner name.

A T4 run validates portability and correctness, but it is not a substitute for RTX 5060 performance data.

## Expected first-run output

Successful execution should show:

- one or more CUDA devices;
- PASS for `cuda.device_query`;
- PASS for `cuda.vector_add`;
- PASS for `cuda.matmul`;
- zero fatal Compute Sanitizer findings;
- vector-add effective bandwidth;
- naive and tiled matrix-multiplication GFLOP/s and speedup.

Do not hard-code expected performance numbers: clocks, driver, power limits, thermal state, operating system, and background load materially affect results.

## Recording a benchmark

A benchmark result should be attached to a commit or PR and include the complete validation log from `results/gpu-validation.log`.

Suggested PR section:

```text
GPU: GeForce RTX 5060
CC: 12.0
Driver:
CUDA:
OS:
Commit:

Vector add:
  N:
  kernel ms:
  effective GB/s:

Matmul:
  N:
  naive GFLOP/s:
  tiled GFLOP/s:
  speedup:

Compute Sanitizer:
  memcheck:
  racecheck:
  initcheck:
  synccheck:
```

## Troubleshooting

### `no kernel image is available for execution on the device`

The binary was not compiled for the installed GPU. Reconfigure with the correct architecture.

RTX 5060:

```bash
-DACCELERATEWORLD_CUDA_ARCHITECTURES=120
```

T4:

```bash
-DACCELERATEWORLD_CUDA_ARCHITECTURES=75
```

### `CUDA driver version is insufficient for CUDA runtime version`

Update the NVIDIA driver or use a CUDA Toolkit/runtime compatible with the installed driver.

### Docker cannot see the GPU

Verify the host NVIDIA driver and NVIDIA Container Toolkit, then test:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```
