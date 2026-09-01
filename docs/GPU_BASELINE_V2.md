# GPU Baseline v2

GPU Baseline v2 makes Accelerateworld a cross-generation RTX benchmark project instead of an RTX 5060-specific lab.

## Supported RTX generations

| RTX generation | Architecture | Compute capability | Native SM |
|---|---|---:|---:|
| RTX 20 | Turing | 7.5 | 75 |
| RTX 30 | Ampere | 8.6 | 86 |
| RTX 40 | Ada | 8.9 | 89 |
| RTX 50 | Blackwell | 12.0 | 120 |

RTX 5060 remains the first reference GPU, but it is not a special execution path. It resolves through the same Blackwell `sm_120` profile used by the rest of RTX 50.

The authoritative model-to-compute-capability mapping should be checked against NVIDIA's CUDA GPU Compute Capability page when new GPU families appear:

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

## Design

The baseline separates three concerns:

1. **portable compile coverage**: CI compiles `sm_75;sm_86;sm_89;sm_120`;
2. **runtime detection**: the physical GPU is discovered with `nvidia-smi` and mapped to `hardware/rtx_capabilities.json`;
3. **feature gating**: each benchmark declares required capabilities in `benchmarks/manifest.json` and is skipped rather than incorrectly executed when a feature is unavailable.

This is important for later BF16, FP8, FP4, attention and quantization experiments where support differs by architecture.

## Detect a GPU

```bash
python scripts/detect_gpu.py
```

Example fields:

```json
{
  "name": "NVIDIA GeForce RTX 4090",
  "compute_capability": "8.9",
  "sm": "89",
  "generation": "Ada",
  "rtx_series": "RTX 40",
  "features": {
    "cuda": true,
    "tensor_core": true,
    "tf32": true,
    "bf16": true,
    "fp8": true,
    "fp4": false
  },
  "reference_gpu": false
}
```

## Run the baseline

Linux / WSL2:

```bash
python3 scripts/run_gpu_baseline.py
```

Windows:

```powershell
python scripts/run_gpu_baseline.py
```

The runner automatically:

1. detects the selected RTX GPU;
2. resolves the native SM architecture;
3. configures a Release build for only that SM;
4. builds all current native CUDA experiments;
5. runs CTest correctness tests;
6. runs capability-appropriate Compute Sanitizer checks;
7. runs the canonical benchmark manifest with feature gates;
8. writes immutable evidence under `results/gpu-baselines/<gpu>/<timestamp>-<commit>/`.

The legacy wrappers remain available:

```bash
bash scripts/run_gpu_validation.sh
```

```powershell
.\scripts\run_gpu_validation.ps1
```

They delegate to GPU Baseline v2.

## CMake presets

Generation presets are available for explicit compile validation:

```bash
cmake --preset rtx20   # sm_75
cmake --preset rtx30   # sm_86
cmake --preset rtx40   # sm_89
cmake --preset rtx50   # sm_120
```

The `rtx5060` preset remains as a compatibility/reference alias for `rtx50`.

## Unified evidence schema

Every complete baseline emits `baseline.json` following:

```text
benchmarks/gpu-baseline-v2.schema.json
```

Important fields include:

- repository commit;
- exact GPU model and PCI bus id;
- driver, VRAM, compute capability and SM target;
- architecture generation and feature map;
- compiler/runtime tool versions;
- validation step status;
- benchmark status, requirements and missing features;
- parsed benchmark metrics plus raw stdout/stderr;
- final success state.

A skipped unsupported feature is different from a failed supported feature.

## Cross-GPU comparison

After collecting baseline files from multiple GPUs:

```bash
python scripts/compare_gpu_baselines.py \
  results/gpu-baselines/geforce-rtx-3090/*/baseline.json \
  results/gpu-baselines/geforce-rtx-4090/*/baseline.json \
  results/gpu-baselines/geforce-rtx-5060/*/baseline.json \
  --markdown results/rtx-comparison.md \
  --json results/rtx-comparison.json
```

The comparison uses each benchmark's declared `primary_metric`, so the same workload definition can be compared across RTX generations without copying values by hand.

## Feature gates for future LLM kernels

Future benchmark entries should declare capabilities explicitly. Examples:

```json
{
  "id": "rope_fp16",
  "requires": ["cuda"]
}
```

```json
{
  "id": "attention_bf16",
  "requires": ["cuda", "bf16"]
}
```

```json
{
  "id": "attention_fp8",
  "requires": ["cuda", "tensor_core", "fp8"]
}
```

```json
{
  "id": "blackwell_fp4_gemm",
  "requires": ["cuda", "tensor_core", "fp4", "blackwell"]
}
```

This means the planned sequence `RoPE -> Online Softmax -> FlashAttention-style -> KV Cache -> Minimal Decoder Runtime` can use the same hardware capability layer from the first implementation instead of being rewritten for each GPU generation.

## Adding a future RTX generation

When NVIDIA introduces a new RTX compute capability:

1. verify the official compute capability;
2. add one architecture entry to `hardware/rtx_capabilities.json`;
3. add the new SM to portable CMake/CI after the selected CUDA Toolkit supports it;
4. define conservative feature flags;
5. collect a real baseline before publishing performance claims.
