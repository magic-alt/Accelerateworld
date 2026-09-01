# Python GPU Environment

The Python layer is intentionally isolated from the native CMake build.

As of 2026-09-01, PyTorch 2.13.0 is the current stable release and official CUDA 13.0 wheels are available. `requirements-gpu.txt` selects that CUDA 13.0 wheel index and also permits PyPI packages such as Triton and Ninja.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r python/requirements-gpu.txt
```

Check the environment before benchmarking:

```bash
python - <<'PY'
import torch
import triton
print(torch.__version__)
print(torch.version.cuda)
print(triton.__version__)
print(torch.cuda.get_device_name())
PY
```

The repository does not assume that a CUDA wheel being installable proves GPU compatibility. Runtime evidence still requires executing the workloads on the target GPU.
