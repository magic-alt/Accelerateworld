# Benchmark Suite

`manifest.json` defines the canonical native-CUDA benchmark shapes and commands. The goal is to keep workload definitions stable so results from different commits or GPUs remain interpretable.

After building the project on a GPU machine:

```bash
python scripts/run_benchmark_suite.py \
  --build-dir build/gpu \
  --output results/native-benchmark.json
```

The result JSON records:

- UTC timestamp and git commit;
- operating system and Python version;
- `nvidia-smi` GPU/driver/VRAM information;
- `nvcc --version`;
- exact command for every benchmark;
- process return code;
- raw stdout/stderr;
- host-observed elapsed time.

Kernel-specific metrics remain printed by the benchmark executable itself. A later metrics-schema milestone will promote latency/bandwidth/GFLOP/s into normalized JSON fields for automatic regression checks.

Do not compare two result files unless their workload definitions, dtype and timing boundaries are equivalent. See `docs/BENCHMARKING.md`.
