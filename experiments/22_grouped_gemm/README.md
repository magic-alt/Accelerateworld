# Persistent / Grouped GEMM Lab

Experiment 22 moves from selecting a good kernel for **one GEMM shape** to scheduling **many heterogeneous GEMMs** with a persistent CUDA grid.

The implementation uses the repository's pinned **NVIDIA CUTLASS 4.7.0** and its real `cutlass::gemm::device::GemmGrouped` scheduler. It does not emulate persistence with a toy work queue: CUTLASS launches an occupancy-bounded set of CTAs and each CTA repeatedly asks the grouped problem visitor for another logical GEMM tile until the group is exhausted.

```text
many M/N/K problems
        ↓
CUTLASS grouped problem metadata
        ↓
occupancy-bounded persistent CTAs
        ↓
problem visitor / tile scheduler
        ↓
next logical GEMM tile
        ↓
Tensor Core mainloop
        ↓
repeat until no work remains
```

## Questions

1. When is one persistent grouped launch faster than launching one GEMM kernel per problem?
2. How many logical tiles are consumed by each launched CTA (`total_tiles / persistent_ctas`)?
3. What is the cost difference between `GroupScheduleMode::kDeviceOnly` and `kHostPrecompute`?
4. Does sorting heterogeneous problems by descending K improve scheduler/load-balancing behavior?
5. How does the same scheduler behave for MoE-style expert GEMMs and decode-style small-M GEMMs?

## Controlled kernel configuration

Scheduling is the independent variable in this stage, so the serial and grouped paths use the same controlled FP16 TensorOp geometry:

```text
minimum architecture = SM75
A / B                = FP16
accumulator / output = FP32
A layout             = row-major
B layout             = column-major
C/D layout           = row-major
threadblock           = 128 x 128 x 32
warp                  = 64 x 64 x 32
instruction           = CUTLASS SM75 default FP16 TensorOp
pipeline stages       = CUTLASS SM75 default
```

The SM75 programming model is intentional. Experiment 21 already studies architecture-specific GEMM kernel selection; experiment 22 keeps one common Tensor Core kernel contract so RTX 20/30/40/50 can compare **dispatch and scheduling**, not different native kernel families.

## Execution modes

### `serial`

Conventional reference: one `cutlass::gemm::device::Gemm` launch per problem, on one CUDA stream. A group of 16 problems therefore creates 16 GEMM launches per measured iteration.

### `device`

One `device::GemmGrouped` launch using `GroupScheduleMode::kDeviceOnly`. Scheduling metadata is walked on the GPU. `Gemm::sufficient()` selects the persistent CTA count from the total tile count and occupancy; when there are more logical tiles than CTAs, each CTA executes multiple tiles over its lifetime.

### `host`

One grouped launch using `GroupScheduleMode::kHostPrecompute`. CUTLASS prepares scheduler workspace on the host and uploads it during initialization. Initialization latency and scheduler workspace bytes are reported separately and remain outside steady-state CUDA Event timing.

### `sorted`

The same device scheduler after stable descending-K problem ordering. CUTLASS documents K ordering as a useful grouped-GEMM scheduling heuristic when per-tile K work is heterogeneous. The benchmark measures it rather than assuming it is always beneficial.

`--mode all` runs all four paths. Speedup values are printed only when the serial reference is present in the same process.

## Workloads

| Preset | What varies | Runtime analogy |
|---|---|---|
| `heterogeneous` | M/N/K | batched heterogeneous GEMMs, mixed projection shapes |
| `moe` | expert token M plus alternating projection N/K | routed MoE expert FFN GEMMs |
| `decode` | M = 1/2/4/8/... with larger N/K | low-batch autoregressive decode |

These are scheduler microbenchmarks, not claims that the shape list exactly reproduces a specific production model.

```bash
# canonical heterogeneous comparison
./build/release/bin/aw_grouped_gemm \
  --workload heterogeneous --groups 16 --warmup 2 --iterations 10 --mode all

# routed-expert style group
./build/release/bin/aw_grouped_gemm \
  --workload moe --groups 32 --warmup 2 --iterations 10 --mode all

# small-M decode pressure
./build/release/bin/aw_grouped_gemm \
  --workload decode --groups 32 --warmup 2 --iterations 20 --mode all
```

## Metrics

The executable reports, per path:

- launch count per measured iteration;
- persistent CTA count and total logical tile count;
- average tiles consumed per persistent CTA;
- scheduler workspace and initialization latency;
- steady-state CUDA Event latency and aggregate GFLOP/s;
- speedup versus the serial per-problem launch reference;
- sampled correctness error against FP64 CPU accumulation of the actual narrowed FP16 inputs.

The canonical GPU Baseline v2 metric is `grouped_device_throughput`. The same raw log retains serial, host-precompute and K-sorted measurements for interpretation.

## Why this connects to MoE and decode runtimes

A production MoE layer does not normally present one large homogeneous GEMM. Token routing creates a collection of expert problems with uneven M, and some experts may receive very little work. Launching and scheduling each GEMM independently can turn host launch overhead and tail imbalance into a significant part of latency.

Likewise, autoregressive decode often drives small-M matrix products. The arithmetic work per individual GEMM becomes small enough that dispatch overhead and the ability to keep resident CTAs supplied with useful tiles matter more than they do for one large square GEMM.

This lab therefore establishes the scheduler concepts used later by inference-runtime work:

```text
routing / batching
      ↓
heterogeneous problem list
      ↓
persistent grouped dispatch
      ↓
load-balanced tile consumption
```

It deliberately stops before implementing token routing, expert permutation, fused activations or a complete MoE runtime; those are separate system layers.

## Correctness and timing boundary

Inputs are deterministic. Every execution path writes FP32 output and is checked on sampled coordinates against a CPU reference accumulated in FP64 from the actual FP16 values.

Allocation, H2D setup, pointer-array construction, scheduler initialization and D2H validation are outside CUDA Event timing. Only repeated GEMM execution is timed. This prevents host preprocessing from being silently mixed into the steady-state kernel metric while still reporting grouped initialization cost separately.

## Evidence boundary

Hosted CI can prove:

- CUTLASS 4.7.0 template/API compatibility for `device::GemmGrouped`;
- portable compile coverage for the configured RTX 20/30/40/50 targets;
- CTest discovery;
- GPU Baseline v2 manifest/feature gating.

Only a physical RTX GPU can establish:

- grouped-vs-serial speedup;
- persistent CTA occupancy and runtime efficiency;
- device-vs-host scheduler winner;
- whether K sorting helps a given problem distribution;
- MoE/decode shape throughput;
- Compute Sanitizer runtime evidence.

No physical-GPU performance result is fabricated by this stage.
