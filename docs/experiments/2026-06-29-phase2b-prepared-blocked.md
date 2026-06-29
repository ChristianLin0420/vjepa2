# Phase 2b geometry distillation — prepared for Slurm execution

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase2b-jepa-geometry-distillation-v1` |
| Stage / status | `geometry student / implementation ready, Slurm validation pending` |
| Evidence level | `implementation-only` |
| Dataset manifest | `tum-rgbd-fr1-xyz-phase2b` version `1.0.1` |
| Intended hardware | NVIDIA A100 80 GB PCIe |
| W&B | Formal run not initialized; online preflight is the next gate. |
| Decision | Submit static/unit/CUDA tests, then real-model preflight, then the unchanged formal protocol. |

## Objective

Measure whether a compact multi-layer V-JEPA geometry student offers a useful accuracy/runtime/memory trade-off against
the frozen VGGT-1B teacher, a non-JEPA RGB baseline, and a final-layer-only V-JEPA ablation. Every learned variant uses
the same probe capacity, data, losses, seeds, selection policy, and held-out metrics.

## Completed implementation

- checksum-pinned official TUM RGB-D Freiburg1 XYZ archive;
- chronological, non-overlapping 64-frame training, 16-frame validation, and 8-frame test split;
- raw RGB plus image-coordinate baseline with no pretrained representation;
- frozen final-layer V-JEPA 2.1 ablation;
- frozen four-layer V-JEPA 2.1 candidate;
- official VGGT-1B BF16 teacher and aligned auxiliary distillation target;
- compact shared depth/log-variance probe architecture;
- metric and median-aligned AbsRel, RMSE, and delta metrics;
- validation-fitted log-depth variance calibration and held-out NLL;
- three seeds per learned variant and validation-only checkpoint selection;
- encoder/head/total latency, peak memory, parameter count, failures, and checkpoint records;
- versioned `jepa4d-phase2b-comparison-v1` JSON schema designed to accept future variants;
- W&B training curves, seed-level comparison table, aggregate summaries, and bundled checkpoint/report artifact path.

## Immutable comparison policy

| Variant | Role | Representation | Trainable component |
|---|---|---|---|
| `vggt_teacher` | teacher baseline | Official VGGT-1B | None |
| `rgb_probe` | non-JEPA baseline | RGB plus normalized XY | Shared compact probe |
| `vjepa_final` | ablation | Frozen normalized final V-JEPA layer | Shared compact probe |
| `vjepa_multilayer` | proposed method | Frozen standardized layers 2, 5, 8, and 11 | Shared compact probe |

The primary promotion metric is metric AbsRel on the chronological test split. Secondary evidence includes aligned depth,
uncertainty NLL before/after validation-only calibration, latency, memory, parameters, per-seed variation, and failures.
No test metric participates in checkpoint selection.

## Pre-execution protocol correction for Slurm

The first code review on the Slurm host found issues that would have made the original prepared comparison misleading.
They were corrected before any Phase-2b optimization result existed, so this is a pre-registered protocol amendment rather
than a post-result change:

- every encoder now receives independent `B=N,V=1,T=1` samples; batching no longer turns eight frames into one
  multi-view VGGT scene;
- RGB and depth use the same center-square crop; RGB is resized bilinearly and depth with nearest-neighbor interpolation;
- VGGT is evaluated directly at 518 px, while a separate 24 px tensor is used only for probe supervision;
- one VGGT metric scale is fitted on training pixels and frozen before validation/test; per-test-frame scale alignment is
  reported only under `aligned_*`;
- invalid/non-positive predictions on valid target pixels fail the run instead of disappearing from metric denominators;
- layers 2, 5, 8, and 11 are standardized from training statistics and averaged, keeping the multi-layer and final-layer
  probes exactly parameter-matched;
- the RGB baseline is train-normalized and explicitly described as a non-JEPA representation with the same VGGT-assisted
  supervision, not as a teacher-free baseline;
- V-JEPA and VGGT are unloaded between profiles; encoder throughput, batch-1 head latency, inference memory, and training
  memory are labeled separately;
- all nine learned runs must finish at 60 epochs with seeds 0/1/2, online W&B, checkpoint/normalization hashes, finite
  metrics, per-frame diagnostics, and a passing strict postflight validator.
- the first real archive audit found RGB index 202 had no ground-truth pose within the fixed 30 ms association tolerance;
  before model execution it was replaced by the nearest unused valid index 203 and the manifest was bumped to 1.0.1.

The formal Slurm allocation is one task and one GPU with 16 CPUs, 220 GiB RAM, and a four-hour limit. A real-model
preflight must first pass on an equal-or-smaller-memory GPU and upload its checkpoint/report artifact to online W&B.

## Slurm gate attempts

- job `29586467` passed Ruff, mypy, 72 tests, environment/repository fingerprinting, and a sustained A100-SXM4-80GB
  CUDA/GEMM check, producing the required test receipt;
- the first real-model preflight, job `29586489`, failed before VGGT and before W&B initialization because its original
  near-bitwise V-JEPA batch-equivalence tolerance rejected normal GPU reduction drift: only 1,612 of 3,538,944 layer-2
  values exceeded the old bound and the worst absolute difference was 0.00212;
- before any optimization or model-quality result, the gate was amended to use explicit numerical-equivalence tolerances
  (`rtol=0.01`, `atol=0.003` for FP32 V-JEPA; PyTorch BF16-scale tolerances for VGGT) and to record maximum/mean error,
  RMSE, relative RMSE, reference RMS, and cosine similarity. Large content/batching changes still fail the gate.
- retry `29586608` passed all V-JEPA layers (relative RMSE at most `1.7e-5`) and showed VGGT was globally equivalent
  (relative RMSE `0.00172`, cosine `0.9999986`), but elementwise all-close rejected 458 of 2,146,592 BF16 depth pixels;
  the final pre-result gate therefore bounds outlier fraction, global relative RMSE, and cosine similarity together while
  retaining maximum error as a visible diagnostic instead of allowing one isolated pixel to veto an equivalent batch.
- preflight `29586767` then passed real models, optimizer/checkpoint reload, report generation, and online W&B artifact
  upload. The first formal job, `29586856`, passed content authorization but failed before W&B/model optimization because
  the CUDA driver UUID wrapper was not JSON serializable. The UUID is now normalized to text, the exact formal environment
  snapshot runs in preflight, and all post-output initialization is covered by the local failure recorder.

## Historical single-host GPU blocker

Immediately before the first training launch, the A100 had passed earlier Phase-2 quality runs. At Phase-2b launch time:

- `nvidia-smi` returned `Unable to determine the device handle ... Unknown Error`;
- `lspci` reported the GA100 at PCI revision `ff`;
- PyTorch reported `cuda_available=false` and zero devices;
- the runner rejected execution before model loading or W&B initialization.

This was the previously diagnosed PCIe link-loss condition, not a model, token, dependency, or training failure. No CPU
fallback was allowed because it would invalidate the required GPU runtime/memory comparison. The relaunch now uses the
approved Slurm partitions, with CUDA health rechecked inside every allocation. No result has yet been promoted.

## Verification before commit

- Ruff: pass across `jepa4d`, `scripts`, and `slurm`;
- mypy: pass across 89 Phase-2b-relevant source files;
- JEPA-4D pytest: 75 passed, one expected local-model skip, one third-party deprecation warning;
- Slurm `--test-only`: test, preflight, and formal entrypoints accepted by the scheduler;
- credential scan: supplied W&B and Hugging Face tokens absent from tracked files;
- login environment: Python 3.12 dependency check passes;
- assets: V-JEPA, pinned compatibility source, VGGT-1B, and TUM archive/extraction identities verified;
- comparison runner and formal wrappers fail closed without CUDA, passing receipts, or online W&B.

## Resume command

From the repository root after login preparation:

```bash
test_job=$(sbatch --parsable slurm/phase2b_tests.sbatch)
preflight_job=$(sbatch --parsable --dependency="afterok:${test_job}" slurm/phase2b_preflight.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" slurm/phase2b_train.sbatch)
```

## Remaining work

1. Pass the submitted static/unit/sustained-CUDA test receipt.
2. Pass real V-JEPA/VGGT inference, optimizer/reload/report, and online-W&B preflight.
3. Execute all nine learned runs (three variants × three seeds) and the VGGT teacher on the pinned split.
4. Verify zero missing seeds, finite predictions, checkpoint hashes, failure records, and local/W&B artifact parity.
5. Add numerical comparison tables and per-variant insights to this record, `INDEX.md`, and `INSIGHTS.md`.
6. Mark Phase 2b complete only if the accuracy/runtime/memory gate is evaluated; record a negative outcome honestly.

## Claim boundary

The committed state proves that the comparison implementation and reporting contracts pass tests. It contains no Phase-2b
model-quality result. The one-sequence protocol will remain sequence-level evidence even after execution and must not be
presented as cross-dataset generalization.
