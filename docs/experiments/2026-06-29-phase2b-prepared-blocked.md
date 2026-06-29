# Phase 2b geometry distillation — completed on Slurm

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase2b-jepa-geometry-distillation-v1` |
| Stage / status | `geometry student / completed; final-layer student selected` |
| Evidence level | `sequence-level` |
| Dataset manifest | `tum-rgbd-fr1-xyz-phase2b` version `1.0.1` |
| Hardware | NVIDIA A100-SXM4-80GB, Slurm `polar4` |
| W&B | [formal run `ikh4ptrb`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/ikh4ptrb) |
| Decision | Use the final V-JEPA layer for the default student; do not promote the multi-layer average on the primary metric. |

## Objective

Measure whether a compact multi-layer V-JEPA geometry student offers a useful accuracy/runtime/memory trade-off against
the frozen VGGT-1B teacher, a non-JEPA RGB baseline, and a final-layer-only V-JEPA ablation. Every learned variant uses
the same probe capacity, data, losses, seeds, selection policy, and held-out metrics.

## Result

All nine 60-epoch learned runs completed with zero failures. Values below are held-out test mean ± population standard
deviation over three seeds; the frozen teacher has one deterministic row. Lower is better for AbsRel/RMSE/NLL and latency.

| Variant | Metric AbsRel ↓ | RMSE m ↓ | Delta-1 ↑ | Aligned AbsRel ↓ | Calibrated NLL ↓ | ms/frame ↓ | Peak inference GiB ↓ | Total / trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGT-1B teacher | 0.12034 | 0.17777 | 0.94548 | **0.03262** | — | 148.766 | 12.604 | 1,256.54M / 0 |
| RGB+XY probe | 0.19417 ± 0.02083 | 0.27467 ± 0.01694 | 0.68364 ± 0.04150 | 0.12500 ± 0.00600 | -1.057 ± 0.115 | **7.266** | **0.032** | 0.0376M / 0.0376M |
| V-JEPA final-layer probe | **0.07523 ± 0.00384** | 0.14492 ± 0.01180 | 0.92743 ± 0.01676 | 0.06784 ± 0.00547 | -2.016 ± 0.250 | 17.931 | 0.873 | 86.92M / 0.0864M |
| V-JEPA four-layer average | 0.07857 ± 0.00538 | **0.13634 ± 0.00358** | **0.93721 ± 0.00723** | **0.06625 ± 0.00226** | **-2.028 ± 0.135** | 17.931 | 0.873 | 86.92M / 0.0864M |

Peak inference memory is the encoder inference peak for pretrained encoders and the head-only inference peak for RGB.

The final-layer probe reduced the preregistered primary metric AbsRel by 61.3% versus RGB and 37.5% versus the
training-scale-frozen VGGT teacher, while running 8.30× faster than VGGT with 14.44× lower encoder peak memory and 14.46×
fewer total parameters. The teacher remained much stronger after per-frame scale alignment (0.03262 versus 0.06784), so
the student does not replace VGGT where scale-agnostic geometric fidelity is the priority.

The multi-layer average had 4.44% worse primary AbsRel than the final-layer probe at identical latency, memory, channels,
and trainable capacity. It improved RMSE by 5.92%, aligned AbsRel by 2.34%, Delta-1, and calibrated NLL, with lower seed
variation on several secondary metrics. Because the registered promotion metric is raw metric AbsRel, this is mixed
evidence—not evidence of a statistically established regression—and is insufficient to promote the multi-layer average.
The final layer becomes the default; learned layer fusion is a future ablation rather than a silent replacement.

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
- final receipt job `29587134` passed 75 tests and sustained CUDA; preflight `29587173` passed and uploaded
  [W&B run `ybzskoo5`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/ybzskoo5); formal job `29587255` then completed
  in 3m42s with all ten result rows, nine checkpoints, and zero failures.

## Historical single-host GPU blocker

Immediately before the first training launch, the A100 had passed earlier Phase-2 quality runs. At Phase-2b launch time:

- `nvidia-smi` returned `Unable to determine the device handle ... Unknown Error`;
- `lspci` reported the GA100 at PCI revision `ff`;
- PyTorch reported `cuda_available=false` and zero devices;
- the runner rejected execution before model loading or W&B initialization.

This was the previously diagnosed PCIe link-loss condition, not a model, token, dependency, or training failure. No CPU
fallback was allowed because it would invalidate the required GPU runtime/memory comparison. The relaunch now uses the
approved Slurm partitions, with CUDA health rechecked inside every allocation. The content-bound Slurm relaunch completed.

## Verification and artifact audit

- Ruff: pass across `jepa4d`, `scripts`, and `slurm`;
- mypy: pass across 89 Phase-2b-relevant source files;
- JEPA-4D pytest: 75 passed, one expected local-model skip, one third-party deprecation warning;
- Slurm `--test-only`: test, preflight, and formal entrypoints accepted by the scheduler;
- credential scan: supplied W&B and Hugging Face tokens absent from tracked files;
- login environment: Python 3.12 dependency check passes;
- assets: V-JEPA, pinned compatibility source, VGGT-1B, and TUM archive/extraction identities verified;
- comparison runner and formal wrappers fail closed without CUDA, passing receipts, or online W&B;
- final Slurm state: `COMPLETED`, exit `0:0`, 10 comparison rows, nine unique checkpoint hashes, zero failures;
- internal and external postflight: pass, with every manifested file's byte count and SHA-256 verified;
- local comparison SHA-256: `01214c8b577441c93af8f293feff2c32160492bad002e1be3a8e66828fa52b80`;
- W&B API audit: run state `finished`, result `success`, 10 variants, zero failures, nine checkpoint files;
- W&B artifact `ikh4ptrb-phase2b-comparison:v0`: local/remote digest
  `2d7bb637cccbb190478758342740b774`, with the comparison, manifest, and self-contained report present.

## Artifacts and dashboard guide

- local comparison: `outputs/jepa4d_phase2b/tum_rgbd_v1_f600e6f/comparison.json`;
- self-contained diagnostics: `outputs/jepa4d_phase2b/tum_rgbd_v1_f600e6f/geometry_student_report.html`;
- immutable file hashes: `outputs/jepa4d_phase2b/tum_rgbd_v1_f600e6f/artifact_manifest.json`;
- backend receipt: `outputs/jepa4d_phase2b/tum_rgbd_v1_f600e6f/wandb_artifact_receipt.json`;
- W&B comparison tables show seed metrics/runtime/parameters; training-history panels use independent per-seed epoch axes;
  diagnostic media show held-out prediction, target, relative error, validation prediction/error, and uncertainty;
  accuracy-latency and summary panels support the final-versus-multilayer decision.

## Reproduction / rerun command

From the repository root, prepare the reusable environment and verified public assets on the login node, then submit the
GPU gates and formal training as dependent Slurm jobs:

```bash
bash slurm/prepare_phase2b_login.sh
test_job=$(sbatch --parsable slurm/phase2b_tests.sbatch)
preflight_job=$(sbatch --parsable --dependency="afterok:${test_job}" slurm/phase2b_preflight.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" slurm/phase2b_train.sbatch)
```

## Follow-up

1. Keep the final-layer probe as the default compact geometry student for downstream integration.
2. Test learned or validation-selected layer fusion rather than the fixed standardized average.
3. Repeat on independent sequences/scenes before claiming generalization or statistical superiority.
4. Evaluate whether the teacher's substantially better aligned fidelity matters more than student speed for each consumer.
5. Carry the same content-bound Slurm/W&B/postflight contract into later model-quality gates.

## Claim boundary

This is a real training and held-out comparison on one named TUM RGB-D sequence, not a cross-dataset benchmark. Three
optimization seeds quantify probe variation but are not independent scenes. The result supports the reported operating
point, the final-layer default decision, and measured A100 latency/memory; it must not be presented as broad geometry
generalization or proof that fixed multi-layer fusion is universally worse.
