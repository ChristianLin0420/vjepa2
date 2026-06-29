# Phase 2b geometry distillation — prepared, GPU execution blocked

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase2b-jepa-geometry-distillation-v1` |
| Stage / status | `geometry student / implementation ready, execution blocked` |
| Evidence level | `implementation-only` |
| Dataset manifest | `tum-rgbd-fr1-xyz-phase2b` version `1.0.0` |
| Intended hardware | NVIDIA A100 80 GB PCIe |
| W&B | Not initialized; no run or partial metrics exist. |
| Decision | Preserve the protocol and resume only after stable CUDA recovery. |

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

## GPU blocker

Immediately before the first training launch, the A100 had passed earlier Phase-2 quality runs. At Phase-2b launch time:

- `nvidia-smi` returned `Unable to determine the device handle ... Unknown Error`;
- `lspci` reported the GA100 at PCI revision `ff`;
- PyTorch reported `cuda_available=false` and zero devices;
- the runner rejected execution before model loading or W&B initialization.

This is the previously diagnosed PCIe link-loss condition, not a model, token, dependency, or training failure. No CPU
fallback was allowed because it would invalidate the required GPU runtime/memory comparison. No result was promoted.

## Verification before commit

- Ruff: pass;
- mypy: pass across 80 source files;
- JEPA-4D pytest: 57 passed, one third-party deprecation warning;
- credential scan: supplied W&B and Hugging Face tokens absent from tracked files;
- comparison runner exits before side effects when CUDA is unavailable.

## Resume command

After the host restores the device and `python scripts/check_cuda.py` passes under sustained load:

```bash
python scripts/run_phase2b_geometry_distillation.py \
  --dataset-root /path/to/rgbd_dataset_freiburg1_xyz \
  --archive /path/to/rgbd_dataset_freiburg1_xyz.tgz \
  --manifest jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml \
  --output outputs/jepa4d_phase2b/tum_rgbd_v1 \
  --device cuda:0 --epochs 60 --wandb \
  --wandb-project jepa4d-worldmodel \
  --run-name phase2b-jepa-geometry-distillation-v1
```

## Remaining work

1. Restore/reboot the host and require stable CUDA/NVML plus sustained allocation and compute.
2. Execute all nine learned runs (three variants × three seeds) and the VGGT teacher on the pinned split.
3. Verify zero missing seeds, finite predictions, checkpoint hashes, failure records, and local/W&B artifact parity.
4. Download the finished W&B summary/config/metadata into the audit workspace.
5. Add numerical comparison tables and per-variant insights to this record, `INDEX.md`, and `INSIGHTS.md`.
6. Mark Phase 2b complete only if the accuracy/runtime/memory gate is evaluated; record a negative outcome honestly.

## Claim boundary

The committed state proves that the comparison implementation and reporting contracts pass tests. It contains no Phase-2b
model-quality result. The one-sequence protocol will remain sequence-level evidence even after execution and must not be
presented as cross-dataset generalization.
