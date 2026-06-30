# Wave A governed consumed-TUM geometry official-mini

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-06-30-wave-a-tum-official-mini` |
| Stage | geometry validation infrastructure |
| Status | complete |
| Evidence level | sequence-level consumed regression / integration |
| Parent | Wave A geometry readiness implementation |
| Timestamp | `2026-06-30T15:56:10Z` |
| Git commit / dirty | `dff5f6a86c8713951ef8b284058a50127193f739` / false |
| Slurm | job `29662550`, `polar4`, A100-SXM4-80GB, `COMPLETED 0:0`, 96 seconds |
| W&B | [run `b7yzbpfo`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/b7yzbpfo) |

## Question and decision

- Objective: obtain the first real terminal Slurm + online-W&B + strict-postflight receipt for the hash-bound Wave A
  consumed-Phase-2b regression runtime.
- Hypothesis: the exact eight registered TUM Freiburg1 XYZ test frames can pass registry/ledger authorization, verified
  archive extraction, official local VGGT-1B inference, aggregate-only reporting, and content-addressed terminal checks.
- Success criteria: 1/1 cell complete; exactly eight frames; every aggregate finite; prediction finite fraction `1.0`;
  immutable dashboard, metric, execution, postflight, W&B, and terminal receipts validate; W&B terminal status is `pass`.
- Decision: the exact consumed-TUM regression/integration path has real execution evidence. This does not unblock formal
  training, promote an architecture, or provide fresh external confirmation.

## Stage results and insights

| Stage | Status | Evidence | Insight / decision |
|---|---|---|---|
| allocation | pass | one node/task/GPU, approved account/partition | Scheduler policy was enforced before data access. |
| governance | pass | access receipt `19955ba2...d82b` | Only the registered consumed Phase 2b regression operation was authorized. |
| archive/model | pass | verified Freiburg1 XYZ archive and hash-stable local VGGT tree | The run used the intended source/model identities. |
| inference/metrics | pass | exactly eight frames, 16 finite quality aggregates, two resource diagnostics | The regression surface is complete. |
| dashboard/W&B | pass | immutable dashboard plus preliminary and terminal artifacts | Aggregate-only observability works online. |
| strict postflight | pass | terminal content address `f575762f...d1d32` | The full receipt chain validates after backend finalization. |

## Reproduction configuration

```bash
export JEPA4D_TUM_ARCHIVE="$PWD/checkpoints/datasets/rgbd_dataset_freiburg1_xyz.tgz"
export JEPA4D_VGGT_CHECKPOINT="$PWD/checkpoints/phase2b_assets/VGGT-1B"
export JEPA4D_WANDB_ENTITY=crlc112358
export JEPA4D_WANDB_PROJECT=jepa4d-worldmodel
bash slurm/submit_geometry_official_mini.sh
```

The governed split is `tum-rgbd.phase2b-freiburg1-xyz-test`, with registered frame indices
`660, 677, 694, 711, 728, 745, 762, 779`. The official VGGT-1B local checkpoint ran in CUDA FP32. Authentication came from
the submitter-owned protected home credential and was absent from the job export and artifacts.

## W&B dashboard reading guide

| Namespace | What it answers | Observed result |
|---|---|---|
| `validation/gate/integrity/*` | Did completeness, finiteness, and frame accounting pass? | all three conditions `1` |
| `validation/quality/*/tum-rgbd.phase2b-freiburg1-xyz-test` | What are the aggregate consumed-regression metrics? | 16 finite values; finite fraction `1.0` |
| `validation/resource/*` | What did this exact smoke consume? | 9.069 s inference, 7.384 GiB CUDA peak |
| `validation/governance/*` | Which registry, ledger, metric, and status identities govern the run? | all hashes populated |
| `validation/postflight/status` | Did the resumed terminal publication pass? | `pass` |

The W&B backend audit found state `finished`, 46 summary keys, completeness `1.0`, zero missing cells, and two artifacts:
the preliminary governed-validation artifact digest `8fbe0759dff0408828a2a30a564ed2ed` and terminal artifact digest
`d3bc6475c9fc6d1c9eda1e99bb5fee1b`.

## Numerical results

All depth and point metrics are aligned under the declared regression protocol; pose uses one sequence-level Sim(3)
alignment. These are consumed single-sequence regression values, not metric monocular or transfer claims.

| Metric | Value |
|---|---:|
| aligned AbsRel | `0.0450468` |
| aligned RMSE | `0.150635 m` |
| aligned Delta-1 / Delta-2 / Delta-3 | `0.959026 / 0.983899 / 0.997052` |
| aligned point mean / median error | `0.064933 / 0.018469 m` |
| aligned point within 5 cm / 10 cm | `0.783636 / 0.855582` |
| Sim(3) pose ATE mean / RMSE | `0.008847 / 0.009757 m` |
| Sim(3) relative translation mean | `0.012983 m` |
| Sim(3) rotation mean | `9.82821 deg` |
| finite fraction / evaluated frames | `1.0 / 8` |
| inference runtime / CUDA peak | `9.069 s / 7.384 GiB` |

## Artifacts

All paths are under `outputs/geometry-official-mini/tum-mini-dff5f6a8-20260630T155556Z-702920a3/` and remain ignored
local execution evidence.

| Artifact | Identity | Purpose |
|---|---|---|
| `metrics/metric-gate-6056...e2e7.json` | content address `6056cb78...e2e7`; file SHA-256 `014e1c94...de12` | exact aggregate metrics and gates |
| `dashboard/validation-dashboard-8a1c...e9b9/` | generation `8a1ccaaa...e9b9`; HTML SHA-256 `54522ea2...2553` | immutable human-readable report |
| `execution/execution-receipt-d90b...bcf6.json` | content address `d90b8818...bcf6` | binds governed execution artifacts |
| `postflight/postflight-b7a2...aeaa.json` | content address `b7a20194...aeaa` | strict postflight receipt |
| `wandb-final/wandb-final-cf61...1861.json` | content address `cf614e66...1861` | backend-confirmed terminal upload |
| `terminal/terminal-f575...d1d32.json` | content address `f575762f...d1d32`; file SHA-256 `23789a74...8055` | terminal success root |

## Failures and supersession

No execution or postflight failure occurred. The only emitted warning is an upstream VGGT deprecated-autocast warning; it
did not change the zero exit status, finite metrics, or strict receipt validation.

## Claim boundary and limitations

This run supports aggregate regression behavior on one already-consumed TUM recording and proves the end-to-end governed
runtime, dashboard, W&B, and postflight integration. It does not support fresh transfer, external validation, model
selection, architecture promotion, sample-level disclosure, raw-target publication, metric single-image scale,
deployment, or speed promotion. SUN training remains policy-blocked and DIODE remains sealed.

## Next experiments

| Priority | Experiment | Promotion criterion | Dependency |
|---|---|---|---|
| P0 | target-free Phase 2b and Phase 2c Wave A manifests | selected/rejected IDs and isolation receipts validate | registry migration |
| P1 | governed Phase 2c consumed-regression runtime | terminal hash chain equivalent to this smoke | portable Phase 2c manifest |
| P2 | SUN Phase 2g preregistration | legal, selector, invalid-depth, target separation, metrics, and DAG all frozen | remaining Wave A blockers |
