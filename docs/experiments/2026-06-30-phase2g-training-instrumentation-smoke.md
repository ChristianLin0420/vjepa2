# Phase 2g synthetic training-observability smoke

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-06-30-phase2g-synthetic-training-observability` |
| Stage | geometry training infrastructure |
| Status | complete; supersedes failed job `29662324` |
| Evidence level | `contract-only` (raw receipt label: `integration-smoke`) |
| Parent | Phase 2g quality-first proposal |
| Timestamp | `2026-06-30T15:53:40Z` |
| Git commit / dirty | `dff5f6a86c8713951ef8b284058a50127193f739` / false |
| Slurm | job `29662431`, `polar4`, A100-SXM4-80GB, `COMPLETED 0:0`, 31 seconds |
| W&B | [run `uy296b4i`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/uy296b4i) |

## Question and decision

- Objective: verify the M0-M3 optimization, gradient-firewall, checkpoint, GPU-telemetry, and online-W&B boundaries before
  any real-data Phase 2g training is authorized.
- Hypothesis: every arm can complete three deterministic synthetic steps with finite logs, a nonzero update, zero
  forbidden-gradient leakage, exact checkpoint reload, and a backend-confirmed online artifact.
- Success criteria: exactly 12 optimizer rows, three per arm; every declared log finite; all four checkpoints reload
  exactly; forbidden-gradient maximum exactly zero; Slurm and W&B finish successfully; no dataset, cache, target artifact,
  credential, or scientific claim crosses the boundary.
- Decision: the bounded training-observability path is ready for later authorized engineering use. This does not authorize
  or substitute for Phase 2g real-data training.

## Stage results and insights

| Stage | Implementation | Status | Evidence | Insight / decision |
|---|---|---|---|---|
| synthetic input | deterministic paired 24x24 feature/target generator | pass | `synthetic_inputs_only=true`, `dataset_or_cache_access=false` | The smoke cannot open a dataset or cache. |
| optimization | M0, M1, M2, M3 via the registered Phase 2f/2g head and loss contracts | pass | 3 steps/arm, 12 total | Every architecture path updates parameters. |
| gradient safety | exact objective-to-owner firewall | pass | maximum forbidden norm `0.0` for all arms | Detached branch ownership holds in the CUDA run. |
| persistence | schema-bound checkpoint save and exact in-run reload | pass | four exact reloads and SHA-256 identities | The generated state survives the checkpoint boundary exactly. |
| observability | local JSONL, one-second Slurm telemetry, per-step `nvidia-smi`, online W&B | pass | 199 populated W&B history keys and 19 scheduler telemetry rows | Loss, gradient, optimizer, throughput, memory, power, temperature, utilization, and clocks are visible. |

## Reproduction configuration

```bash
export JEPA4D_WANDB_ENTITY=crlc112358
export JEPA4D_WANDB_PROJECT=jepa4d-worldmodel
export JEPA4D_MAX_STEPS=3
bash slurm/submit_phase2g_training_smoke.sh
```

Authentication was read from the submitter-owned mode-0600 home credential. No API key was exported to Slurm, persisted,
or printed. The job used seed `260630`, AdamW at `1e-3`, weight decay `1e-4`, gradient clip `5.0`, two synthetic source
groups with two views each, and the exact M0-M3 parameter counts `86,402 / 92,820 / 92,916 / 93,685`.

## W&B dashboard reading guide

| Namespace | What it records | Observed result |
|---|---|---|
| `arms/M*/loss/*` | all monolithic or factorized shape/scale/field/NLL terms | 21 distinct loss keys across the arm matrix |
| `arms/M*/gradients/*` | owned and forbidden objective-to-parameter norms | 14 distinct gradient keys; forbidden maximum `0.0` |
| `arms/M*/optimizer/*` | LR, clip threshold/coefficient, unclipped norm, update norm | finite and populated for every step |
| `arms/M*/timing/*`, `throughput/*` | synchronized step time and throughput | finite for all 12 rows |
| `arms/M*/memory/*`, `nvidia_smi/*` | CUDA allocation plus utilization, memory, temperature, power, and clocks | complete seven-field hardware sample per row |

The backend audit found state `finished`, 12 history rows, 199 populated history keys (`43/51/51/51` under M0/M1/M2/M3),
and one versioned artifact with digest `04ba2ee1fadf71c4117693c3a1d2ac3d`.

## Numerical results

These values are instrumentation diagnostics on generated tensors, not model-quality measurements.

| Arm | Steps | Final total objective | Final update norm | Forbidden-gradient max | Reload |
|---|---:|---:|---:|---:|---|
| M0 | 3 | `-0.0734323` | `0.212194` | `0.0` | exact |
| M1 | 3 | `0.412849` | `0.199613` | `0.0` | exact |
| M2 | 3 | `0.411206` | `0.199002` | `0.0` | exact |
| M3 | 3 | `0.436053` | `0.197663` | `0.0` | exact |

The A100 per-step telemetry ranged from 537-597 MiB used, 36 C, 94.76-95.89 W, and 0-4% instantaneous utilization;
the job is deliberately too short for a throughput or efficiency claim.

## Artifacts

| Artifact | Identity | Purpose |
|---|---|---|
| `outputs/phase2g-training-smoke/p2g-smoke-exec-dff5f6a8-20260630T155328Z-5d976eef/training_receipt.json` | file SHA-256 `f3f88d2f9cbae5da8d9510fed8aea35a1216bebf43fdd6dbdc402bb736d565dc` | runner aggregate receipt; not independently postflight-validated |
| same output, `steps.jsonl` | file SHA-256 `c3576d73564da7a12d85142e0f499488621a28734d6b81c54d173a154b63c176` | 12 sanitized scalar rows |
| same output, `wandb_receipt.json` | file SHA-256 `4990492d93dd0db0b5a14fab1f0c130bbbd3531612dddd93ee1f3ebf0990d22c` | backend-confirmed run and artifact identity |
| M0 checkpoint | SHA-256 `70915e8fd2ddd1e373c559cc6b5046788ea6f08c55a8c1adc8ddcc85f4ed7880` | exact reload evidence |
| M1 checkpoint | SHA-256 `6116568e6cefc63d934f49ebe1c4dc6c3e43acf5fb1083418102b0dd2ac03c12` | exact reload evidence |
| M2 checkpoint | SHA-256 `e36999a93206a5d0d6f2e8bf076d1e8f78eccc825c2560184643e35477e8058b` | exact reload evidence |
| M3 checkpoint | SHA-256 `50c6df993667ec146ae93055cee63ac40bddd2f4d26ea0f431b459da2449940c` | exact reload evidence |
| W&B artifact | `phase2g-instrumentation-smoke-...-uy296b4i:v0`, digest `04ba2ee1fadf71c4117693c3a1d2ac3d` | durable steps and checkpoints |

## Failures and supersession

Job `29662324` at commit `062b975` / W&B run
[`jelzi967`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/jelzi967) failed before persisting an optimizer row because
PyTorch returned the allocated UUID without the `GPU-` prefix required by `nvidia-smi --id`. It wrote no runner or Slurm
success marker and is not evidence. Commit `dff5f6a` normalized the UUID, added the exact regression test, and job
`29662431` / W&B run `uy296b4i` supersedes the failed attempt.

## Claim boundary and limitations

This run proves only that the synthetic optimizer and observability contracts execute together on one A100. The raw
`integration-smoke` receipt label is mapped to the repository's canonical `contract-only` evidence level. It contains no
real image, dataset, feature cache, depth target, pretrained model or input checkpoint, validation split, checkpoint
selection, or architecture comparison. It cannot support accuracy, convergence, calibration, transfer, latency,
efficiency, architecture-quality, promotion, or deployment claims, and it does not authorize any Phase 2g-A job.

The current Slurm wrapper does not independently revalidate the runner receipt in a strict terminal postflight, and the
W&B artifact was uploaded before the final local training receipt existed, so it does not bind that receipt. Those are
formal evidence-pipeline blockers, not failures of this bounded synthetic contract smoke. Formal SUN training remains
policy-blocked and DIODE remains sealed.

## Next experiments

| Priority | Experiment | Promotion criterion | Dependency |
|---|---|---|---|
| P0 | governed consumed-TUM official-mini regression | strict terminal Slurm/W&B/postflight receipt | completed separately as job `29662550` |
| P1 | portable Phase 2b/2c Wave A manifests and governed Phase 2c runtime | target-free membership and isolation receipts pass | registry migration |
| P2 | freeze legal, selector, invalid-depth, metrics, and DAG prerequisites for Phase 2g | reviewed hash-bound preregistration | SUN policy blockers resolved |
