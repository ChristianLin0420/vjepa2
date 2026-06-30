# Phase 2g governed synthetic training-observability smoke

## Experiment metadata

| Field | Governed v2 result |
|---|---|
| Experiment ID | `p2g-smoke-exec-11bef3c1-20260630T175107Z-798001b1` |
| Stage | geometry training infrastructure |
| Status | complete; governed v2 terminal pass, with v1 history retained below |
| Evidence level | `contract-only` (raw receipt label: `integration-smoke`) |
| Parent | [Phase 2g quality-first proposal](2026-06-29-phase2g-quality-first-proposal.md) |
| Started | `2026-06-30T17:51:07Z` |
| Git commit / worktree | `11bef3c1b4439acc452941ca34aedec4ffaa7300` / clean |
| Slurm | job `29672691`, `polar4`, A100-SXM4-80GB, `COMPLETED 0:0`, `00:01:22` |
| Allocation | 1 node, 1 task, 8 CPUs, 32 GiB, 1 GPU; 30-minute limit |
| W&B | [run `qsoqhk22`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/qsoqhk22), state `finished` |
| Terminal identity | canonical payload SHA-256 `9a328e96d826f98409d94d245483729e4eb2d165a90acc1457276132e5ebcd5f` |

## Question and decision

- Objective: verify the M0-M3 optimizer, gradient firewall, checkpoint, GPU telemetry, online W&B, independent postflight,
  and terminal evidence boundaries before any real-data Phase 2g training is authorized.
- Success criteria: exactly 12 finite optimizer rows and three per arm; four exact reloads; zero forbidden-gradient
  leakage; exact scheduler, commit, configuration, checkpoint, and artifact identities; backend download-and-hash
  verification; terminal W&B publication; and `SUCCESS` written only after a content-addressed terminal pass.
- Decision: the governed synthetic training-observability contract passes. The v2 run closes the v1 evidence-binding
  weakness, but it does not test architecture quality and does not authorize Phase 2g-A formal training.

## Evidence flow

```text
Slurm allocation 29672691: exact resources + clean commit
                         |
                         v
M0-M3: 12/12 steps + 4 exact reloads + forbidden-gradient max 0.0
                         |
                         v
preliminary W&B artifact: 5/5 files downloaded and SHA-256 matched
                         |
                         v
postflight 03245ce1...: scheduler/config/metrics/checkpoints/backend PASS
                         |
                         v
terminal W&B artifact 8ef71093...: 8 bound evidence files, run qsoqhk22
                         |
                         v
terminal 9a328e96...: PASS -> SUCCESS marker
```

## Stage results and insights

| Stage | Status | Exact evidence | Interpretation |
|---|---|---|---|
| allocation governance | pass | approved account/partition; 1 node, 1 task, 8 CPUs, 32 GiB, 1 GPU; no array | The job ran inside the frozen short diagnostic envelope. |
| synthetic input | pass | `synthetic_inputs_only=true`, `dataset_or_cache_access=false` | No dataset, cache, pretrained model, or input checkpoint entered the run. |
| optimization | pass | M0-M3 each ran 3 steps; 12/12 total; every final update norm is nonzero | Every registered architecture path executes and updates parameters. |
| gradient safety | pass | all firewall flags true; maximum forbidden norm exactly `0.0` | Objective ownership holds for this controlled CUDA fixture. |
| persistence | pass | 4/4 exact in-run reloads; checkpoint identities match receipts and W&B | Generated state survives the checkpoint boundary exactly. |
| online observability | pass | 12 W&B history rows, 223 populated keys, state `finished` | Objectives, diagnostics, gradients, optimizer state, architecture, and resources are inspectable. |
| independent postflight | pass | preliminary artifact creator/version/digest/manifest and all 5 downloaded files match local SHA-256 identities | Local evidence is bound to the actual online backend artifact. |
| terminal publication | pass | finalized 8-file W&B artifact plus content-addressed postflight, final-W&B, and terminal receipts | `SUCCESS` represents the complete governed evidence chain, not runner exit alone. |

## Frozen configuration and runtime identity

| Item | Value |
|---|---|
| synthetic fixture | seed `260630`; 2 source groups × 2 views; 24×24; input dimension 768 |
| optimizer | AdamW, LR `1e-3`, weight decay `1e-4`, clip threshold `5.0` |
| determinism | CPU/CUDA seeded by arm and step; bitwise reproducibility explicitly not claimed |
| parameter counts M0/M1/M2/M3 | `86,402 / 92,820 / 92,916 / 93,685` |
| configuration SHA-256 | `0821c563c7c412bac1189de807ffe4ff3668652f8188af3a3797b13241d188a9` |
| runner SHA-256 | `3beb2a5159cac655cfe4d21dfb5755fed93c902e788177ec0d0448563b13ebb6` |
| model module SHA-256 | `39f7e8925b6d36ebf6cdfbe0d003444a59ec7f94a4c5b4ea2b7202d4bd062de7` |
| training module SHA-256 | `5195b5890e63866926ee0694b9c24564a81220253ab7f0ee6eccc854b94cc2a6` |
| environment | Python 3.12.13; PyTorch 2.7.1+cu118; cuDNN 9.1; W&B 0.28.0 |

Authentication came from the submitter-owned mode-0600 home credential. The Slurm environment reported no exported W&B
key or Hugging Face token, and neither credential was persisted or printed.

## W&B dashboard reading guide

| Namespace | What it records | Governed v2 observation |
|---|---|---|
| `arms/M*/objective/*` | optimized monolithic or factorized terms | finite for all 12 rows; optimized terms are separated from diagnostics |
| `arms/M*/diagnostic/*` | non-optimized joint-NLL, scale, and field checks | diagnostic-only labels prevent false loss interpretation |
| `arms/M*/gradients/*` | owned and forbidden objective-to-parameter norms | forbidden maximum `0.0` |
| `arms/M*/optimizer/*` | LR, clip threshold/coefficient, pre/post-clip norm, update norm | finite; 0/12 steps clipped because the maximum pre-clip norm was `1.017025` |
| `arms/M*/architecture/*` | component and total parameter counts | exact registered counts on every arm |
| `arms/M*/resource_diagnostic/*` | timing, throughput, CUDA memory, and `nvidia-smi` | present for every row; explicitly diagnostic-only |
| `validation/*` summary | terminal postflight state and bound receipt identities | `postflight-pass`, 12 steps, firewall pass, synthetic-only |

The live backend audit found 12 history rows and 223 populated history keys: `49/57/57/57` under M0/M1/M2/M3 plus
`arm`, `arm_step`, and `global_step`. The finalized summary has 236 populated keys. The run owns two versioned artifacts:
the preliminary 5-file artifact (`4726b0d3ff0ed7d66f2f66bf5957c3e3`) and terminal 8-file artifact
(`8ef7109367e39968533158b1437063a2`).

## Numerical diagnostics

These are generated-fixture instrumentation diagnostics, not model-quality measurements and not comparable architecture
scores.

| Arm | Objective step 0 → 2 | Pre-clip gradient range | Clipped | Final update norm | Forbidden max | Reload |
|---|---:|---:|---:|---:|---:|---|
| M0 | `0.135865 → -0.073432` | `0.894082–1.017025` | 0/3 | `0.212194` | `0.0` | exact |
| M1 | `0.459260 → 0.412848` | `0.276735–0.332128` | 0/3 | `0.199613` | `0.0` | exact |
| M2 | `0.440670 → 0.411206` | `0.262455–0.355341` | 0/3 | `0.199003` | `0.0` | exact |
| M3 | `0.489768 → 0.436053` | `0.290321–0.362946` | 0/3 | `0.197663` | `0.0` | exact |

Training-runner elapsed time was `18.809` seconds. Per-step A100 telemetry ranged over 537-597 MiB used, 37 °C,
90.59-90.78 W, and 0-3% instantaneous utilization; process peak allocation/reservation was 45,878,272/71,303,168 bytes.
The run is deliberately too short and synthetic for throughput, latency, memory-efficiency, or deployment claims.

## Governed v2 artifacts

Base path: `outputs/phase2g-training-smoke/p2g-smoke-exec-11bef3c1-20260630T175107Z-798001b1`

| Artifact | Verified identity | Role |
|---|---|---|
| `training_receipt.json` | file SHA-256 `00c694773a968d2ecbed400263308929989890f517796f7bd056eaeb960772b6` | pending-postflight runner aggregate |
| `steps.jsonl` | file SHA-256 `8d7fd6148d6efd562f7bb8ca4f723041b6a4f6d4337c8424ca1790f65dc716a3` | 12 sanitized scalar rows |
| `wandb_receipt.json` | file SHA-256 `d663a4dc7901424743bd64b5f8211fcd5733a9fd0cad60740d4c564dd2794d9d` | preliminary online identity and 5-file manifest |
| M0 checkpoint | file SHA-256 `c360b3fa4283149f0c4ec2893740e10880a2c37bb8051fa78c5e688d68e598a4` | exact reload evidence |
| M1 checkpoint | file SHA-256 `c5ffd8cdcfb84ef4c965070c1d315a6f61cd49a2b141028f4998ea2aecedd817` | exact reload evidence |
| M2 checkpoint | file SHA-256 `921206fff15eff681c1317a8b7569d7d47fa7447aa56df38683dbe255ee25ae0` | exact reload evidence |
| M3 checkpoint | file SHA-256 `c7a186a06fc98c059a556ec63d763a8e5cb5e73ce7fbaffcb8d625d8f2d71dd9` | exact reload evidence |
| postflight receipt | canonical payload SHA-256 `03245ce1ca00509f39c0d6d8d6df2f3b0d7d94a94693299d43f51c8c4f15549b` | independent local/backend verification pass |
| final-W&B receipt | canonical payload SHA-256 `da7931f200d75a9fd2978d42e848c1984b0b2ff631802879d8cb7232b28ed627` | finalized terminal artifact identity |
| terminal receipt | canonical payload SHA-256 `9a328e96d826f98409d94d245483729e4eb2d165a90acc1457276132e5ebcd5f` | binds training, preliminary, postflight, final W&B, run, and terminal artifact |

## Execution lineage and supersession

| Version | Commit | Slurm / W&B | Outcome | Evidence decision |
|---|---|---|---|---|
| first attempt | `062b975` | job `29662324` / [run `jelzi967`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/jelzi967) | failed before an optimizer row because bare UUID selection was invalid for `nvidia-smi --id` | no success marker; not evidence |
| v1 smoke | `dff5f6a` | job `29662431` / [run `uy296b4i`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/uy296b4i) | 12 steps, four reloads, zero forbidden gradients; no independent terminal postflight | valid bounded runner smoke; superseded as strongest evidence |
| governed v2 | `11bef3c` | job `29672691` / [run `qsoqhk22`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/qsoqhk22) | exact allocation, postflight, backend round trip, terminal publication, and terminal receipt pass | current strongest `contract-only` evidence |

The v1 immutable output remains at
`outputs/phase2g-training-smoke/p2g-smoke-exec-dff5f6a8-20260630T155328Z-5d976eef`. Its training, steps, and W&B
receipt file hashes are respectively `f3f88d2f9cbae5da8d9510fed8aea35a1216bebf43fdd6dbdc402bb736d565dc`,
`c3576d73564da7a12d85142e0f499488621a28734d6b81c54d173a154b63c176`, and
`4990492d93dd0db0b5a14fab1f0c130bbbd3531612dddd93ee1f3ebf0990d22c`.

## Claim boundary and remaining gates

The raw `integration-smoke` receipt label maps to the repository's canonical `contract-only` level. Governed v2 proves
that the bounded synthetic optimizer and evidence pipeline execute together on one A100. It used no real image, dataset,
cache, archive, depth target, pretrained model, input checkpoint, validation split, checkpoint selection, or architecture
comparison. Its objective values cannot support accuracy, convergence, calibration, transfer, causal-camera,
architecture-quality, efficiency, promotion, or deployment claims.

This run closes the earlier independent-postflight and local/W&B-binding engineering gaps for the synthetic smoke. It does
not clear Phase 2g-A's data/legal, split, manifest, metric, causal-control, preregistration, or formal-training gates. The
real run did not trigger gradient clipping, and bitwise reproducibility is not claimed. Formal SUN training remains
policy-blocked, no Phase 2g-A job is authorized, and DIODE remains sealed.

## Next experiments

| Priority | Experiment | Promotion criterion | Dependency |
|---|---|---|---|
| P0 | promote target-free per-split Phase 2b/2c Wave A manifests and add equivalent governed Phase 2c runtime coverage | portable membership/isolation receipts and terminal Phase 2c evidence pass | geometry-readiness migration step 1 |
| P1 | complete SUN license review and freeze an identity-only selector that cannot receive depth paths | reviewed license records plus target-opaque selection receipt | geometry-readiness migration step 2 |
| P2 | freeze Phase 2g-A invalid-depth, metric, control, and DAG specifications | reviewed, hash-bound preregistration with DIODE opacity | portable SUN membership artifacts |
| P3 | adapt the governed terminal contract to the formal cache/training/evaluation artifact graph | strict per-task and aggregate terminal receipts without scientific-output leakage | frozen Phase 2g-A protocol |
| P4 | train M0-M3 fairly for quality and causal mechanism | all frozen health, completeness, held-out-family, and mechanism gates pass | explicit Phase 2g-A authorization |
