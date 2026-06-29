# Phase 2c cross-sequence geometry and learned fusion — completed on Slurm

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase2c-cross-sequence-fusion-9a8f8f0cb5fb-20260629T120903Z` |
| Stage | geometry student |
| Status | complete; learned fusion not promoted |
| Evidence level | sequence-level training on two held-out recordings |
| Parent | [frozen Phase 2c preregistration](2026-06-29-phase2c-cross-sequence-preregistered.md) |
| Timestamp | `2026-06-29T12:21:45Z` |
| Git commit / dirty | `9a8f8f0cb5fbe8aa55b609845d9204df885f0d95` / clean |
| Dataset split hash | `e5fbb372b1858d1a1783b78a3bee8948d1a6f40a6926d689729437b99ed14862` |
| Hardware | NVIDIA A100-SXM4-80GB, Slurm `polar4` |
| W&B | [formal run `mfquwgbw`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/mfquwgbw) |
| Decision | Retain `vjepa_final`; the candidate failed only the preregistered latency condition. |

## Question and decision

- Objective: test whether the Phase-2b final-layer student transfers from Freiburg-1 training through Freiburg-2
  validation to two held-out Freiburg-3 recordings, and whether three learned residual layer gates improve it
  without losing its efficiency.
- Hypothesis: learned fusion will lower the equal-weight two-sequence macro metric AbsRel, avoid a greater-than-5%
  regression on either sequence, and remain within 1.10× final-layer latency and memory.
- Success criteria: all six frozen promotion conditions in `promotion_gate.json` must pass.
- Decision: keep the final layer as the formal default. Learned fusion improved quality on both sequence means and passed
  memory/integrity gates, but its measured mean latency was 1.1655× final-only and exceeded the 1.10× limit.

## Stage results and insights

| Stage | Implementation | Status | Inputs | Outputs | Evidence | Insight / decision |
|---|---|---|---|---|---|---|
| input | exact TUM cross-sequence bundle | pass | two Freiburg-1 train, one Freiburg-2 validation, two Freiburg-3 test sequences | 128/64/128 frames | full archives, extracted files, associations, and split hash verified | No frame replacement or path aliasing occurred. |
| features | frozen V-JEPA 2.1 ViT-B and VGGT-1B | pass | center-cropped RGB | final/layer-2/5/8 tokens and teacher depth | finite tensors; chunk-size checks; exact model hashes | Cross-family absolute scale, rather than aligned shape, is the dominant geometry failure. |
| optimization | shared probe plus optional three gates | pass | train-only normalized features and frozen teacher scale | 12 checkpoints and 720 epoch rows | three seeds × four learned variants × 60 epochs | Candidate coefficients stayed close to final-only and were not a stable layer hierarchy. |
| evaluation | equal-weight held-out sequence macro | pass | two Freiburg-3 recordings | 13 result rows, 26 sequence rows, 1,664 frame rows | internal and external postflight both passed | RGB and fixed averaging remain serious baselines on this shift. |
| promotion | six-condition operational gate | fail candidate / retain reference | final versus learned fusion | `retain_final_layer` | five conditions pass; latency fails | Do not override the registered rule because quality improved. |

## Numerical results

Values are held-out means ± sample standard deviation over three optimization seeds. The frozen teacher has one row.
Lower is better except Delta-1. Only the three V-JEPA latency values share the preregistered co-resident batch-1 policy;
teacher and RGB timing are fallback diagnostics and are not used by the promotion rule.

| Variant | Macro metric AbsRel ↓ | Aligned AbsRel ↓ | RMSE m ↓ | Delta-1 ↑ | Abs log-scale error ↓ | Calibrated NLL ↓ | V-JEPA E2E ms/frame ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGT-1B teacher | 0.44141 | **0.03816** | 1.08076 | 0.10581 | 0.59980 | — | — |
| RGB+XY probe | **0.40425 ± 0.00966** | 0.25928 ± 0.01399 | 1.20065 ± 0.02904 | **0.20535 ± 0.01385** | **0.45958 ± 0.03236** | **0.26763** | — |
| V-JEPA final | 0.43807 ± 0.01982 | **0.15962 ± 0.00378** | 1.22101 ± 0.03312 | 0.15257 ± 0.02082 | 0.59697 ± 0.04740 | 1.59251 | **35.95064** |
| V-JEPA fixed four-layer mean | **0.41054 ± 0.00563** | 0.16588 ± 0.00413 | 1.19783 ± 0.01097 | 0.15324 ± 0.00589 | **0.52320 ± 0.01331** | **1.15737** | 46.56554 |
| V-JEPA learned residual fusion | 0.41801 ± 0.02285 | 0.16047 ± 0.00587 | **1.17878 ± 0.05055** | **0.16263 ± 0.01400** | 0.55636 ± 0.04754 | 2.14915 | 41.90130 |

The learned candidate lowered final-layer macro AbsRel by 4.58% and RMSE by 3.46%, but aligned AbsRel was 0.53%
worse and absolute log-scale error was 6.80% better. The primary gain is therefore mainly improved metric-scale transfer,
not a clear improvement in relative depth shape.

The result also changes the baseline story. RGB has the best raw primary mean, but its aligned AbsRel is 62.4% worse
than final-layer V-JEPA. Fixed averaging is the best and most seed-stable V-JEPA primary result, 6.28% below final and
1.82% below learned fusion, but it gives up aligned fidelity and measured latency. Neither baseline can be dismissed.

### Held-out sequence means

| Variant | Freiburg-3 long office AbsRel ↓ | Freiburg-3 structure/texture far AbsRel ↓ |
|---|---:|---:|
| VGGT-1B teacher | 0.37170 | 0.51112 |
| RGB+XY probe | 0.35468 ± 0.00973 | **0.45383 ± 0.01736** |
| V-JEPA final | 0.32003 ± 0.02190 | 0.55611 ± 0.02257 |
| V-JEPA fixed mean | 0.32761 ± 0.00341 | 0.49347 ± 0.01131 |
| V-JEPA learned fusion | **0.30955 ± 0.01431** | 0.52646 ± 0.03820 |

Learned fusion improved the registered final baseline by 3.27% on long office and 5.33% on structure/texture far, so
the per-sequence guard passed. The far recording nevertheless remains a scale-transfer failure: learned raw/aligned
AbsRel is 0.52646/0.14256 and raw/aligned Delta-1 is 0.00016/0.80898. The teacher shows the same qualitative split—very
strong aligned geometry and weak metric transfer—so scene and camera-family effects remain confounded.

### Frozen promotion decision

| Condition | Observation | Result |
|---|---:|---|
| Candidate primary strictly lower | 0.41801 versus 0.43807; -4.58% | pass |
| No sequence regresses by more than 5% | -3.27% and -5.33% | pass |
| Latency at most 1.10× final | 41.90130 / 35.95064 = **1.16552×** | **fail** |
| Peak inference memory at most 1.10× final | 0.42459 / 0.41871 = 1.01404× | pass |
| All results finite, valid, and checkpointed | 13 rows, 12 exact checkpoints | pass |
| Zero failures | 0 | pass |

The latency repetitions were noisy: final ranged from 33.36 to 45.49 ms and learned fusion from 34.03 to 66.84 ms.
The measurements still satisfy the frozen protocol and therefore drive the operational decision. A future profiling-only
confirmation must be preregistered and randomized/interleaved; it cannot retroactively turn this result into a promotion.

## Fusion and uncertainty diagnostics

| Seed | Final coefficient | Layer 2 | Layer 5 | Layer 8 | Test macro AbsRel |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.98295 | +0.00973 | +0.01328 | -0.00595 | 0.42899 |
| 1 | 0.97568 | +0.00977 | +0.00888 | +0.00567 | 0.43329 |
| 2 | 1.00147 | +0.00246 | -0.00292 | -0.00101 | 0.39174 |

The candidate stayed near final-only, far from the 0.25-per-layer fixed mean, and layer-5/8 signs were inconsistent.
This supports only a small residual-perturbation effect on these recordings, not a stable learned hierarchy preference.

Validation-fitted variance multipliers remained greater than one, indicating under-dispersed raw uncertainty. Calibration
reduced final NLL from 33.4043 to 1.5925 and learned NLL from 32.1261 to 2.1492, but learned calibrated NLL was 35.0%
worse than final and 85.7% worse than the fixed mean. On the learned candidate it was 0.1012 for long office versus
4.1971 for structure/texture far. Better depth accuracy did not produce better transferable uncertainty.

## Reproduction configuration

The login node staged the Python 3.12 environment and fully verified public assets. GPU health, preflight, training, and
profiling ran only in Slurm. No credential is stored in the command or artifacts.

```bash
export JEPA4D_REPO_ROOT="$PWD"
export JEPA4D_DATASET_PARENT="$PWD/checkpoints/datasets"
export JEPA4D_MANIFEST="$PWD/jepa4d/config/benchmarks/manifests/tum_rgbd_phase2c_cross_sequence_v1.yaml"
export JEPA4D_TEST_REPORT="$PWD/outputs/phase2c-gates/9a8f8f0cb5fb-20260629T120903Z/tests.json"
export JEPA4D_PREFLIGHT_REPORT="$PWD/outputs/phase2c-gates/9a8f8f0cb5fb-20260629T120903Z/preflight.json"
export JEPA4D_OUTPUT_DIR="$PWD/outputs/jepa4d_phase2c/tum_rgbd_cross_sequence_9a8f8f0cb5fb-20260629T120903Z"
test_job=$(sbatch --parsable slurm/phase2c_tests.sbatch)
preflight_job=$(sbatch --parsable --dependency="afterok:${test_job}" slurm/phase2c_preflight.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${preflight_job}" slurm/phase2c_train.sbatch)
```

The final chain used jobs `29590017` (full tests), `29590019` (static/type shard), `29590020` (protocol/adversarial
shard), `29590022` (real-model preflight), and `29590023` (formal training/postflight). All completed with exit `0:0`.

## W&B dashboard reading guide

| Panel / namespace | What it answers | Observed result | Action |
|---|---|---|---|
| `training/<variant>/seed_<n>/...` | Did every loss, gradient, validation score, and fusion gate remain finite through 60 epochs? | 720 complete rows; zero failed seeds | Keep checkpoint selection validation-only. |
| seed and per-sequence comparison tables | Is an aggregate hiding a sequence or seed collapse? | Candidate improves both sequence means but has higher seed SD than fixed averaging. | Carry all baselines to fresh sequences. |
| fusion coefficients | Did the candidate learn a stable nontrivial mixture? | Coefficients remain tiny and signs vary. | Do not claim a learned hierarchy. |
| accuracy/latency/memory panels | Does quality survive the registered efficiency gate? | Quality and memory pass; latency fails at 1.1655×. | Retain final by rule. |
| held-out depth/error/uncertainty media | Are failures predominantly shape, scale, or calibration? | Far-sequence aligned geometry is much better than raw metric depth; calibrated uncertainty still shifts. | Target scale transfer and richer calibration. |

## Artifacts

| Path / artifact | Type | Checksum / version | Purpose |
|---|---|---|---|
| `outputs/jepa4d_phase2c/tum_rgbd_cross_sequence_9a8f8f0cb5fb-20260629T120903Z/comparison.json` | comparison JSON | `ba9ede9ded475fa178d8cc3d00230bd7d35fa214d8fef61b6d87f7bac6a1f8ce` | canonical result rows and metrics |
| `.../promotion_gate.json` | decision JSON | `fde3a2efceaa34abdf56199c9062f86fd92ce8c9192b3f4c0d5fea2571d1808f` | independently recomputable decision |
| `.../geometry_student_report.html` | self-contained interactive report | `ddc96804930ce7b45e2bca85b4f5b16b6776190b53a929664f4d5b310a0634a8` | sequence, seed, fusion, runtime, and qualitative diagnostics |
| `.../artifact_manifest.json` | complete file manifest | `3e06c11452c8bdf8ea2564f30afdf083ca89000edfa13775c9dab9d5eebb4918` | byte and SHA-256 audit for 59 files |
| `mfquwgbw-phase2c-comparison:v0` | W&B artifact | digest `6125d213c8a20c9d600751a4afc3e52c` | backend-confirmed immutable snapshot |

The W&B API independently reports run `mfquwgbw` as `finished`, decision `retain_final_layer`, and the same artifact
name, version, and digest. The final external postflight reports 13 rows, 12 checkpoints, 26 sequence rows, 1,664 frame
rows, zero failures, and zero errors.

## Failures and supersession

The first formal execution, job `29589613` / W&B `cajhcmbz`, completed optimization and artifact generation but exited
failed because the postflight looked for the literal text `https://cdn.plot.ly`. Plotly's fully inlined JavaScript contains
that string as an unused topojson default even when the HTML has no external script tag. The run was not promoted.

Commit `9a8f8f0` changed the validator to reject actual external `<script src=...>` dependencies, added acceptance coverage
for the inline Plotly bundle and a real-CDN rejection test, reran 111 tests, and launched a completely fresh receipt,
preflight, output directory, W&B run, and training job. No checkpoint or completion state from the failed run was reused.

## Claim boundary and limitations

- Supported: on the two named Freiburg-3 recordings, learned fusion lowered the three-seed mean primary metric and both
  sequence means, but exceeded the frozen latency allowance and was not promoted.
- Not supported: population-level or cross-dataset superiority, statistical significance, separation of camera-family
  from scene shift, or a stable preferred intermediate layer.
- The 64 frames per recording are temporally correlated. Frame-level tests or bootstrap intervals would be
  pseudoreplication; three seeds measure optimization variation, not scene variation.
- The two test sequences are now consumed for these design claims and must not be reused as fresh confirmation data.
- Unknown overlap with frozen encoder pretraining data remains a limitation.

## Next experiments

| Priority | Experiment | Uncertainty reduced | Promotion criterion | Dependency |
|---|---|---|---|---|
| P0 | Same-checkpoint fusion attribution: learned gates, zero gates, and fixed-equivalent gates | Did fusion itself help, or did joint probe training/checkpoint selection create the gain? | Learned gates improve the same probe/checkpoint under an immutable evaluation path | Existing checkpoints; no retraining or new quality claim |
| P0 | Fresh rotated-family or external-sequence comparison of RGB, final, fixed, and learned variants | Does the ranking survive independent scenes and camera families? | Lower macro metric on multiple independent held-out recordings without sequence collapse | New licensed split; no reuse of Freiburg-3 claims |
| P0 | Preregistered profiling-only confirmation with randomized/interleaved order and more repetitions | Is the observed 1.1655× candidate latency structural or measurement noise? | Stable confidence interval and predefined efficiency decision | Frozen checkpoints; decision remains unchanged until complete |
| P1 | Intrinsics/scale-aware head and richer validation-only uncertainty calibration | Can scale and coverage transfer improve without sacrificing aligned shape? | Raw metric, scale error, NLL, and coverage improve on fresh cameras | Independent train/validation/test camera families |
