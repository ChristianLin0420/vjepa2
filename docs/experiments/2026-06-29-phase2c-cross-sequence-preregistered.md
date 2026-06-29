# Phase 2c cross-sequence geometry and learned fusion — preregistered

## Status

Protocol and promotion criteria are frozen before any Phase-2c model inference or optimization. Assets may be downloaded,
hashed, associated, and audited on the login node; all CUDA tests, real-model preflight, profiling, and training must run
inside the approved Slurm allocations. At preregistration time, this record contained no Phase-2c model-quality result.

Post-run update: formal Slurm job `29590023` completed with zero failures and passing internal/external postflight. The
candidate improved macro AbsRel from 0.43807 to 0.41801 and improved both held-out sequence means, but its 1.1655× latency
exceeded the frozen 1.10× ceiling. The decision is `retain_final_layer`. Full metrics, W&B/artifact identities, failure
history, limitations, and next experiments are recorded in the
[completed Phase 2c result](2026-06-29-phase2c-cross-sequence.md); the protocol below remains unchanged.

## Question

Does the Phase-2b final-layer V-JEPA geometry student generalize across unseen sequences and camera families, and can a
three-parameter learned residual layer fusion improve its primary metric without losing the measured efficiency advantage?

## Immutable data protocol

The official TUM RGB-D archives are split by complete sequence and camera family:

| Role | Camera family | Sequence | Frames |
|---|---|---|---:|
| Training | Freiburg 1 | `freiburg1_xyz` | 64 |
| Training | Freiburg 1 | `freiburg1_floor` | 64 |
| Validation | Freiburg 2 | `freiburg2_xyz` | 64 |
| Test | Freiburg 3 | `freiburg3_long_office_household` | 64 |
| Test | Freiburg 3 | `freiburg3_structure_texture_far` | 64 |

The bundle manifest pins all source URLs, exact bytes, locally verified SHA-256 identities, roles, camera families, and
selected RGB indices. RGB-depth and RGB-pose associations are independently formed by global greedy minimum timestamp
error with one-to-one ownership and a strict 20 ms bound. From the intersection of valid RGB indices, 64 frames are chosen
by deterministic rank-midpoint quantiles. The runner recomputes this selection and rejects any mismatch before loading a
model. No failed or difficult frame may be replaced after inference.

The split deliberately combines unseen scene and unseen camera-family shift. A failure cannot identify which factor is
responsible. A later rotated-family study is required to separate them.

## Variants and fairness

| Variant | Role | Trainable capacity |
|---|---|---:|
| `vggt_teacher` | frozen teacher baseline | 0 |
| `rgb_probe` | non-pretrained representation baseline | shared compact probe |
| `vjepa_final` | registered reference/default | shared compact probe |
| `vjepa_multilayer` | fixed four-layer average control | shared compact probe |
| `vjepa_learned_fusion` | promotion candidate | shared compact probe + 3 scalar gates |

All learned variants use seeds 0/1/2, 60 epochs, identical batches, optimizer/loss settings, VGGT auxiliary supervision,
and validation-only checkpoint selection. RGB and every V-JEPA layer are normalized only from the pooled Freiburg-1
training frames. The VGGT metric scale is fitted once on pooled training frames and frozen. Freiburg-3 targets are used
only for final evaluation.

The learned candidate uses standardized final features `F` and layers `I2`, `I5`, and `I8`:

```text
F + Σ_l tanh(g_l) / 3 × (I_l - F)
```

All gates initialize to zero, making the initial representation and prediction exactly equal to the paired final-layer
control while giving every gate an immediate gradient. The coefficients are signed and individually bounded to ±1/3.
The fixed four-layer average is contained in this family. Probe initialization hashes, optimizer ownership, gate curves,
best-checkpoint coefficients, strict checkpoint reload, and the three-parameter delta must all be verified.

## Metrics and decision rule

The primary metric is the equal-weight macro mean of metric AbsRel from the two test-sequence means. Frames or seeds are
not treated as independent scenes. Secondary evidence includes per-sequence metric and aligned depth metrics, absolute
log scale error, validation-fitted uncertainty NLL, Delta-1, RMSE, failures, parameters, latency, and peak inference/training
memory.

Efficiency is measured with a co-resident batch-1 encoder→train-frozen normalization→fusion/probe path. The timing
boundary starts from a preloaded `RGBInputBatch` and includes device transfer plus model preprocessing, but excludes file
decode and model loading. Every V-JEPA seed/path receives 30 warm-up iterations followed by three repetitions of 30
measured iterations; median latency and actual co-resident peak allocation drive the 10% promotion conditions. Isolated
encoder/head profiles remain diagnostic only and are not added together for promotion.

The learned fusion is promoted over `vjepa_final` only if all conditions hold:

1. its three-seed mean primary macro AbsRel is strictly lower;
2. its three-seed mean AbsRel on neither held-out sequence regresses by more than 5% relative;
3. its end-to-end latency and peak inference memory are each at most 10% above the separately profiled final-only path;
4. all three seeds finish with finite metrics, valid bounded coefficients, exact checkpoints, and zero recorded failures.

This is an operational promotion rule, not a statistical-superiority claim. Two test sequences remain insufficient for
population-level inference.

## Logging and artifacts

The formal job requires online W&B. It logs independent per-seed epoch axes, all losses and gradient norms, gate and
effective coefficient curves, per-sequence and per-frame tables, raw/aligned/scale-error comparisons, runtime/memory,
accuracy-latency, and deterministic held-out prediction/target/error/uncertainty panels. Dense diagnostics are bounded to
midpoint and worst-error panels so the artifact does not grow with every evaluated frame.

Local JSON, compact NPZ diagnostics, checkpoints, a self-contained Plotly HTML report, file manifest, completion gate,
Slurm telemetry, and the W&B backend artifact receipt remain the durable source of record. Formal completion requires a
strict local postflight, successful backend upload, and a second receipt-bound postflight.

## Claim boundary

A pass supports generalization to the two named Freiburg-3 recordings under this operating point. It does not establish
cross-dataset geometry quality, separate scene from sensor shift, prove statistical superiority, or rule out overlap with
the frozen encoders' unknown pretraining data.
