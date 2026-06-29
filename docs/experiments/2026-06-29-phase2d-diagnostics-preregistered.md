# Phase 2d causal attribution, calibration audit, and latency confirmation — preregistered

## Status

This protocol is frozen before any Phase-2d GPU inference. Phase 2d is diagnostic and profiling-only: it cannot
retroactively change the Phase-2c `retain_final_layer` decision, and the consumed Freiburg-3 recordings cannot support a
new generalization claim. Login-node work is limited to code, environment, public-asset staging, hashing, and CPU tests;
all real-model inference and timing run in approved Slurm allocations with online W&B.

## Questions

1. Did the learned gates themselves cause the Phase-2c quality change when the probe, normalization, and checkpoint are
   held fixed?
2. Is the cross-family gap explained mostly by a global scale, an affine correction, or a spatially varying scale field?
3. Were camera intrinsics transformed consistently through crop/resize, and which calibration facts remain undeclared?
4. Is the observed 1.1655x learned/final latency ratio structural under independent randomized/interleaved jobs?

## Frozen source

- Phase-2c formal output:
  `outputs/jepa4d_phase2c/tum_rgbd_cross_sequence_9a8f8f0cb5fb-20260629T120903Z`;
- Phase-2c W&B run: `mfquwgbw`;
- source split hash: `e5fbb372b1858d1a1783b78a3bee8948d1a6f40a6926d689729437b99ed14862`;
- source decision: `retain_final_layer`;
- Phase-2d execution commit: recorded in the completed result before submission.

All source checkpoint, normalization, split, and artifact hashes must verify before a diagnostic result is accepted.

## Experiment A — same-checkpoint fusion attribution

For every learned-fusion seed, reload the exact selected checkpoint and train-fitted normalization. Recompute Freiburg-2
validation features for uncertainty calibration and all 128 Freiburg-3 test predictions. The probe weights remain fixed.

Interventions are:

- original learned gates;
- all gates zero;
- fixed-average-equivalent gates, `g = atanh(0.75)`;
- every non-identity intermediate-layer permutation;
- every non-empty sign-flip subset.

Report equal-sequence macro and per-sequence raw/aligned AbsRel, RMSE, Delta-1, absolute log-scale error, validation-only
calibrated NLL, prediction change from the original checkpoint, per-layer residual norm relative to the final feature, and
total residual contribution. Persist full test tensors in schema `jepa4d-phase2d-depth-predictions-v1` for the audit below.

The causal comparison is original versus zero with the same probe. Fixed-equivalent gates with the learned probe test
co-adaptation; they are not equivalent to retraining the separately selected fixed-fusion probe.

## Experiment B — calibration and scale oracle audit

Recompute the center-square crop and 384/518 resize intrinsics using the half-pixel convention. Record original image/depth
dimensions, original/transformed K, asymmetric horizontal/vertical FoV, RGB-depth size agreement, and provenance for
distortion, registration, integer depth scale, and any device-specific depth correction. Missing provenance is reported as
`unknown_not_declared`, never inferred.

On the full Phase-2c test predictions, evaluate diagnostic-only target-fitted corrections:

- per-sequence scalar;
- per-image scalar;
- per-sequence affine;
- bounded low-resolution spatial scale.

These are upper-bound mechanism probes, not deployable methods. Also generate correct-K, wrong-K, and shuffled-K controls;
only a K-conditioned model may claim an executed causal effect.

## Experiment C — independent latency confirmation

Run 12 independent Slurm allocations. Each allocation uses one frozen seed-0 checkpoint set, 30 warmups per path, then 30
randomized blocks of 100 serial batch-one iterations. Every block independently shuffles path order and saves the schedule.

Deployment paths:

- final-only layer capture plus final probe;
- all-layer capture plus final probe;
- all-layer capture plus fixed-average probe;
- all-layer capture plus learned-fusion probe.

Arithmetic-only paths operate on the identical precomputed all-layer tensor and include final, fixed, learned, zero-gate,
and fixed-equivalent-gate heads. Synchronized host wall time and CUDA-event device time are both saved. Each job records GPU
UUID/name, clocks, temperature, power, utilization, software versions, raw blocks, p50/p90/p95, and peak memory telemetry.

The resampling unit for the aggregate confidence interval is the independent Slurm job, not an inner timing iteration.
The confirmation label is `within_1.10x` only if the 95% cluster-bootstrap upper bound of the learned/final paired ratio is
at most 1.10. Otherwise it is `exceeds_or_uncertain_1.10x`. Either result leaves Phase 2c unchanged.

## Logging and visualization contract

Every GPU job requires online W&B plus local JSON/CSV/NPZ and self-contained HTML. Reports must provide:

- top-level decision cards and explicit claim boundaries;
- seed/sequence drill-down rather than pooled-frame-only numbers;
- gate coefficients and residual contribution plots;
- raw versus aligned versus oracle-corrected scale views;
- camera K/FoV audit tables;
- paired latency distributions, tails, job-level ratios, and schedule-position diagnostics;
- bounded qualitative panels without growing artifacts with every frame.

W&B is the interactive comparison surface. Local files, hashes, source identities, and Slurm telemetry are the durable
record. External JavaScript/CDN dependencies are forbidden in promoted HTML.

## Completion gate

Phase 2d completes only if all three experiments finish, source hashes match, all values are finite, the full prediction
scope is explicit, all 12 latency jobs and their aggregate are present, online W&B artifacts are confirmed, and strict
postflight finds zero failures. Scientific interpretation is deferred to the completed experiment record.

## Post-execution addendum

The frozen protocol completed under commit `160207418112bb18c8d6d1c4c6c8b7082ea8d114`; all canonical jobs completed
`0:0`, strict postflight passed, and final W&B run
[q1m52wi1](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/q1m52wi1) uploaded the hash-bound aggregate. The protocol
above is unchanged. Results and interpretation are recorded in
[2026-06-29-phase2d-diagnostics.md](2026-06-29-phase2d-diagnostics.md).
