# Phase 2e factorized shape/scale geometry on fresh sensor families — preregistered

## Status

The protocol below is frozen before any Phase-2e model feature extraction, optimization, validation selection, or test
evaluation. Public assets and the selected-file manifest were downloaded, decoded, hashed, and audited on the login node.
All real-model work must run in approved Slurm allocations with a clean, test-receipted commit and online W&B.

Phase 2e is a new experiment. It does not reuse the consumed Freiburg-3 sequences as confirmation data.

## Question

Can explicit factorization of relative depth shape and metric scale improve cross-sensor monocular metric depth, while
preserving aligned geometry, uncertainty quality, and compact-probe efficiency? Does known camera information have a
causal effect under wrong-K and shuffled-K controls?

## Official dataset and frozen split

Dataset: official SUN RGB-D V1 from Princeton. The source contains four RGB-D sensor families and requires citation of the
SUN RGB-D, NYU Depth v2, Berkeley B3DO, and SUN3D papers. The official archive contains no explicit license text found by
the preparation audit, so it is retained for internal research and is not redistributed.

- archive URL: `https://rgbd.cs.princeton.edu/data/SUNRGBD.zip`;
- archive bytes: `6,885,481,608`;
- archive SHA-256: `1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c`;
- selected split hash: `d1815109fa0b34dd2270f1da616d4ff65beaa41fcf437b7d75ead557a1ab75c7`;
- manifest: `jepa4d/config/benchmarks/manifests/sun_rgbd_phase2e_sensor_blocked_v1.yaml`.
- manifest SHA-256: `174716f4f1bd4a4b709f2a4b1a1cd4dca4fd17ef34cc543c1fc8985b75b44c92`.

The sensor-blocked roles are immutable:

| Role | Sensor | Samples | Use |
|---|---|---:|---|
| Training | Kinect v1 | 192 | optimization only |
| Training | Asus Xtion | 192 | optimization only; equal sensor weighting |
| Validation | Intel RealSense | 128 | checkpoint selection and uncertainty calibration |
| Test | Kinect v2 | 128 | one final evaluation after every shard is complete |

Selection uses sorted leaf identity and deterministic rank-midpoint quantiles. It does not inspect target values. RGB,
encoded depth, and intrinsics are file-hashed for every selected sample. Split paths, group IDs, and sample IDs do not
overlap. Test depth may be decoded before training only for integrity/schema validation; no test metric or model-dependent
selection is allowed before final evaluation.

SUN RGB-D depth is decoded exactly as the official toolbox:

```text
bitor(bitshift(raw, -3), bitshift(raw, 13)) / 1000 metres
```

The official 8 m clamp is explicitly enabled. Zero depth is invalid.

## Feature-cache boundary

A single real-model Slurm job builds two physically separate hash-bound files:

- `train_validation_cache.pt`: the only cache accessible to pilot/formal training;
- `test_cache.pt`: accessible only to the final evaluator.

V-JEPA 2.1 ViT-B is frozen. Final-layer 24x24 grids are normalized using training samples only. Each training sample has
two paired views: center-square and a deterministic 0.85 crop. Validation/test use only center-square. RGB is cached at
96x96 for the small scale branch; metric target depth is 24x24. Intrinsics are updated exactly through crop/resize using
the half-pixel convention.

Official VGGT-1B BF16 produces a spatially centered log-depth teacher only for the two training views. No VGGT metric scale
is fitted, and VGGT never sees validation or test targets in this experiment.

## Model family

The factorized prediction is exactly:

```text
log_depth(x, y) = centered_shape(x, y) + global_log_scale
```

The shape branch consumes frozen dense V-JEPA features and, where registered, normalized camera rays. The scale branch can
consume pooled V-JEPA, a tiny RGB encoder, normalized intrinsics, and a ray summary. Every probe uses hidden width 64 and
remains approximately 0.1M trainable parameters.

Pre-run architecture consistency audit: before any Phase-2e optimization or held-out evaluation, the original
16-dimensional pooled-V-JEPA projection and width-32 scale MLP were found to make the fixed candidate 1.1770x the
reference parameter count, so the registered 1.10x resource gate could never pass. The scale branch was therefore frozen
at an 8-dimensional pooled-V-JEPA projection and width 24. This gives 95,003 candidate versus 86,402 reference parameters
(1.0995x), preserving hidden width 64 and making the gate informative. No data metric or training result informed this
preflight correction.

Eight variants are frozen:

1. `monolithic_final` — non-factorized final-layer reference;
2. `factorized_bias` — factorized with one learned global scale bias;
3. `factorized_vjepa` — pooled V-JEPA scale;
4. `factorized_rgb` — tiny RGB scale branch;
5. `factorized_vjepa_rgb` — pooled V-JEPA plus RGB scale;
6. `factorized_vjepa_k` — V-JEPA plus known rays/intrinsics/ray summary;
7. `factorized_full` — V-JEPA + RGB + known rays/intrinsics;
8. `factorized_full_teacher` — full factorization plus centered VGGT shape distillation; fixed promotion candidate.

The fixed candidate is chosen before validation. Validation rankings of the remaining variants are explanatory ablations,
not a replacement candidate.

## Optimization and parallel execution

Every variant uses seeds 0/1/2, AdamW (`lr=0.002`, `weight_decay=1e-4`), gradient clipping at 5, batch size 8 groups, and
60 epochs. A two-epoch pilot of the reference and candidate must pass first. Formal variants are divided across four
parallel Slurm shards; each shard receives only the train/validation cache.

The base geometry loss contains heteroscedastic log-depth NLL, scale-invariant residual, and spatial-gradient terms.
Factorized variants additionally receive direct global-log-scale supervision, centered GT-shape supervision, paired-view
scale consistency, and—only for the registered teacher variant—centered VGGT shape distillation. Checkpoints are selected
only by minimum validation raw AbsRel, then strictly reloaded with exact prediction equality.

After every shard, the final evaluator verifies all 24 checkpoints and W&B receipts before opening `test_cache.pt`.

## Metrics and causal controls

Primary metric: mean raw metric AbsRel over the 128 untouched Kinect-v2 samples. Secondary metrics include raw/aligned
RMSE and Delta-1, per-image aligned AbsRel, absolute log-scale error, raw/calibrated log-depth NLL, reliability/coverage,
AUSE/risk-coverage, predicted-versus-target global scale, parameter count, and synchronized head latency.

Every K-conditioned checkpoint is evaluated three ways without retraining:

- correct K;
- deterministic wrong focal/principal-point K;
- deterministic shuffled K across test samples.

Correct K must improve the camera-conditioned candidate over both negative controls before camera metadata is interpreted
as causal evidence.

## Frozen promotion gate

`factorized_full_teacher` is promoted over `monolithic_final` only if all conditions hold across the three-seed means:

1. raw metric AbsRel is strictly lower;
2. absolute log-scale error is strictly lower;
3. aligned AbsRel does not regress by more than 2% relative;
4. validation-calibrated test NLL is strictly lower;
5. correct-K raw AbsRel is lower than both wrong-K and shuffled-K controls;
6. head latency and trainable parameters are each no more than 1.10x the reference;
7. all 24 runs, exact checkpoints, metrics, controls, W&B uploads, and postflight checks are finite and complete with zero
   recorded failures.

This is an operational decision, not a statistical-superiority or universal camera-generalization claim. Seeds quantify
optimization variation; they are not independent sensors. SUN RGB-D sensor identity remains confounded with source and
scene composition.

## Logging and visualization contract

Online W&B and durable local files are both mandatory. Training logs independent epoch axes, every loss component, gradient
norms, validation raw/aligned/scale/NLL metrics, checkpoint selection, timing, and GPU telemetry. Final reports include:

- top-level gate cards and claim boundaries;
- variant/seed rankings and uncertainty bars;
- raw-versus-aligned and scale-error comparisons;
- predicted-versus-target scale plots;
- correct/wrong/shuffled K controls;
- uncertainty reliability and risk-coverage curves;
- bounded prediction/target/error/uncertainty examples;
- parameter/latency comparisons;
- manifests, hashes, checkpoints, JSON/CSV/NPZ, self-contained HTML, and W&B artifact receipts.

The promoted record must identify the exact execution commit and job dependency graph. No external script/CDN dependency is
allowed in final HTML.

## Post-execution addendum

The final clean DAG completed under commit `547ecc1ee2d2cc40863c179b625ce7952010a94d`; all eight canonical jobs
completed `0:0`, strict postflight passed, and final W&B run
[89ugevtp](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/89ugevtp) uploaded the hash-bound evaluation. The execution
completed but the fixed scientific promotion gate failed with five passes and four failures, so
`factorized_full_teacher` is not promoted. The protocol above is unchanged. Exact results, the non-identifiable
constant-`K` shuffle, retry history, and
claim boundaries are recorded in [2026-06-29-phase2e-sunrgbd.md](2026-06-29-phase2e-sunrgbd.md).
