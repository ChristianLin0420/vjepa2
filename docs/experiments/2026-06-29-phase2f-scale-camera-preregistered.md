# Phase 2f detached metric scale and identifiable camera controls — preregistered

## Status and claim boundary

**Status:** frozen and authorized for implementation, but not yet authorized to open external-final targets.

This document freezes the Phase-2f protocol before any Phase-2f Slurm experiment or DIODE target load. A semantic change
to the data, model arms, losses, transforms, ordering, selection rule, final gate, or opacity contract creates a new
protocol version; it may not be described as the experiment preregistered here.

Phase 2f asks whether separating shape and metric-scale gradients repairs the Phase-2e scale regression, and whether a
cheap camera summary has an identifiable causal effect without the 9.36x head-latency failure of the dense-ray design.
It does not test a universal camera model. Seeds measure optimization variation, SUN RGB-D camera identity remains
confounded with source and scene composition, and the external confirmation is one labeled public dataset rather than a
hidden benchmark server.

The Phase-2f proposal contained one contradiction: it combined “open the external final once” with target-scored
updated/stale/wrong/permuted-`K` ablations on that final. This preregistration resolves it as follows:

- camera causality and latency are development-only qualification gates completed before survivor selection;
- the external final evaluates `M0` and exactly one frozen survivor once, with no target-scored `K` ablation;
- any camera claim is therefore **development-qualified only**, not externally confirmed as causal.

## Frozen evidence and assets

The design follows the completed Phase-2d and Phase-2e records. Phase 2d found negligible same-checkpoint learned-gate
effect. Phase 2e found improved shape but worse scale/calibration, a constant-`K` shuffle, and unacceptable head latency.
The Phase-2e `factorized_full_teacher` checkpoint is not a starting checkpoint and is not promoted.

The frozen encoder is the same V-JEPA 2.1 ViT-B, final-layer 24x24 grid used in Phase 2e:

| Asset | Frozen identity |
|---|---|
| V-JEPA checkpoint directory content manifest | `8c61f645d6252d619acdd15bca42f210fc27768050cc9995ebaa98cf6d779908` |
| Matched V-JEPA implementation content manifest | `2479dbf282e31821dddfea7b8f26b4aee629b762c8fad4023d1f57a7e3f55d8c` |
| SUN RGB-D archive SHA-256 | `1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c` |
| Phase-2e SUN manifest SHA-256 | `174716f4f1bd4a4b709f2a4b1a1cd4dca4fd17ef34cc543c1fc8985b75b44c92` |
| Phase-2e SUN split hash | `d1815109fa0b34dd2270f1da616d4ff65beaa41fcf437b7d75ead557a1ab75c7` |

VGGT/teacher predictions are excluded. No Phase-2e test result is confirmation evidence in Phase 2f; all four SUN RGB-D
families are development data.

## Development data and rotations

From each existing Phase-2e sensor list, select exactly 128 samples by sorted sample ID followed by deterministic
rank-midpoint selection. Selection must not decode or summarize depth. The four immutable rotations are:

| Rotation | Train families, 128 each | Validation family, 128 | Development-test family, 128 |
|---|---|---|---|
| `R0` | `kv1`, `xtion` | `realsense` | `kv2` |
| `R1` | `xtion`, `realsense` | `kv2` | `kv1` |
| `R2` | `realsense`, `kv2` | `kv1` | `xtion` |
| `R3` | `kv2`, `kv1` | `xtion` | `realsense` |

Each training sample has two views: center-square and a centered 0.85 crop. Validation and ordinary development-test
metrics use center-square only. RGB is resized to 384x384 for V-JEPA and 96x96 for the scale branch. Targets are reduced
to 24x24 by mask-weighted area interpolation: interpolate `depth * valid` and `valid` separately, divide where the
interpolated valid mass is at least 0.25, and mark all other cells invalid. Intrinsics use the half-pixel
`align_corners=False` crop/resize convention. Feature normalization is fitted independently from each rotation's two
training families and never from validation or development-test data.

The primary development aggregate first averages frames within each held-out family, then averages the four family
means. Repeated transformed views never count as independent samples.

## Frozen paired camera-causality suite

For every selected SUN frame, form a 384x384 center-square base and the following eight deterministic profiles. RGB uses
bilinear interpolation; target handling follows the mask-weighted rule above. Crop tuples are `(top,left,height,width)`
in the 384x384 base.

| Profile | Image transform |
|---|---|
| `P0` | identity |
| `P1` | crop `(29,29,326,326)`, resize to 384x384 |
| `P2` | crop `(0,29,326,326)`, resize to 384x384 |
| `P3` | crop `(58,29,326,326)`, resize to 384x384 |
| `P4` | crop `(29,0,326,326)`, resize to 384x384 |
| `P5` | crop `(29,58,326,326)`, resize to 384x384 |
| `P6` | resize to 326x326, pad 29 pixels on every side |
| `P7` | resize to 326x384, pad 29 pixels at top and bottom |

For each transformed RGB input evaluate, without retraining:

- `updated`: analytically transformed `K`;
- `stale`: base-image `K` unchanged;
- `wrong`: updated `K`, then multiply `fx` and `fy` by 1.25 and shift `(cx,cy)` by `(+38.4,-38.4)` pixels;
- `permuted`: within the eight profiles of the same source frame, replace profile indices by the fixed seed-260629
  derangement `[5,6,3,2,1,7,0,4]`.

The validator must prove that all eight analytic matrices are distinct per source matrix, the permutation is bijective,
100% of profile assignments change, and camera-conditioned output deltas exceed `1e-6` metres in mean absolute value.
Camera quality comparisons exclude identity profile `P0` and use equal-family macro raw AbsRel. This suite is development
only and is never run against DIODE targets.

## Frozen model arms

All arms consume the same frozen, normalized final-layer V-JEPA grid. Hidden width is 64, GroupNorm has eight groups,
and dense output is 24x24. `M0` must contain exactly 86,402 trainable parameters before any other arm is qualified.

| Arm | Shape path | Metric-scale path | Camera path |
|---|---|---|---|
| `M0_monolithic` | Phase-2e `monolithic_final`, unchanged | entangled dense log depth | none |
| `M1_detached_global` | centered dense shape | pooled V-JEPA: 768->8 linear, then 8->24->2 MLP | none |
| `M2_canonical_k` | same as M1 | M1 scale input plus four normalized `K` values, then 12->24->2 MLP | `[log(fx/W),log(fy/H),(cx+0.5)/W-0.5,(cy+0.5)/H-0.5]` |
| `M3_coarse_scale_field` | same as M1 | M2 global scale plus zero-mean 4x4 residual log-scale field | M2 summary only |

The two scale outputs are global log scale and global log-scale variance. `M3` obtains its 4x4 field by adaptive-average
pooling the frozen 768-channel grid to 4x4 and applying a 1x1 `768->1` projection. The field is bilinearly upsampled,
spatially mean-centered, and hard-bounded as `0.25 * tanh(raw)` log units. It has no dense ray construction. A camera
prompt and every teacher/RGB-only/bias-only arm are excluded from this experiment.

For M1-M3, predicted log depth is `z_centered + s_global + r_field`, with `r_field=0` for M1/M2. Total predictive
variance is the sum in variance space of positive dense-shape and global-scale variances. Every parameter and operation
belongs to exactly one timed component.

## Gradient firewall and exact optimization

For valid target log depth `y` and predicted centered shape `z`, define:

```text
y_centered = y - median_valid(y)
s_star     = median_valid(y - stopgrad(z))
```

`M1`-`M3` use only the following weighted objective:

| Term | Definition | Weight |
|---|---|---:|
| centered shape | Smooth-L1(`z`, `y_centered`), beta 0.1 | 1.0 |
| shape gradients | mean Smooth-L1 of valid x/y first differences | 0.25 |
| shape Gaussian NLL | centered residual with dense shape variance | 0.10 |
| global scale | Smooth-L1(`s_global`, `s_star`), beta 0.1 | 1.0 |
| scale Gaussian NLL | `s_global - s_star` with global variance | 0.10 |
| paired-view scale consistency | Smooth-L1 between the two training-view scales | 0.10 |
| M3 field target | Smooth-L1(field, mean-centered `stopgrad(y-z-s_star)`) | 0.25 |
| M3 field total variation | mean absolute x/y first differences | 0.01 |

Shape losses cannot update global-scale/field parameters. Scale/field losses cannot update shape parameters; `s_star`
always receives detached `z`. No composed metric-depth loss may bypass this firewall. Unit tests require the forbidden
autograd gradients to be absent or bitwise zero, and every training epoch logs both allowed and forbidden gradient norms.
Any forbidden norm greater than zero is a hard failure.

All formal runs use seeds `0,1,2`, AdamW, learning rate `0.002`, weight decay `1e-4`, batch size eight source groups,
gradient clipping at 5.0, and 60 epochs. There is no scheduler or early stopping. Checkpoint selection uses lowest
validation raw AbsRel, then lower validation absolute log-scale error, then the earlier epoch. A strict reload must produce
bitwise-equal CPU state dictionaries and `allclose(rtol=0, atol=0)` predictions on a fixed two-sample validation batch.

Uncertainty calibration is fit on the registered validation family only. Promotion NLL uses a single positive variance
multiplier per checkpoint, `mean(residual^2 / predicted_variance)`, clipped to `[1e-3,1e3]`. An empirical regression
calibration curve is reported for coverage analysis but cannot change point predictions or the promotion NLL. AUSE and
risk-coverage use the uncalibrated uncertainty ranking and are judged separately from NLL.

## Latency-first qualification and pilot gate

No formal 60-epoch training can start until latency aggregation passes. Twelve independent Slurm allocations benchmark
all four instantiated arms on the same immutable development-cache batch using seed 260629 initialization. Weights do not
affect the operation graph. Each job performs 30 warmups per path followed by 30 independently randomized blocks of 100
serial batch-one iterations. It saves each schedule and both synchronized host-wall and CUDA-event time.

The timed components are V-JEPA encoding, feature normalization, pooling, `K` summary, dense shape decoder, global scale
head, coarse field, composition, complete head, and complete encoder-plus-head. The hard efficiency quantity is complete
head wall time; end-to-end time is secondary. The independent job is the resampling unit. A paired cluster bootstrap with
100,000 resamples and seed 260629 produces the ratio confidence interval. All 12 jobs must use
`NVIDIA A100-SXM4-80GB`; mixed GPU models are a protocol failure, not pooled evidence.

An arm is latency-qualified only when:

- parameters are at most 95,042 (`1.10x` M0); at most 90,722 (`1.05x`) is the registered target;
- the upper 95% paired job-cluster CI for complete-head wall-time ratio is at most 1.10;
- all values are finite, all 12 replica receipts validate, and peak allocation is recorded.

Only latency-qualified arms enter a ten-epoch seed-0 pilot. M0 always runs as the reference. The pilot requires finite
losses/gradients, exact reload equality, zero forbidden cross-gradient, and no schema/hash failure. M2/M3 additionally
require updated `K` to have strictly lower paired-suite raw AbsRel than stale, wrong, and permuted `K`, plus the permutation
and output-delta checks above. A failed arm is removed before formal training; a failed M0 stops the DAG.

## Formal training and survivor selection

For each arm admitted by the pilot gate, train all 12 `(rotation, seed)` combinations. M0 runs all 12 combinations.
Disqualified predeclared jobs write a validated `skipped_not_qualified` receipt and perform no optimizer step.

An arm is eligible as the single external-final survivor only if its four-family/three-seed development mean satisfies all
of the following against paired M0 runs:

1. raw metric AbsRel is strictly lower;
2. absolute log-scale error is strictly lower;
3. aligned AbsRel is at most `1.02x` M0;
4. validation-calibrated NLL is strictly lower;
5. AUSE is no worse;
6. its frozen parameter and latency qualification remains valid;
7. for M2/M3, updated `K` strictly beats stale, wrong, and permuted `K` on the formal paired-suite aggregate;
8. every expected run, checkpoint, W&B upload, hash, and receipt is finite and complete.

Among eligible arms choose the lowest development raw AbsRel; ties within `1e-12` break by lower scale error, then lower
aligned AbsRel, then fixed order M1, M2, M3. If no arm is eligible, `final_authorized=false`: the external-final guard job
writes a skip receipt without touching the DIODE archive, and Phase 2f ends with no promoted model.

## Fresh external final and target opacity

The external final is the complete official [DIODE depth validation archive](https://diode-dataset.org/diode-dataset.github.io),
not a selected mini split. The official archive contains 612 frames: 220 indoor and 392 outdoor, across three scenes per
domain. DIODE provides RGB, metric-depth arrays, validity masks, and camera intrinsics at 1024x768 and is released under
the MIT license.

| Frozen item | Identity |
|---|---|
| Archive URL | `https://diode-dataset.s3.amazonaws.com/val.tar.gz` |
| Archive bytes | `2,774,625,282` |
| Official archive MD5 | `5c895d09201b88973c8fe4552a67dd85` |
| Devkit commit | `8b1765b7d801a5f5e2877c434ffe164e62ce8c90` |
| `diode_meta.json` SHA-256 | `ea293e1e8eb5615430353291ea9b798d8e75b6672abfd90d185069a3f53b1288` |
| `intrinsics.txt` SHA-256 | `ba3c845f0ca40173196bcdf8ce66b03be431840b077665bf85172f156b930b02` |
| Devkit `LICENSE` SHA-256 | `bb83d5a21f4b0d0dd6a024a41e4f3719cda0fcf0093b03f1c536931a7f396a58` |

The asset-seal job may stream the compressed archive to verify bytes, MD5, and SHA-256, but may not list, extract, load,
summarize, visualize, or cache any depth/mask value. It uploads only a metadata receipt. The archive path is absent from
development-cache, latency, pilot, formal, and selector environments. Their loaders must reject `diode`, `final`, or any
external-target key/path. No DIODE feature cache is built before survivor selection.

The external evaluator receives the sealed archive only after a hash-bound selector receipt names exactly one survivor.
Immediately before archive extraction it atomically creates an immutable `FRESH_FINAL_OPENED.json` containing execution,
commit, selector, Slurm job, and UTC identities. It extracts only to node-local `$SLURM_TMPDIR`, streams M0 and the survivor
through the same preprocessing, and deletes temporary tensors on exit. It evaluates all 12 rotation/seed checkpoints of
each of the two arms in this single job; this is one prespecified evaluation, not model selection.

DIODE preprocessing is center-square 768x768 followed by 384x384 RGB and 24x24 mask-weighted target reduction. The
provided validity mask, finite depth, and `depth > 0` define validity; no depth clipping is allowed. The primary final
metric is the equal-domain macro raw AbsRel: average frames within indoor and outdoor, then average the two domain means.
Per-scene, per-domain, pooled-frame, and per-checkpoint values are secondary and all are reported.

The final confirms the survivor over M0 only if the paired 12-checkpoint means satisfy all conditions:

1. equal-domain raw AbsRel is strictly lower;
2. equal-domain absolute log-scale error is strictly lower;
3. equal-domain aligned AbsRel is at most `1.02x` M0;
4. frozen validation-calibrated NLL is strictly lower;
5. AUSE is no worse;
6. every sample/checkpoint is finite and complete, with exact source and output hashes;
7. the already frozen development latency ratio and parameter ratio remain within `1.10x`.

No DIODE target-scored updated/stale/wrong/permuted-`K` result is computed. Failure of this scientific gate is a valid
completed result and does not authorize another DIODE run.

## Strict Slurm DAG

All jobs use account `edgeai_tao-ptm_image-foundation-model-clip`, partition fallback
`polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3`, one node, one task, and at most `04:00:00`. GPU jobs
request one GPU, 16 CPUs, and 160 GB RAM. The test, aggregates, selector, and postflight may request fewer resources but
remain Slurm jobs. No experiment, target decode, model inference, timing, or optimization runs on the login node.

```text
T tests
├── A sealed DIODE archive audit (compressed bytes only) ───────────────────────────────┐
└── C SUN development cache -> Q static/model audit -> L00..L11 latency replicas       │
                                              -> LA latency gate -> P0..P3 pilots       │
                                              -> PG pilot gate -> F arm×rotation×seed  │
                                              -> S development selector ────────────────┤
                                                                                       v
                                                                            E final guard/eval
                                                                                       |
                                                                            Z strict postflight
```

All jobs are initially submitted with `--hold`. The submitter writes one canonical dependency graph atomically after all
job IDs are known, including execution ID, full commit, preregistration hash, sbatch/source hashes, requested resources,
job IDs, dependencies, expected outputs, and skip semantics. It then releases every submitted job in one operation; only
`T` is initially runnable, while every other job remains dependency-blocked through `afterok`. A job must verify that the
graph contains its own job ID and exact parents before work. Downstream jobs validate parent SUCCESS, receipt, W&B, file
hashes, commit, test receipt, graph, and scientific allowlist before reading an artifact.

Formal jobs are predeclared for all `4 arms x 4 rotations x 3 seeds`; only M0 and the pilot allowlist train. This keeps the
DAG immutable while preserving fail-closed adaptive elimination. `E` opens the final only when `S` records exactly one
survivor and `final_authorized=true`; otherwise it records a no-target-open skip.

## Test receipt and execution provenance

`T` runs these commands, with the execution commit's resolved file list recorded verbatim in the receipt:

```text
bash -n slurm/phase2f_*.sbatch slurm/submit_phase2f.sh
python -m compileall -q jepa4d scripts slurm
python -m ruff format --check jepa4d scripts slurm
python -m ruff check jepa4d scripts slurm
python -m mypy jepa4d scripts/phase2f_*.py slurm/*phase2f*.py
python -m pytest jepa4d/tests -ra -q
python scripts/check_cuda.py --device 0 --stress-seconds 20 --matrix-size 4096 --allocation-mib 1024
```

Tests must cover transform/K algebra, eight-matrix diversity, permutation validity, gradient-firewall isolation,
target-path rejection, cache/shard schemas, exact reload, W&B receipt, fresh-final sentinel behavior, dependency
validation, and strict postflight. The atomic UTF-8
`outputs/phase2f-gates/<execution_id>/tests.json` records every command/argv, exit code, duration, pytest counts, source
hashes, CUDA result, environment, full clean commit, Slurm identity, dependency-graph hash, and its own schema/status.
Every downstream job requires `status=pass`, the same full commit, and the exact test-receipt SHA-256.

Every cache, latency replica/aggregate, pilot, gate, shard, selector, final, skip, and postflight receipt embeds a complete
`execution_provenance` object; a pointer to another receipt is insufficient. The object contains at least:

- schema/version, phase, execution ID, receipt kind, UTC start/end, exact argv, canonical config and config SHA-256;
- full git commit/branch, `git_status` clean flag and status-file hash, preregistration path/hash;
- test-receipt path/hash/job/commit and dependency-graph path/hash;
- Slurm job/array ID, job name, account, requested partition list, actual allocated partition, node, CPUs, memory, GPU
  UUID/name, CUDA/driver/runtime, Python/package-lock hashes;
- every source input path, bytes, SHA-256, schema, producer job, producer W&B artifact/run, and parent receipt hash;
- seeds, arm, rotation, split/sample-manifest hashes, transform version, and normalization identity where applicable;
- all output paths, bytes, SHA-256, schemas, and W&B run/artifact receipt fields.

Receipts are written atomically with explicit UTF-8 and `allow_nan=false`. Secrets, tokens, `.netrc`, and environment
values containing credentials are never serialized or printed.

## W&B, local logging, and visualizations

Online W&B is mandatory: entity `crlc112358`, project `jepa4d-worldmodel`, group
`phase2f-<execution_id>`, `WANDB_MODE=online`. Job types are exactly `tests`, `asset-seal`, `dev-cache`, `static-audit`,
`latency-replica`, `latency-aggregate`, `pilot`, `pilot-gate`, `formal`, `selection`, `external-final`, and `postflight`.
Run names include execution ID, stage, arm/rotation/seed or replica, and Slurm job ID. Each run logs commit, graph/test
hashes, config, actual hardware, parent artifact IDs, and a terminal `status`. A job succeeds only after the run is
finished online, the artifact is uploaded, and a local receipt records run ID/URL plus artifact name/version/ID/digest.

Artifacts use `phase2f-<stage>-<execution_id>` with arm/rotation/seed or replica suffixes. Raw SUN/DIODE targets, full
feature caches, model credentials, and the sealed DIODE archive are never uploaded. Hash-bound receipts, checkpoints,
metrics, reports, bounded panels, and small prediction summaries are uploaded.

The durable local root is `outputs/jepa4d_phase2f/<execution_id>/`; gates live under
`outputs/phase2f-gates/<execution_id>/`. Required local outputs are JSON, JSONL, CSV, NPZ, PNG, checkpoints, file
manifests, W&B receipts, and self-contained HTML with no CDN/external JavaScript. Training logs one explicit epoch axis per
arm/rotation/seed, every loss term, allowed/forbidden gradient norms, learning rate, validation raw/aligned/scale/NLL,
checkpoint choice, epoch time, peak memory, and 15-second GPU telemetry.

Aggregate reports contain gate cards and claim boundaries; family/domain/seed uncertainty bars; raw versus aligned and
scale-error plots; predicted versus optimal log-scale scatter and residual histograms; uncertainty reliability/coverage
and risk-coverage/AUSE; paired K-control deltas; component and tail-latency plots; parameter/memory tables; and retry/
provenance tables. Development panels use 16 sample IDs selected before training by lowest SHA-256 within each family.
External-final panels use the six lowest-hash indoor and six lowest-hash outdoor metadata IDs, fixed before target open,
and show RGB, M0/survivor prediction, target, error, uncertainty, and M3 field when applicable. No panel is selected by
error or target content.

## Postflight, failure, and relaunch rules

Each job writes into a new temporary stage directory and atomically promotes it only after local validation and W&B
upload. `SUCCESS` is written last. A scheduler `COMPLETED 0:0`, validated receipt, online W&B receipt, and SUCCESS are all
required; wrapper text alone is never authoritative. `Z` rehashes every promoted file, recursively validates every
receipt/parent/graph edge and W&B identity, checks expected executed/skipped job counts, and emits separate
`integrity_status` and `scientific_gate` fields. Integrity pass never implies scientific promotion.

Relaunches obey these rules:

1. Never overwrite or append to a failed execution. Archive it under `outputs/failed_attempts/<execution_id>-<reason>` and
   create a new execution ID, graph, W&B runs, and job names.
2. A code/config/preregistration change requires a new clean commit, full test job, and complete DAG. No cache or shard is
   reused across commits.
3. With an unchanged commit, a validated immutable upstream artifact may be reused only when all bytes/hashes match and
   the new child embeds it under `reused_parent_artifacts`; the failed node and its full downstream subtree rerun.
4. Timeout, preemption, node failure, validator failure, missing W&B upload, missing SUCCESS, NaN/Inf, reload mismatch,
   provenance mismatch, or partial output all fail closed. A scientific threshold miss records a completed negative result
   and is never “retried” with different settings.
5. The external-final job may relaunch only if `FRESH_FINAL_OPENED.json` does not exist and its receipt attests
   `fresh_final_opened=false`. Once the sentinel exists, any success or failure consumes DIODE for this protocol. A crash,
   W&B outage, or incomplete report after that point yields `external_final_consumed_inconclusive`; no DIODE rerun or
   post-hoc metric recomputation is allowed. Another confirmation would require a newly preregistered external dataset.

The only promotion label allowed is `promote_<survivor>` when both strict postflight integrity and the frozen external
scientific gate pass. Every other completed outcome is `retain_M0`, `no_survivor`, or
`external_final_consumed_inconclusive`, with the exact reason preserved.
