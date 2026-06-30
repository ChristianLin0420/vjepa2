# Phase 2g proposal — quality-first detached scale and identifiable camera conditioning

## Status

**Proposed, not preregistered, and not authorized for execution. Queued behind the common validation-foundation gate.**

This is the geometry-stage proposal under the [Phase 2 validation plan](../validation/PHASE2_GEOMETRY.md), not the
project-wide experiment plan. Before preregistration, complete the applicable Wave-A registry, dataset-role, metric, and
artifact tasks in the [systematic validation plan](../VALIDATION_PLAN.md).

### Engineering preflight completed outside Phase 2g-A

A separately authorized, synthetic no-data instrumentation smoke completed on 2026-06-30. It exercised three optimizer
steps for each M0-M3 arm, the zero-tolerance gradient firewall, checkpoint reload, allocated-GPU telemetry, local receipts,
and online W&B. The first attempt, job `29662324` at commit `062b975`, failed on bare-UUID `nvidia-smi` selection; commit
`dff5f6a` fixed and tested the selector, and replacement job `29662431` completed. The append-only result is
[Phase 2g synthetic training-instrumentation smoke](2026-06-30-phase2g-training-instrumentation-smoke.md).

This preflight used generated tensors and no dataset, cache, archive, pretrained model, or checkpoint input. Its raw
receipt label is `integration-smoke`; the experiment ledger maps it to the canonical `contract-only` evidence level. It is
not part of the Phase 2g-A scientific DAG, does not answer H1-H5, clears none of the data/legal/preregistration items below,
and does not authorize real-data training or DIODE access.

This plan records the revised research priority after Phase 2f. Phase 2f remains an honest latency-first result: its
current eager M1-M3 implementations failed the frozen head-runtime gate and therefore were not trained. It does not answer
whether those architectures improve geometry, metric scale, camera causality, or uncertainty.

Phase 2g separates scientific architecture validation from later implementation optimization:

```text
Phase 2g-A: train and evaluate every healthy architecture for quality and mechanism
                              |
                              v
                 freeze one development survivor
                              |
                              v
Phase 2g-B: optimize that survivor for latency with prediction/metric parity
                              |
                              v
separate preregistration: one-shot external confirmation, only if still justified
```

No speed, parameter-count, throughput, or memory threshold may eliminate an arm in Phase 2g-A. Resource metrics remain
mandatory diagnostics so the later optimization target is known.

## 1. Questions and hypotheses

| ID | Question | Falsifiable hypothesis |
|---|---|---|
| H1 | Does separating shape and global-scale gradients improve cross-family metric depth? | M1 improves equal-family raw AbsRel and absolute log-scale error over M0 without meaningful aligned-shape regression. |
| H2 | Does canonical K add useful, sample-specific camera information? | Trained M2 improves over M1 and updated K beats stale, wrong, and permuted K under paired controls. |
| H3 | Does a bounded coarse scale field correct residual spatial scale structure? | M3 improves over M2 and the same checkpoint performs worse when its field is zeroed. |
| H4 | Do factorized uncertainty components remain useful and calibrated? | Validation-calibrated NLL improves or remains close while AUSE and coverage do not regress materially. |
| H5 | Are gains consistent across camera families rather than driven by one family? | A candidate improves raw AbsRel in at least three of four held-out families and avoids a large regression in any family. |

The primary objective is architecture quality. Phase 2g-A will still complete if every candidate is slow, provided training,
evaluation, controls, artifacts, and integrity checks are valid.

## 2. Claim boundary

Phase 2g-A may support only development claims on the frozen SUN RGB-D camera-family protocol. Camera family remains
confounded with data source, scene composition, and capture conditions. Four families limit population inference; three
seeds measure optimizer variation, not independent-camera uncertainty.

A winner is called a **development survivor**, not externally confirmed, deployment-ready, or universally metric. DIODE
remains sealed throughout this phase. Speed becomes a hard decision only in a subsequent, separately frozen optimization
experiment after a scientific survivor exists.

## 3. Data protocol

### 3.1 Expanded balanced development set

Select exactly 1,024 samples from each existing SUN RGB-D family:

- Kinect v1 (`kv1`);
- Kinect v2 (`kv2`);
- Intel RealSense (`realsense`);
- Asus Xtion (`xtion`).

Selection is target-value-blind after a mechanical validity screen:

1. sort every immutable family sample ID and freeze a deterministic rank order before depth decode;
2. decode only enough target metadata to mark whether the source has at least 100 finite pixels in `0.1 < depth < 10.0`;
3. take the first 1,024 mechanically eligible IDs in the frozen rank order;
4. persist every rejected ID and the boolean failure reason, but no depth value, histogram, aggregate statistic, or preview;
5. abort the protocol if any family has fewer than 1,024 eligible samples rather than changing the threshold or selecting
   replacements after model results.

The manifest records source paths, selected and rejected IDs, family, RGB/K identities, bytes, hashes, and the frozen
eligibility rule. Model predictions, error, scene difficulty, and target-depth distributions cannot influence inclusion.

Scaling the measured Phase 2f cache by eight gives an expected 4,096-sample cache of about 51 GiB (55 GB decimal): roughly
33.8 GiB frozen features, 17.1 GiB inputs, and 0.11 GiB targets. Store immutable per-family/per-profile shards so workers
read only their authorized split rather than loading the full cache.

### 3.2 Four frozen rotations

| Rotation | Training families | Validation family | Frozen held-out development family |
|---|---|---|---|
| R0 | kv1 + xtion | RealSense | kv2 |
| R1 | xtion + RealSense | kv2 | kv1 |
| R2 | RealSense + kv2 | kv1 | xtion |
| R3 | kv2 + kv1 | xtion | RealSense |

Training/tuning workers receive only training and validation targets. They do not receive the held-out-family target shard
or path. After checkpoint selection and hashing, separate immutable evaluation jobs receive the corresponding held-out
shard. Development metrics cannot influence an already frozen checkpoint.

### 3.3 Preprocessing retained from Phase 2f

- frozen V-JEPA 2.1 ViT-B final-layer 24x24 feature grid;
- 384x384 center-square RGB for V-JEPA and 96x96 RGB where the scale path requires it;
- two training views per source: center-square and centered 0.85 crop;
- validation and ordinary held-out metrics on the center-square view;
- target validity from finite `0.1 < depth < 10.0` metres;
- mask-weighted 24x24 target reduction with minimum valid mass 0.25;
- half-pixel `align_corners=False` crop/resize intrinsics convention;
- feature normalization fitted independently on each rotation's two training families only;
- eight deterministic P0-P7 profiles for the paired camera suite.

The primary aggregate averages frames within each held-out family, averages seeds within each rotation, and then gives all
four families equal weight.

## 4. Model matrix and causal contrasts

| Arm | Parameters | Shape path | Scale/camera path | Primary contrast |
|---|---:|---|---|---|
| M0 monolithic | 86,402 | Historical monolithic decoder | Entangled metric log depth, no K | Operational/reference system |
| M1 detached global | 92,820 | Centered shape | Detached pooled V-JEPA global scale | M1 vs M0 tests the full separated training system |
| M2 canonical K | 92,916 | Same as M1 | M1 scale plus four normalized K values | M2 vs M1 isolates canonical camera conditioning |
| M3 coarse field | 93,685 | Same as M1 | M2 global scale plus bounded zero-mean 4x4 field | M3 vs M2 isolates the coarse field |

Parameter counts are strict architecture identity checks, not qualification limits. M0 uses the historical monolithic loss
while M1-M3 share the separated objective. Therefore M0-versus-M1 measures architecture plus training-system change;
M1-versus-M2 and M2-versus-M3 are cleaner mechanism contrasts.

Retain the Phase 2f gradient firewall and exact loss weights. Do not introduce teacher loss, RGB-only scale, a dense-ray
path, or a learned camera prompt in this phase.

## 5. Fair hyperparameter selection

Every arm and rotation receives the same prespecified learning-rate search:

| Setting | Value |
|---|---|
| Learning rates | `{5e-4, 1e-3, 2e-3}` |
| Tuning seed | `260629` |
| Tuning epochs | 20 |
| Optimizer | AdamW |
| Weight decay | `1e-4` |
| Batch | 8 source groups, two views per group |
| Gradient clipping | 5.0 |
| Scheduler / early stopping | None |

Tuning may eliminate a run only for health/integrity failure:

- any NaN/Inf in loss, gradients, output, or metric;
- forbidden cross-branch gradient not bitwise zero;
- expected allowed branch receives no gradient;
- model remains bitwise identical to initialization;
- strict checkpoint/state/output reload mismatch;
- mean total objective over the final 10% of steps is not lower than the first 10%;
- schema, source, dependency, W&B, or artifact validation failure.

Do not eliminate an arm for short-run validation quality, speed, memory, or parameter count. Among healthy learning rates,
choose lowest validation raw AbsRel, then lower validation absolute log-scale error, then the lower learning rate.

With 1,024 samples in each of two training families, one epoch contains 256 source-group optimizer steps. Each tuning job
therefore executes 5,120 steps. The 48 tuning cells total 245,760 optimizer steps.

## 6. Formal training and held-out evaluation

Train all `4 arms x 4 rotations x 3 seeds` for 60 epochs using the selected learning rate for that arm/rotation.

| Setting | Value |
|---|---|
| Seeds | `0,1,2` |
| Formal epochs | 60 |
| Steps per run | 15,360 |
| Formal optimizer steps | 737,280 total |
| Tuning plus formal steps | 983,040 total |
| Checkpoint selection | Lowest validation raw AbsRel, then lower scale error, then earlier epoch |

Formal training jobs cannot load held-out-family targets. Once selected checkpoints and receipts are immutable, 48
separate evaluation jobs compute held-out metrics and interventions. A failed run remains a failure; it is not converted to
an exclusion that makes an incomplete arm look stronger.

## 7. Quality metrics and aggregation

The shared definitions and cross-phase caveats are in [the metric guide](../METRICS.md). Phase 2g retains the Phase 2f
median-log-residual alignment and introduces a new schema for any added metric rather than silently reusing an old label.

### Primary and key secondary metrics

| Metric | Role | Direction |
|---|---|---:|
| Equal-family raw AbsRel | Primary architecture-quality endpoint | Lower |
| Absolute and signed log-scale error | Primary scale diagnosis | Lower absolute; signed toward zero |
| Aligned AbsRel | Shape non-inferiority | Lower |
| Raw/aligned RMSE | Large physical and shape-error diagnosis | Lower |
| Delta-1 | Threshold accuracy | Higher |
| Validation-calibrated log-depth NLL | Uncertainty magnitude | Lower |
| Per-frame/family AUSE and risk-coverage | Uncertainty ranking | Lower |
| Empirical 50/80/90/95% coverage and reliability error | Interval calibration | Closer to nominal / lower error |
| Predicted-versus-optimal scale correlation and residuals | Scale-head mechanism | Higher correlation, centered narrow residuals |
| Valid frames/pixels and failure count | Completeness | Complete / zero unexplained failures |

Fit one positive variance multiplier on validation pixels only, clip it under the frozen schema, and reuse it unchanged on
held-out evaluation. Never fit point predictions, scale correction, variance calibration, or thresholds on held-out labels.

### Variation and descriptive intervals

Persist per-frame values, per-seed family means, paired candidate-minus-M0 differences, and three-seed sample SD for each
family. Add a 100,000-resample paired hierarchical bootstrap with seed 260629:

1. preserve candidate/reference pairing;
2. average optimizer seeds within the paired unit;
3. resample the four family clusters with replacement;
4. resample frames within each selected family;
5. recompute the equal-family effect.

The interval is descriptive because there are only four families. Hard eligibility uses frozen effect sizes and family
consistency, not a population-significance claim.

## 8. Candidate quality eligibility

Compare every candidate with paired M0 under the four-family mean after seed averaging. A candidate is quality-eligible
only if all conditions hold:

| Condition | Proposed threshold |
|---|---:|
| Raw AbsRel | Candidate/M0 `<= 0.98` |
| Absolute log-scale error | Candidate/M0 `<= 0.95` |
| Aligned AbsRel | Candidate/M0 `<= 1.02` |
| Calibrated NLL | Candidate minus M0 `<= +0.02` |
| AUSE | Candidate/M0 `<= 1.02` |
| Family consistency | Raw AbsRel improves in at least 3/4 families |
| Worst-family protection | Candidate raw AbsRel `<= 1.05x` M0 in every family |
| Completeness | All 12 formal/evaluation cells, checkpoints, metrics, receipts, and uploads finite and valid |

Hierarchical architecture gates additionally enforce the hypotheses in Section 1:

| Arm | Additional comparison |
|---|---:|
| M2 | Equal-family raw AbsRel `M2/M1 <= 0.99`, in addition to its K-control gates |
| M3 | Equal-family raw AbsRel `M3/M2 <= 0.99`, in addition to its zero-field gate |

These margins prioritize depth and scale improvement while avoiding promotion on floating-point noise. They are proposed,
not frozen, until the preregistration is approved.

## 9. Mechanism gates

### 9.1 Camera conditioning for M2/M3

Exclude identity profile P0 from the quality comparison and aggregate P1-P7 within each family. Updated K must satisfy:

| Gate | Proposed threshold |
|---|---:|
| Versus stale K raw AbsRel | `updated/stale <= 0.99` |
| Versus wrong K raw AbsRel | `updated/wrong <= 0.99` |
| Versus permuted K raw AbsRel | `updated/permuted <= 0.99` |
| Family sign consistency | Updated wins in at least 3/4 families for every control |
| Distinct analytic K | Exactly 8/source |
| Permutation assignment/matrix change | 100% |
| Mean absolute prediction delta | Greater than `1e-6` metres/control |

M0 and M1 are structural negative controls: their frozen configs must declare `consumes_intrinsics=false`, contain no
camera parameters, and reject an evaluator call that attempts to pass K. The evaluator records their camera conditions as
`not_applicable_nonconsumer` rather than fabricating K-replacement predictions. Sensitivity alone is not usefulness for
M2/M3; the updated condition must improve target-scored quality.

### 9.2 Scale and field mechanisms

- M1: log learned global-scale performance versus a fixed train-median scale; report prediction/optimal-scale correlation
  and residual distributions. This is diagnostic, not an extra eligibility gate.
- M3: evaluate the same checkpoint with its scale field zeroed. Full M3 must achieve raw AbsRel `<= 0.99x` the zero-field
  intervention and improve in at least three of four families.
- Every intervention reuses the selected checkpoint. No control is retrained.

## 10. Development survivor selection

Build the eligible list only after all 48 frozen held-out evaluations finish.

1. Keep candidates satisfying quality eligibility and their mechanism gates.
2. Select lowest equal-family raw AbsRel.
3. If candidates are within 0.5% relative raw AbsRel, select lower scale error.
4. If scale error is within 1% relative, select lower calibrated NLL.
5. If still tied, use the fixed simplicity order M1, M2, M3.
6. If no candidate qualifies, retain M0 and end without a development survivor.

The selector always records `external_final_authorized=false`. Phase 2g-A cannot open an external final even when it finds
a survivor.

## 11. DIODE opacity

Phase 2g must not open, list, extract, hash-stream again, load, cache, summarize, visualize, or expose the DIODE archive path
to a worker. A metadata-only opacity job validates the prior Phase 2f sealed receipt and asserts:

- no `FRESH_FINAL_OPENED.json` exists;
- no DIODE target/feature cache exists;
- Phase 2f recorded `fresh_final_opened=false` and `archive_touched=false`;
- development loaders reject `diode`, `final`, and `external` target/path keys;
- the dependency graph gives no development job an external archive path.

A terminal guard reasserts the same conditions after selection. External confirmation requires a separate preregistration
after the development survivor and any allowed implementation optimization are frozen.

## 12. Slurm DAG and resource policy

All GPU work runs through Slurm. Login-node work is limited to code inspection, environment construction, metadata-only
staging, submission, and monitoring. Use account `edgeai_tao-ptm_image-foundation-model-clip` and partition fallback
`polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3`. Every job requests one GPU because the approved
partitions require it, and no job requests more than four hours.

The proposed graph has 11 held base submissions and 152 logical tasks:

```text
p2gq-T tests
   |-- p2gq-O DIODE metadata-only opacity audit --|
   `-- p2gq-C expanded SUN cache -----------------+--> p2gq-Q architecture/cache audit
                                                        |
                                               p2gq-H[0-47]%8 tuning/health
                                                        |
                                                   p2gq-HG selector
                                                        |
                                               p2gq-F[0-47]%8 formal train
                                                        |
                                               p2gq-V[0-47]%8 held-out/control eval
                                                        |
                                                   p2gq-S selector
                                                        |
                                               p2gq-G external-seal guard
                                                        |
                                                   p2gq-Z postflight
```

H, F, and V arrays are sequential and each uses `%8`; the maximum concurrent running allocation is therefore eight. O and
C may overlap, for a peak of two at that stage. Do not replace logical arrays with hundreds of independent `sbatch`
submissions.

| Stage | CPU | Memory | Time | Runtime constraint |
|---|---:|---:|---:|---|
| T | 16 | 160 GB | 1:30 | Full tests, CUDA health, real-model equivalence, W&B contract |
| O | 8 | 32 GB | 0:30 | Metadata-only; no external archive access |
| C | 16 | 160 GB | 4:00 | A100-SXM4-80GB; sharded cache |
| Q | 8 | 64 GB | 1:00 | Architecture/count/cache audit |
| H array | 16 | 160 GB | 4:00 each | A100-SXM4-80GB tuning cells |
| HG | 8 | 64 GB | 1:00 | Health-only LR selection |
| F array | 16 | 160 GB | 4:00 each | A100-SXM4-80GB formal training |
| V array | 16 | 160 GB | 4:00 each | A100-SXM4-80GB held-out/control evaluation |
| S | 16 | 64 GB | 2:00 | Quality/mechanism aggregation and selection |
| G | 8 | 32 GB | 0:30 | Reassert external opacity |
| Z | 16 | 64 GB | 2:00 | Strict content, scheduler, W&B, and status audit |

All submissions start held. The submitter writes one atomic dependency graph containing execution ID, clean commit,
proposal/preregistration hash, source/sbatch hashes, logical array mappings, dependencies, resources, outputs, and skip/
failure semantics before releasing jobs.

## 13. Logging, W&B, and visualization

Online W&B is mandatory under entity `crlc112358`, project `jepa4d-worldmodel`, and group
`phase2g-quality-<execution_id>`. Every logical task receives a unique semantic run name and artifact. SUCCESS is written
only after the run finishes online, its artifact uploads, and a local receipt records run/artifact identity and digest.

Do not serialize credentials, raw target arrays, large feature caches, or protected archives to W&B.

### Per-epoch/per-run logging

- every shape, scale, field, consistency, and NLL loss term;
- allowed and forbidden gradient norms by parameter group;
- learning rate, gradient clipping, parameter/update norms;
- validation raw/aligned AbsRel, signed/absolute scale error, NLL, AUSE, coverage;
- checkpoint rank and selected epoch;
- throughput, epoch time, peak allocated/reserved GPU memory;
- 15-second GPU utilization, memory, temperature, power, and clock telemetry;
- exact split/cache/config/code/dependency identities.

Speed and memory panels are descriptive in Phase 2g-A and cannot affect health or quality selection.

### Required aggregate visualizations

1. per-family/per-seed forest plots with paired M0 differences;
2. raw versus aligned AbsRel and scale-error plots;
3. signed scale residual distributions and predicted-versus-optimal scale scatter;
4. reliability/coverage and risk-coverage/AUSE curves;
5. P1-P7 updated/stale/wrong/permuted K deltas;
6. M3 full versus zero-field comparisons and field distributions;
7. fixed qualitative RGB/target/prediction/error/uncertainty/field panels;
8. loss and allowed/forbidden-gradient curves;
9. descriptive epoch time, memory, throughput, and component-resource tables;
10. provenance, failure, retry, and completeness dashboards.

Select 16 qualitative sample IDs per family by lowest SHA-256 before training. Never select a panel because its target or
error looks favorable. Persist immutable JSON, JSONL, CSV, NPZ, checkpoints, PNG, self-contained HTML, manifests, W&B
receipts, and SHA-256 identities locally.

## 14. Failure and relaunch rules

- A failed array cell blocks the dependent aggregate. Do not silently omit it.
- Infrastructure retry is allowed only when no trustworthy scientific output/receipt exists; retain original scheduler
  history and use the same logical identity where supported.
- A semantic code, data, loss, metric, split, selection, or gate change creates a new protocol/execution lineage.
- Partial tuning/formal metrics cannot select a model.
- Candidate underperformance is a valid result and does not authorize architecture or threshold changes within the run.
- No-survivor ends Phase 2g-A with M0 retained and DIODE sealed.

## 15. Decisions after Phase 2g-A

| Result | Decision |
|---|---|
| M1 wins | Detached global scale is useful without camera conditioning; optimize M1 implementation later. |
| M2 beats M1 and passes K controls | Canonical camera conditioning is causally useful; freeze M2 as the development survivor. |
| M3 beats M2 and zero-field | Spatially varying scale adds value; freeze M3 with its field constraints. |
| Candidate improves accuracy but not uncertainty | Preserve the point predictor; treat uncertainty/calibration as a separate repair before external confirmation. |
| Candidate wins only one family | Diagnose family-specific transfer; do not promote. |
| No candidate passes | Retain M0 and redirect work toward supervision/data rather than speed optimization. |

Only after a development survivor exists should Phase 2g-B optimize precomputation, operation fusion, compilation, or CUDA
Graph capture. It must demonstrate prediction parity and no material metric regression. Only after that winner is frozen
should a separate one-shot DIODE confirmation be considered.

## 16. Items to freeze before preregistration

The synthetic instrumentation preflight closes only the bounded optimizer/firewall/checkpoint/logging wiring check. Every
scientific and governance item below remains open:

- applicable validation-registry and consumed-test-ledger entries from the master plan;
- selected 1,024-sample family manifests and hashes;
- cache shard schema, expected sizes, and source identities;
- new metric schema for RMSE/Delta-1/reliability additions;
- exact hierarchical-bootstrap implementation and tests;
- quality and mechanism thresholds in Sections 8-9;
- all commands, resource requests, array mappings, and expected outputs;
- fixed qualitative IDs and visualization schema;
- exact clean execution commit and environment lock;
- strict DIODE path-denial and opacity assertions;
- online W&B artifact schemas and terminal postflight checks.

Until these items are frozen and reviewed, this document remains a proposal: no Phase 2g-A data, cache, tuning, formal
training, held-out evaluation, selection, or external job may be submitted. A synthetic engineering preflight is not an
exception to, substitute for, or partial execution of that scientific DAG.
