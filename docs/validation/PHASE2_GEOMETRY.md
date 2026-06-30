# Phase 2 geometry validation specification

## Status and decision boundary

**Status:** proposed validation plan; not preregistered and not authorized for execution.

The objective is to select a geometry architecture that recovers useful relative shape, metric scale, calibrated
uncertainty, and—where claimed—causally useful camera conditioning across sensor families. Scientific quality is evaluated
before implementation speed. Latency, throughput, memory, and component profiles remain mandatory diagnostics, but they
cannot prevent a healthy candidate from training or eliminate it before the quality winner is frozen.

This specification preserves three different data roles:

```text
SUN RGB-D: reusable development and architecture selection
TUM RGB-D: already consumed historical evidence and regression only
DIODE: sealed external confirmation; no access before a frozen winner and separate preregistration
```

## 1. Current actual evidence

### TUM RGB-D — consumed historical evidence

- The official VGGT-1B teacher was measured on eight Freiburg1 XYZ frames split into four calibration and four test
  frames. Median-aligned test AbsRel was `0.043210`, aligned RMSE `0.116805 m`, and Sim(3)-aligned ATE RMSE
  `0.016874 m`; this is one-sequence aligned evidence, not metric monocular scale.
- Phase 2b then consumed a chronological `64/16/8` Freiburg1 XYZ train/validation/test split. The frozen final-layer
  V-JEPA probe reached raw AbsRel `0.07523 +/- 0.00384`, versus RGB `0.19417` and the frozen teacher `0.12034`, and was
  selected over a fixed four-layer average.
- Phases 2c and 2d consumed Freiburg1/2/3 recordings for cross-sequence training, selection, evaluation, and causal
  diagnostics. They identified absolute scale transfer as the dominant gap.

TUM is therefore not fresh external evidence. It remains useful for deterministic regression, camera-convention tests,
and comparison with the historical pipeline, but it cannot select or externally confirm the next architecture.

### SUN RGB-D — consumed development evidence

- Phase 2e trained factorized models on sensor-blocked SUN RGB-D and opened its held-out Kinect-v2 split. The candidate
  improved raw/aligned AbsRel but worsened scale error and calibrated NLL, so it was not promoted.
- Phase 2f cached 128 samples from each of `kv1`, `kv2`, `realsense`, and `xtion`. M0 completed four held-out-family
  rotations and three seeds, with equal-run means `0.202083` raw AbsRel, `0.146821` aligned AbsRel, `0.118882` absolute
  log-scale error, `-0.781757` validation-calibrated NLL, and `0.072549` AUSE.
- M1-M3 performed zero optimizer steps because Phase 2f's frozen latency-first gate rejected their eager implementations.
  Phase 2f therefore provides no candidate-quality or camera-causality result.

SUN is a reusable **development** resource. All four families have influenced protocol design, and no SUN result may be
described as untouched external confirmation.

### DIODE — sealed and unconsumed

The Phase 2f asset audit streamed only the compressed DIODE validation archive bytes and verified size
`2,774,625,282` and SHA-256 `8e847e0923c57c221533c0040a49fc37a547af08f0a78ab235fdbf91dc362374`.
It did not list or extract the archive or load, summarize, visualize, or cache targets. No `FRESH_FINAL_OPENED.json`
exists, and no DIODE target/feature cache exists. DIODE remains sealed for a future one-shot comparison.

The promoted evidence and boundaries are recorded in [the experiment index](../experiments/INDEX.md),
[Phase 2](../experiments/2026-06-29-phase2-geometry.md),
[Phase 2b](../experiments/2026-06-29-phase2b-prepared-blocked.md),
[Phase 2e](../experiments/2026-06-29-phase2e-sunrgbd.md), and
[Phase 2f](../experiments/2026-06-29-phase2f-scale-camera.md).

## 2. Dataset roles, access, and licensing

### Dataset A — SUN RGB-D development

**Role:** training, validation, hyperparameter selection, architecture comparison, camera-family rotation, causal
camera/field interventions, and uncertainty calibration during development.

Primary source: [official SUN RGB-D project and download page](https://rgbd.cs.princeton.edu/).

The official project describes 10,335 RGB-D images captured by four sensors and explicitly notes that the collection
incorporates NYU Depth v2, Berkeley B3DO, and SUN3D data, each of which must also be cited.

Access and license caveats:

- the visible official page does not publish one blanket license covering every constituent image. Do not infer CC0 or
  another permissive license from Kaggle or other mirrors;
- before expanding the cache, record the official download identity and review the original NYU Depth v2, B3DO, SUN3D,
  SUN RGB-D annotation/toolbox, and redistribution terms applicable to the selected samples;
- raw RGB/depth, previews, and source paths remain local restricted data and are excluded from W&B artifacts;
- all previous Phase 2e/2f SUN samples and family labels are development-consumed, regardless of whether a particular
  candidate trained on them.

### Regression source R — TUM RGB-D historical evidence

**Role:** checksum-pinned regression of the teacher/student pipeline, camera frame convention, alignment, exporters, and
historical metric continuity. It cannot select or confirm the next model.

Primary source: [official TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).

Access and license caveats:

- unless a sequence states otherwise, TUM publishes benchmark data under **CC BY 4.0** and accompanying source code under
  BSD-2-Clause; retain attribution and the per-sequence exception check;
- Freiburg1/2/3 sequences used in Phases 2-2d are consumed. A new frame subset from those recordings is not independent;
- TUM RGB-D provides registered RGB/depth and trajectory data, but historical JEPA-4D depth/point results include
  target-derived scale alignment and pose includes Sim(3) alignment. They are not raw metric-scale claims.

### Dataset B — DIODE sealed external confirmation

**Role:** one-shot, cross-dataset indoor/outdoor confirmation after exactly one SUN-selected architecture and its
calibration are frozen.

Primary sources:

- [official DIODE dataset site](https://diode-dataset.org/);
- [official DIODE development toolkit](https://github.com/diode-dataset/diode-devkit).

The official release provides RGB, depth maps, validity masks, train/validation enumeration, and indoor/outdoor domains.
The official site states that the dataset and code are released under the **MIT license**.

Opacity rules:

- before external authorization, no job may receive the DIODE path; list, extract, sample, decode, load, summarize,
  visualize, or create feature/target caches; hashing the archive again is also forbidden because its identity is frozen;
- metadata-only guards may validate the prior sealed receipt and absence of an opening sentinel without touching the
  archive;
- DIODE may be opened only by a separately preregistered evaluation DAG after one survivor, code revision, checkpoint,
  calibration multiplier, metric schema, indoor/outdoor aggregation, and failure policy are immutable;
- opening is irreversible. A failed scientific result does not authorize retraining, threshold changes, or a second
  survivor comparison on DIODE.

## 3. Frozen SUN development split policy

1. Use the four registered families: Kinect v1, Kinect v2, Intel RealSense, and Asus Xtion.
2. Select the balanced manifest by immutable sample ID without looking at target values, predictions, or previews. Freeze
   the exact selection algorithm, eligible-source audit, and insufficient-valid-depth policy before decoding.
3. Preserve the four rotations:

| Rotation | Training families | Validation family | Held-out development family |
|---|---|---|---|
| `R0` | kv1 + xtion | RealSense | kv2 |
| `R1` | xtion + RealSense | kv2 | kv1 |
| `R2` | RealSense + kv2 | kv1 | xtion |
| `R3` | kv2 + kv1 | xtion | RealSense |

4. Fit feature normalization from training data only. Select hyperparameters/checkpoints and fit the positive variance
   multiplier on validation data only. Formal workers cannot receive held-out-family targets; immutable evaluation jobs
   receive them only after checkpoint hashing.
5. Run three optimizer seeds per arm/rotation. Seeds quantify optimization variability, not independent sensors.
6. Preserve explicit view/crop identity and updated intrinsics. The paired camera suite uses eight deterministic profiles
   and updated, stale, wrong, and permuted K conditions.
7. Average frames within family, seeds within rotation, and the four held-out families equally. Do not let family size or
   valid-pixel count silently dominate the primary endpoint.
8. Persist per-frame values and failures. Missing/invalid samples follow the preregistered fail/replace rule and are never
   omitted after predictions are visible.

## 4. Models, baselines, and causal contrasts

| Arm | Role | Definition | Required contrast |
|---|---|---|---|
| `M0` | operational baseline | historical monolithic metric-depth/log-variance head | candidate versus M0 |
| `M1` | detached scale | centered shape plus detached pooled-feature global scale | M1 versus M0 |
| `M2` | camera-conditioned scale | M1 plus four canonical normalized-K values | M2 versus M1 and K controls |
| `M3` | spatial correction | M2 plus bounded zero-mean 4x4 scale field | M3 versus M2 and zero-field intervention |
| `RGB` | non-JEPA baseline | RGB plus image-coordinate probe under the same split | representation-value floor |
| `VGGT` | frozen teacher/reference | official VGGT-1B, no development tuning | shape/teacher reference |

M0 versus M1 changes both architecture and training objective. M1 versus M2 and M2 versus M3 are the cleaner mechanism
contrasts. A survivor selector must enforce these hierarchical contrasts: M2 cannot support a camera-architecture claim
without improving on M1, and M3 cannot support a scale-field claim without improving on M2.

The gradient firewall, exact loss weights, field amplitude, camera transforms, optimizer search, seed policy, and
checkpoint selector are frozen before execution. Every healthy arm receives the same quality budget. A run may be excluded
only for declared health/integrity failures, never for early quality, speed, memory, or parameter count.

Same-checkpoint mechanism tests:

- M2/M3 updated K versus stale, wrong, and permuted K on nonidentity profiles;
- M0/M1 evaluation-path invariance to camera-condition changes without passing unsupported K into their model API;
- M3 full output versus zeroed scale field;
- learned global scale versus the frozen train-median scale diagnostic.

Sensitivity alone is insufficient: the correct condition must improve target-scored quality.

## 5. Metrics and aggregation

Exact formulas and cross-phase incompatibilities are in [the metric guide](../METRICS.md). Phase 2 validation reports:

### Primary quality

- equal-family raw AbsRel: metric-depth architecture endpoint;
- absolute and signed log-scale error: scale transfer and near/far bias;
- median-log-residual aligned AbsRel: relative-shape diagnosis, never metric-scale performance.

### Secondary geometry

- raw and aligned RMSE in metres;
- Delta-1/2/3 accuracy;
- valid frame/pixel counts and typed failure counts;
- TUM-only aligned point error and Sim(3)-aligned camera metrics for historical regression.

### Uncertainty

- validation-only variance multiplier, frozen before held-out evaluation;
- log-depth Gaussian NLL under one versioned convention;
- empirical 50/80/90/95% interval coverage and reliability error;
- per-frame/family AUSE and risk-coverage curves.

### Mechanisms

- predicted-versus-optimal global log-scale correlation and residual distribution;
- updated/stale/wrong/permuted-K raw-error ratios, family sign consistency, assignment/matrix change fractions, and output
  deltas;
- full-M3 versus zero-field paired effects and field amplitude/total variation.

### Descriptive resources

- trainable/total parameters, epoch and job duration, throughput, head and encoder-plus-head p50/p95 latency, peak
  allocated/reserved memory, and component profiles.

Quality and mechanism select the winner. Resource measurements become a hard gate only in a later, separately frozen
implementation-optimization experiment.

## 6. SUN development promotion gates

Proposed margins below must be frozen in the preregistration before labeled execution. Every comparison is paired with the
corresponding M0 rotation/seed and then aggregated equally across families.

### Integrity and completeness

- all arm x rotation x seed training and immutable evaluation cells complete with finite metrics, exact reload, valid
  receipts, online W&B artifacts, and no forbidden gradient;
- no held-out target influences optimization, selection, calibration, qualitative selection, or thresholds;
- TUM is regression-only and DIODE remains sealed throughout SUN development.

### Candidate quality

- candidate/M0 raw AbsRel `<= 0.98`;
- candidate/M0 absolute log-scale error `<= 0.95`;
- candidate/M0 aligned AbsRel `<= 1.02`;
- candidate minus M0 calibrated NLL `<= +0.02`;
- candidate/M0 AUSE `<= 1.02`;
- raw AbsRel improves in at least three of four families;
- no family raw AbsRel exceeds `1.05x` its paired M0 value.

### Hierarchical mechanisms

- M2 must improve the frozen primary endpoint over M1 and updated K must beat stale, wrong, and permuted K by the
  preregistered margin in at least three of four families for every control;
- M3 must improve the frozen primary endpoint over M2, and full M3 must beat its same-checkpoint zero-field intervention
  by the preregistered margin in at least three of four families;
- all eight updated K values per source are analytically distinct, permutation assignment/matrix change is 100%, and every
  required output delta exceeds numerical tolerance.

Select the lowest equal-family raw AbsRel among eligible candidates, then scale error, calibrated NLL, and the frozen
simplicity order. If none qualifies, retain M0 and do not open DIODE.

## 7. One-shot DIODE confirmation gate

External evaluation is a separate experiment, not an automatic continuation. Before opening the archive, freeze:

- exactly one SUN development architecture survivor;
- one final survivor checkpoint and one paired M0 checkpoint built by the same prespecified external-fit recipe: retrain
  each architecture once from a fixed seed on all four SUN families except a hash-selected calibration slice, use no
  DIODE signal for early stopping or selection, and fit the variance multiplier only on that SUN calibration slice;
- code/commit, environment, manifests, archive identity, preprocessing, valid-mask/depth-range rules, calibration,
  metrics, indoor/outdoor aggregation, thresholds, qualitative IDs, failure policy, and report schema;
- a selector receipt explicitly authorizing one opening and creating `FRESH_FINAL_OPENED.json` atomically at first access.

The primary external endpoint is equal-domain raw AbsRel over indoor and outdoor DIODE validation samples. The external
preregistration must freeze the exact official enumeration, capture-group/scene key if available, per-sample then
per-group/domain aggregation, bootstrap unit, and numerical margins before first access. If no reliable capture-group key
exists, the image is the independent unit and the claim is limited accordingly. Promotion beyond development requires the
survivor to improve the paired M0 raw AbsRel and absolute scale error by the frozen margins in **both** domains, avoid an
aligned-shape, NLL, AUSE, or coverage regression beyond tolerance, and complete every declared sample without unexplained
omission. Target-derived alignment is diagnostic only.

If external confirmation fails, report the result and retain the development-only claim. Do not tune on DIODE or reopen
it for another candidate under the same lineage.

## 8. W&B and durable visual logging

Every logical task has one unique online W&B run and local run/artifact receipt. Required panels include:

- total/component losses, learning rate, gradient/update norms, clipping, and allowed/forbidden branch gradients;
- validation and held-out raw/aligned metrics by family, seed, arm, and epoch;
- signed/absolute scale-residual distributions and predicted-versus-optimal scale scatter;
- reliability/coverage and risk-coverage/AUSE curves;
- P1-P7 updated/stale/wrong/permuted-K paired deltas;
- full/zero-field paired results and bounded field distributions;
- fixed-ID RGB, target, prediction, raw/aligned error, uncertainty, and field panels;
- failures, completeness, checkpoint selection, provenance, and descriptive resource/telemetry panels.

Persist versioned JSON/JSONL, CSV, NPZ, checkpoints, manifests, PNG, self-contained HTML, Slurm logs, W&B receipts, and
SHA-256 identities locally. Raw RGB/depth, protected archives, credentials, and large caches are not W&B artifacts.

## 9. Slurm and resource policy

- all GPU cache extraction, training, profiling, and evaluation runs through Slurm; login-node work is metadata-only;
- every base submission begins held until an atomic dependency graph records execution ID, clean commit,
  proposal/preregistration hash, sources, array mappings, resources, dependencies, outputs, and failure semantics;
- tuning, formal training, and evaluation arrays are sequential and use `%8`; no execution path exceeds **eight concurrent
  RUNNING allocations**;
- arrays, not hundreds of independent `sbatch` calls, represent the logical model/rotation/seed matrix;
- accounting records both the exact RUNNING-allocation peak and transient COMPLETING display rows;
- infrastructure retry is allowed only before trustworthy scientific output exists, with the same logical identity and
  retained scheduler history.

## 10. Staged TODO

### A. Data and legal freeze

- [ ] Complete constituent-license review for every selected SUN RGB-D sample.
- [ ] Freeze the balanced SUN selection algorithm and insufficient-valid-depth policy without target-driven cherry-picking.
- [ ] Revalidate consumed TUM identities as regression-only.
- [ ] Validate DIODE opacity from prior metadata without touching the archive.

### B. Protocol implementation

- [x] Exercise the M0-M3 optimizer, zero-tolerance gradient firewall, checkpoint reload, allocated-GPU telemetry, online
  W&B, independent backend download/hash verification, and content-addressed terminal publication with generated tensors
  only. The [2026-06-30 governed instrumentation smoke](../experiments/2026-06-30-phase2g-training-instrumentation-smoke.md)
  passed as job `29672691` with terminal identity `9a328e96...cd5f`. It is `contract-only` and clears no real-data, quality,
  legal, split, or preregistration gate.
- [ ] Implement the expanded sharded SUN cache and target-separation audit.
- [ ] Add raw/aligned RMSE, Delta metrics, reliability error, scale correlation, and per-frame persistence under a new
  metric schema.
- [ ] Implement M0/M1 evaluation-path invariance, M2/M3 K controls, and M3 zero-field intervention.
- [ ] Freeze model identities, learning-rate search, health rules, losses, gradients, selection margins, and Slurm DAG.
- [ ] Convert this specification into a hash-bound preregistration.

### C. Quality-first development

- [ ] Run tests, cache construction, architecture audit, and health-only tuning.
- [ ] Train every healthy M0-M3 arm across four rotations and three seeds.
- [ ] Run immutable held-out-family and mechanism evaluations.
- [ ] Complete metric, content, scheduler, W&B, license, and artifact postflight.
- [ ] Apply quality and hierarchical-mechanism gates; freeze one survivor or retain M0.

### D. After selection

- [ ] Optimize only the frozen quality winner using precomputation, fusion, compilation, or graph capture.
- [ ] Require prediction and metric parity between scientific and optimized implementations.
- [ ] If still justified, write and approve a separate one-shot DIODE preregistration.
- [ ] Freeze the all-SUN external-fit/calibration manifest, fixed seed, aggregation unit, and external effect margins.
- [ ] Open DIODE exactly once, publish the result whether positive or negative, and preserve the terminal audit.

## 11. Claim boundary

SUN results support development claims only on the named four-family protocol; camera family is confounded with source,
scene, and capture conditions. TUM results are consumed single-/few-sequence evidence and remain aligned historical
regressions. DIODE supports an external indoor/outdoor result only after the sealed one-shot protocol executes. None of
these alone establishes universal metric monocular depth, robot-safe probability calibration, full 3D reconstruction,
or real-world planning success. Three seeds measure optimizer variability, not sensor-population uncertainty. A camera
input is causally useful only when the correct camera improves target-scored quality under same-checkpoint controls.
Latency and memory remain hardware/protocol specific and do not override architecture quality during discovery.
