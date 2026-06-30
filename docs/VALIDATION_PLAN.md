# JEPA-4D systematic validation and experiment plan

## Status and purpose

**Living project plan. Dataset choices and gates become binding only when copied into a hash-bound stage
preregistration.**

JEPA-4D is an RGB-first substrate for robot perception, geometric belief, persistent memory, and verified planning. Its
product is a queryable, uncertainty-aware belief over a changing world: geometry, objects, identities, regions,
observations, time, task evidence, predictions, and verification state. A depth map, point cloud, feature tensor, object
list, database, or action sequence is only one intermediate output.

The project therefore cannot be validated by one geometry number or one robot success rate. It needs a stagewise pipeline
that answers two questions simultaneously:

1. Does each stage work on common labeled data under a frozen protocol?
2. Does the composed system retain those properties and improve closed-loop task behavior?

This document defines that common pipeline. Stage files define dataset-specific TODOs:

- [Phase 0 infrastructure and contracts](validation/PHASE0_INFRASTRUCTURE.md)
- [Phase 1 representation](validation/PHASE1_REPRESENTATION.md)
- [Phase 2 geometry](validation/PHASE2_GEOMETRY.md)
- [Phase 3 grounding and identity](validation/PHASE3_GROUNDING_IDENTITY.md)
- [Phase 4 persistent memory](validation/PHASE4_MEMORY.md)
- [Phase 5 dynamics and planning](validation/PHASE5_DYNAMICS_PLANNING.md)
- [Phase 6 system benchmark](validation/PHASE6_SYSTEM.md)

Use [METRICS.md](METRICS.md) for formulas, aggregation, calibration, units, and cross-phase incompatibilities. Historical
results and decisions remain in the [experiment index](experiments/INDEX.md) and [insight ledger](experiments/INSIGHTS.md).

## 1. Current evidence baseline

The baseline is intentionally candid. “Real model” does not imply “real benchmark,” and a perfect fixture score does not
imply model quality.

| Stage | Strongest input actually evaluated | Strongest current evidence | Main validation gap |
|---|---|---|---|
| Phase 0 contracts | Unit and deterministic fixtures | Contract-only | No model-quality question; maintain infrastructure invariants |
| Phase 1 representation | Eight generated RGB-gradient frames through real V-JEPA 2.1 | Integration | No natural-video labeled validation set |
| Phase 2 teacher | Eight official TUM RGB-D Freiburg1 XYZ frames | Official mini subset | Aligned single-sequence geometry only |
| Phase 2b student | TUM Freiburg1 XYZ, 64/16/8 chronological split | Sequence-level training | No independent scene or dataset |
| Phase 2c cross-camera | Five TUM recordings, two train/one validation/two test | Sequence-level transfer | One dataset family; Freiburg3 test is consumed |
| Phase 2d diagnostics | Reused Phase 2c predictions/test frames | Mechanism diagnostic | No fresh generalization evidence |
| Phase 2e factorization | SUN RGB-D sensor-blocked 384 train/128 validation/128 kv2 test | Benchmark | kv2 test is consumed; factorized candidate failed |
| Phase 2f detached scale/camera | Four SUN families, 128 each, four rotations; only M0 trained | Development benchmark | M1-M3 quality and camera mechanisms untested; DIODE sealed |
| Phase 3 grounding | Generated rectangles and one unlabeled repository illustration | Integration | No detection, phrase-grounding, or mask benchmark |
| Identity | One subsampled DAVIS 2017 `dogs-scale` sequence | Sequence-level | Same-sequence tuning; no held-out sequence suite |
| Phase 4 memory | Eight deterministic mug updates | Contract-only | No perception-driven episodic or scene-memory benchmark |
| Phase 5 planning | One deterministic pick/place recovery episode | Contract-only | No learned dynamics or repeated named-simulator episodes |
| Phase 6 harness | Generated 376-byte contract asset, six stages x five repeats | Contract-only | No official per-stage adapters or system benchmark |

No DIODE performance result exists. The archive was byte-hash audited but never listed, extracted, loaded, summarized, or
scored. Preserve that state until an independently selected geometry survivor is frozen.

## 2. Minimum dataset portfolio

Every learned stage must eventually have at least two common evaluation sources or benchmark suites with explicit roles:

- **Dataset A1 — primary development benchmark:** training/validation and repeatable held-out development evaluation;
- **Dataset A2 — complementary development benchmark, when needed:** a different labeled capability that may be fitted or
  selected jointly with A1; it does not support an independent-transfer claim;
- **Dataset B — transfer/external benchmark, when that claim is sought:** architecture and decision rules are frozen
  before formal targets are opened; task-specific probe fitting is allowed only if its recipe was frozen on Dataset A;
- **optional Dataset C — stress/safety:** distribution shift, occlusion, long horizon, or rare failure modes.

Two complementary sources satisfy the minimum coverage requirement, but not the L2 cross-dataset promotion level. A
stage must use an independent Dataset B without architecture, threshold, or calibrator retuning before claiming transfer.

The initial portfolio is:

| Stage | Primary development | Second common source and role | Optional independent/stress source | Purpose |
|---|---|---|---|---|
| Representation | Something-Something V2 (A1) | EPIC-KITCHENS-100 (A2, complementary anticipation) | Ego4D (future B/C) | Temporal discrimination, egocentric anticipation, long-form transfer |
| Geometry | Existing SUN RGB-D four-family development rotations | DIODE validation, still sealed | Consumed TUM RGB-D regression suite | Cross-sensor metric depth, external indoor/outdoor transfer, pose/camera regression |
| Grounding | COCO 2017 + RefCOCO/RefCOCO+ development suite | Flickr30K Entities | LVIS | Detection/masks and referring expressions, independent-image phrase transfer, open-vocabulary long tail |
| Identity/tracking | DAVIS 2017 (A1) | MOT17 (A2, complementary crowded tracking) | YouTube-VIS or PointOdyssey (future B/C) | Mask identity, crowded box tracking, long-occlusion/category stress |
| Persistent memory | Ego4D episodic-memory and EgoTracks (A1) | ScanNet v2 (A2, complementary scene memory) | RoboMME/RoboMemArena when frozen | Temporal retrieval/re-detection, persistent scene/object memory, robot memory stress |
| Dynamics | D4RL transition sanity plus RoboNet visual trajectories (A1/A2) | ManiSkill3 demonstrations (complementary embodiment) | DROID or another version-pinned B/C source | Action conditioning, visual rollout, cross-view/platform/task transfer |
| Planning | ManiSkill3 fixed task suite | LIBERO fixed suites | BEHAVIOR-1K, CALVIN, or RoboCasa | Closed-loop manipulation, language/task transfer, long-horizon recovery |
| System | Phase-6-reserved ManiSkill3 composed tasks | Phase-6-reserved LIBERO tasks/suites | Robo4D-JEPA attribution tracks and BEHAVIOR-1K | End-to-end attribution and verified long-horizon execution |

This table is a planning commitment, not evidence that every dataset is downloaded, licensed for every use, or already
implemented. Each stage plan records access and license constraints. A preregistration must pin the exact release,
official split, subset rule, checksums, and permitted artifact handling.

## 3. Common experiment lifecycle

Every stage uses the same state machine. Skipping a state requires an explicit justification in the preregistration.

```text
S0 question and claim boundary
  -> S1 dataset/license/access audit
  -> S2 immutable manifest, split, hashes, and target-opacity rules
  -> S3 contract tests and tiny official smoke
  -> S4 reference baselines and metric sanity
  -> S5 health-only pilot / hyperparameter selection
  -> S6 formal development training and frozen checkpoint selection
  -> S7 held-out development evaluation and same-checkpoint mechanisms
  -> S8 one development survivor or explicit no-survivor
  -> S9 separate external confirmation, if authorized
  -> S10 strict postflight, report, insight update, and next decision
```

### S0 — question before implementation

Freeze:

- one primary scientific question;
- hypotheses and falsifying outcomes;
- what changes if the result is positive, negative, or mixed;
- evidence level and prohibited claims;
- model family, reference, intervention, and selection unit;
- whether efficiency is descriptive or a hard operational constraint.

Architecture discovery is quality-first. Speed, memory, and throughput are always logged, but they do not eliminate an
untrained architecture unless the explicit research question is deployment feasibility. After a quality survivor is
frozen, optimization becomes a separate parity-constrained experiment.

### S1 — dataset and license audit

Before download or target inspection, record:

- official project/paper and canonical source URL;
- release/version, split names, archive bytes/checksums when published;
- license, terms, citation, redistribution, privacy, and takedown rules;
- credentials or click-through access requirements without storing secrets;
- expected samples/scenes/subjects/episodes and modalities;
- whether labels are public, server-scored, or hidden;
- storage, decode, cache, and estimated compute costs.

If terms do not clearly permit the planned use, stop and select a different source. Never upload raw restricted data to
W&B or a repository.

### S2 — immutable manifest and split

Prefer official splits. When a development subset is necessary:

1. freeze an ordered source-ID list before model results;
2. apply only prespecified mechanical eligibility checks;
3. split by the independent unit—scene, video, subject, camera, or episode—not frame;
4. persist rejected IDs/reasons and all selected file hashes;
5. make train, validation, development-test, and external-test roles explicit;
6. prevent training jobs from receiving development-test/external target paths;
7. add the split to a consumed-test ledger after opening.

Repeated frames, crops, views, or seeds are not independent scenes. They may improve optimization or diagnostics but do
not increase the population sample count.

### S3 — contract and tiny official smoke

Run on the smallest official subset that exercises the real loader/model/metric path:

- decode and shape/dtype/range checks;
- coordinate/frame and timestamp tests;
- valid-label and missing-data behavior;
- exact deterministic preprocessing and seed checks;
- metric unit tests on analytically known cases;
- real checkpoint load and finite forward/backward pass;
- strict checkpoint reload equality;
- W&B/local artifact round-trip;
- secret and raw-target upload denial.

Smoke scores are never promoted as quality results.

### S4 — reference baselines

Every quality experiment includes:

1. a trivial or heuristic baseline that exposes dataset leakage or metric defects;
2. the strongest current JEPA-4D baseline;
3. one common published/pretrained reference when licensing and compute permit;
4. same preprocessing, valid masks, metrics, and aggregation for every comparable model;
5. paired sample-level outputs so differences can be audited.

Baselines cannot be selectively dropped after seeing results.

### S5 — health-only pilot and tuning

Pilot gates may reject only broken runs:

- non-finite losses, outputs, gradients, or metrics;
- missing expected gradients or forbidden gradient flow;
- model unchanged from initialization;
- exact reload mismatch;
- clearly non-learning objective under a prespecified check;
- loader, schema, hash, dependency, W&B, or artifact failure.

Select hyperparameters only on validation data. Do not use development-test/external labels. Give each arm an equal and
prespecified search budget; otherwise architecture and tuning effort are confounded.

### S6 — formal development training

Freeze:

- seeds, epochs/steps, optimizer, scheduler, batch and precision;
- validation checkpoint key and tie breaks;
- failure/retry rules;
- expected logical cell matrix;
- per-stage resource envelope;
- complete epoch/step logging schema.

Every declared cell must end as a valid success, legal skip defined before execution, or visible failure. Partial matrices
cannot select a model.

### S7 — held-out evaluation and mechanism tests

Evaluate immutable checkpoints in separate jobs. Report primary quality, calibration, safety, and completeness metrics.
Mechanism claims require same-checkpoint interventions when possible:

- remove/zero the claimed component;
- provide correct, stale, wrong, and permuted metadata controls;
- compare memory enabled/disabled under identical observations;
- compare planner verification/recovery enabled/disabled with the same episode seeds;
- prove the control changes the treatment before interpreting target scores.

### S8 — development survivor

Use frozen effect-size and non-inferiority gates, not a post-hoc weighted score. Report an eligible list and deterministic
tie breaks. No-survivor is a valid result and ends the phase without opening an external test.

### S9 — external confirmation

External evaluation is a separate preregistration. It receives exactly the frozen reference and one frozen survivor,
with no model, threshold, calibrator, or preprocessing changes after opening. One-shot final data is not used for
explanatory ablations. A failed external gate remains the final result.

### S10 — postflight and decision record

Verify all logical jobs, receipts, W&B identities, parent/output hashes, checkpoints, manifests, expected failures/skips,
and external-target sentinels. Update:

- the stage result record;
- [INDEX.md](experiments/INDEX.md);
- [INSIGHTS.md](experiments/INSIGHTS.md);
- this validation matrix and the corresponding stage TODO;
- the consumed-test ledger;
- the next uncertainty and stop decision.

## 4. Metric and statistical rules

Every result must state formula/version, direction, unit, valid denominator, alignment, calibration split, aggregation, and
uncertainty unit. The [metric guide](METRICS.md) is mandatory reading.

Common rules:

- primary aggregation uses independent scenes/videos/subjects/episodes, not pooled pixels or frames;
- report per-unit values and paired candidate/reference differences;
- seed SD is optimizer variation, not a population confidence interval;
- bootstrap the independent cluster and preserve pairing;
- a small number of clusters produces descriptive intervals, not broad significance claims;
- calibrated uncertainty and uncertainty ranking are separate gates;
- report failures and coverage; never improve a metric by silently dropping samples;
- define safety metrics and false-acceptance costs before closed-loop evaluation;
- use exact same metric implementation for reference and candidate.

## 5. Logging and visualization contract

Online W&B is the interactive comparison surface; local immutable artifacts are the source of record. Every run logs:

- clean commit, branch, config/schema and data/checkpoint hashes;
- Slurm allocation, node, GPU UUID/model, CUDA/driver, environment lock;
- source/parent artifact identities;
- explicit step/epoch/episode axes;
- training losses, learning rate, allowed/forbidden gradients, update norms;
- validation and held-out metrics with direction and units;
- throughput, elapsed time, peak memory, and periodic GPU telemetry;
- failure category, retry/requeue history, and terminal status.

Every formal aggregate includes:

1. gate/status cards;
2. paired per-scene/video/episode forest plots;
3. mean plus distribution, not mean alone;
4. calibration/reliability and risk-coverage when uncertainty exists;
5. fixed qualitative panels selected before training;
6. failure taxonomy and worst-case examples selected by frozen rules;
7. resource diagnostics separated visually from quality selection;
8. provenance and completeness tables.

Persist JSON/JSONL, CSV, NPZ or Parquet as appropriate, checkpoints, PNG, self-contained HTML, manifest, receipt, and
SHA-256 identities. Do not serialize credentials or raw restricted targets.

## 6. Slurm execution contract

The login node may inspect code, build the pinned environment, stage metadata/assets, submit, and monitor. GPU model
execution, training, profiling, and formal evaluation run only inside Slurm allocations.

All stage plans inherit:

- account `edgeai_tao-ptm_image-foundation-model-clip`;
- partition fallback `polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3`;
- one node/task/GPU unless a preregistration justifies otherwise;
- no logical task longer than `04:00:00`;
- distinct semantic names such as `j4d-p3-ground-<commit>`;
- held submissions and an atomic, hash-bound dependency graph;
- job arrays instead of hundreds of independent submissions;
- array concurrency caps and stage dependencies that guarantee at most **eight RUNNING allocations globally**;
- checkpoint/resume chunks for work exceeding four hours, with chunk lineage and exact step continuity;
- infrastructure retries only when no trustworthy scientific output exists;
- strict terminal `sacct`, receipt, hash, artifact, and W&B audit.

The graph records base-submission count and logical-task count separately. `RUNNING+COMPLETING` display lag is reported but
does not redefine the allocation-overlap invariant.

## 7. Stage promotion levels

Each stage advances through the same levels:

| Level | Minimum evidence | Permitted status |
|---|---|---|
| L0 contract | Deterministic fixtures, unit tests, schemas, real loader smoke | Implementation ready for quality work |
| L1 single-dataset development | Common Dataset A, frozen train/val/dev-test, baselines | Development-quality evidence |
| L2 cross-dataset transfer | Frozen Dataset B without retuning | Transfer evidence |
| L3 mechanism | Same-checkpoint causal controls and calibrated uncertainty | Bounded mechanism claim |
| L4 operational | Frozen survivor meets speed/memory/reliability envelope | Deployment candidate for composition |
| L5 composed system | Repeated named-environment episodes with attribution | System-level evidence |

Operational efficiency follows scientific selection unless the stage's explicit question is feasibility. L4 cannot turn a
scientifically weak model into a promoted model, and L1/L2 cannot be described as robot-task success.

## 8. Project execution waves

The stages are connected but not fully serial. Dataset and adapter preparation can proceed in parallel under the global
eight-allocation policy.

### Wave A — benchmark foundations

- [ ] Freeze this master plan and stage file template.
- [ ] Complete license/access/storage audit for every declared A1/A2/B/C source.
- [ ] Add a machine-readable validation registry and consumed-test ledger.
- [ ] Add manifest schema for scene/video/subject/episode split units.
- [ ] Add shared paired-bootstrap and failure-taxonomy utilities.
- [ ] Add dashboard/report templates that label evidence level and data role.

### Wave B — representation and geometry

- [ ] Establish labeled natural-video V-JEPA baselines on two common datasets.
- [ ] Complete quality-first geometry architecture development without speed elimination.
- [ ] Freeze at most one geometry survivor.
- [ ] Run a separate external geometry confirmation only after selection.
- [ ] Optimize efficiency only for the frozen quality survivor.

### Wave C — grounding and identity

- [ ] Add labeled detection/mask/referring-expression adapters.
- [ ] Separate teacher-box/GT-mask association from end-to-end perception.
- [ ] Evaluate identity across multiple held-out videos, occlusion lengths, and same-category instances.
- [ ] Calibrate object/association confidence.

### Wave D — persistent memory

- [ ] Drive memory from frozen perception outputs on versioned sequences.
- [ ] Measure last-seen, temporal retrieval, scene-graph change, identity survival, and compression/task curves.
- [ ] Add database growth, replay throughput, and latency percentiles.

### Wave E — dynamics and planning

- [ ] Train/evaluate action-conditioned prediction on versioned offline trajectories.
- [ ] Freeze uncertainty and value calibration on held-out tasks.
- [ ] Evaluate repeated named-simulator episodes with seeded disturbances.
- [ ] Measure false verification, recovery, collision/control failure, and task success.

### Wave F — system benchmark

- [ ] Compose only stage survivors with frozen interfaces.
- [ ] Run end-to-end episodes with stagewise attribution and counterfactual component disablement.
- [ ] Release versioned Robo4D-JEPA descriptors/assets only when licensing permits.
- [ ] Add evaluation-server validation, sandboxing, resource limits, and leaderboard policy.

## 9. Master TODO and stop rules

Before any large formal graph:

- [ ] At least two declared common sources exist, are accessible, and have approved terms; Dataset B is mandatory for L2.
- [ ] Exact independent split unit and counts are frozen.
- [ ] Baselines, metrics, calibration, and aggregation are implemented and tested.
- [ ] Primary question, quality gate, mechanism gate, and prohibited claims are written.
- [ ] W&B/local logging and fixed visualizations pass an official-mini smoke.
- [ ] Slurm graph is dry-run validated with no more than eight running allocations.
- [ ] External target paths are denied until selector authorization.
- [ ] Compute/storage budget and four-hour chunking are feasible.

Stop or redesign when:

- the dataset license/access no longer supports the plan;
- controls do not change the intended treatment;
- results depend on one scene/video/episode or on test-tuned thresholds;
- a stage cannot beat its trivial/current baseline on Dataset A;
- gains disappear on Dataset B;
- uncertainty is unsafe for the downstream decision;
- the composed system cannot attribute failures to a stage;
- repeated infrastructure instability invalidates comparability.

No stage advances because “the pipeline completed.” Advancement requires the registered scientific and integrity gates,
with no-survivor and stopped directions retained as first-class results.
