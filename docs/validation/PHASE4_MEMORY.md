# Phase 4 validation plan: episodic and persistent memory

Status: proposed external validation plan, 2026-06-30. This document does not promote the current implementation.

## Decision and claim boundary

Phase 4 must validate two capabilities separately:

1. episodic retrieval: find when and where prior evidence answers a language or visual query;
2. persistent scene and object identity memory: preserve the same scene/object belief through disappearance, re-entry,
   viewpoint change, persistence, replay, and bounded compression.

The current code proves storage and query contracts on deterministic fixtures. It does not yet prove learned retrieval,
identity association, scene-scale accuracy, calibrated confidence, or task benefit. “Identity” below means object-instance
identity, never biometric or person identification.

## Current evidence

| Evidence | Result | What it does not establish |
|---|---|---|
| `FourDMemoryCore`, SQLite WAL persistence, snapshots, and replay | Atomic updates, monotonic time, rollback, reload/replay parity, and bounded LOD are tested. | Long-duration reliability or migration safety. |
| `memory_smoke.yaml` | Fixture history/reference/query recall and parity are 1.0; tiny in-process query latency is about 0.05 ms. | External retrieval quality or capacity latency. |
| Initial Phase 4 demo | Eight revisions, one object, seven observations/events, exact reload/replay parity. | Tracking, re-identification, or metric map quality. |
| W&B run [`fa9r6n1c`](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fa9r6n1c) | Finished run with revision timelines, object/event tables, snapshot, SQLite, and HTML artifacts. | Benchmark-grade model evidence. |

The existing evidence remains the contract gate; external results must never be merged into or described as those fixture
scores.

## External datasets and benchmark roles

| Source | Role in Phase 4 | Access and license boundary |
|---|---|---|
| [Ego4D Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/) | Primary episodic retrieval lane: NLQ for temporal language retrieval and VQ2D for last-seen object localization. The official task has language, visual, and moments queries over past egocentric video. | Data and annotations require acceptance of the [Ego4D license agreement](https://ego4d-data.org/docs/start-here/); approval can take about 48 hours and issued AWS credentials expire. Download only the benchmark subset. Benchmark code is in the [official MIT-licensed repository](https://github.com/EGO4D/episodic-memory); the code license does not replace the data agreement. |
| [EgoTracks](https://github.com/EGO4D/episodic-memory/tree/main/EgoTracks) | Primary persistent object-identity lane: long-term tracking and re-detection through hand interaction, occlusion, exit, re-entry, scale, and viewpoint changes. Use the official STARK/EgoSTARK setup as a comparison. | Clips and annotations use the Ego4D gated download and data agreement. The benchmark implementation inherits the parent repository's MIT code license. See the [primary EgoTracks paper](https://arxiv.org/abs/2301.03213). |
| [ScanNet v2](https://www.scan-net.org/) | Persistent scene lane: replay RGB-D sequences into memory, associate instance-level objects in a common scene, and evaluate scene/object queries against poses, reconstructions, and instance annotations. | Data uses the custom ScanNet Terms of Use; code is MIT. Access must be approved and recorded before download. ScanNet reports 2.5M RGB-D views across more than 1,500 scans. Use the official split and [repository](https://github.com/ScanNet/ScanNet). |

No dataset may enter training or evaluation until its agreement, permitted use, version, split, source URL, byte count,
and SHA-256 are present in an immutable manifest. Raw restricted media must not be uploaded to W&B or committed to Git.
Ego4D/EgoTracks and ScanNet are complementary A1/A2 development lanes for episodic/object and scene memory; neither is
treated as a no-retuning transfer confirmation of the other. Phase 4 therefore remains L1 until a separately frozen
robot-memory or cross-domain Dataset B has a stable task definition, official release, and audited terms.

## Frozen evaluation lanes

| Lane | Input and split | Required output |
|---|---|---|
| M0 contract | Existing deterministic fixtures and corruption tests. | Exact snapshot/replay parity, rollback, monotonic rejection, bounded LOD, and finite metrics. |
| M1 episodic retrieval | Ego4D official train/validation split; test labels remain sealed. NLQ and VQ2D are reported separately. | Ranked temporal windows; VQ tracks/last occurrence; query latency and evidence references. |
| M2 persistent identity | EgoTracks official train/validation split with long-gap and re-entry strata. | Object-present/absent decisions, boxes, re-detection events, aliases/splits/merges, and confidence. |
| M3 persistent scene | ScanNet official train/validation scenes, split by scene rather than frame. | Scene graph, stable object IDs within a scan, observation links, pose-grounded query results, and replay artifacts. |
| M4 systems stress | Increasing episode length, object count, query load, and LOD budget on held-out development sequences. | Database growth, write/read/query percentiles, replay time, crash recovery, and quality-versus-compression curves. |

Splits are identity- and scene-disjoint. Frame-neighbor leakage, query-answer leakage, and validation-driven test selection
are prohibited. A pilot subset may set thresholds once; formal results use a separately hashed split and configuration.

## Baselines and ablations

- recency-only and random-window retrieval;
- frozen V-JEPA feature nearest-neighbor retrieval with no durable memory;
- current structured text query and supplied-ID scene graph as contract/oracle controls, clearly labeled as such;
- official Ego4D NLQ/VQ2D reference code;
- STARK and EgoSTARK for EgoTracks;
- category-only, appearance-only, geometry-only, and appearance-plus-geometry association;
- no event log, no confidence decay, no alias/split/merge handling, and fixed LOD-budget ablations;
- ScanNet last-observation and pose-grounded fusion baselines.

All learned methods use identical frozen inputs and split manifests. Parameter count, cache identity, runtime, and peak
memory are reported beside quality.

## Metrics

### Episodic retrieval

- NLQ official recall at K and temporal IoU thresholds, mean IoU, and per-template/per-duration breakdowns;
- VQ2D official spatiotemporal localization and recovery metrics using the official evaluator;
- last-seen temporal error, evidence-reference precision/recall, and abstention coverage-risk;
- p50/p95/p99 query latency and peak resident/GPU memory at fixed index sizes.

### Persistent identity and scene memory

- EgoTracks official success, precision, and normalized precision, plus re-detection recall after absence-duration bins;
- false merge, false split, identity switch, IDF1/HOTA where the evaluation representation supports multiple identities;
- ScanNet 3D instance AP at the official IoU thresholds and object-query recall at fixed spatial tolerances;
- last-seen pose error, state/event precision and recall, history/reference recall, and confidence calibration;
- exact reload/replay parity, corruption detection, bytes per observation/object/event, replay seconds, and LOD
  quality-versus-size area under the curve.

Report macro means across videos/scenes first, then micro totals. Use paired bootstrap confidence intervals over the
video/scene unit, not individual frames.

## Staged TODO

- [ ] **M0 — contract hardening:** add identity alias/split/merge/tombstone records, frame-transform validation, persisted
   embeddings, schema migration, process-safe writer ownership, and crash-injection tests.
- [ ] **M1 — sealed data preparation:** obtain approvals; create content-addressed manifests and scene/video-disjoint splits;
   audit that restricted media and test labels cannot enter W&B artifacts.
- [ ] **M2 — adapters and baselines:** implement Ego4D NLQ/VQ2D, EgoTracks, and ScanNet adapters plus official evaluators;
   reproduce one published/reference baseline before evaluating JEPA-4D memory.
- [ ] **M3 — pilots:** run small train/dev jobs to establish capacity and latency limits. Freeze metric directions,
   confidence threshold, LOD budgets, and formal seeds in a committed preregistration.
- [ ] **M4 — formal arrays:** run dataset × method × seed jobs with held submissions and immutable receipts.
- [ ] **M5 — aggregation:** compute paired confidence intervals, failure strata, calibration, capacity curves, and visual
   reports without opening sealed test labels during development.
- [ ] **M6 — postflight:** verify every expected receipt, source hash, W&B artifact, scheduler terminal state, and claim
   boundary before promotion.

## Logging and visual evidence

Use online W&B under `jepa4d-worldmodel`, one group per immutable execution ID. Every run records Git commit, dirty state,
dataset/split/config hashes, scheduler IDs, node/GPU identity, seed, parents, and exact output hashes.

Required W&B panels and matching local files:

- NLQ/VQ recall by duration, query type, and IoU; `episodic_metrics.csv`, `.npz`, `.png`, and `report.html`;
- retrieval timelines with query, ground truth, prediction, evidence thumbnails, and abstentions;
- identity survival/re-detection versus gap length, false merge/split matrices, and annotated failure videos;
- scene graph and map-XY trajectories, confidence histories, and object evidence drill-down;
- database bytes/replay/query percentiles versus episode length and LOD budget;
- calibration/reliability and coverage-risk plots;
- object/event/query tables that link every metric back to immutable evidence references.

Raw licensed video, faces, voices, location metadata, and unrestricted database dumps remain local. W&B receives only
approved derived metrics, redacted visualizations, manifests, and small reports.

## Slurm execution policy

- Login nodes are limited to code inspection, environment construction, manifest generation, and `sbatch` submission.
- Account: `edgeai_tao-ptm_image-foundation-model-clip`.
- Partitions: `polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3`.
- Each job has a maximum wall time of four hours and writes stdout, stderr, telemetry, and a terminal receipt.
- Submit tests, data audit/cache, baselines, formal arrays, aggregation, selection, and postflight held; atomically write
  the dependency graph before release; use `afterok` dependencies.
- Arrays use `%8` or less. Across the whole execution, expanded Slurm `RUNNING` tasks must never exceed eight. Monitor
  `RUNNING` continuously and record `COMPLETING` separately because epilog rows can transiently exceed the array throttle.
- A missing receipt, nonzero exit, source-hash mismatch, offline W&B run, or concurrency violation fails closed. Operator
  requeues are logged with reason/restart count and cannot silently replace a logical job.

## Promotion gates

Phase 4 is promoted from contract-only to externally validated only when all conditions hold:

1. all contract, corruption, migration, replay, and deterministic rerun tests pass;
2. official evaluator reproduction matches the chosen reference within a preregistered tolerance;
3. the memory method improves the primary metric over recency/no-memory and frozen-feature baselines with a paired 95%
   bootstrap confidence interval above zero on both episodic retrieval and persistent identity/scene lanes;
4. no preregistered safety metric regresses: false merge, false high-confidence retrieval, sealed-label access, and
   evidence/provenance violations must remain within their frozen bounds;
5. exact reload/replay parity is 100%, all formal values are finite, and every formal job has a scheduler/W&B/local
   receipt triple;
6. p95/p99 latency, database growth, replay time, and LOD quality stay within limits frozen before formal evaluation;
7. independent postflight validates splits, licenses, 8-task concurrency, hashes, and the report's claims.

Passing this phase authorizes simulator research queries only. It does not authorize person identification, surveillance,
real-world autonomous action, or safety-critical use.
