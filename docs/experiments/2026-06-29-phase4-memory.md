# Phase 4 initial persistent-memory experiment

## Objective

Validate that sequential object evidence becomes bounded local context, durable global history, episodic events,
transactional SQLite state, reloadable snapshots, replayable event logs, planner-safe queries, compressed copies,
interactive reports, and readable multi-step W&B timelines.

## Code state

- baseline before Phase 4: `23bb994` on `origin/main`;
- date: 2026-06-29 UTC;
- Phase 4 committed separately after final regression;
- runtime: CPU-safe deterministic observation fixture.

## Fixture

The demo generates eight monotonically timestamped updates for `demo-mug-001`. Seven contain one mug observation with a
slowly changing map pose, confidence, visible/dynamic state, and unique frame reference. The middle update contains no
object, modeling an unobserved/occluded frame without fabricating evidence.

This is an interface and persistence experiment, not a tracker evaluation. The object ID and pose trajectory are supplied
by the fixture.

## Results

- memory revisions: 8;
- global objects: 1;
- object history entries: 7;
- unique observation references: 7;
- episodic discovery/reobservation events: 7;
- query matches for `red mug`: 1;
- LOD history entries after compression: 4;
- SQLite current records: 9;
- append-only event rows: 22;
- snapshots: 8;
- schema version: 2;
- snapshot reload equals event replay: true.

The deliberately empty update contributes one active-map record and one snapshot but no object or episodic event.

## Stagewise benchmark

`jepa4d/config/benchmarks/memory_smoke.yaml` runs representation, geometry, object grounding, and memory adapters. The
memory adapter performs five observations and reports:

- history recall: 1.0;
- observation-reference recall: 1.0;
- query recall: 1.0;
- snapshot reload parity: 1.0;
- event replay parity: 1.0;
- query latency: approximately 0.05 ms on this tiny in-process fixture.

Latency is not a capacity result. It excludes process/network overhead and must be remeasured over realistic graph and
embedding sizes with percentiles.

## W&B

- project: `jepa4d-worldmodel`;
- run name: `phase4-incremental-memory-demo`;
- promoted run ID: `fa9r6n1c`;
- URL: <https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fa9r6n1c>.

The run contains an eight-step revision timeline for inserted/updated objects, local/global counts, events, persistence
records, confidence, and history size; final object/event tables; snapshot metadata; and run/file-scoped artifacts. It
supersedes `orz1oten`, whose correlated-evidence confidence aggregation approached certainty too aggressively and whose
same-stem JSON/database artifacts had ambiguous names.

## Artifacts

Ignored local outputs under `outputs/phase4_memory_demo/`:

- `memory.json`: complete latest snapshot;
- `memory.db`: WAL-backed SQLite database;
- `metrics.json`: aggregate results and per-update outcomes;
- `report.html`: interactive trajectory/confidence report;
- `EXPERIMENT.md`: generated local record.

## Verification

- Ruff formatting and lint;
- mypy across the JEPA-4D package;
- 35 JEPA-4D unit tests after adding the memory benchmark;
- standalone Phase 4 demo;
- four-stage stagewise smoke benchmark;
- W&B online artifact and timeline run;
- full upstream regression before commit;
- TOML/YAML and credential scans.

## Interpretation

The experiment demonstrates that the memory substrate is no longer append-only scaffolding. Current records, event
history, snapshots, queries, compression, and reports agree on a deterministic sequential fixture. Atomic rollback and
non-duplicating confidence decay are separately unit tested.

It does not demonstrate object identity under occlusion, metric map correctness, calibrated confidence, scene-scale
storage, or robot task improvement. Those require external ground truth and long-duration episodes.

## Next experiment

Use a versioned RGB video with two visually similar mugs, real Phase 3 detections, and manually labeled visibility/identity.
Measure false merge/split rate, IDF1, last-seen retrieval, event accuracy, database growth, replay time, p50/p95/p99 query
latency, and accuracy after multiple LOD budgets.
