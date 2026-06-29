# Phase 4 initial persistent-memory experiment

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `2026-06-29-memory-incremental-replay-v1` |
| Stage / status | `memory / complete` |
| Evidence level | `contract-only` |
| Promoted W&B run | [fa9r6n1c](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/fa9r6n1c) |
| Decision | Keep the persistence/replay architecture and replace fixture observations with real sequence association. |

## W&B dashboard reading guide

| Panel | What it answers | Observation | Insight / decision |
|---|---|---|---|
| memory revision and update timelines | Does every observation advance state deterministically? | Eight revisions are visible, including an intentionally empty update. | Occlusion must not fabricate an observation. |
| mean confidence and history-entry timelines | How does evidence accumulate or decay? | Seven observations produce a readable history progression. | Retain explicit time axes and confidence components. |
| episodic-event timeline | Are changes recorded as queryable events? | Event count evolves with the fixture. | Events and current object state serve different query needs. |
| object and event tables | Can a user audit individual records? | IDs, timestamps, states, and evidence are inspectable. | Tables are essential complements to aggregate curves. |
| persistence-record counts | Do snapshots and events survive storage? | Reload and replay produce equal serialized snapshots. | Promote persistence contract; do not infer tracking quality. |

## Stage insights and decisions

| Stage | Evidence | Insight | Decision |
|---|---|---|---|
| Incremental update | Revision/history timelines | State changes are visible at each step. | Keep step-indexed logging for real episodes. |
| Occlusion | One empty update | Absence of evidence is represented without hallucination. | Add confidence decay/last-seen policies under longer gaps. |
| Persistence | Reload/replay equality | Snapshot and event-sourced paths agree on this fixture. | Expand corruption and migration tests. |
| LOD/query | Compressed history and mug query | Queryability survives the basic lifecycle. | Measure compression-versus-task accuracy on real episodes. |

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

## Downloaded W&B record snapshot

The promoted run record was downloaded through the W&B API on 2026-06-29 and verified in `finished` state. Its persisted
summary reports revision 8, one global/local object, seven history entries, seven episodic events, four compressed history
entries, one query match, and exact reload/replay parity. W&B lists eight logged artifacts covering object/event tables,
the memory snapshot and metrics, SQLite memory, interactive report, experiment record, and history.
