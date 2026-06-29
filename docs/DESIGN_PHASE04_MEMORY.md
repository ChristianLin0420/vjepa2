# Phase 4 design: persistent hierarchical 4D memory

## Scope

Phase 4 makes memory an explicit, durable product instead of an in-process list. It synchronizes a bounded robot-centric
working set, a global temporal scene graph, episodic events, a vector index, task state, append-only update evidence, and
versioned snapshots. The implementation accepts Phase 3 object slots and geometry beliefs but remains usable with mocks
for deterministic CPU tests.

This phase establishes storage and query invariants. It does not yet solve global loop closure, distributed multi-robot
consistency, learned place recognition, probabilistic data association, or production database migration.

## Two synchronized memory levels

```text
observation update
  geometry + object slots + robot state + timestamp
              │
              ├── ActiveLocalMap (base/odom working set)
              │     radius filter, distance, stale pruning, bounded summaries
              │
              ├── SceneGraph (map-frame durable belief)
              │     object state, evidence refs, trajectory/confidence history
              │
              ├── EpisodicMemory
              │     discovered/reobserved events with evidence
              │
              ├── VectorIndex
              │     normalized appearance retrieval
              │
              └── SQLite transaction
                    current records + append-only event log + snapshot
```

All five in-memory views are updated by `FourDMemoryCore.update`. SQLite receives the externally durable subset in one
`BEGIN IMMEDIATE` transaction. The memory revision advances exactly once per accepted timestamp, including an empty
observation update.

## Time invariant

Updates must be monotonic. A timestamp older than `last_update_time` raises before mutation. This prevents silent history
reordering and makes event replay deterministic. Out-of-order robotics streams must be reordered or explicitly handled by
a future reconciliation layer rather than slipped into the current state.

## Active local map

`ActiveLocalMap` is a high-frequency working set with configurable radius, frame, stale timeout, maximum observation
summaries, and current local objects. For each object it stores ID, category, position, robot distance, confidence,
last-seen time, and dynamic flag.

Distance uses `pose_robot` when available and falls back to `pose_map`. The robot origin is taken from the first three
elements of `RobotState.base_pose`; absent state means origin zero. This fallback is an explicit approximation because a
map-frame point should normally be transformed through a frame graph before radius comparison.

Objects outside the radius are not placed in active memory, but remain eligible for the global scene graph. Stale local
objects are removed after `stale_after_s`. Observation summaries are bounded independently of objects.

## Global scene graph

`SceneObject` retains:

- category, description, region, affordances, and states;
- first/last seen timestamps and confidence timestamp;
- unique observation references and their count;
- latest 2D/3D/robot/map pose metadata;
- bounded `ObjectHistoryEntry` records with timestamp, confidence, pose, state, and evidence refs.

An upsert creates a history entry even when evidence refers to an existing object. Evidence references are deduplicated.
Repeated confidence uses a 0.7/0.3 exponential moving average because adjacent views are correlated; noisy-OR would
incorrectly approach certainty after several similar frames.

The current ID still comes from Phase 3 association. A future identity manager must represent split, merge, alias, and
re-identification decisions without rewriting historical evidence.

## Confidence decay

`decay_confidence(timestamp, half_life_s)` exponentially decays from `confidence_timestamp`, then advances that timestamp.
Calling decay twice at the same time is idempotent. The last-seen timestamp remains evidence time and is never rewritten
by decay.

Confidence is a heuristic ranking belief, not a calibrated probability. Decay half-life must be benchmarked by object
class, scene dynamics, and task cost before it affects safety decisions.

## Episodic events

Each observed slot yields either `object_discovered` or `object_reobserved`. Event IDs are deterministic UUID5 values over
revision, timestamp, object ID, and event type. Events retain evidence references, current confidence, and pose. The
container supports entity, type, and time-range filtering and a configurable global bound.

An empty update currently records active-map/persistence progress but does not invent an occlusion event. Explicit
occlusion requires visibility/frustum reasoning and belongs in the tracking model.

## Vector index

The dependency-free cosine index stores Phase 3 visual embeddings by object ID. It provides a FAISS-compatible conceptual
boundary while keeping CI offline. Embeddings are not yet embedded in snapshots, so vector state is rebuilt only when a
future persistence schema includes compact vectors. Text query APIs currently use structured fields.

## SQLite schema version 2

Four tables are created idempotently:

1. `records`: latest JSON record per ID and kind;
2. `event_log`: append-only sequence, timestamp, operation, kind, ID, and payload;
3. `snapshots`: complete serialized memory by revision;
4. `metadata`: schema version.

WAL mode and normal synchronous mode support concurrent readers and practical durability. JSON rejects NaN so corrupted
numeric beliefs fail the transaction. Tests deliberately insert one valid and one NaN record and verify both roll back.

`commit_update` writes current records, event-log rows, and an optional snapshot atomically. The database can recover by
loading the latest snapshot or by replaying event rows from sequence zero. The two paths must produce identical
serialized state in tests and demos.

## Snapshots and replay

`FourDMemorySnapshot` includes map/robot frames, timestamp, revision, active map, scene graph, episodic events, task state,
and uncertainty summary. Serialization is plain JSON data.

Snapshot loading is the fast path. Replay is the audit path: object rows restore successive graph states, event rows
restore unique episodic records, and active-map rows mark revisions and restore working memory. Replay parity detects
serialization omissions and transaction ordering defects.

## Level-of-detail policy

`LODPolicy` creates a compressed copy without mutating live memory. It bounds per-object history, events, and local
observation summaries. Task-protected entity IDs receive twice the ordinary history budget, and their events survive even
when old. Compression reports removed counts in the uncertainty summary.

Current LOD reduces JSON history but does not downsample geometric arrays because those are not yet stored in snapshots.
Future policies should measure compression ratio against query/task accuracy.

## Query boundary

`WorldModelQueryAPI` now exposes local objects, update time, global object ordering by confidence/recency, graph history,
episodic history, observation counts, last-seen time, uncertainty, affordances, regions, topology, and verification action.
Raw JEPA tensors, point maps, masks, and SQLite rows remain outside planner APIs.

## Observability

Every memory revision can log inserted/updated objects, local/global counts, event count, persistence records, mean
confidence, and history size against `memory/revision`. Final W&B tables summarize objects and events. Local interactive
HTML plots map-XY trajectories and confidence histories and embeds the full snapshot/persistence metadata.

Artifacts include snapshot JSON, metrics JSON, SQLite, interactive HTML, and Markdown experiment record. Artifact names
include W&B run ID and full filename to avoid collisions such as `memory.json` versus `memory.db`.

## Benchmark and tests

The memory smoke benchmark performs five updates and checks history/reference recall, text query recall, reload parity,
event replay parity, and query latency. Perfect smoke scores are interface invariants, not memory-quality results.

Unit coverage includes incremental merge, event types, vector retrieval, radius filtering, robot origin, stale pruning,
transaction rollback, event sequence ordering, snapshot/replay parity, monotonic rejection, idempotent decay, LOD bounds,
non-mutation, local query context, and history queries.

## Known limitations

- active-map fallback mixes frames when no transform is available;
- Phase 3 IDs are not durable under arbitrary reassociation or batch reorder;
- SQLite is single-node and embeddings remain in memory;
- snapshots are written every update and may become expensive;
- event payloads store successive full object records rather than compact deltas;
- no deletion/tombstone or schema migration runner exists;
- no region inference, place hierarchy, frame graph, or loop closure exists;
- no concurrency control protects one in-memory core from multiple writers;
- uncertainty is scalar and does not preserve pose covariance provenance.

## Next implementation steps

1. Add durable identity alias/split/merge/tombstone operations.
2. Introduce frame transforms and reject untransformable radius comparisons.
3. Persist vector embeddings in a versioned array/vector backend.
4. Add periodic rather than per-update snapshots and snapshot/event sequence watermarks.
5. Add building/floor/room/place hierarchy and learned region assignment.
6. Evaluate NaVQA/SG3D-style queries and RoboMME/RoboMemArena memory tasks.
7. Measure database growth, replay time, query percentiles, and LOD accuracy curves.
8. Add process-safe writer ownership and migration tests before robotics deployment.

## Phase 4 exit gate

The substrate is complete when deterministic updates, crash-safe transactions, reload/replay parity, bounded working
memory, explicit uncertainty, typed queries, stagewise smoke metrics, human reports, and CPU CI all pass. Model-quality
completion additionally requires external labeled memory benchmarks, calibrated temporal confidence, robust identity
events, and long-duration storage profiling.
