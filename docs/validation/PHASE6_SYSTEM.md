# Phase 6 validation plan — composed system benchmark

## Status and role

**Plan status:** draft. The current Phase 6 result is contract-only.

Phase 6 does not replace stage evaluation. It verifies that frozen stage survivors compose into an uncertainty-aware world
model and that end-to-end task outcomes can still be attributed to representation, geometry, grounding, identity, memory,
dynamics, planning, verification, control, or infrastructure.

The current `robo4d-jepa-contract` asset is a generated 376-byte fixture. Six deterministic adapters ran five times each,
validating manifest checks, bootstrap/report plumbing, typed failures, local artifacts, and W&B. It contains no official
model-quality ground truth and cannot support a system-performance claim.

## 1. System objective

Evaluate the complete loop:

```text
observe RGB
  -> build representation/geometry/object evidence
  -> update persistent 4D memory
  -> answer typed queries
  -> predict and plan
  -> execute
  -> observe fresh evidence
  -> verify or reject
  -> update memory and recover
```

The system succeeds only when task completion, safety, memory use, uncertainty handling, and stagewise evidence are all
valid. A final task success without a correct persistent identity or with a false verification is not a clean success.

## 2. Benchmark portfolio

Phase 6 consumes every frozen official stage adapter and adds at least two common closed-loop environments.

| Role | Benchmark | Unit | Purpose | Source/access note |
|---|---|---|---|---|
| Development system A | [ManiSkill3](https://maniskill.readthedocs.io/en/latest/) fixed manipulation tasks | Episode/task/scene seed | Reproducible state and visual manipulation, perturbations, parallel evaluation | Pin release, assets, simulator, renderer, task configs, and evaluation mode |
| Transfer system B | [LIBERO](https://libero-project.github.io/) fixed benchmark suites | Demonstration/task/episode seed | Language-conditioned spatial, object, goal, and long-horizon transfer | Pin repository/data release and suite split; obey dataset terms |
| Optional stress C | [BEHAVIOR-1K](https://behavior.stanford.edu/behavior-1k) task subset | Activity/house/episode | Long-horizon household state, memory, and recovery stress | Large assets and simulator-specific terms require a separate audit |
| Custom attribution | Robo4D-JEPA held-out episodes | Scene/episode | Explicit hidden-state memory, evidence, verification, and failure labels | Generated or redistributable assets only; never call the current contract fixture a benchmark |

ManiSkill3 and LIBERO are the minimum common system pair. Optional BEHAVIOR-1K work begins only after the first two adapters
and their licenses/storage requirements are stable.

Their Phase-6 task/scene/suite IDs must be reserved before Phase-5 formal evaluation and inaccessible to Phase-5 model,
threshold, prompt, controller, or planner selection. New episode seeds inside a task already inspected in Phase 5 are not
independent system tasks. If disjoint reservations were not made, reuse is allowed only as `consumed_regression` evidence;
an independent Phase-6 claim then requires a newly frozen benchmark or custom held-out track.

## 3. Required upstream admission

No component enters Phase 6 merely because its code path runs. The system manifest records the admitted version and stage
evidence:

| Component | Minimum admission |
|---|---|
| Representation | Labeled natural-video Dataset A result plus transfer evidence or an explicit limitation |
| Geometry | Frozen metric-depth survivor, cross-dataset result, calibrated uncertainty, and coordinate convention |
| Grounding | Labeled box/mask/referring-expression result with confidence calibration |
| Identity | Multi-video held-out tracking result with merge/switch/fragment metrics |
| Memory | Perception-driven sequence result with retrieval/change/identity and scaling metrics |
| Dynamics | Held-out action-conditioned prediction with calibrated uncertainty/value |
| Planner | Repeated named-simulator result with verification, attribution, safety, and recovery |

An oracle component may enter as a diagnostic baseline but must be labeled. Missing stage admission blocks a full-system
promotion claim, although integration work may continue at contract level.

## 4. Dataset and episode split policy

- Pin simulator, benchmark, task, asset, renderer, physics, controller, camera, and observation versions.
- Verify every task/scene/suite against the Phase-5 consumed ledger and reject overlap from the independent aggregate.
- Use official task/suite splits when provided.
- Split generated variations by scene/house/task, not only random episode seed.
- Reserve task/scene combinations for held-out transfer; do not tune thresholds on them.
- Freeze language instructions and paraphrase policy before evaluation.
- Store exact initial state, randomization seed, object identities, target predicates, action bounds, and time limit.
- Run reference and candidate on paired initial states/seeds.
- Record deterministic replay inputs, but do not assume the physics rollout is bitwise identical across hardware.
- Treat a simulator or asset-version change as a new benchmark revision.

Minimum formal coverage per selected task is proposed as 50 paired episodes for development and 100 paired episodes for a
separately frozen external suite, subject to preregistered power/resource analysis. Report task/scene macros; do not pool
thousands of time steps as independent episodes.

## 5. Robo4D-JEPA custom tracks

The custom benchmark should add failure attribution that generic success suites may not expose.

| Track | Episode design | Primary uncertainty |
|---|---|---|
| A single-image bootstrap | One initial RGB observation, query, and optional next-view action | Does uncertainty trigger a useful observation rather than hallucinated metric belief? |
| B multi-view 4D memory | Revisited rooms/objects with occlusion, motion, and state changes | Are identity, last-seen, relations, and evidence preserved? |
| C delayed hidden state | Critical object/state observed early but unavailable near decision time | Does persistent memory improve the later action? |
| D verified long horizon | Multi-room task with injected perception/control failures | Does the system reject unsafe beliefs, attribute failure, and recover within bounds? |

Assets must be generated or redistributable with a versioned license/manifest. Hidden evaluation labels remain unavailable
to agents and model-selection jobs.

## 6. Baselines and ablations

| ID | System | Purpose |
|---|---|---|
| B0 | Scripted/environment reference where available | Task and evaluator sanity |
| B1 | Current policy with no persistent memory | Quantify memory value |
| B2 | Current policy with oracle perception/state | Upper-bound planning/control independent of perception |
| B3 | Frozen previous JEPA-4D system | Regression anchor |
| C | Frozen composed candidate | Primary comparison |

Same-seed counterfactual ablations:

- no geometry or aligned-only geometry where the task permits;
- no appearance identity versus no spatial identity;
- no persistent memory / current observation only;
- no uncertainty rejection;
- no verification action;
- no failure attribution/replanning;
- oracle perception, identity, or memory one component at a time.

Disable one mechanism without retraining whenever possible. A full retrain comparison is an architecture-system effect, not
a clean causal intervention.

## 7. Metrics and aggregation

### Primary system metrics

| Metric | Direction | Aggregation |
|---|---:|---|
| Verified task success | Higher | Episode, then equal task/scene macro |
| Unsafe false success | Lower; target zero | Episode rate with claimed success but violated target/safety predicate |
| Collision/constraint violation | Lower | Episode rate and counts, severity strata |
| Normalized subgoal progress | Higher | Episode then task macro |

### World-model and memory metrics

| Metric | Direction |
|---|---:|
| Object/identity query precision, recall, F1 | Higher |
| Last-seen temporal/spatial error | Lower |
| Relation/state-change F1 | Higher |
| ID switches, false merges, fragments | Lower |
| Evidence completeness and stale-evidence use | Higher completeness / lower stale use |
| Memory-dependent success delta over no-memory | Higher |

### Verification and recovery metrics

| Metric | Direction |
|---|---:|
| False verification accept rate | Lower; safety-critical |
| False rejection rate | Lower subject to safety |
| Recovery success after attributable failure | Higher |
| Correct failure attribution | Higher |
| Replans and verification actions | Lower for matched safety/success |
| Time/steps to recovery | Lower |

### Operational diagnostics

Report observation-to-belief, memory update/query, planning, control, and verification latency; throughput; peak GPU/CPU
memory; database growth; token/object/event counts; and total episode wall time. These diagnose feasibility after quality
and safety. They do not compensate for incorrect beliefs or unsafe success.

Use paired task/scene/episode effects and cluster bootstrap over tasks/scenes. Seeds within one scene measure stochastic
rollout variation, not environment diversity. Report failure counts and censored/time-limit episodes explicitly.

## 8. Experiment ladder and TODO

### L0 contract replay

- [x] Versioned generated manifest and byte/SHA validation ([contract evidence](../experiments/2026-06-29-phase6-benchmark-harness.md)).
- [x] Six deterministic stage adapters, typed failures, reports, and W&B ([contract evidence](../experiments/2026-06-29-phase6-benchmark-harness.md)).
- [ ] Add exact episode-trace schema with stage evidence references.
- [ ] Add deterministic evaluation of success predicates independent of planner claims.

### L1 named-simulator integration

- [ ] Pin ManiSkill3 release/assets and complete license/storage audit.
- [ ] Bind the Phase-6-reserved ManiSkill/LIBERO IDs and prove denial from all Phase-5 jobs.
- [ ] Select 3-5 tasks covering pick/place, articulation, state change, and occlusion.
- [ ] Run oracle-state and current JEPA-4D integration smoke.
- [ ] Pin LIBERO release/data and implement suite loader/action/controller adapter.
- [ ] Validate exact reset seeds, task predicates, timeouts, and video/state capture.

### L2 repeated baseline evaluation

- [ ] Freeze paired episode seeds and task/scene split.
- [ ] Run B0-B3 baselines with at most eight concurrent allocations.
- [ ] Verify stage metrics and environment success predicates agree.
- [ ] Produce per-task distributions, videos, event traces, and failure taxonomy.

### L3 composed candidate

- [ ] Admit only frozen stage survivors and bind all component hashes.
- [ ] Run the complete candidate matrix on development tasks.
- [ ] Execute no-memory/no-verification/no-replanning and oracle-component interventions.
- [ ] Select at most one composed survivor on frozen quality/safety gates.

### L4 transfer and custom memory tracks

- [ ] Evaluate the survivor on frozen LIBERO transfer suites without retuning.
- [ ] Create licensed/generated Robo4D-JEPA tracks A-D with hidden labels.
- [ ] Measure memory-dependent success and attribution under controlled failures.

### L5 external/system release

- [ ] Run a separately preregistered held-out suite once.
- [ ] Publish complete schema, manifests, evaluator, baselines, and license documentation where permitted.
- [ ] Add sandboxing, upload validation, resource limits, and leaderboard policy before accepting external submissions.

## 9. Promotion gates

A composed development survivor must satisfy all frozen stage gates plus:

- statistically and practically meaningful verified-success improvement over B3;
- unsafe false-success and collision rates no worse than the reference safety envelope;
- nonnegative memory-dependent success delta with improved memory query metrics;
- successful recovery for prespecified injected failures without excessive replanning;
- complete stage-evidence references for every verified subgoal;
- no unexplained stage regression beyond registered non-inferiority margins;
- all expected episodes, traces, videos, receipts, hashes, and W&B artifacts complete.

No survivor means retain the previous composed system. Do not hide a stage failure behind aggregate task success.

## 10. Logging and visualization

Each episode emits:

- environment/task/scene/seed and exact component identities;
- synchronized RGB, optional privileged evaluator state, actions, observations, beliefs, queries, and memory revisions;
- subgoal/behavior-tree transitions and evidence used for verification;
- uncertainty thresholds and accept/reject decisions;
- failures, attribution, replans, recoveries, collisions, timeouts, and terminal predicates;
- per-stage and cumulative latency/resource telemetry.

Aggregate reports include success/safety forest plots, stage-failure Sankey/confusion views, memory query/identity tables,
verification calibration, recovery timelines, paired ablation effects, resource breakdown, and fixed episode videos. Keep
privileged evaluator state separate from runtime model inputs and redact restricted assets from W&B.

## 11. Slurm execution

Use task/scene/seed arrays capped at `%8`; sequentially gate baseline, candidate, ablation, and aggregation arrays so no
more than eight allocations run globally. Use distinct `j4d-p6-*` names, a hash-bound graph, and tasks no longer than four
hours. Long episodes or training use exact checkpoint/resume chunks. Simulator rendering/model inference never runs on the
login node.

## 12. Claim boundary

- Simulator success is not real-robot safety or transfer.
- Oracle-state baselines are diagnostic upper bounds.
- One task suite does not establish general household intelligence.
- Generated Robo4D-JEPA tracks establish only the frozen scenario distribution.
- End-to-end success cannot replace stagewise geometry, identity, memory, calibration, and safety metrics.
- Real robot actuation remains out of scope until repeated simulator and safety gates pass.
