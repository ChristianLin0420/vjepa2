# JEPA-4D implementation roadmap

## Guiding rule

Advance only when the previous phase passes unit, integration, artifact, and documentation gates. A new model integration
does not compensate for unstable contracts. Mocks remain available at every heavy boundary so CI stays offline and CPU
runnable. Future quality work follows the [systematic validation plan](VALIDATION_PLAN.md) and its
[stage-specific dataset/TODO plans](validation/README.md); historical completion statements below retain their original
claim boundaries.

## Phase 0 — repository and contracts: complete

Delivered package structure, typed RGB/token/geometry/object/memory contracts, deterministic mocks, query API, FastAPI
service, packaging, Ruff, mypy, pytest, pre-commit, CI, demos, and experiment ledger.

Exit evidence: JEPA-4D tests pass offline; server health and query endpoints respond; no heavy checkpoint is needed.

## Phase 1 — V-JEPA 2.1 substrate: complete

Delivered four-mode RGB loading, view/time identity, ViT-B/L/g/G configuration, native and HF-compatible checkpoints,
dense/global/multi-layer tokens, tubelet validity, PyTorch/Zarr output, PCA and temporal reports, and W&B diagnostics.

Exit evidence: mock and real ViT-B inference pass; feature artifacts are finite and reproducible; intermediate layers are
available; upstream tests remain green.

## Phase 2 — geometry belief teacher baseline: complete

Delivered deterministic mock, official VGGT package/checkpoint boundary, real single/multi-view CPU smoke tests, camera
matrices, depth/point mean and log-variance, tracks, confidence policy, NPZ/PLY export, interactive 3D reports, geometry
W&B panels, configuration, benchmark smoke test, and documentation.

Completion evidence now includes a checksum-pinned official TUM RGB-D Freiburg1 XYZ mini subset, deterministic
calibration/test frames, per-frame aligned depth and point metrics, Sim(3)-aligned camera metrics, held-out variance
calibration, explicit camera-from-world numerical tests, COLMAP text export, categorized per-sample failures, and A100
FP32/BF16 latency/memory scaling from one to eight frames. The official VGGT-1B teacher run is logged as W&B
`rcpsxq6g`; its bounded claim is an official single-sequence mini-baseline, not cross-dataset generalization or metric
single-image scale.

VGGT-Omega and MapAnything are deferred as separately named future backends: replacing the pinned VGGT-1B teacher would
invalidate the baseline. Bundle adjustment remains optional post-processing rather than a Phase-2 exit requirement.
Broader multi-scene/dataset evaluation remains Phase-6 benchmark expansion; the learned-student comparison is Phase 2b.

## Phase 2b — JEPA metric-depth distillation v1: complete

The completed v1 gate trains a lightweight metric-depth/log-variance head over frozen V-JEPA 2.1 layers using licensed
ground truth and VGGT auxiliary targets. Broader point, pose, track, cross-view reprojection, temporal-consistency, and
selective-encoder-tuning objectives remain explicitly deferred extensions rather than v1 deliverables.

Gate: distilled head must report accuracy/runtime/memory trade-offs against the frozen teacher and a non-JEPA baseline.

Completion evidence: the versioned TUM RGB-D chronological 64/16/8 train/validation/test split, compact metric-depth and
log-variance probe, VGGT auxiliary distillation, RGB+coordinate non-JEPA baseline, final-layer V-JEPA ablation,
four-layer V-JEPA candidate, three-seed 60-epoch training, validation-only checkpoint selection, held-out calibration,
latency/memory measurement, W&B tables/media/artifacts, and extensible comparison JSON schema all completed under Slurm.
Content-bound tests and real-model preflight passed before formal job `29587255`; the job produced ten result rows, nine
checkpoints, zero failures, a passing local/remote artifact audit, and finished W&B run `ikh4ptrb`.

The final-layer student is selected: its held-out AbsRel is 0.07523 ± 0.00384 versus RGB 0.19417 and frozen VGGT 0.12034,
while it runs 8.30× faster with 14.44× lower encoder peak memory than VGGT. The fixed four-layer average is not promoted
because its primary AbsRel is 4.44% worse at identical capacity and runtime, despite improvements in RMSE, aligned AbsRel,
Delta-1, and calibrated NLL. This is one-sequence evidence; independent scenes and learned layer fusion remain future gates.

## Phase 2c — cross-sequence geometry and learned fusion: complete

The camera-family-blocked gate trains on two Freiburg-1 recordings, selects checkpoints on Freiburg-2, and evaluates an
equal-weight macro over two held-out Freiburg-3 recordings. It adds exact bundle identities, train-only normalization,
three-seed learned residual fusion, per-sequence metrics, co-resident end-to-end profiling, online W&B, and strict
content-bound postflight.

Learned fusion improves final-layer macro AbsRel from 0.43807 to 0.41801 and improves both sequence means, but its measured
latency is 1.1655× final-only, above the frozen 1.10× threshold. It is therefore not promoted. Fixed averaging reaches
0.41054 and RGB reaches 0.40425, while V-JEPA and VGGT retain much better aligned geometry than RGB. The evidence points
to unseen-camera metric-scale transfer, not only representation shape, as the next geometry target.

Historical next gate, subsequently addressed by Phases 2d-2f: evaluate RGB, final, fixed, and learned variants on fresh
rotated camera-family or external sequences; add a preregistered randomized/interleaved latency confirmation and
scale-aware modeling without reusing Freiburg-3 as fresh confirmation data.

## Phase 2d — fusion, scale, and latency diagnostics: complete

Same-checkpoint interventions showed that zeroing the learned fusion gates changes raw AbsRel by only `0.000081`, while a
target-fitted per-image scale oracle reduces raw AbsRel by `61.61%`. Twelve independent A100 jobs measured learned/final
latency at `1.02262x` with a tight paired interval. The evidence rejects a learned-gate causal explanation and identifies
metric-scale transfer as the dominant modeling gap. Scale-oracle results remain target-fitted diagnostics, not deployable
performance.

## Phase 2e — factorized SUN RGB-D sensor transfer: complete, not promoted

Eight variants across three seeds completed under a sensor-blocked protocol. The fixed candidate improves held-out raw and
aligned AbsRel by `2.67%` and `6.22%`, but worsens absolute log-scale error by `13.82%`, worsens calibrated NLL, and reaches
`9.3578x` baseline head latency. Its shuffled-K control is vacuous because the held-out kv2 split contains one repeated K.
The candidate is not promoted; the opened Phase 2e kv2 set is development-only for future work.

## Phase 2f — detached scale/camera latency-first screen: complete, no survivor

The four-arm preregistration, 12 independent A100 latency replicas, 12 M0 camera-family training runs, guarded DIODE final,
and strict 73-task postflight completed. M1/M2/M3 fail the frozen head-latency gate at
`1.681x/3.606x/4.362x`; only M0 trains. This is an implementation/runtime rejection, not candidate-quality evidence.
Selector output is no survivor, and DIODE remains sealed. The graph used 12 base submissions and never exceeded eight
concurrent running allocations.

## Phase 2g — quality-first detached scale and camera conditioning: proposed

The revised next gate trains and evaluates every healthy M0-M3 arm before speed becomes a hard decision. It expands the
balanced SUN RGB-D development protocol, uses equal health-only hyperparameter selection, executes all four rotations and
three seeds, evaluates updated/stale/wrong/permuted-K and zero-field interventions, and selects only on frozen quality and
mechanism criteria. Efficiency remains a mandatory descriptive diagnostic. A later experiment optimizes only a frozen
quality survivor with prediction/metric parity; external confirmation remains separately preregistered and DIODE stays
sealed throughout Phase 2g. See the [full proposal](experiments/2026-06-29-phase2g-quality-first-proposal.md) and
[metric guide](METRICS.md). It is the geometry-stage proposal, not blanket authorization or the project-wide first task;
the shared validation registry, dataset-role audit, and Phase-2 prerequisites must be frozen first.

## Phase 3 — object slots and grounding: initial substrate complete

Delivered deterministic and real GroundingDINO detection backends, box masks, optional SAM2 image-prompt boundary,
V-JEPA token pooling, geometry centroids, cross-view/time association, deterministic IDs, evidence references,
affordance/state priors, JSON/NPZ/SQLite/scene-graph persistence, query CLI, interactive report, W&B diagnostics, and a
stagewise object-grounding smoke benchmark. The real GroundingDINO CPU path and W&B artifact flow have been exercised.

Remaining before declaring model-quality completion:

- pin, install, and exercise SAM2 image/video checkpoints;
- replace batch-local greedy IDs with transactional incremental identity ownership;
- evaluate AP, mask IoU, HOTA/IDF1, ID switches, false merges/splits, and calibration on labeled data;
- add ontology/language embedding normalization and localized descriptions;
- train compact JEPA slot, state, affordance, and uncertainty heads;
- propagate geometry covariance/frame provenance into slot poses;
- add explicit occlusion, split, merge, deletion, and re-identification events.

Initial identity evidence: a real V-JEPA2.1 ablation on DAVIS `dogs-scale` shows appearance-only F1 0.609 versus RGB
0.374, but IoU-only reaches 0.768 and current default/swept weighted fusion does not exceed it. This rejects treating raw
box-pooled V-JEPA tokens as a sufficient identity embedding and prioritizes mask-weighted multi-layer projection plus
motion-aware global assignment. See `DESIGN_IDENTITY_EVALUATION.md`.

Gate: object and identity metrics pass on static, cross-view, video, and occlusion fixtures; teacher dependencies remain
optional; no planner consumes raw masks or tensors.

## Phase 4 — persistent 4D memory: initial substrate complete

Delivered a bounded robot-centric active map, temporal object scene graph, observation/evidence histories, deterministic
episodic events, in-memory cosine retrieval, monotonic revisions, confidence decay, SQLite WAL records/event log/snapshots,
atomic update transactions, snapshot reload, event replay, task-aware LOD compression, expanded query APIs, interactive
memory report, W&B revision timelines, stagewise memory benchmark, demo, tests, and documentation.

Remaining before model-quality completion:

- durable identity split/merge/alias/tombstone operations;
- explicit frame graph and loop closure;
- building/floor/room/place hierarchy and region inference;
- persisted FAISS/Chroma/LanceDB-compatible embeddings and array storage;
- periodic snapshot policy, sequence watermarks, migrations, and concurrent writer ownership;
- external memory QA/scene-graph/robotics benchmarks and long-duration scaling;
- calibrated state/pose uncertainty and active verification based on task cost.

Gate: crash-safe reload, deterministic replay, bounded memory growth, query latency targets, and benchmarked
compression-versus-task curves.

## Phase 5 — latent dynamics and verified planning

Connect V-JEPA 2-AC or JEPA-WM prediction, action/proprioception conditioning, uncertainty/value heads, CEM/MPPI, symbolic
task graphs, behavior trees, observation-driven verification, and replanning. Add a mock robot before LeRobot or ROS 2.

Initial substrate complete: deterministic and trainable action-conditioned dynamics boundaries, uncertainty/value
outputs, multi-step rollout, seeded bounded CEM, dependency-checked task graphs, behavior-tree primitives, a stateful mock
robot, confidence-gated observation verification, failure attribution, bounded retry/replanning, JSON traces, CLI/HTTP
execution, and a deterministic recovery benchmark. The A100 CUDA contract smoke passes.

Remaining before model-quality or robot-execution completion:

- load and evaluate real V-JEPA 2-AC or JEPA-WM weights;
- train and calibrate dynamics, uncertainty, and value heads on action-conditioned data;
- connect continuous MPC proposals to symbolic skills, constraints, and collision checking;
- evaluate repeated episodes in a named simulator with latency and safety metrics;
- add asynchronous behavior-tree execution, cancellation, timeouts, and operator escalation;
- implement LeRobot/ROS 2 only after the simulator gate passes.

Gate: deterministic simulation episodes show subgoal evidence, failure attribution, safe uncertainty thresholds, and
recovery—not merely open-loop action generation.

## Phase 6 — benchmark expansion

Implement one official-small-subset adapter per stage, JSON/HTML aggregation, confidence intervals, and failure dashboards.
Then build Robo4D-JEPA with versioned assets and evaluation server.

Initial substrate complete: versioned dataset manifests with size/SHA-256 validation, a generated Robo4D-JEPA contract
asset, repeated six-stage execution, seeded bootstrap intervals, typed failure records, unified JSON/failure/HTML/Markdown
reports, an evaluation CLI, W&B tables/artifacts, and regression tests.

Remaining before Phase-6 model-quality completion:

- add one licensed, revision-pinned official mini subset and protocol adapter per stage;
- compute intervals over independent scenes/episodes and add paired baseline comparisons;
- expand Robo4D-JEPA from descriptors to distributable RGB/world-memory/execution episodes;
- implement evaluation-server schema validation, sandboxing, resource limits, and leaderboard policy;
- document dataset privacy, retention, license, and takedown procedures.

## Cross-cutting work

- dependency lock and container images for CPU, CUDA, and robotics;
- checkpoint manifests with hashes and licenses;
- structured logging and profiling;
- data privacy and retention controls;
- numerical tests for frames and uncertainty;
- model cards and risk assessment;
- migration notes for every public contract change.

## Explicit non-goals until gates pass

- real robot actuation;
- monolithic language-model-to-motor control;
- claims of metric single-image mapping;
- end-to-end finetuning without stage baselines;
- deploying heuristic confidence as a safety guarantee.
