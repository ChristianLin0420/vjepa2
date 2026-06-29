# JEPA-4D implementation roadmap

## Guiding rule

Advance only when the previous phase passes unit, integration, artifact, and documentation gates. A new model integration
does not compensate for unstable contracts. Mocks remain available at every heavy boundary so CI stays offline and CPU
runnable.

## Phase 0 — repository and contracts: complete

Delivered package structure, typed RGB/token/geometry/object/memory contracts, deterministic mocks, query API, FastAPI
service, packaging, Ruff, mypy, pytest, pre-commit, CI, demos, and experiment ledger.

Exit evidence: JEPA-4D tests pass offline; server health and query endpoints respond; no heavy checkpoint is needed.

## Phase 1 — V-JEPA 2.1 substrate: complete

Delivered four-mode RGB loading, view/time identity, ViT-B/L/g/G configuration, native and HF-compatible checkpoints,
dense/global/multi-layer tokens, tubelet validity, PyTorch/Zarr output, PCA and temporal reports, and W&B diagnostics.

Exit evidence: mock and real ViT-B inference pass; feature artifacts are finite and reproducible; intermediate layers are
available; upstream tests remain green.

## Phase 2 — geometry belief: implemented substrate, calibration work remains

Delivered deterministic mock, official VGGT package/checkpoint boundary, real single/multi-view CPU smoke tests, camera
matrices, depth/point mean and log-variance, tracks, confidence policy, NPZ/PLY export, interactive 3D reports, geometry
W&B panels, configuration, benchmark smoke test, and documentation.

Remaining before declaring model-quality completion:

- run official datasets with immutable manifests;
- validate coordinate conventions against known cameras;
- add pose/depth/point metrics and per-sample failure reports;
- calibrate uncertainty on held-out splits;
- add COLMAP export and optional bundle adjustment;
- profile CUDA precision, memory, and view-count scaling;
- decide whether VGGT-Omega or MapAnything merits a separate backend rather than silently replacing VGGT.

## Phase 2b — JEPA geometry distillation

Train a lightweight head over frozen or selectively tuned V-JEPA 2.1 layers using VGGT/DUSt3R/MASt3R pseudo-labels and
ground truth where licensed. Losses: scale-invariant depth, point L1/Huber, pose geodesic/translation, track loss, Gaussian
NLL, confidence calibration, cross-view reprojection, and temporal consistency.

Gate: distilled head must report accuracy/runtime/memory trade-offs against the frozen teacher and a non-JEPA baseline.

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

Gate: deterministic simulation episodes show subgoal evidence, failure attribution, safe uncertainty thresholds, and
recovery—not merely open-loop action generation.

## Phase 6 — benchmark expansion

Implement one official-small-subset adapter per stage, JSON/HTML aggregation, confidence intervals, and failure dashboards.
Then build Robo4D-JEPA with versioned assets and evaluation server.

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
