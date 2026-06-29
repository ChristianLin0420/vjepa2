# JEPA-4D benchmark specification

## 1. Evaluation philosophy

JEPA-4D is decomposed because a single robot task-success number cannot identify whether a failure originated in visual
representation, geometry, tracking, object identity, memory, planning, or control. Every stage therefore has an adapter,
dataset protocol, metrics, uncertainty analysis, runtime report, and failure taxonomy. Closed-loop evaluation is added
only after individual stages are measurable.

Mocks test contracts and infrastructure. They are never included in model-quality tables. Real experiment records must
state model revision, checkpoint hash, input preprocessing, calibration availability, device, precision, and whether
alignment to ground truth was performed.

## 2. Adapter contract

Every benchmark implements `BenchmarkAdapter`:

```python
name: str
stage: str
requires_runtime_depth: bool
supports_single_image: bool
supports_multiview: bool
supports_video: bool
prepare(config)
run(model_or_system, split)
compute_metrics(predictions, ground_truth)
report() -> dict
```

Reports must be JSON-serializable and include input counts, success/failure counts, aggregate and per-scene metrics,
latency percentiles, peak memory, calibration availability, model identifiers, and categorized failures.

## 3. Phase 1 representation evaluation

### Datasets

- Something-Something-V2: temporal action discrimination.
- EPIC-KITCHENS-100: action anticipation and retrieval.
- Ego4D short-term anticipation: future object interaction.
- NYUv2: frozen dense-feature linear depth probe.
- TartanDrive: trajectory/pose probe and temporal consistency.

### Metrics

- top-1/top-5 or mean class accuracy;
- recall@k and mean average precision;
- depth RMSE, AbsRel, and delta thresholds for frozen probes;
- ATE/RPE for trajectory probes;
- adjacent-token cosine, cycle consistency, and occlusion recovery;
- throughput, latency p50/p95, peak accelerator memory, and artifact size.

### Required ablations

- V-JEPA 2 versus V-JEPA 2.1;
- final versus multi-layer tokens;
- framewise versus tubelet-aware temporal features;
- frozen versus selective unfreezing;
- view/time identity enabled versus disabled.

## 4. Phase 2 single-image geometry belief

### Datasets

NYUv2, ScanNet++, ARKitScenes, and Hypersim evaluate indoor monocular geometry. Co3D evaluates object-centric priors.
Ground-truth depth is used only for evaluation, never as runtime input.

### Geometry metrics

- AbsRel: mean `|d-d*|/d*`;
- squared relative error;
- RMSE and log RMSE;
- delta accuracy at 1.25, 1.25², and 1.25³;
- surface-normal angular error when normals are derived;
- point-cloud Chamfer/F-score when reference geometry exists.

Both scale-aware and median/least-squares aligned results must be reported. Alignment can diagnose shape quality but cannot
be presented as metric-scale performance.

### Uncertainty metrics

- Gaussian NLL from predicted mean/log-variance;
- expected calibration error over binned confidence;
- sparsification error and area under the sparsification-error curve;
- risk-coverage curves;
- correlation between confidence and true error;
- frequency of unsafe overconfidence above task thresholds.

### Active observation protocol

Given one image, estimate belief and propose a next view. Add a second observed view and measure reduction in depth, pose,
or point error. Report improvement per meter moved, per second, and per additional frame. A method that simply requests
many views is not efficient.

## 5. Phase 2 multi-view geometry

### Datasets

- DTU: controlled reconstruction and camera evaluation.
- Tanks and Temples: realistic scene reconstruction.
- ETH3D: high-quality geometry and camera trajectories.
- ScanNet++ and ARKitScenes: indoor view sets.
- TUM RGB-D, EuRoC, KITTI, and TartanAir: RGB-runtime odometry/SLAM-style evaluation.

### Camera metrics

- absolute trajectory error after clearly stated alignment;
- relative pose translational and rotational error;
- camera rotation geodesic error;
- focal length/principal-point error;
- pairwise pose AUC at standard angular thresholds.

### Dense metrics

- depth AbsRel/RMSE/delta;
- point accuracy and completeness;
- F-score at dataset-specific distance thresholds;
- normal consistency;
- reprojection and photometric consistency;
- track endpoint error and survival.

### Confidence calibration

Current `scale_confidence`, `pose_confidence`, and `reconstruction_confidence` are heuristic policy inputs. Before use in
robot safety decisions, fit calibration only on a training/calibration split and freeze it before testing. Report raw and
calibrated curves. Calibration must not use test labels.

## 6. Dynamic 4D tracking

TAPVid-3D, PointOdyssey, Dynamic Replica, and DAVIS/TAP-Vid-DAVIS cover long-range 3D or fallback 2D tracking. Metrics are
3D average position distance, visible/occluded accuracy, trajectory survival, world-frame consistency, and dynamic/static
separation. Report performance by occlusion length and camera/object motion magnitude.

## 7. Semantic scene graph and memory

DAAAM, NaVQA, OC-NaVQA, SG3D, ConceptGraphs-style evaluation, HOV-SG-style hierarchy, and ReMEmbR NaVQA assess object,
region, relation, and temporal memory. Metrics include object localization, region topology, relation F1, last-seen
retrieval, temporal QA, query latency, memory bytes/hour, and task accuracy versus compression.

Object identity evaluation must include merges, splits, reappearance after occlusion, and changed state. A correct class
with the wrong persistent identity is a memory failure.

## 8. Robotic memory and execution

RoboMME and RoboMemArena are prioritized for memory-dependent manipulation. Later execution uses VLABench, RoboCasa365,
LIBERO, ManiSkill, and CALVIN. Metrics include task success, normalized subgoal progress, memory recall, unnecessary
revisits, verification actions, replanning count, collision/control failures, latency, and evidence quality.

## 9. Robo4D-JEPA custom benchmark

Four tracks are planned:

1. single-image bootstrap and next-view utility;
2. multi-view 4D object-region-event memory;
3. delayed and hidden-state memory-dependent tasks;
4. long-horizon multi-room execution with verification and replanning.

Every episode emits a machine-readable event trace. Metrics include uncertainty calibration, query F1, spatial/temporal
relation accuracy, success, subgoal progress, verification efficiency, replanning quality, and real-time feasibility.

## 10. Current smoke benchmarks

`geometry-smoke` runs deterministic single- and two-view inputs and checks finite geometry, point count, and positive
multi-view confidence gain. This is an API regression test only. Phase 1 smoke similarly checks finite features, shape,
temporal cosine, and runtime.

Run:

```bash
python scripts/run_eval_stagewise.py
pytest jepa4d/tests -q
```

## 11. Failure taxonomy

Every failed sample receives one primary and optional contributing labels:

- `input_decode` or synchronization;
- `representation` correspondence/semantics;
- `geometry_depth`, `geometry_pose`, or `geometry_scale`;
- `tracking` or data association;
- `object_grounding` or state classification;
- `memory_insert`, `memory_retrieval`, or stale belief;
- `planning_grounding`, search, or symbolic decomposition;
- `verification` false acceptance/rejection;
- `control` or collision;
- `infrastructure` OOM, timeout, dependency, or service failure.

Unknown is permitted but must be inspected; it cannot become the dominant category.

## 12. Reproducibility checklist

- immutable config and random seed;
- Git commit and dirty-state record;
- model/checkpoint ID and SHA-256;
- dependency lock or environment export;
- hardware, precision, and memory;
- input manifest and dataset revision;
- calibration/alignment policy;
- metric implementation version;
- per-sample predictions and failures;
- W&B URL plus local Markdown and JSON/HTML reports.
