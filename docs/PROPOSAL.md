# JEPA-4D research proposal

## Implementation status update: Phase 4

The proposal now has an implemented semantic evidence boundary. GroundingDINO detections or deterministic mocks become
typed observations; optional SAM2 prompting refines masks; V-JEPA tokens supply appearance evidence; geometry beliefs
supply conservative centroids; association creates persistent-in-result object slots; and JSON, NPZ, SQLite, scene graph,
interactive HTML, and W&B outputs make each run inspectable. A real GroundingDINO CPU smoke test completed successfully.

The repository now also has a Phase 4 persistent-memory substrate: bounded robot-centric context, temporal object history,
episodic evidence, vector retrieval, atomic SQLite records/event log/snapshots, reload/replay parity, confidence decay,
task-aware LOD, planner-safe history queries, benchmark smoke tests, interactive reports, and W&B revision timelines.

This narrows the next research question. The immediate challenge is no longer whether observations can be persisted, but
whether identity, relations, and uncertainty remain reliable under occlusion, repeated categories, camera motion,
long-duration updates, and compression. Current deterministic persistence is infrastructure evidence, not proof of memory
quality or task improvement.

## Abstract

JEPA-4D investigates whether pretrained V-JEPA 2.1 dense spatiotemporal features can form a shared substrate for RGB-first
robot geometry, persistent object/event memory, latent dynamics, and verified long-horizon planning. The system combines
learned teachers only where first principles require information absent from JEPA alone: geometry, segmentation and
language grounding, action-conditioned dynamics, and robot execution. Runtime interfaces remain explicit and queryable.

## Problem

Robot perception systems often optimize isolated reconstruction or recognition while long-horizon tasks fail because the
robot cannot retain object identity, know when a belief is uncertain, retrieve prior evidence, or verify a changed state.
Conversely, policy-first systems may hide perception and memory inside activations that cannot be inspected or queried.
JEPA-4D treats task-relevant memory as the product and representation, geometry, and planning as measurable contributors.

## Central hypothesis

Dense V-JEPA 2.1 features provide temporally stable semantics and correspondence structure that reduce the cost of
building geometry-aware, persistent robot memory. Explicit geometric and epistemic adapters can convert this substrate into
beliefs suitable for structured planning without forcing the planner to consume raw tensors.

## Research questions

1. Which V-JEPA layers best support depth, pose, tracking, identity, and affordance prediction?
2. Does a JEPA-conditioned geometry student retain VGGT quality with lower latency and memory?
3. Do geometry plus JEPA features improve identity survival through occlusion over appearance or mask-IoU baselines?
4. Can uncertainty-driven observation reduce failure and unnecessary revisits?
5. Which memory compression policy preserves task-relevant evidence over hours?
6. Does latent prediction improve short-horizon action selection once memory and verification are controlled?

## Falsifiable hypotheses

- H1: multi-layer V-JEPA features improve temporally consistent depth/track metrics over final-layer-only features.
- H2: a distilled geometry head achieves a favorable accuracy-latency Pareto point relative to VGGT and a DINO baseline.
- H3: geometry-aware JEPA association increases correct re-identification after occlusion.
- H4: calibrated uncertainty reduces unsafe false verification at fixed observation cost.
- H5: hierarchical memory improves delayed-query accuracy per stored byte over raw replay and flat vector retrieval.
- H6: verified task graphs recover from injected perception/memory failures more often than open-loop plans.

Failure to beat the specified baseline rejects the corresponding hypothesis; architectural complexity alone is not a
positive result.

## Architecture rationale

V-JEPA is the representation core because it is trained to predict latent video structure and exposes dense features.
VGGT is a Phase 2 teacher because metric camera/depth/point inference is underdetermined by JEPA features alone. Object
teachers provide mask and language supervision. The memory core stores explicit entities and evidence. Dynamics predicts
future latent/object state. The planner uses typed queries and verification rather than raw activations.

## Single-image policy

One image initializes a belief, never a complete map. The system must expose weak scale/pose confidence unless calibration
or a legitimate metric prior exists. Evaluation separates scale-aligned shape from metric accuracy and measures the value
of the next observation. Hallucinated hidden geometry is a failure even if visually plausible.

## Experimental program

### Representation

Probe final and intermediate V-JEPA layers on temporal classification, anticipation, depth, and trajectory tasks. Measure
frozen, LoRA, and selective-unfreezing regimes.

### Geometry

Benchmark VGGT teacher output, uncertainty calibration, view-count scaling, and coordinate correctness. Train JEPA
students with depth/point/pose/track/NLL losses. Compare with RGB geometry baselines under identical preprocessing.

### Object and memory

Associate teacher masks using appearance, JEPA, geometry, and combinations. Replay long sequences with controlled object
movement and occlusion. Measure identity, temporal QA, query latency, storage, and compression.

### Planning

Start with deterministic simulation. Compare open-loop, query-only, verification, and verification-plus-latent-MPC systems.
Inject stage failures and require attribution.

## Required ablations

- no JEPA, final JEPA, and multi-layer JEPA;
- no geometry versus teacher versus distilled geometry;
- flat vector memory versus graph versus graph plus episodic events;
- uncalibrated versus calibrated uncertainty;
- no verification versus fixed verification versus uncertainty-driven verification;
- symbolic-only versus latent-MPC-assisted execution.

## Success criteria

Success requires stage improvements with confidence intervals, reproducible artifacts, bounded runtime/memory, calibrated
beliefs, and closed-loop gains traceable to measured stages. A compelling visualization without quantitative memory or
task improvement is insufficient.

## Risks

- teacher bias may be distilled into JEPA heads;
- relative geometry may be mistaken for metric scale;
- similar objects may cause persistent identity errors;
- stale memory may be more harmful than no memory;
- W&B or interactive reports may conceal non-reproducible preprocessing;
- heavy models may prevent robot-rate operation;
- benchmark leakage may invalidate uncertainty calibration.

Mitigations include independent ground truth, explicit alignment policy, confidence thresholds, evidence timestamps,
failure traces, immutable configs, and lightweight students.

## Responsible-use boundary

The project targets research in indoor perception, memory, and manipulation. It does not authorize military, surveillance,
harmful, critical-infrastructure, or heavy-machinery deployment. External model licenses and acceptable-use policies apply.
Real robot work requires separate safety review, workspace limits, collision monitoring, and human emergency control.

## Expected contributions

1. a stable RGB/view/time and geometry-belief API around V-JEPA 2.1;
2. a benchmarked geometry distillation recipe;
3. queryable hierarchical 4D robot memory with uncertainty and evidence;
4. a verified planning interface joining latent and symbolic reasoning;
5. stagewise and closed-loop evaluation with failure attribution.
