# JEPA-4D experiment insights and decision ledger

This is the compact cross-experiment synthesis. Detailed reproduction commands, artifacts, W&B panels, numerical results,
and limitations remain in the linked phase records and the [experiment index](INDEX.md). Update this file only when a
result changes a design decision; do not copy every scalar into it.

## Evidence chain

| Stage | Strongest current evidence | What was learned | Decision | Next gate |
|---|---|---|---|---|
| Representation | Real V-JEPA 2.1 integration | Local ViT-B produces finite multi-layer image/video features and supports a real CUDA handoff. | Keep V-JEPA as the representation substrate. | Official frozen-probe subsets and layer ablations. |
| Geometry | Official TUM RGB-D mini subset on A100 | VGGT gives strong aligned depth/point and translation structure on one sequence; orientation is weaker, raw variance needs large rescaling, and BF16 is much faster. | Freeze VGGT-1B as the measured teacher while preserving the aligned/single-sequence claim boundary. | Run the prepared Phase-2b teacher/RGB/final/multi-layer comparison after stable CUDA recovery. |
| Grounding | Real GroundingDINO integration | Teacher detections can become slots with JEPA and geometry evidence. Bootstrap association is not durable identity. | Preserve explicit evidence and verification boundaries. | SAM2 plus labeled detection/segmentation/tracking evaluation. |
| Identity | DAVIS sequence-level ablation | V-JEPA appearance beats RGB appearance, but IoU-only remains stronger than current fusion. | Learn mask-weighted projections and motion-aware assignment; retain IoU. | Multiple sequences, occlusion strata, global assignment. |
| Memory | Deterministic persistence lifecycle | Snapshot reload, event replay, histories, queries, and LOD agree on controlled updates. | Keep atomic records plus append-only events. | Long-duration real sequences, concurrency, and task-retention curves. |
| Planning | Deterministic closed-loop recovery plus real feature handoff | Explicit verification rejects low confidence; attributed control failure recovers with one bounded replan. | Keep evidence-gated task transitions. | Trained dynamics and repeated named-simulator episodes. |
| Benchmarking | Versioned six-stage contract suite | One manifest and report pipeline now produces intervals, typed failures, JSON/HTML/Markdown, and W&B artifacts. | Require this reporting contract for every future benchmark. | One official, licensed mini subset per stage. |
| Infrastructure | Kernel/NVIDIA diagnostics | A100 intermittency is a PCIe link-loss failure (`Xid 79: GPU has fallen off the bus`), not a Python or CUDA-package issue. FLR, isolated bridge reset, unbind, and remove/rescan did not restore link state. | Require host reboot, then preserve health gates before every GPU run. | Reboot, validate sustained CUDA load, then investigate platform power/link/driver stability if Xid 79 recurs. |

## Overall current status

The repository now has an end-to-end, inspectable contract pipeline through representation, geometry, grounding, identity,
persistent memory, verified planning, and benchmark aggregation. Phases 0–1 are implementation-complete at their stated
scope. The Phase-2 VGGT teacher baseline now has bounded official-subset quality evidence. Phases 3–6 have useful initial
substrates and real integration evidence where noted, but are not model-quality or production complete. Phase 2b is
implementation-ready but has no training result because the A100 link failed before its first W&B run initialized.

The dominant scientific gaps are no longer interface construction. They are calibrated geometry/object uncertainty,
identity under occlusion, long-duration memory quality, trained action-conditioned dynamics, simulator/hardware safety,
and official independent benchmark subsets. The dominant infrastructure blocker is the unstable A100 PCIe link.

Priority order:

1. reboot and validate sustained A100 health; escalate recurring Xid 79 as a platform/firmware/power/link issue;
2. run the pinned Phase-2b comparison without changing its split or promotion criteria;
3. add licensed official mini subsets for representation and the remaining stages;
4. train or integrate real action-conditioned dynamics and evaluate repeated simulator recovery episodes;
5. improve mask-weighted identity projection and measure long-sequence memory/calibration quality.

## Cross-cutting insights

1. Integration evidence and quality evidence must remain separate. A real checkpoint completing a path proves
   interoperability, not accuracy or task value.
2. Explicit uncertainty is useful only when paired with a threshold, verification action, and false-acceptance metric.
3. Raw pretrained appearance features help identity but do not yet displace geometric/IoU continuity.
4. Current deterministic persistence and planning results validate invariants and recovery control flow, not open-world
   memory or robot safety.
5. Benchmark confidence intervals are meaningful only over independent scenes or episodes. Repeating a deterministic
   fixture measures harness variation and must stay labeled contract-only.
6. W&B is the interactive comparison surface; versioned local JSON, HTML, Markdown, manifests, and hashes are the durable
   reproducibility record.
7. The A100 failure is confirmed below the framework layer: PCI revision `ff`, missing data-link active state, and NVIDIA
   Xid 79 persist across all safe userspace reset paths.

## Rejected shortcuts

- Do not label a physically listed but unavailable A100 as a GPU run.
- Do not claim metric monocular scale from heuristic confidence.
- Do not treat a correct category with the wrong persistent ID as success.
- Do not mark a subgoal complete from a control return without fresh observational evidence.
- Do not promote deterministic smoke scores or degenerate intervals as model-quality benchmarks.
- Do not let online dashboards become the only copy of results or decisions.
