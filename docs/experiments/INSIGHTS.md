# JEPA-4D experiment insights and decision ledger

This is the compact cross-experiment synthesis. Detailed reproduction commands, artifacts, W&B panels, numerical results,
and limitations remain in the linked phase records and the [experiment index](INDEX.md). Update this file only when a
result changes a design decision; do not copy every scalar into it.

## Evidence chain

| Stage | Strongest current evidence | What was learned | Decision | Next gate |
|---|---|---|---|---|
| Representation | Real V-JEPA 2.1 integration | Local ViT-B produces finite multi-layer image/video features and supports a real CUDA handoff. | Keep V-JEPA as the representation substrate. | Official frozen-probe subsets and layer ablations. |
| Geometry | Official TUM RGB-D mini subset on A100 | VGGT gives strong aligned depth/point and translation structure on one sequence; orientation is weaker, raw variance needs large rescaling, and BF16 is much faster. | Freeze VGGT-1B as the measured teacher while preserving the aligned/single-sequence claim boundary. | Expand geometry evaluation to independent sequences and scenes. |
| Geometry student | Three-seed TUM RGB-D training and held-out evaluation on Slurm A100 | The final V-JEPA layer reaches 0.07523 ± 0.00384 AbsRel versus RGB 0.19417 and VGGT 0.12034, with 8.30× teacher speedup and 14.44× lower encoder peak memory. The fixed four-layer average is 4.44% worse on the primary metric but improves several secondary metrics. | Use the final layer by default; retain VGGT when aligned fidelity dominates and test learned fusion separately. | Independent scenes plus learned or validation-selected layer fusion. |
| Grounding | Real GroundingDINO integration | Teacher detections can become slots with JEPA and geometry evidence. Bootstrap association is not durable identity. | Preserve explicit evidence and verification boundaries. | SAM2 plus labeled detection/segmentation/tracking evaluation. |
| Identity | DAVIS sequence-level ablation | V-JEPA appearance beats RGB appearance, but IoU-only remains stronger than current fusion. | Learn mask-weighted projections and motion-aware assignment; retain IoU. | Multiple sequences, occlusion strata, global assignment. |
| Memory | Deterministic persistence lifecycle | Snapshot reload, event replay, histories, queries, and LOD agree on controlled updates. | Keep atomic records plus append-only events. | Long-duration real sequences, concurrency, and task-retention curves. |
| Planning | Deterministic closed-loop recovery plus real feature handoff | Explicit verification rejects low confidence; attributed control failure recovers with one bounded replan. | Keep evidence-gated task transitions. | Trained dynamics and repeated named-simulator episodes. |
| Benchmarking | Versioned six-stage contract suite | One manifest and report pipeline now produces intervals, typed failures, JSON/HTML/Markdown, and W&B artifacts. | Require this reporting contract for every future benchmark. | One official, licensed mini subset per stage. |
| Infrastructure | Content-bound Slurm tests, preflight, training, and postflight | The login node can build the pinned Conda environment and stage verified assets; the gated Slurm GPU runs completed successfully with sustained-CUDA and real-model checks. Formal job `29587255` completed on an A100-SXM4-80GB after the historical single-host Xid 79 incident. | Keep CPU-only preparation on the login node and require Slurm allocations for GPU tests and training. | Preserve health/content gates and escalate platform power/link/driver stability only if Xid 79 recurs. |

## Overall current status

The repository now has an end-to-end, inspectable contract pipeline through representation, geometry, grounding, identity,
persistent memory, verified planning, and benchmark aggregation. Phases 0–1 are implementation-complete at their stated
scope. The Phase-2 VGGT teacher baseline and Phase-2b geometry student now have bounded single-sequence quality evidence.
Phase 2b completed all nine registered training runs, selected the final-layer V-JEPA student, and rejected promotion of
the fixed four-layer average on the primary metric. Phases 3–6 have useful initial substrates and real integration evidence
where noted, but are not model-quality or production complete.

The dominant scientific gaps are no longer interface construction. They are calibrated geometry/object uncertainty,
identity under occlusion, long-duration memory quality, trained action-conditioned dynamics, simulator/hardware safety,
and official independent benchmark subsets. The prior single-host A100 failure is no longer a launch blocker because
GPU tests and training now run through content-bound Slurm allocations; recurring Xid 79 remains an infrastructure risk.

Priority order:

1. repeat geometry/student evaluation on independent scenes and test learned or validation-selected layer fusion;
2. add licensed official mini subsets for representation and the remaining stages;
3. train or integrate real action-conditioned dynamics and evaluate repeated simulator recovery episodes;
4. improve mask-weighted identity projection and measure long-sequence memory/calibration quality;
5. preserve Slurm CUDA/content gates and escalate recurring Xid 79 as a platform/firmware/power/link issue.

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
7. Fixed layer averaging is not automatically better than a strong final-layer baseline: Phase 2b lost 4.44% on primary
   AbsRel while improving several secondary metrics, so learned fusion needs its own registered evaluation.
8. Login-node environment preparation and asset staging are appropriate, but GPU health, real-model equivalence,
   optimization, profiling, and formal training belong in reproducible Slurm allocations.

## Rejected shortcuts

- Do not label a physically listed but unavailable A100 as a GPU run.
- Do not claim metric monocular scale from heuristic confidence.
- Do not treat a correct category with the wrong persistent ID as success.
- Do not mark a subgoal complete from a control return without fresh observational evidence.
- Do not promote deterministic smoke scores or degenerate intervals as model-quality benchmarks.
- Do not let online dashboards become the only copy of results or decisions.
