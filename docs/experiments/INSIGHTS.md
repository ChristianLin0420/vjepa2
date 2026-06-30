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
| Cross-family geometry | Camera-family-blocked TUM training/validation and two held-out Freiburg-3 recordings | Learned fusion lowers final-layer macro AbsRel by 4.58% and improves both sequence means, but fixed averaging and RGB have better raw primary means. Gains mainly reduce scale error; aligned shape and uncertainty do not clearly improve. Candidate latency is 1.1655× final under the frozen profile. | Retain final by the registered gate; do not interpret three seeds as independent-scene evidence. | Fresh rotated/external camera families, scale-aware modeling, and an independently registered latency confirmation. |
| Fusion diagnostics | Same-checkpoint interventions, target-fitted scale oracles, and 12 independent A100 latency jobs | Original and zeroed learned gates differ by only `0.000081` raw AbsRel. A per-image scalar oracle reduces raw AbsRel from `0.41801` to `0.16046`; learned/final latency is tightly `1.02262×`. Camera correction provenance remains incomplete. | Reject a learned-gate causal explanation and keep Phase 2c's final-layer decision unchanged. | Treat scale recovery as a separate estimand and complete camera provenance before causal camera claims. |
| Factorized sensor transfer | Eight variants × three seeds on sensor-blocked SUNRGBD, with untouched kv2 final evaluation | The candidate improves raw/aligned AbsRel by `2.67%/6.22%` but worsens scale error by `13.82%`, calibrated NLL by `0.0675`, and head latency to `9.3578×`. Correct and shuffled `K` are identical because all kv2 samples share one intrinsic matrix. | Execution succeeds but promotion fails; do not continue `factorized_full_teacher` unchanged. | [Phase 2f proposal](2026-06-29-phase2f-scale-camera-proposal.md): detached scale, identifiable camera controls, latency-first screening, and a fresh final set. |
| Detached scale/camera screen | Preregistered four-arm development protocol, 12 interleaved A100 latency allocations, and 12 M0 camera-family rotations | M1/M2/M3 pass the parameter cap but fail the frozen head-latency gate at `1.681×/3.606×/4.362×`; only M0 proceeds to training. Its development mean is `0.20208` raw AbsRel, `0.14682` aligned AbsRel, and `0.11888` scale error. No enhanced arm is eligible, so no external-final comparison is opened. | Keep M0 as the operational baseline. This is an implementation/runtime rejection, not evidence that detached scale or camera conditioning cannot improve quality. | [Phase 2g quality-first proposal](2026-06-29-phase2g-quality-first-proposal.md): train and causally evaluate every healthy arm, record efficiency descriptively, then optimize only a frozen quality survivor. |
| Phase 2g instrumentation preflight | Governed synthetic M0-M3 A100 run with independent postflight and terminal online-W&B publication | All four arms complete 12/12 optimizer steps with zero forbidden-gradient leakage and four exact reloads. The backend artifact is downloaded and hash-verified before an 8-file terminal artifact and content-addressed terminal pass are published. This remains `contract-only`, not architecture-quality evidence. | The bounded training and evidence pipeline works; the v1 postflight/binding gap is closed without upgrading the scientific claim. | Finish Phase 2g data governance, manifests, metrics, controls, and preregistration; adapt the terminal contract to the formal DAG before authorizing Phase 2g-A. |
| Grounding | Real GroundingDINO integration | Teacher detections can become slots with JEPA and geometry evidence. Bootstrap association is not durable identity. | Preserve explicit evidence and verification boundaries. | SAM2 plus labeled detection/segmentation/tracking evaluation. |
| Identity | DAVIS sequence-level ablation | V-JEPA appearance beats RGB appearance, but IoU-only remains stronger than current fusion. | Learn mask-weighted projections and motion-aware assignment; retain IoU. | Multiple sequences, occlusion strata, global assignment. |
| Memory | Deterministic persistence lifecycle | Snapshot reload, event replay, histories, queries, and LOD agree on controlled updates. | Keep atomic records plus append-only events. | Long-duration real sequences, concurrency, and task-retention curves. |
| Planning | Deterministic closed-loop recovery plus real feature handoff | Explicit verification rejects low confidence; attributed control failure recovers with one bounded replan. | Keep evidence-gated task transitions. | Trained dynamics and repeated named-simulator episodes. |
| Benchmarking | Versioned six-stage contract suite | One manifest and report pipeline now produces intervals, typed failures, JSON/HTML/Markdown, and W&B artifacts. | Require this reporting contract for every future benchmark. | One official, licensed mini subset per stage. |
| Infrastructure | Content-bound Slurm tests, preflight, training, and postflight | The login node can build the pinned Conda environment and stage verified assets; the gated Slurm GPU runs completed successfully with sustained-CUDA and real-model checks. Formal job `29587255` completed on an A100-SXM4-80GB after the historical single-host Xid 79 incident. | Keep CPU-only preparation on the login node and require Slurm allocations for GPU tests and training. | Preserve health/content gates and escalate platform power/link/driver stability only if Xid 79 recurs. |

## Overall current status

The repository now has an end-to-end, inspectable contract pipeline through representation, geometry, grounding, identity,
persistent memory, verified planning, and benchmark aggregation. Phases 0–1 are implementation-complete at their stated
scope. The Phase-2 VGGT teacher baseline, Phase-2b geometry student, Phase-2c cross-family gate, Phase-2d causal audit,
Phase-2e sensor-blocked benchmark, and Phase-2f latency-first screen now provide a bounded geometry evidence chain. Phase 2d
showed that the learned fusion gates were not the cause of the observed behavior and confirmed that their true latency
overhead is small. Phase 2e showed that explicit factorization can improve held-out shape/raw error, but the current scale
head, camera control, and dense-ray implementation are not promotable. Phase 2f then rejected all three enhanced arms at
the preregistered head-latency gate before formal training; consequently it provides no enhanced-arm quality or camera-
causality result, and the fresh DIODE final set remains sealed. A governed synthetic Phase 2g instrumentation preflight
proved the M0-M3 optimizer, gradient-firewall, checkpoint, GPU-telemetry, independent postflight, backend round trip, and
terminal online-W&B wiring. It used no dataset or pretrained model and clears no Phase 2g-A scientific gate. Phases 3–6
have useful initial substrates and real integration evidence where noted, but are not model-quality or production complete.

The dominant scientific gaps are no longer interface construction. They are identifiable and efficient metric-scale/camera
modeling, calibrated geometry/object uncertainty, identity under occlusion, long-duration memory quality, trained
action-conditioned dynamics, simulator/hardware safety, and official independent benchmark subsets. The prior single-host
A100 failure is no longer a launch blocker because GPU tests and training now run through content-bound Slurm allocations;
recurring Xid 79 remains an infrastructure risk.

The [systematic validation plan](../VALIDATION_PLAN.md) and [stage plans](../validation/README.md) now define the priority
order. Plans do not overwrite the evidence above.

1. Complete Wave A once for the whole project: freeze Dataset A/B roles, access/license records, immutable manifests,
   target-opacity rules, a consumed-test ledger, shared statistics, and the common report/postflight contract.
2. In parallel, build and smoke-test official adapters for representation, grounding/identity, memory, dynamics/planning,
   and the two composed-system environments. These are loader/integration tasks until labeled baselines run.
3. Run Dataset-A reference baselines and health pilots stage by stage. No learned stage advances to formal training without
   a valid baseline, complete metric sanity checks, and a frozen claim boundary.
4. For geometry, the governed synthetic instrumentation preflight is complete. Preregister
   [Phase 2g-A quality-first architecture validation](2026-06-29-phase2g-quality-first-proposal.md) only after the shared
   foundation and Phase-2-specific scientific/governance prerequisites pass, and adapt the proven terminal contract to the
   formal artifact graph. Train M0-M3 fairly; efficiency is descriptive during selection.
5. If Phase 2g-A identifies a development survivor, optimize only that frozen architecture under prediction/metric parity,
   then consider one separately preregistered DIODE confirmation. Otherwise retain M0 and keep DIODE sealed.
6. Promote each remaining stage from Dataset A to its frozen Dataset B only after a development survivor exists; then
   compose admitted survivors in ManiSkill3 and LIBERO with repeated, paired episodes and stagewise attribution.
7. Preserve Slurm CUDA/content gates, semantic arrays, online W&B plus immutable local artifacts, the global eight-running-
   allocation cap, and escalation for recurring Xid 79 as a platform issue.

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
9. Cross-family metric depth can fail mainly through absolute scale even when aligned shape is good. Raw and aligned
   metrics, camera-family roles, and scale-error diagnostics must remain separate.
10. Efficiency decisions near a hard threshold need randomized/interleaved profiling and enough repetitions; noisy
    measurements still govern a frozen operational gate but should motivate a separately registered confirmation.
11. Same-checkpoint interventions are stronger causal evidence than retrained comparisons: Phase 2d's original and zeroed
    gates are effectively identical, so the Phase 2c result must not be credited to gate adaptation.
12. A validation sensor can select the wrong scale mechanism for a new sensor. Bias-only scale was best on RealSense
    validation (`0.26153` raw AbsRel) but collapsed to `0.26972` on kv2 versus the monolithic `0.19407`.
13. A control is useful only if it changes the treatment. Shuffling a single unique `K` is the identity operation; future
    camera claims need paired crop/resize transforms or a genuinely multi-intrinsics test.
14. Parameter count is not a runtime proxy. Phase 2e stayed within the `1.10×` parameter cap but its dense camera/ray path
    took `9.36×` the baseline head latency. Record component profiles throughout, but do not let a diagnostic speed screen
    substitute for scientific architecture validation when quality is the active research question.
15. Uncertainty ranking and calibration are distinct. The candidate improves AUSE while worsening calibrated NLL; both must
    remain separate gates, with calibration fitted only on validation.
16. A completed, postflight-clean pipeline is not a promoted model. Phase 2e execution passed while its operational model
    gate correctly failed.
17. A latency-first screen protects experimental budget but narrows the scientific claim. Phase 2f did not train M1–M3, so
    it rejects their current implementations, not their quality hypotheses or camera causality.
18. The latency estimand can reverse the apparent engineering conclusion. Phase 2f's synchronized head-only ratios are
    `1.681×/3.606×/4.362×`, while descriptive encoder-plus-head ratios are only `1.012×/1.039×/1.049×` because
    the frozen encoder dominates. Both are useful, but only the preregistered head-only interval governs this experiment.
19. Small-kernel overhead can dominate tiny heads. M2/M3 spend about `0.717` ms in the camera transform alone, identifying
    a concrete later target for precomputation, fusion, compilation, or graph capture if either architecture first proves
    scientifically useful.
20. Sensor-family variation exceeds seed variation for the M0 baseline: kv2 is strongest on raw error/NLL/AUSE, xtion is
    dominated by scale transfer, and RealSense retains good scale but the weakest aligned shape. Family-macro reporting is
    therefore more informative than pooled-frame confidence.
21. Scheduler provenance is part of experimental provenance. Phase 2f used 12 named base submissions for 73 logical tasks,
    never exceeded eight concurrent allocations, and records four same-ID operator requeues caused by nested-step stalls.
22. Architecture discovery and deployment optimization answer different questions. Train and compare healthy candidates
    for quality and causal mechanism first; only then make runtime a hard gate for the frozen scientific survivor.
23. A synthetic optimizer smoke validates wiring, not architecture quality or formal-training readiness. Governed v2 now
    binds exact local evidence, an independently downloaded backend artifact, terminal W&B publication, and a
    content-addressed pass; that stronger provenance still cannot upgrade controlled-fixture evidence into a scientific
    result.

## Rejected shortcuts

- Do not label a physically listed but unavailable A100 as a GPU run.
- Do not claim metric monocular scale from heuristic confidence.
- Do not treat a correct category with the wrong persistent ID as success.
- Do not mark a subgoal complete from a control return without fresh observational evidence.
- Do not promote deterministic smoke scores or degenerate intervals as model-quality benchmarks.
- Do not let online dashboards become the only copy of results or decisions.
- Do not claim camera sensitivity from a shuffled-`K` control when the split contains one unique intrinsic matrix.
- Do not infer deployability from parameter count without synchronized component and end-to-end latency.
- Do not reuse the opened Phase 2e kv2 test as an untouched Phase 2f final set.
- Do not reinterpret a frozen head-only latency gate after seeing a more favorable encoder-plus-head ratio.
- Do not claim an enhanced architecture or camera-conditioning hypothesis failed when its formal training was skipped.
- Do not open a reserved final set merely to produce a number after the selector returns no survivor.
- Do not use an implementation-speed rejection as a substitute for evaluating an unresolved architecture-quality hypothesis.
- Do not treat a synthetic optimizer/W&B smoke as authorization for a governed real-data training DAG.
