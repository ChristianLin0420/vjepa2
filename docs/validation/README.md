# JEPA-4D stage validation plans

This directory turns the broad JEPA-4D objective into persistent, stage-specific benchmark and TODO programs. These files
are plans, not completed evidence and not authorization to submit jobs.

| Stage | Plan | Minimum common evaluation portfolio | Current strongest evidence |
|---|---|---|---|
| Phase 0 | [Infrastructure and contracts](PHASE0_INFRASTRUCTURE.md) | Analytic fixtures + tiny official samples from every adapter | Contract-only |
| Phase 1 | [Representation](PHASE1_REPRESENTATION.md) | Something-Something V2 + EPIC-KITCHENS-100 | Real-checkpoint integration on generated frames |
| Phase 2 | [Geometry](PHASE2_GEOMETRY.md) | SUN RGB-D development + sealed DIODE transfer; TUM regression | Development benchmark; no DIODE score |
| Phase 3 | [Grounding and identity](PHASE3_GROUNDING_IDENTITY.md) | COCO/RefCOCO development + Flickr30K Entities transfer; DAVIS 2017 + MOT17 | Grounding integration and one DAVIS sequence |
| Phase 4 | [Persistent memory](PHASE4_MEMORY.md) | Ego4D episodic memory/EgoTracks + ScanNet v2 | Deterministic persistence fixture |
| Phase 5 | [Dynamics and planning](PHASE5_DYNAMICS_PLANNING.md) | D4RL + RoboNet offline dynamics; ManiSkill3/LIBERO closed loop | Deterministic dynamics/planning contract |
| Phase 6 | [Composed system](PHASE6_SYSTEM.md) | ManiSkill3 + LIBERO; Robo4D-JEPA attribution tracks | Generated benchmark-harness contract |

The common state machine, dataset-role policy, logging contract, Slurm constraints, and project execution waves are in
[the master validation plan](../VALIDATION_PLAN.md). Metric formulas and cross-phase boundaries are in
[the metric guide](../METRICS.md). The implemented foundation and remaining blockers are tracked in
[the Wave A implementation record](WAVE_A_FOUNDATION.md). Use [TEMPLATE.md](TEMPLATE.md) when adding a new stage.

## Authority order

When documents differ, use this order:

1. a frozen, hash-bound experiment preregistration for that execution;
2. the relevant stage validation plan;
3. the master validation plan;
4. the broader benchmark/roadmap discussion;
5. an exploratory proposal or notebook.

Historical preregistrations remain authoritative for their completed runs. A new plan does not retroactively change a
past metric, gate, split, or claim.

## Updating a stage plan

- Mark a TODO complete only when a linked receipt/result provides evidence.
- Move an opened test to the consumed-test ledger immediately.
- Add failed, negative, and no-survivor results; never keep only successful branches.
- Update dataset access/license notes when terms or official hosting change.
- Label each source as A1 primary development, A2 complementary development, B transfer/external, or C stress. A2 may
  support a distinct capability but never an independent-transfer claim.
- Record a new protocol version when metrics, splits, baselines, or gates change.
- Link promoted evidence from `docs/experiments/INDEX.md`; do not place exploratory numbers in the evidence map.

## Global execution constraints

- GPU model work runs through Slurm, not the login node.
- Use the approved account/partition fallback and tasks no longer than four hours.
- Use semantic job names and arrays; never exceed eight concurrently RUNNING allocations.
- Online W&B is mandatory for formal jobs, with immutable local JSON/tabular/checkpoint/HTML/PNG receipts and hashes.
- Raw restricted media/targets and credentials are never uploaded or committed.
- Scientific quality and mechanism are established before implementation speed becomes a hard promotion gate.
