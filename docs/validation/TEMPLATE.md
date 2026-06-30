# Stage validation plan template

> Copy this file only when adding a new stage or splitting a stage into a materially different benchmark program. Existing
> stage files should be updated in place so TODOs and decisions remain discoverable.

## Status and ownership

| Field | Value |
|---|---|
| Stage | `<stage>` |
| Plan status | `draft / approved / active / superseded` |
| Current evidence level | `<contract / integration / sequence / benchmark / closed-loop>` |
| Strongest result record | `<relative link>` |
| Next decision | `<one sentence>` |

## Objective and system role

- Stage input:
- Stage output:
- Downstream consumers:
- Primary scientific question:
- Claims explicitly out of scope:

## Current evidence and gaps

| Existing experiment | Dataset/fixture | Split/unit | What it establishes | What remains open |
|---|---|---|---|---|
| `<record>` | `<source>` | `<count>` | `<bounded claim>` | `<gap>` |

## Dataset portfolio

| Role | Dataset/release | Independent unit | Official split | License/access | Intended claim |
|---|---|---|---|---|---|
| Primary development A1 |  |  |  |  |  |
| Complementary development A2 (optional) |  |  |  |  |  |
| Transfer/external B (required for L2) |  |  |  |  |  |
| Optional stress C |  |  |  |  |  |

### Dataset TODO

- [ ] Verify official source, release, license, citation, privacy, and redistribution.
- [ ] Estimate download, extracted storage, cache, and runtime costs.
- [ ] Freeze mechanical eligibility, independent split unit, counts, and seed.
- [ ] Write manifest with selected/rejected IDs and source hashes.
- [ ] Prove target paths are denied to training/selection jobs.
- [ ] Add the opened split to the consumed-test ledger.

## Baselines and model matrix

| ID | Model/system | Purpose | Frozen initialization/source | Trainable components |
|---|---|---|---|---|
| B0 | Trivial/heuristic | Metric sanity |  |  |
| B1 | Current JEPA-4D reference | Regression anchor |  |  |
| B2 | Common published/pretrained reference | External context |  |  |
| C1 | Candidate | Primary hypothesis |  |  |

## Metrics and aggregation

| Metric | Role | Direction | Unit | Aggregation | Gate |
|---|---|---:|---|---|---:|
|  | Primary |  |  |  |  |

State valid denominators, missing-data behavior, alignment, calibration split, and independent resampling unit. Link to
[the common metric guide](../METRICS.md) and version any stage-specific implementation.

## Experiment ladder

### L0 contract and official-mini smoke

- [ ] Real loader/checkpoint/metric forward pass.
- [ ] Analytic metric and coordinate tests.
- [ ] W&B/local artifact round-trip.

### L1 development benchmark

- [ ] Equal-budget health pilot and tuning on train/validation only.
- [ ] Complete formal model/seed/split matrix.
- [ ] Freeze checkpoints before development-test evaluation.

### L2 transfer benchmark

- [ ] Freeze exactly one development survivor.
- [ ] Evaluate Dataset B without retuning.

### L3 mechanism and uncertainty

- [ ] Same-checkpoint intervention for every causal claim.
- [ ] Validation-only calibration and frozen held-out scoring.

### L4 operational optimization

- [ ] Optimize only a quality survivor.
- [ ] Require prediction/metric parity before accepting speed/memory gains.

### L5 composition

- [ ] Feed frozen outputs into the next stage.
- [ ] Measure downstream impact and retain stagewise failure attribution.

## Promotion and stop rules

| Gate | Exact requirement | Failure decision |
|---|---|---|
| Health |  |  |
| Primary quality |  |  |
| Secondary/non-inferiority |  |  |
| Mechanism |  |  |
| Calibration/safety |  |  |
| Integrity |  |  |
| Operational, after quality |  |  |

## Logging and visualization

- [ ] Explicit step/epoch/frame/episode axis and all component losses.
- [ ] Per-independent-unit metrics and paired baseline differences.
- [ ] Mean plus distributions/intervals.
- [ ] Fixed preselected qualitative panels.
- [ ] Calibration, risk, failure, and worst-case views where applicable.
- [ ] Descriptive throughput, memory, latency, and GPU telemetry.
- [ ] JSON/JSONL, CSV/Parquet/NPZ, checkpoint, PNG, self-contained HTML, receipts, hashes.
- [ ] Unique online W&B run/artifact per logical job; no raw restricted targets or credentials.

## Slurm and resource plan

- Base submissions:
- Logical tasks:
- Array mappings and `%8` caps:
- CPU/memory/GPU/time per class:
- Four-hour checkpoint/chunk policy:
- Dependency and failure/retry semantics:

No more than eight allocations may be RUNNING globally. GPU model work never runs on the login node.

## Claim boundary and risks

- Dataset/domain limitations:
- Selection/tuning limitations:
- Calibration/safety limitations:
- Downstream limitations:
- License/privacy risks:
- Infrastructure risks:

## Active TODO and decision log

- [ ] `<next concrete task>`

| Date | Evidence/change | Decision | Record |
|---|---|---|---|
| `<date>` | `<fact>` | `<action>` | `<link>` |
