# Experiment title

> Copy this file for a promoted experiment. Delete instructional text, keep the headings, and add stage-specific
> appendices only when they clarify evidence. Generated per-run records use the same structure.

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `<date>-<stage>-<dataset>-<purpose>-s<seed>` |
| Stage | `representation / geometry / grounding / memory / identity / dynamics / planning / closed-loop` |
| Status | `planned / running / complete / failed / superseded` |
| Evidence level | `contract-only / integration / sequence-level / benchmark / training / closed-loop` |
| Parent / supersedes | `<experiment ID or none>` |
| Timestamp | `<ISO-8601 UTC>` |
| Git commit / dirty | `<commit> / <true-or-false>` |
| W&B project / run | `<project / URL or disabled>` |
| Owner | `<name>` |

## Question and decision

- Objective: What exact uncertainty is this run intended to reduce?
- Hypothesis: State a falsifiable expectation before interpreting plots.
- Success criteria: Give numerical or contract-level thresholds.
- Decision: State what changes—or deliberately does not change—because of this result.

## Stage results and insights

| Stage | Implementation | Status | Inputs | Outputs | Evidence | Insight / decision |
|---|---|---|---|---|---|---|
| input | `<loader>` | pass | `<manifest>` | normalized RGB | checksum/count | `<interpretation>` |
| features | `<model>` | pass | RGB | JEPA tokens | finite/shape/consistency | `<interpretation>` |
| downstream | `<adapter>` | pass | tokens/belief | task output | task metric | `<interpretation>` |

Distinguish observation from interpretation. For example, “F1 = 0.61” is evidence; “appearance helps under overlap” is
an interpretation; “retain appearance in the next tracker” is a decision.

## Reproduction configuration

Record the exact command, resolved config, input manifest and checksum, model source/revision, checkpoint SHA-256,
preprocessing, calibration, device, precision, dependency lock, seed, and relevant environment variables. Never record
credentials.

```bash
# exact command
```

```json
{"resolved_config": "..."}
```

## W&B dashboard reading guide

| Panel / namespace | Type | What it answers | Healthy or expected pattern | Observed result | Interpretation / action |
|---|---|---|---|---|---|
| `pipeline/stage_latency_s` | bar/time series | Which stage dominates latency? | Stable after warm-up | `<value>` | `<action>` |
| `<stage>/<metric>` | scalar/line | Did the primary metric improve? | Defined before run | `<value>` | `<action>` |
| `<stage>/<distribution>` | histogram | Is an average hiding collapse/outliers? | Finite, plausible support | `<shape>` | `<action>` |
| `<stage>/<qualitative>` | image/video/table | Do predictions fail in a systematic spatial or temporal pattern? | Consistent with numeric result | `<observation>` | `<action>` |

Panel rules:

- Group keys by stage: `features/`, `geometry/`, `objects/`, `memory/`, `identity/`, `dynamics/`, `planning/`,
  `training/`, `system/`, and `pipeline/`.
- Every important scalar gets a definition, unit, direction (`higher` or `lower` is better), aggregation, and step axis.
- Pair means with distributions or per-example tables. Pair qualitative media with machine-readable artifacts.
- Training runs show loss components, learning rate, gradient/parameter norms, throughput, memory, validation metrics,
  calibration, and checkpoint-selection score. Inference runs show stage/cumulative latency and output diagnostics.
- Put exploratory and held-out metrics in visibly different namespaces; never select and evaluate on the same data
  without labeling it exploratory.

## Numerical results

| Dataset / split | Variant | Primary metric | Secondary metrics | Runtime | Seed / CI |
|---|---|---|---|---|---|
| `<versioned split>` | `<configuration>` | `<value>` | `<values>` | `<value>` | `<seed or interval>` |

Include metric definitions, alignment/calibration policy, denominators, missing-data handling, and comparison baseline.

## Artifacts

| Path / W&B artifact | Type | Checksum / version | Purpose |
|---|---|---|---|
| `<path>` | metrics JSON | `<sha256>` | machine-readable result |
| `<path>` | interactive HTML | `<sha256>` | linked diagnostic views |

## Failures and supersession

Record the last trustworthy stage, exception or failure mode, impact on partial metrics, remediation, and replacement run.
Do not silently remove failed runs.

## Claim boundary and limitations

- State what this experiment proves.
- State what it does not prove.
- Identify dataset, calibration, scale, sample-size, selection, and hardware limitations.

## Next experiments

| Priority | Experiment | Uncertainty reduced | Promotion criterion | Dependency |
|---|---|---|---|---|
| P0 | `<next run>` | `<question>` | `<threshold>` | `<artifact/code>` |

## Optional stage-specific appendices

Add appendices for per-layer representation probes, geometry calibration, object-level error taxonomy, memory lifecycle,
identity operating-point sweeps, training checkpoint selection, or closed-loop failure attribution. Preserve the canonical
headings above so indexers and future tooling remain stable.
