# Phase 6 versioned benchmark-harness experiment

## Experiment metadata

| Field | Value |
|---|---|
| Experiment ID | `phase6-contract-v0` |
| Stage / status | `benchmark aggregation / complete` |
| Evidence level | `contract-only` |
| Dataset manifest | `robo4d-jepa-contract` version `0.1.0`, revision `fixture-v0` |
| W&B | [63j8m3cp](https://wandb.ai/crlc112358/jepa4d-worldmodel/runs/63j8m3cp) |
| Decision | Require the Phase-6 reporting contract for future official subset adapters. |

## Objective and hypothesis

Validate that one checksum-verified dataset manifest can drive repeated adapters across representation, geometry,
grounding, identity, memory, and planning while producing deterministic confidence intervals, typed failures, local
JSON/HTML/Markdown artifacts, and a synchronized W&B dashboard.

The hypothesis was that every current contract adapter would complete five repetitions with finite, stable metrics and no
uncategorized infrastructure failures.

## Configuration

- suite: `phase6-contract-v0`;
- stages: representation, geometry, object grounding, tracking/identity, memory, planning;
- repetitions: 5 per stage, 30 stage runs total;
- bootstrap: 2,000 seeded resamples at 95% confidence;
- manifest integrity: asset byte size and SHA-256 required;
- runtime: CPU because the A100 remained unavailable at PCI revision `ff`;
- claim boundary: generated deterministic contract fixtures only.

## Results and insights

| Benchmark | Successes | Failures | Mean harness latency | Main insight |
|---|---:|---:|---:|---|
| representation-smoke | 5 | 0 | 17.378 ms | All three input modes remain finite and shape-valid. |
| geometry-smoke | 5 | 0 | 4.648 ms | Multi-view confidence gain remains deterministic at 0.08. |
| object-grounding-smoke | 5 | 0 | 1.820 ms | Association and ID invariants hold on the controlled fixture. |
| identity-association-smoke | 5 | 0 | 23.792 ms | Oracle appearance retains a 0.5614 F1 gap over ambiguous appearance. |
| memory-smoke | 5 | 0 | 16.477 ms | Reload/replay parity and query/history invariants remain 1.0. |
| planning-smoke | 5 | 0 | 0.283 ms | Failure attribution, recovery, and verified progress remain 1.0. |

The suite recorded zero failures. Memory query latency was 0.0776 ms with a bootstrap interval of
`[0.0755, 0.0800]` ms over five executions. Most model/invariant metrics have degenerate intervals because the fixtures are
deterministic. That is the expected result and demonstrates why these intervals cannot be interpreted as population or
model uncertainty.

## W&B dashboard reading guide

| Panel / artifact | Question | Interpretation |
|---|---|---|
| `benchmark/stage_table` | Did every stage and repetition complete? | Six stages completed with five successes and zero failures each. |
| `benchmark/metric_estimates` | Are means, intervals, and sample counts inspectable together? | Contract metrics are stable; degenerate intervals reflect deterministic fixtures. |
| `benchmark/failure_table` | Were failures retained rather than hidden? | Empty for this run; injected-failure unit tests exercise the schema. |
| `benchmark/latency_ms` | Which fixture dominates harness time? | Identity and representation are largest, but these are not capacity benchmarks. |
| JSON/HTML/Markdown artifacts | Can the result be reproduced without W&B? | The complete local record is versioned independently of the dashboard. |

Cloud verification found the run in `finished` state with result `success`, six completed stages, zero failures, and seven
logged table/report artifacts.

## Verification

- Ruff: pass;
- mypy: pass across 76 JEPA-4D source files;
- combined upstream and JEPA-4D pytest: 75 passed, 3 CUDA-dependent tests skipped, 14 warnings;
- manifest SHA-256 and byte-size validation: pass;
- W&B cloud run and seven logged artifacts: verified.

The CUDA skips reflect the host A100 returning to PCI revision `ff`, not test failures.

## Claim boundary and limitations

- The fixture is not an official dataset and contains no model-quality ground truth.
- Repetitions reuse deterministic cases and therefore do not estimate generalization.
- Latencies exclude realistic decoding, large assets, services, and accelerator synchronization.
- No leaderboard or remote submission security boundary exists yet.
- Passing every contract adapter does not close the remaining quality gates in Phases 2–5.

## Next experiments

1. Add licensed, version-pinned official mini subsets for representation and geometry first.
2. Add per-scene predictions and paired baseline comparisons.
3. Expand official adapters across grounding, identity, memory, and simulator planning.
4. Define evaluation-server schemas, validation, resource limits, and leaderboard policy.

## Downloaded W&B record snapshot

The promoted run record was downloaded through the W&B API on 2026-06-29 and verified in `finished` state. Its persisted
summary reports six completed stages, zero total failures, five successes for the final stage, identity evidence gap
0.561404, memory query latency 0.077606 ms, and task/recovery success 1.0. W&B lists eight logged artifacts covering the
stage/metric/failure tables, JSON report, failure record, HTML dashboard, experiment record, and run history. The remote
configuration continues to label this evidence `contract-only`.
