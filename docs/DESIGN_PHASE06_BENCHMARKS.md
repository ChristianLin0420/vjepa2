# Phase 6 design: versioned benchmark aggregation

## Scope and evidence boundary

Phase 6 begins by making benchmark execution reproducible and auditable before adding many datasets. The initial substrate
validates manifests, repeated execution, statistical aggregation, typed failures, dashboards, artifacts, and experiment
narratives across every implemented stage. It does not turn deterministic smoke fixtures into model-quality evidence.

An official-small-subset adapter is complete only when its manifest names the dataset revision, split, license/source,
local assets and hashes; its adapter consumes those assets; and its metrics match a documented protocol. The bundled
Robo4D-JEPA contract fixture is deliberately marked `official: false` and `evidence_level: contract-only`.

## Versioned inputs

`DatasetManifest` records dataset ID, semantic version, revision, split, license, evidence level, official status, source,
and local assets. Asset size and SHA-256 checks run before any adapter. A missing or changed file stops evaluation rather
than silently producing incomparable results.

The first manifest, `robo4d-jepa-contract` version `0.1.0`, contains four generated track descriptors: single-image
bootstrap, multi-view memory, delayed hidden state, and verified recovery. Existing stage adapters provide the executable
contract fixtures while the asset establishes the custom benchmark's versioning boundary.

## Aggregation and uncertainty

Each stage executes for a configured number of repetitions. Metrics must be finite numeric mappings with identical keys.
The runner reports means and deterministic percentile-bootstrap intervals with a seeded local generator. Single samples
produce a degenerate interval rather than fabricated uncertainty.

Repeated deterministic runs measure harness/runtime variability only. They do not estimate population uncertainty,
dataset variation, or model generalization. Official adapters must resample independent scenes or episodes before their
confidence intervals support model comparisons.

## Failure model

Failures are preserved as records rather than crashing the entire suite. Each record owns benchmark, stage, sample ID,
primary taxonomy category, message, and optional contributing categories. The taxonomy covers input, representation,
geometry, tracking, grounding, memory, planning, verification, control, collision, infrastructure, and unknown failures.
Unknown is permitted but must not become a default bucket.

Manifest integrity errors fail before stage execution because running against altered assets would invalidate every
downstream estimate.

## Outputs

The `jepa4d-eval` CLI writes:

- `report.json`: manifest, configuration, stage predictions, means, intervals, latency, and failures;
- `failures.json`: typed per-sample failure records;
- `report.html`: stage, metric-interval, latency, and failure dashboard;
- `EXPERIMENT.md`: durable question, decision, configuration, numerical results, artifacts, limitations, and next action;
- W&B stage/metric/failure tables and versioned artifacts when enabled.

## Remaining Phase-6 work

- add one licensed, version-pinned official mini subset and protocol adapter per stage;
- add per-scene/episode metrics rather than repeated deterministic aggregate metrics;
- add paired comparisons, multiple-comparison policy, and confidence intervals over independent units;
- build evaluation-server upload validation, sandboxing, schema versioning, and leaderboard rules;
- expand Robo4D-JEPA from descriptors to distributable RGB, geometry, memory, and execution episodes;
- add privacy, retention, license, and takedown processes for externally sourced data.
