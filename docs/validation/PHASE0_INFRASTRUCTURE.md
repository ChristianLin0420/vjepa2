# Phase 0 validation plan — infrastructure, provenance, and contracts

## Status and exception

Phase 0 is not a learned-model stage, so it does not own an accuracy dataset. It validates the machinery used by every
declared A1/A2/B/C source: loaders, manifests, target isolation, metrics, Slurm graphs, artifacts, and reports. Its two required data
classes are:

1. generated analytic fixtures with exact expected answers;
2. one tiny official sample from each stage adapter, with no promoted quality claim.

Current package/contracts/tests are complete at their original scope. The remaining work is to turn the growing set of
stage-specific practices into a common validation registry and consumed-test ledger.

## Objectives

- Make every data/model/result identity content-addressable and auditable.
- Prevent train/validation/test leakage by construction.
- Give every stage the same experiment state machine, status semantics, failure taxonomy, and artifact contract.
- Guarantee at most eight concurrently RUNNING Slurm allocations.
- Preserve online W&B observability without making W&B the only record.
- Ensure credentials, raw restricted targets, and protected archives never enter logs or artifacts.

## Validation sources

| Source | Purpose | Permitted claim |
|---|---|---|
| Generated analytic fixtures | Exact shapes, coordinates, metrics, failures, replay, and control flow | Contract correctness |
| Tiny official adapter samples | Real decode/schema/checkpoint/metric compatibility | Loader/integration readiness only |
| Historical receipts/manifests | Migration, backward compatibility, and postflight regression | Provenance compatibility |

## TODO — dataset registry

- [ ] Add one machine-readable registry entry per stage Dataset A1/A2/B/C.
- [ ] Record official URL, release, split, license/access, citation, bytes, hashes, independent unit, and allowed artifact use.
- [ ] Add schema validation and duplicate source/split detection.
- [ ] Add mechanical eligibility and selected/rejected ID receipts.
- [ ] Add a consumed-test ledger with first-open execution/commit/time and allowed future use.
- [ ] Add explicit `development`, `external`, `stress`, and `contract-only` roles.
- [ ] Fail when a test role is passed to training, tuning, calibration, or checkpoint-selection code.
- [ ] Add restricted-data upload denial tests for W&B and local bundles.

## TODO — experiment registry and state machine

- [ ] Represent S0-S10 states from [the master plan](../VALIDATION_PLAN.md) in a versioned schema.
- [ ] Require question, hypotheses, gates, baselines, expected cells, and claim boundary before submission.
- [ ] Record protocol status separately from execution status and scientific gate status.
- [ ] Require explicit legal skip semantics and fail incomplete matrices.
- [ ] Add supersession lineage without deleting failed/negative experiments.
- [ ] Generate INDEX/INSIGHTS/consumed-test updates from validated receipts where safe.

## TODO — metrics and statistics

- [ ] Version every metric implementation and aggregation rule.
- [ ] Add analytic tests for alignment, calibration, AUSE/risk, pose/point, tracking, memory, and planning metrics.
- [ ] Add shared paired cluster-bootstrap utilities with explicit resampling unit.
- [ ] Prevent optimizer seed SD from being labeled population confidence.
- [ ] Persist per-independent-unit values and failure coverage for every macro.
- [ ] Validate candidate/reference pairing before computing differences.

## TODO — Slurm orchestration

- [ ] Provide one common held-graph submitter with semantic array dispatch.
- [ ] Validate account, partition fallback, GPU requirement, time, CPU, memory, and output paths.
- [ ] Prove array/dependency topology caps global RUNNING allocations at eight.
- [ ] Record base submissions and logical tasks separately.
- [ ] Add exact four-hour checkpoint/resume lineage for long work.
- [ ] Batch `sacct` postflight and memoize repeated content hashes.
- [ ] Preserve same-ID operator-requeue history and distinguish it from scientific retry.
- [ ] Deny GPU model execution on the login node.

## TODO — observability and artifacts

- [ ] Standardize online W&B group/job/run/artifact naming by stage/execution/logical cell.
- [ ] Require terminal W&B upload receipt before SUCCESS.
- [ ] Standardize JSON/JSONL, CSV/Parquet/NPZ, checkpoint, PNG, self-contained HTML, manifest, and receipt roles.
- [ ] Add fixed visualization selection before training.
- [ ] Add report cards for data role, evidence level, gate outcome, completeness, and claim boundary.
- [ ] Add secret/credential and raw-target scans to preflight/postflight.
- [ ] Test offline interruption and idempotent online artifact recovery without duplicate scientific runs.

## Promotion gate

Phase 0 validation infrastructure is ready for a formal stage only when:

- its declared A1/A2/B/C registry entries and manifests validate;
- target-role denial tests pass;
- metric analytic tests pass;
- mock and tiny-official adapter tests pass;
- dry-run graph identities/dependencies/resources are complete;
- simulated accounting never exceeds eight running allocations;
- W&B/local artifact round-trip and strict postflight pass;
- no credential or restricted raw target is serialized.

Infrastructure completion never promotes a model. It authorizes the stage to collect quality evidence.
