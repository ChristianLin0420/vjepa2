# Phase 0 validation plan — infrastructure, provenance, and contracts

## Status and exception

Phase 0 is not a learned-model stage, so it does not own an accuracy dataset. It validates the machinery used by every
declared A1/A2/B/C source: loaders, manifests, target isolation, metrics, Slurm graphs, artifacts, and reports. Its two required data
classes are:

1. generated analytic fixtures with exact expected answers;
2. one tiny official sample from each stage adapter, with no promoted quality claim.

The common Wave-A registry, consumed-test ledger, access controller, split-manifest contract, statistical utilities,
failure taxonomy, and governed report surface are implemented. Remaining work is to complete the declared dataset
portfolio and source audits, create real adapter manifests, and wire these contracts into every formal runner and Slurm
graph; the foundation alone does not authorize model training.

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

- [ ] Add one machine-readable registry entry per stage Dataset A1/A2/B/C; the initial 12-family portfolio is incomplete.
- [ ] Complete every source's official release, license/access approval, bytes, hashes, and storage estimate; pending
  blockers are recorded rather than treated as approval.
- [x] Add strict schema validation and duplicate YAML/source/split/physical-identity detection.
- [x] Add target-field screening, isolation attestation, and selected/rejected split-manifest receipts; formal execution
  still requires physical target-path denial.
- [x] Add a consumed-test ledger with first-open execution/commit/time and allowed future use.
- [x] Add explicit A1/A2/B/C/contract roles and sealed/consumed/unavailable target state.
- [x] Fail when held-out targets are passed to training, tuning, calibration, or checkpoint-selection code.
- [x] Add restricted-data upload denial tests for W&B/repository artifacts.
- [x] Require a registry-approved Ed25519 selector, exact model/protocol/calibrator bindings, and atomic consumption for a
  sealed external target.
- [ ] Provision a trusted signing public key and externally append-only event store; the checked-in DIODE authority stays
  blocked until this deployment-specific trust root exists.

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
- [x] Add shared paired cluster-bootstrap utilities with explicit resampling unit.
- [x] Keep optimizer-seed spread, unregistered cluster intervals, and registered intervals with fewer than five clusters
  descriptive rather than labeling them population confidence.
- [ ] Persist per-independent-unit values and failure coverage for every macro.
- [x] Validate candidate/reference pairing before computing differences.

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
- [x] Add a fixed stage-agnostic visualization declaration contract.
- [x] Add report cards for data role, evidence level, gate outcome, completeness, and claim boundary.
- [x] Make formal failure artifacts always pseudonymize sample IDs and reject every raw-disclosure authorization input;
  no policy authority is inferred from a caller-supplied record.
- [ ] Add secret/credential and raw-target scans to preflight/postflight.
- [ ] Define and security-review cryptographic policy-authority verification before enabling any raw sample-ID artifact
  path; leave raw disclosure disabled until that separate integration exists.
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
- sealed external evaluation has an approved signer and externally append-only consumption store;
- no credential or restricted raw target is serialized.

Infrastructure completion never promotes a model. It authorizes the stage to collect quality evidence.

See [WAVE_A_FOUNDATION.md](WAVE_A_FOUNDATION.md) for the implemented code paths, operator commands, verification, and
remaining audit blockers.
