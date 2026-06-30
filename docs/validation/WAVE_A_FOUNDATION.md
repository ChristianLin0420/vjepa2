# Wave A implementation — validation foundations

## Status

**Foundation code implemented and Phase 2 metadata audit recorded; portfolio expansion remains in progress.**

Wave A started from pushed commit `21af2f3`. This implementation adds the fail-closed machinery required before dataset
downloads, adapter qualification, or formal GPU experiments. It does not authorize training, open a held-out target, or
claim that pending dataset terms have been approved.

## Delivered contracts

| Contract | Source of truth | What it prevents or records |
|---|---|---|
| Dataset access | [`configs/validation/dataset_registry.yaml`](../../configs/validation/dataset_registry.yaml), [`registry.py`](../../jepa4d/validation/registry.py), [`access.py`](../../jepa4d/validation/access.py) | A1/A2/B/C/contract roles, audit blockers, split purpose, artifact policy, ledger-aware authorization, and Ed25519-authenticated one-shot external authority |
| Consumed-test ledger | [`configs/validation/consumed_test_ledger.yaml`](../../configs/validation/consumed_test_ledger.yaml), [`ledger.py`](../../jepa4d/validation/ledger.py) | First-open lineage, sealed/unopened state, canonical event-store identity, atomic consumption, and explicitly permitted future diagnostic use |
| Split manifest | [`split_manifest.py`](../../jepa4d/validation/split_manifest.py) | Mechanical selection, source and physical-unit hashes, selected/rejected units, target-field screening, path-denial attestation, registry binding, and cross-manifest isolation |
| Statistics | [`statistics.py`](../../jepa4d/evaluation/statistics.py) | Manifest-bound paired equal-cluster bootstrap, descriptive small-sample status, replay identities, and separation of seed spread from population intervals |
| Failure/status taxonomy | [`failure_taxonomy.py`](../../jepa4d/evaluation/failure_taxonomy.py) | Typed failure attribution and separate protocol, execution, scientific, and evidence status |
| Dashboard/report | [`validation_dashboard.py`](../../jepa4d/visualization/validation_dashboard.py) | Visible dataset role, evidence, gate, completeness, claim boundary, and separate quality/resource panels |

The checked-in JSON schemas are under [`configs/validation/schemas/`](../../configs/validation/schemas/). They cover the
registry, ledger, split manifest, sealed-target selector receipt, and sealed-target authorization. Content hashes normalize
unordered collections before hashing, so registry and ledger identities are stable across Python hash seeds.

## Initial portfolio state

The initial registry contains 12 dataset families, 16 globally unique splits, and 15 held-out ledger targets.

- Consumed: Phase 2e SUN Kinect-v2, Phase 2f SUN four-family development, TUM Phase 2b/2c held-out data, and DAVIS
  `dogs-scale`.
- Sealed and unopened: DIODE validation.
- Operationally blocked DIODE authority: the official dataset license, archive identity, access, and storage record are
  present, but no trusted signing public key or externally append-only event store is provisioned.
- Planned and unavailable: sources whose license/access/storage/hash or task-manifest audit is incomplete.
- Shared Phase 5/6 sources: ManiSkill3 and LIBERO have distinct placeholder split identities, but these names alone do
  not establish task/scene/seed disjointness. That claim remains blocked until populated manifests pass the cross-manifest
  selected-unit and independent-cluster verifier.

This is not yet the complete master portfolio. Registry rows or distinct split identities still need to be added for
RefCOCO/RefCOCO+, Flickr30K Entities, MOT17, EgoTracks, ScanNet v2, D4RL, RoboNet, and the custom Phase-6 Robo4D-JEPA
attribution tracks, plus any optional Dataset B/C adopted later. Pending entries remain denied for data access; the
registry reports the blocker rather than inventing a license, byte count, or checksum.

## Operator commands

After installing the editable package, use `jepa4d-validation`; the module form works directly from the repository:

```bash
export JEPA4D_VALIDATION_STATE_ROOT=/approved/shared/validation-state

python -m jepa4d.cli.validation_registry validate \
  --registry configs/validation/dataset_registry.yaml \
  --ledger configs/validation/consumed_test_ledger.yaml

python -m jepa4d.cli.validation_registry freeze \
  --registry configs/validation/dataset_registry.yaml \
  --ledger configs/validation/consumed_test_ledger.yaml \
  --output outputs/validation/frozen
```

The event directory is derived only from the ledger's root-environment-variable and relative-path contract; callers
cannot substitute an arbitrary directory. Use `authorize` for ordinary registered operations and `consume` atomically at
the first real target read. Metadata audit and reporting cannot create a consumption event. A sealed external target is
never granted by `authorize`: the only grant occurs inside the lock-held `consume` transition, and only when the event
store is declared externally append-only.

Sealed selection uses a registry-approved Ed25519 public key. The private key is supplied only to the trusted selector
process through an environment variable and is never committed, printed, or passed to an evaluation job. Its signed
receipt binds `final_authorized=true`, the survivor, a clean-commit attestation, preregistration, checkpoint, config,
distinct calibrator, registry, base ledger, event-store specification, and resolved store instance. `issue-sealed-authorization`
revalidates those bindings, and `consume` records the authorization identity before target bytes may be read.

Split-manifest screening and reviewer attestation are defense in depth, not mathematical proof that targets were hidden.
Formal selectors must still run with target paths physically denied, and the manifest binds the resulting path-denial
receipt. Phase-5/Phase-6 reservations must also pass the selected-unit and independent-cluster disjointness verifier.

## Pushed Wave A baseline verification

The following counts describe pushed commit `2d43ff9`, not the current post-Wave-A working tree:

- The integrated Wave A suite passes: 143 tests.
- The complete repository suite passes: 399 tests in 24:25, with one pre-existing Starlette/httpx deprecation warning.
- Ruff lint/format and focused mypy pass.
- Checked registry/ledger source-record and local manifest hashes match their files.
- Registry and ledger hashes match under different `PYTHONHASHSEED` values.
- Final semantic identities are registry
  `4dd75d40bed2228e3b60fb68d8ec325b6a48bf7b0be2d197ab64334ff4c7b11a` and ledger
  `1b05f49f621d0cba35fce73075b6475265be9e4ee2f256a3bb2636cef4e1df63`.
- CLI validation correctly reports `schema-valid-but-operationally-blocked`; schema generation and content-addressed
  freeze complete without opening dataset data.
- Adversarial tests reject forged/wrong-key/revoked selectors, changed ledger or event-store roots, rollbackable sealed
  consumption, duplicate/noncanonical JSON, physical split reuse, caller-invented inferential clusters, and target text
  in formal failure artifacts.
- No W&B/Hugging Face credentials are stored in code, configs, schemas, or reports.
- Formal failure artifacts always pseudonymize sample identifiers. Raw sample-ID disclosure requests fail closed because
  Wave A does not yet have a cryptographically verified policy-authority integration; a self-issued approval record is
  not authorization.

## Remaining Wave A TODO

- [ ] Add every common Dataset A1/A2/B/C row from the master portfolio.
- [ ] Complete source-specific legal approval and record exact terms rather than only the current blocker.
- [ ] Freeze archive bytes, extracted/cache estimates, retention, and hashes for every planned source.
- [ ] Create content-addressed split manifests with selected/rejected IDs for the first adapters.
- [ ] Integrate registry authorization into every formal runner/Slurm graph. The consumed Phase 2b smoke now has a
  test-covered authorization/access-decision path; first-open consumption receipts remain required only for unopened
  targets, and all other formal runners still need migration.
- [ ] Provision a security-reviewed selector signing authority and trusted externally append-only event store; keep DIODE
  sealed until both are represented by a new hash-bound ledger/registry version.
- [ ] Record the first terminal Slurm + online-W&B + strict-postflight receipt for the hash-bound, test-covered Phase 2b
  official smoke.
  The aggregate-only publisher and content-addressed dashboard are implemented, but mocked upload tests are not execution
  evidence.
- [ ] Add automated credential, restricted-path, raw-target, and unsafe sample-identifier scans to formal
  preflight/postflight; the current clean repository scan is point-in-time evidence only.
- [ ] Keep raw sample-ID artifact disclosure disabled unless a future design verifies signed, policy-bound authority and
  receives an explicit security review; pseudonymous identifiers remain the default even after such a design exists.
- [ ] Add S0-S10 experiment-state receipts and generate safe ledger/INDEX/INSIGHTS updates from postflight.

Within the Phase 2 geometry scope covered here, formal training, SUN development expansion, and DIODE external evaluation
remain blocked. The only newly implemented GPU path in this change is the exact consumed Phase 2b TUM regression smoke:
it must run through its clean-commit Slurm wrapper, registry/ledger authorization, verified-archive extraction,
aggregate-only online W&B publisher, and strict postflight. Any terminal receipt is integration evidence, not
architecture-quality promotion or fresh transfer evidence.
