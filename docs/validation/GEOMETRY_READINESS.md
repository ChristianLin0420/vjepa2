# Phase 2 geometry metadata readiness

## Outcome

The checked-in Phase 2 readiness pack is **metadata-audit ready, partially implemented for consumed Phase 2b regression,
and blocked for architecture development or external confirmation**. It is deliberately not a preregistration, dataset
grant, target-opening receipt, training authorization, or external-evaluation authorization.

The machine-readable source is
[`configs/validation/geometry/phase2_readiness_v1.yaml`](../../configs/validation/geometry/phase2_readiness_v1.yaml).
It binds the exact checked-in registry, consumed-target ledger, and proposed Phase 2 specification by both file identity
and normalized model identity where applicable. It also binds the exact Phase 2b runner, safe W&B/dashboard path, Slurm
wrapper/postflight implementation, and their focused test sources by tracked file path and SHA-256.

Loading the pack reads checked-in metadata and bound source/test text only. It does not resolve `JEPA4D_DATA_ROOT`,
`JEPA4D_CACHE_ROOT`, or `JEPA4D_DIODE_SEALED_ROOT`, and it never reads an archive, extracted dataset, cache, prediction,
or target.

## Gate summary

| Gate | Status | What the status means |
|---|---|---|
| Repository metadata audit | `audit-ready` | Registry, ledger, specification, tracked legacy manifests, hashes, and recorded gaps can be checked without data access. |
| Consumed TUM regression | `partial-runtime-implemented` | A hash-bound, test-covered runner authorizes and evaluates only the consumed Phase 2b split from its verified archive. Phase 2c has no equivalent runtime, both manifests remain legacy, and no terminal Slurm/W&B/postflight receipt is bound into this pack. |
| SUN development | `policy-blocked` | SUN constituent licensing, identity-only selection, invalid-depth policy, per-split Wave A manifests, governed target separation, and formal preregistration remain incomplete. |
| DIODE external | `sealed-blocked` | DIODE remains sealed; the signer, append-only store, survivor, calibrator, exact enumeration, governed adapter, and external preregistration are absent. |

Every gate has `execution_ready: false` and `pack_authorizes_data_access: false`. The pack itself never grants access. The
separate runtime controller may authorize the exact ledger-permitted consumed Phase 2b regression; that exception is an
integration/regression diagnostic only and does not make the portfolio or a scientific-promotion gate execution-ready.

## Runtime implementation binding

The readiness pack binds the exact tracked bytes for the Phase 2b runner, safe W&B publisher, immutable dashboard,
Slurm allocation wrapper, submission guard, strict postflight, and their focused runner/W&B/dashboard tests. Repository
validation fails if any bound file is missing from Git or its SHA-256 changes. This proves which implementation and tests
the partial-runtime status describes; it does not prove that a real job ran or that the tests passed in a particular CI
environment. A real execution claim still requires the terminal receipt produced after strict postflight and terminal
online-W&B publication.

## Legacy manifest gaps

The registry currently binds valid historical file hashes, but those files are not yet Wave A per-split manifests:

- `sun-rgbd.phase2e-kv2-test` points to a tracked 640-sample Phase 2e file containing train, validation, and test rows plus
  target-derived depth statistics, while the registry row denotes only 128 Kinect-v2 development-test scenes.
- `sun-rgbd.phase2f-four-family-development` points to an ignored local cache receipt under `outputs/`. Its hash is known,
  but it is unavailable in a clean clone and contains cache/execution paths rather than a portable target-free 512-scene
  membership.
- `tum-rgbd.phase2b-freiburg1-xyz-test` points to a tracked file containing train, validation, and test indices for one
  recording rather than one registered split with typed units and an isolation receipt.
- `tum-rgbd.phase2c-freiburg3-test` points to a tracked five-sequence train/validation/test protocol rather than only the
  two registered Freiburg3 regression recordings.

These gaps are recorded as facts. The readiness pack does not fabricate replacement manifests or attestations.

## Historical overlap

SUN Phase 2f selected 128 samples from each family from the already consumed Phase 2e manifest:

| Family | Phase 2e selected | Phase 2f selected | Exact overlap | Relationship |
|---|---:|---:|---:|---|
| Kinect v1 | 192 | 128 | 128 | Phase 2f strict subset of Phase 2e |
| Kinect v2 | 128 | 128 | 128 | all Phase 2e membership reused |
| RealSense | 128 | 128 | 128 | all Phase 2e membership reused |
| Xtion | 192 | 128 | 128 | Phase 2f strict subset of Phase 2e |

The historical Phase 2f receipt asserts this membership and commits it by selection SHA-256
`01a8c4577289034db86b63c4f6e9eaef9afd7aa636a9d952e9878dae3758bca6`, but the receipt and individual IDs are absent
from a clean clone. The counts are therefore a recorded historical assertion, not independently recomputable Wave A
membership proof.

TUM Phase 2c reused eight `freiburg1_xyz` indices for training:

- prior Phase 2b train: `35, 172, 248, 384, 483`;
- prior Phase 2b validation: `520, 631`;
- prior Phase 2b test: `779`.

Therefore SUN and TUM provide consumed development/regression evidence only. Neither supports a fresh external claim.

## Validation API

```python
from pathlib import Path

from jepa4d.validation.geometry_readiness import load_and_validate_geometry_readiness

root = Path(".").resolve()
pack = load_and_validate_geometry_readiness(
    root / "configs/validation/geometry/phase2_readiness_v1.yaml",
    root,
)
print(pack.status_by_gate())
```

Validation fails closed when:

- the registry, ledger, Phase 2 specification, or tracked legacy manifest changes bytes;
- normalized registry/ledger identity or event-store policy changes;
- a bound Phase 2b runtime, W&B/dashboard, Slurm/postflight, or focused test file changes bytes or is absent from Git;
- a registered geometry split changes its manifest path/hash, operations, target state, or ledger state;
- a required legacy file is absent from Git, or the ignored Phase 2f receipt becomes tracked without updating its status;
- the SUN legal blocker, DIODE signer blocker, or local non-append-only ledger blocker no longer matches the bound state.

## Next migration order

1. Commit and test the governed Phase 2b smoke, obtain its first terminal Slurm/online-W&B/postflight receipt, then promote
   target-free per-split Phase 2b and Phase 2c Wave A manifests and add equivalent Phase 2c runtime coverage.
2. Complete SUN constituent-license review, then implement an identity-only selector that cannot receive depth paths.
3. Promote portable SUN Phase 2e and Phase 2f membership artifacts; keep target validation in a separately authorized path.
4. Freeze the Phase 2 scientific preregistration and complete all SUN development gates.
5. Only after one survivor is frozen, provision the DIODE signer and externally append-only store, then write a separate
   external preregistration. DIODE remains sealed until that atomic signed first-open transition.

## Claim boundary

This pack supports repository-metadata auditability, an exact binding to the partial Phase 2b runtime source/test files, a
recomputed TUM overlap statement, and an explicit record of the historical-but-clean-clone-unverifiable SUN overlap
assertion. It does not itself authorize data access, prove a test execution, establish a real terminal job receipt,
authorize regression execution, permit model fitting, decode targets, access caches, or authorize external evaluation.
The separate controller can authorize the exact consumed Phase 2b regression under its ledger future-use rule. In
particular, nothing here permits listing, extracting, sampling, decoding, rehashing, summarizing, visualizing, or caching
DIODE content.
