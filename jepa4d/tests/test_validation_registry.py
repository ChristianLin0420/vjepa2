from __future__ import annotations

import base64
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from jepa4d.validation._content import canonical_json, sha256_value, verify_content_addressed_json
from jepa4d.validation.access import DatasetAccessController
from jepa4d.validation.ledger import ConsumedTestLedger
from jepa4d.validation.registry import (
    AccessDenied,
    AccessOperation,
    DataRole,
    DatasetRegistry,
    freeze_registry,
)

RAW_RULES = [
    {
        "artifact": "raw-data",
        "local": "restricted",
        "wandb": "deny",
        "repository": "deny",
        "notes": "Raw samples remain in approved storage.",
    },
    {
        "artifact": "raw-targets",
        "local": "restricted",
        "wandb": "deny",
        "repository": "deny",
        "notes": "Targets never leave approved storage.",
    },
]


@pytest.fixture(autouse=True)
def _canonical_event_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEPA4D_TEST_EVENT_ROOT", str(tmp_path / "validation-state"))


def dataset_entry(
    *,
    dataset_id: str = "fixture.dataset",
    role: str = "A1",
    split_id: str = "fixture.dataset.dev-test",
    purpose: str = "development-test",
    target_state: str = "open",
    operations: list[str] | None = None,
    selection_rule: str = "ordered IDs v1",
) -> dict:
    return {
        "dataset_id": dataset_id,
        "display_name": "Fixture dataset",
        "stages": ["phase1"],
        "role": role,
        "status": "active-development",
        "claim_use": "Unit-test fixture only.",
        "source": {
            "official_url": f"https://example.org/{dataset_id}",
            "version": "v1",
            "citation": "Fixture citation.",
        },
        "license": {
            "status": "approved",
            "name": "Fixture license",
            "terms_url": "https://example.org/terms",
            "redistribution": "prohibited",
            "privacy_notes": "No personal data.",
        },
        "access": {
            "status": "approved",
            "method": "public",
            "reviewed_at": "2026-06-30",
            "reviewer": "test-suite",
        },
        **(
            {
                "sealed_authority": {
                    "status": "pending",
                    "blocker": "Test signer has not been provisioned.",
                }
            }
            if target_state == "sealed"
            else {}
        ),
        "storage": {
            "status": "approved",
            "expected_bytes": 4,
            "raw_root_env": "JEPA4D_DATA_ROOT",
            "cache_root_env": "JEPA4D_CACHE_ROOT",
            "retention": "Test lifetime.",
        },
        "hashes": [
            {
                "name": "fixture",
                "kind": "fixture",
                "status": "verified",
                "sha256": "a" * 64,
                "bytes": 4,
                "provenance": "test suite",
            }
        ],
        "splits": [
            {
                "split_id": split_id,
                "official_name": "dev-test",
                "purpose": purpose,
                "independent_unit": "video",
                "target_state": target_state,
                "targets_present": True,
                "selection_rule": selection_rule,
                "id_manifest": "manifests/fixture.txt",
                "id_manifest_sha256": "b" * 64,
                "expected_units": 2,
                "allowed_operations": operations or ["development-evaluation", "reporting"],
                **(
                    {"seal_condition": "One frozen survivor and a hash-bound authorization receipt."}
                    if target_state == "sealed"
                    else {}
                ),
            }
        ],
        "artifact_rules": deepcopy(RAW_RULES),
    }


def approve_test_sealed_authority(entry: dict, private_key: Ed25519PrivateKey) -> None:
    """Approve an ephemeral runtime key; no private signing material is stored in the repository."""
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    entry["sealed_authority"] = {
        "status": "approved",
        "key_id": "ephemeral-test-ed25519-v1",
        "public_key_ed25519_base64": base64.b64encode(public_key).decode("ascii"),
        "approved_at": "2026-06-30",
        "approved_by": "test-suite-runtime-fixture",
    }


def registry_ledger_value(*targets: dict) -> dict:
    return {
        "schema_version": "jepa4d-consumed-test-ledger-v1",
        "ledger_version": "registry-test-v1",
        "event_store": {
            "root_env": "JEPA4D_TEST_EVENT_ROOT",
            "relative_path": "events/registry-test-v1",
            "durability": "local-filesystem-best-effort",
            "externally_append_only": False,
            "deployment_blocker": "Test-only local store; no external append-only durability.",
        },
        "targets": list(targets),
    }


def registry_value(*datasets: dict) -> dict:
    return {
        "schema_version": "jepa4d-validation-registry-v1",
        "portfolio_version": "test-v1",
        "datasets": list(datasets),
    }


def test_canonical_json_sorts_unordered_collections() -> None:
    assert canonical_json({"operations": frozenset({"reporting", "metadata-audit"})}) == (
        b'{"operations":["metadata-audit","reporting"]}'
    )


@pytest.mark.parametrize("role", [value.value for value in DataRole])
def test_all_registered_roles_validate(role: str) -> None:
    entry = dataset_entry(role=role)
    if role == "contract-only":
        entry["status"] = "contract-only"
        entry["splits"][0].update(
            purpose="contract",
            target_state="not-applicable",
            targets_present=False,
            allowed_operations=["decode-smoke", "reporting"],
        )
    if role in {"B", "C"}:
        entry["splits"][0]["allowed_operations"] = ["external-evaluation", "reporting"]
    assert DatasetRegistry.model_validate(registry_value(entry)).datasets[0].role.value == role


def test_shared_dataset_can_serve_unique_phase5_phase6_splits() -> None:
    entry = dataset_entry()
    entry["stages"] = ["phase5", "phase6"]
    entry["splits"].append(
        {
            **deepcopy(entry["splits"][0]),
            "split_id": "fixture.dataset.phase6-reserved",
            "official_name": "phase6-reserved",
            "selection_rule": "disjoint reserved IDs v1",
            "id_manifest": "manifests/phase6.txt",
            "id_manifest_sha256": "c" * 64,
        }
    )
    registry = DatasetRegistry.model_validate(registry_value(entry))
    assert registry.datasets[0].stages == ("phase5", "phase6")
    assert len({split.split_id for split in registry.datasets[0].splits}) == 2


def test_duplicate_physical_identity_is_rejected() -> None:
    first = dataset_entry(dataset_id="fixture.first", split_id="fixture.first.test")
    second = dataset_entry(dataset_id="fixture.second", split_id="fixture.second.test")
    second["source"] = deepcopy(first["source"])
    for entry in (first, second):
        entry["splits"][0]["id_manifest"] = None
        entry["splits"][0]["id_manifest_sha256"] = None
    with pytest.raises(ValidationError, match="duplicate physical split identity"):
        DatasetRegistry.model_validate(registry_value(first, second))


def test_target_and_raw_artifact_leakage_are_schema_errors() -> None:
    entry = dataset_entry(operations=["training", "development-evaluation"])
    with pytest.raises(ValidationError, match="held-out target split permits selection"):
        DatasetRegistry.model_validate(registry_value(entry))

    entry = dataset_entry()
    entry["artifact_rules"][0]["wandb"] = "allowed"
    with pytest.raises(ValidationError, match="raw data/targets may not be unconditionally"):
        DatasetRegistry.model_validate(registry_value(entry))

    entry = dataset_entry()
    entry["artifact_rules"][0]["repository"] = "allowed"
    with pytest.raises(ValidationError, match="non-contract raw data/targets must be denied"):
        DatasetRegistry.model_validate(registry_value(entry))

    entry = dataset_entry()
    entry["artifact_rules"] = [entry["artifact_rules"][0]]
    with pytest.raises(ValidationError, match="missing mandatory raw-artifact denial"):
        DatasetRegistry.model_validate(registry_value(entry))


def test_generated_contract_data_may_be_repository_tracked() -> None:
    entry = dataset_entry(role="contract-only", purpose="contract", target_state="not-applicable")
    entry["status"] = "contract-only"
    entry["splits"][0].update(targets_present=False, allowed_operations=["decode-smoke", "reporting"])
    entry["artifact_rules"][0]["repository"] = "allowed"
    assert DatasetRegistry.model_validate(registry_value(entry)).datasets[0].role is DataRole.CONTRACT_ONLY


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.yaml"
    source.write_text(
        "schema_version: jepa4d-validation-registry-v1\n"
        "portfolio_version: first\n"
        "portfolio_version: second\n"
        "datasets: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML key 'portfolio_version'"):
        DatasetRegistry.load(source)


def test_content_addressed_json_rejects_duplicate_keys_and_noncanonical_whitespace(tmp_path: Path) -> None:
    digest = sha256_value({"value": 1})
    artifact = tmp_path / f"fixture-{digest}.json"
    artifact.write_text('{"value":1,"value":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_content_addressed_json(artifact, prefix="fixture")
    artifact.write_text('{ "value": 1 }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-canonical JSON encoding"):
        verify_content_addressed_json(artifact, prefix="fixture")


def test_transfer_target_denies_selection_and_plain_authorize_cannot_open_sealed_target() -> None:
    entry = dataset_entry(
        role="B",
        target_state="sealed",
        purpose="external-test",
        operations=["metadata-audit", "external-evaluation", "reporting"],
    )
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(
        registry_ledger_value(
            {
                "dataset_id": entry["dataset_id"],
                "split_id": entry["splits"][0]["split_id"],
                "state": "sealed-unopened",
                "seal_evidence": "Test seal receipt.",
                "notes": "Fixture.",
            }
        )
    )
    controller = DatasetAccessController(registry=registry, ledger=ledger)
    with pytest.raises(AccessDenied, match="forbidden for transfer/external"):
        controller.authorize(entry["dataset_id"], entry["splits"][0]["split_id"], AccessOperation.TRAINING)
    metadata = controller.authorize(
        entry["dataset_id"], entry["splits"][0]["split_id"], AccessOperation.METADATA_AUDIT
    )
    assert metadata.authorized and not metadata.grants_data_access
    with pytest.raises(AccessDenied, match="atomic first-open/consume"):
        controller.authorize(entry["dataset_id"], entry["splits"][0]["split_id"], AccessOperation.EXTERNAL_EVALUATION)
    with pytest.raises(AccessDenied, match="atomic first-open/consume"):
        controller.authorize(
            entry["dataset_id"],
            entry["splits"][0]["split_id"],
            AccessOperation.EXTERNAL_EVALUATION,
            sealed_authorization=Path("sealed-target-authorization-" + "d" * 64 + ".json"),
        )


def test_pending_audit_blocks_data_access_but_not_metadata_audit() -> None:
    entry = dataset_entry(operations=["metadata-audit", "development-evaluation"])
    entry["access"] = {
        "status": "pending",
        "method": "request",
        "blocker": "Data-use approval has not been recorded.",
    }
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(
        registry_ledger_value(
            {
                "dataset_id": entry["dataset_id"],
                "split_id": entry["splits"][0]["split_id"],
                "state": "available-unopened",
                "notes": "Fixture.",
            }
        )
    )
    controller = DatasetAccessController(registry=registry, ledger=ledger)
    with pytest.raises(AccessDenied, match="audit is incomplete"):
        controller.authorize(entry["dataset_id"], entry["splits"][0]["split_id"], "development-evaluation")
    assert not controller.authorize(
        entry["dataset_id"], entry["splits"][0]["split_id"], "metadata-audit"
    ).grants_data_access


def test_freeze_is_deterministic_and_detects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "registry.yaml"
    source.write_text(yaml.safe_dump(registry_value(dataset_entry()), sort_keys=False), encoding="utf-8")
    snapshot, receipt = freeze_registry(source, tmp_path / "frozen")
    repeated_snapshot, repeated_receipt = freeze_registry(source, tmp_path / "frozen")
    assert snapshot.sha256 == repeated_snapshot.sha256
    assert receipt.sha256 == repeated_receipt.sha256
    assert verify_content_addressed_json(snapshot.path, prefix="dataset-registry")["registry_sha256"]
    value = json.loads(snapshot.path.read_text())
    value["registry"]["portfolio_version"] = "tampered"
    snapshot.path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="content digest mismatch"):
        verify_content_addressed_json(snapshot.path, prefix="dataset-registry")


def test_checked_in_registry_and_json_schema_match_runtime_models() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = DatasetRegistry.load(root / "configs/validation/dataset_registry.yaml")
    assert len(registry.datasets) >= 10
    assert {role.value for role in DataRole} <= {entry.role.value for entry in registry.datasets}
    assert len({split.split_id for entry in registry.datasets for split in entry.splits}) == sum(
        len(entry.splits) for entry in registry.datasets
    )
    schema = json.loads((root / "configs/validation/schemas/dataset-registry.schema.json").read_text())
    assert schema == DatasetRegistry.model_json_schema()
