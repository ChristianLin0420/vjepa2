from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from jepa4d.cli.validation_registry import _atomic_write_text, app
from jepa4d.tests.test_consumed_test_ledger import ledger_value, unopened_target
from jepa4d.tests.test_validation_registry import (
    approve_test_sealed_authority,
    dataset_entry,
    registry_value,
)
from jepa4d.validation._content import write_content_addressed_json
from jepa4d.validation.access import (
    ArtifactBinding,
    DatasetAccessController,
    verify_sealed_target_authorization,
    verify_sealed_target_selector_receipt,
    write_sealed_target_authorization,
    write_sealed_target_selector_receipt,
)
from jepa4d.validation.ledger import ConsumedTestLedger, ConsumptionEvent, FirstOpenLineage, FutureUse
from jepa4d.validation.registry import AccessDenied, DatasetRegistry


@pytest.fixture(autouse=True)
def _canonical_event_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEPA4D_TEST_EVENT_ROOT", str(tmp_path / "validation-state"))


def _sealed_context(
    tmp_path: Path,
) -> tuple[dict, DatasetRegistry, ConsumedTestLedger, dict[str, Path], Ed25519PrivateKey]:
    entry = dataset_entry(
        dataset_id="fixture.external",
        split_id="fixture.external.test",
        role="B",
        purpose="external-test",
        target_state="sealed",
        operations=["metadata-audit", "external-evaluation", "reporting"],
    )
    private_key = Ed25519PrivateKey.generate()
    approve_test_sealed_authority(entry, private_key)
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(
        ledger_value(unopened_target(entry, state="sealed-unopened"), externally_append_only=True)
    )
    sources: dict[str, Path] = {}
    for name in ("preregistration", "checkpoint", "config", "calibrator"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"immutable-{name}".encode())
        sources[name] = path
    selector = write_sealed_target_selector_receipt(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        survivor="candidate-m1",
        git_commit="a" * 40,
        preregistration=sources["preregistration"],
        checkpoint=sources["checkpoint"],
        config=sources["config"],
        calibrator=sources["calibrator"],
        private_key=private_key,
        output_dir=tmp_path / "selector",
    )
    sources["selector"] = selector.path
    return entry, registry, ledger, sources, private_key


def _write_yaml_models(
    tmp_path: Path,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
) -> tuple[Path, Path]:
    registry_path = tmp_path / "registry.yaml"
    ledger_path = tmp_path / "ledger.yaml"
    registry_path.write_text(
        yaml.safe_dump(registry.model_dump(mode="json", by_alias=True), sort_keys=False),
        encoding="utf-8",
    )
    ledger_path.write_text(yaml.safe_dump(ledger.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return registry_path, ledger_path


def test_typed_sealed_authorization_binds_all_inputs_and_is_deterministic(tmp_path: Path) -> None:
    entry, registry, ledger, sources, _ = _sealed_context(tmp_path)
    first = write_sealed_target_authorization(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        survivor="candidate-m1",
        output_dir=tmp_path / "authority",
        preregistration=sources["preregistration"],
        selector=sources["selector"],
        checkpoint=sources["checkpoint"],
        config=sources["config"],
        calibrator=sources["calibrator"],
    )
    repeated = write_sealed_target_authorization(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        survivor="candidate-m1",
        output_dir=tmp_path / "authority",
        preregistration=sources["preregistration"],
        selector=sources["selector"],
        checkpoint=sources["checkpoint"],
        config=sources["config"],
        calibrator=sources["calibrator"],
    )
    assert repeated.sha256 == first.sha256
    authorization, identity = verify_sealed_target_authorization(
        first.path,
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
    )
    assert identity.sha256 == first.sha256
    assert authorization.registry_sha256 == registry.sha256
    assert authorization.survivor == "candidate-m1"
    assert authorization.intent == "one-shot-external-evaluation"
    assert authorization.use_limit == 1
    for name, source in sources.items():
        binding = getattr(authorization, name)
        assert binding.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert binding.bytes == source.stat().st_size


def test_sealed_authorization_rejects_tamper_wrong_target_and_stale_registry(tmp_path: Path) -> None:
    entry, registry, ledger, sources, _ = _sealed_context(tmp_path)
    artifact = write_sealed_target_authorization(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        survivor="candidate-m1",
        output_dir=tmp_path / "authority",
        **sources,
    )
    with pytest.raises(AccessDenied, match="target mismatch"):
        verify_sealed_target_authorization(
            artifact.path,
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id="another.split",
        )
    stale_registry = registry.model_copy(update={"portfolio_version": "changed"})
    with pytest.raises(AccessDenied, match="different registry snapshot"):
        verify_sealed_target_authorization(
            artifact.path,
            registry=stale_registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )
    payload = json.loads(artifact.path.read_text())
    payload["survivor"] = "tampered"
    artifact.path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="content digest mismatch"):
        verify_sealed_target_authorization(
            artifact.path,
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )


def test_authorization_rejects_random_or_inconsistent_selector_receipt(tmp_path: Path) -> None:
    entry, registry, ledger, sources, _ = _sealed_context(tmp_path)
    random_selector = tmp_path / "selector.json"
    random_selector.write_text('{"final_authorized": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="content-addressed.*filename"):
        write_sealed_target_authorization(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            survivor="candidate-m1",
            preregistration=sources["preregistration"],
            selector=random_selector,
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            calibrator=sources["calibrator"],
            output_dir=tmp_path / "authority-random",
        )
    with pytest.raises(ValueError, match="does not authorize.*survivor/artifacts"):
        write_sealed_target_authorization(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            survivor="different-survivor",
            preregistration=sources["preregistration"],
            selector=sources["selector"],
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            calibrator=sources["calibrator"],
            output_dir=tmp_path / "authority-mismatch",
        )
    sources["checkpoint"].write_bytes(b"changed-after-selection")
    with pytest.raises(ValueError, match="does not authorize.*artifacts"):
        write_sealed_target_authorization(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            survivor="candidate-m1",
            output_dir=tmp_path / "authority-checkpoint-mismatch",
            **sources,
        )


def test_selector_signature_wrong_key_and_revoked_authority_are_rejected(tmp_path: Path) -> None:
    entry, registry, ledger, sources, _ = _sealed_context(tmp_path)
    with pytest.raises(ValueError, match="private key does not match"):
        write_sealed_target_selector_receipt(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            survivor="candidate-m1",
            git_commit="a" * 40,
            preregistration=sources["preregistration"],
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            calibrator=sources["calibrator"],
            private_key=Ed25519PrivateKey.generate(),
            output_dir=tmp_path / "wrong-key-selector",
        )

    signed_value = json.loads(sources["selector"].read_text())
    signature = bytearray(base64.b64decode(signed_value["signature_ed25519_base64"], validate=True))
    signature[0] ^= 1
    signed_value["signature_ed25519_base64"] = base64.b64encode(signature).decode("ascii")
    tampered = write_content_addressed_json(
        signed_value, tmp_path / "tampered-selector", prefix="sealed-target-selector"
    )
    with pytest.raises(AccessDenied, match="Ed25519 verification failed"):
        verify_sealed_target_selector_receipt(
            tampered.path,
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )

    revoked_entry = deepcopy(entry)
    revoked_entry["sealed_authority"] = {
        "status": "revoked",
        "blocker": "The test authority was revoked.",
    }
    revoked_registry = DatasetRegistry.model_validate(registry_value(revoked_entry))
    revoked_ledger = ConsumedTestLedger.model_validate(
        ledger_value(unopened_target(revoked_entry, state="sealed-unopened"), externally_append_only=True)
    )
    with pytest.raises(AccessDenied, match="authority is not approved"):
        verify_sealed_target_selector_receipt(
            sources["selector"],
            registry=revoked_registry,
            ledger=revoked_ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )


def test_signed_authorization_is_bound_to_exact_ledger_store_and_atomic_consume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, registry, ledger, sources, _ = _sealed_context(tmp_path)
    artifact = write_sealed_target_authorization(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        survivor="candidate-m1",
        output_dir=tmp_path / "authority",
        **sources,
    )
    with pytest.raises(AccessDenied, match="atomic first-open/consume"):
        DatasetAccessController(registry=registry, ledger=ledger).authorize(
            entry["dataset_id"],
            entry["splits"][0]["split_id"],
            "external-evaluation",
            sealed_authorization=artifact.path,
        )

    wrong_ledger = ledger.model_copy(update={"ledger_version": "different-ledger-v2"})
    with pytest.raises(AccessDenied, match="different base ledger"):
        verify_sealed_target_authorization(
            artifact.path,
            registry=registry,
            ledger=wrong_ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )
    wrong_store_value = ledger_value(
        unopened_target(entry, state="sealed-unopened"),
        externally_append_only=True,
    )
    wrong_store_value["event_store"]["relative_path"] = "events/different-approved-store"
    wrong_store = ConsumedTestLedger.model_validate(wrong_store_value)
    with pytest.raises(AccessDenied, match="different canonical event store"):
        verify_sealed_target_authorization(
            artifact.path,
            registry=registry,
            ledger=wrong_store,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )
    monkeypatch.setenv("JEPA4D_TEST_EVENT_ROOT", str(tmp_path / "fresh-rollback-store"))
    with pytest.raises(AccessDenied, match="different resolved event-store instance"):
        verify_sealed_target_authorization(
            artifact.path,
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
        )


def test_calibrator_must_be_distinct_and_artifact_hashing_detects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, registry, ledger, sources, private_key = _sealed_context(tmp_path)
    with pytest.raises(ValueError, match="calibrator must be a distinct"):
        write_sealed_target_selector_receipt(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            survivor="candidate-m1",
            git_commit="a" * 40,
            preregistration=sources["preregistration"],
            checkpoint=sources["checkpoint"],
            config=sources["config"],
            calibrator=sources["config"],
            private_key=private_key,
            output_dir=tmp_path / "duplicate-calibrator",
        )

    import jepa4d.validation.access as access_module

    original_fstat = access_module.os.fstat
    calls = 0

    def changed_second_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        value = original_fstat(descriptor)
        if calls != 2:
            return value
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns + 1,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr(access_module.os, "fstat", changed_second_fstat)
    with pytest.raises(ValueError, match="changed while hashing"):
        ArtifactBinding.from_file(sources["checkpoint"])


@pytest.mark.parametrize("state", ["sealed-unopened", "planned-unavailable", "server-only-unopened"])
@pytest.mark.parametrize("operation", ["metadata-audit", "reporting"])
def test_non_data_operations_remain_non_data_for_opaque_states(
    tmp_path: Path,
    state: str,
    operation: str,
) -> None:
    target_state = {
        "sealed-unopened": "sealed",
        "planned-unavailable": "unavailable",
        "server-only-unopened": "server-only",
    }[state]
    entry = dataset_entry(
        dataset_id=f"fixture.{state}",
        split_id=f"fixture.{state}.test",
        role="B",
        purpose="external-test",
        target_state=target_state,
        operations=["metadata-audit", "external-evaluation", "reporting"]
        if target_state == "sealed"
        else ["metadata-audit", "reporting"],
    )
    registry = DatasetRegistry.model_validate(registry_value(entry))
    target = unopened_target(entry, state=state)
    ledger = ConsumedTestLedger.model_validate(ledger_value(target))
    decision = DatasetAccessController(registry=registry, ledger=ledger).authorize(
        entry["dataset_id"], entry["splits"][0]["split_id"], operation
    )
    assert decision.authorized and not decision.grants_data_access


def test_non_held_out_training_split_does_not_require_ledger_row() -> None:
    entry = dataset_entry(
        split_id="fixture.dataset.train",
        purpose="train",
        operations=["training", "decode-smoke", "reporting"],
    )
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value())
    decision = DatasetAccessController(registry=registry, ledger=ledger).authorize(
        entry["dataset_id"], entry["splits"][0]["split_id"], "training"
    )
    assert decision.authorized and decision.grants_data_access
    assert decision.ledger_state is None


def test_cli_authorize_requires_ledger_events_and_validate_checks_event_lineage(tmp_path: Path) -> None:
    entry = dataset_entry()
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    registry_path, ledger_path = _write_yaml_models(tmp_path, registry, ledger)
    runner = CliRunner()

    missing_context = runner.invoke(
        app,
        [
            "authorize",
            "--registry",
            str(registry_path),
            "--dataset",
            entry["dataset_id"],
            "--split",
            entry["splits"][0]["split_id"],
            "--operation",
            "reporting",
        ],
    )
    assert missing_context.exit_code != 0
    authorized = runner.invoke(
        app,
        [
            "authorize",
            "--registry",
            str(registry_path),
            "--ledger",
            str(ledger_path),
            "--dataset",
            entry["dataset_id"],
            "--split",
            entry["splits"][0]["split_id"],
            "--operation",
            "reporting",
        ],
    )
    assert authorized.exit_code == 0, authorized.output
    assert json.loads(authorized.output)["grants_data_access"] is False

    event = ConsumptionEvent(
        schema_version="jepa4d-consumed-test-event-v1",
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        prior_state="available-unopened",
        open_operation="development-evaluation",
        first_open=FirstOpenLineage(
            opened_at="2026-06-30T08:00:00Z",
            time_precision="exact",
            experiment_id="mismatched-event",
            git_commit="abcdef1",
            source_record="receipt.json",
            registry_sha256="0" * 64,
            notes="Fixture.",
        ),
        permitted_future_uses=frozenset({FutureUse.REPORTING}),
        registry_sha256="0" * 64,
        base_ledger_sha256=ledger.sha256,
        event_store_sha256=ledger.event_store.sha256,
        resolved_instance_sha256=ledger.event_store.resolved_instance_sha256,
    )
    write_content_addressed_json(event, ledger.event_store.resolve(), prefix="consumption")
    validation = runner.invoke(
        app,
        [
            "validate",
            "--registry",
            str(registry_path),
            "--ledger",
            str(ledger_path),
        ],
    )
    assert validation.exit_code != 0
    assert "registry_sha256 mismatch" in str(validation.exception)


def test_cli_status_and_atomic_schema_replacement(tmp_path: Path) -> None:
    target = tmp_path / "schema.json"
    target.write_text("partial", encoding="utf-8")
    _atomic_write_text(target, "complete\n")
    assert target.read_text() == "complete\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
    schema_dir = tmp_path / "schemas"
    schema_result = CliRunner().invoke(app, ["schema", "--output", str(schema_dir)])
    assert schema_result.exit_code == 0, schema_result.output
    assert {
        "dataset-registry.schema.json",
        "consumed-test-ledger.schema.json",
        "split-manifest.schema.json",
        "sealed-target-authorization.schema.json",
        "sealed-target-selector-receipt.schema.json",
    } <= {path.name for path in schema_dir.iterdir()}
    assert list(schema_dir.glob(".*.tmp")) == []

    entry = dataset_entry()
    entry["license"] = {
        "status": "pending",
        "redistribution": "pending",
        "privacy_notes": "Pending audit.",
        "blocker": "License review pending.",
    }
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    registry_path, ledger_path = _write_yaml_models(tmp_path, registry, ledger)
    result = CliRunner().invoke(
        app,
        ["validate", "--registry", str(registry_path), "--ledger", str(ledger_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_status"] == "valid"
    assert payload["portfolio_operational_status"] == "blocked"
    assert payload["status"] == "schema-valid-but-operationally-blocked"
    assert entry["dataset_id"] in payload["dataset_audit_blockers"]


def test_cli_signed_selector_and_authorization_require_ledger_and_ephemeral_key_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, registry, ledger, sources, private_key = _sealed_context(tmp_path)
    registry_path, ledger_path = _write_yaml_models(tmp_path, registry, ledger)
    private_key_value = base64.b64encode(
        private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode("ascii")
    monkeypatch.setenv("JEPA4D_EPHEMERAL_TEST_SIGNER", private_key_value)
    runner = CliRunner()
    selector_result = runner.invoke(
        app,
        [
            "issue-sealed-selector",
            "--registry",
            str(registry_path),
            "--ledger",
            str(ledger_path),
            "--dataset",
            entry["dataset_id"],
            "--split",
            entry["splits"][0]["split_id"],
            "--survivor",
            "candidate-m1",
            "--git-commit",
            "a" * 40,
            "--preregistration",
            str(sources["preregistration"]),
            "--checkpoint",
            str(sources["checkpoint"]),
            "--config",
            str(sources["config"]),
            "--calibrator",
            str(sources["calibrator"]),
            "--private-key-env",
            "JEPA4D_EPHEMERAL_TEST_SIGNER",
            "--output",
            str(tmp_path / "cli-selector"),
        ],
    )
    assert selector_result.exit_code == 0, selector_result.output
    assert private_key_value not in selector_result.output
    selector_path = json.loads(selector_result.output)["selector"]
    authorization_result = runner.invoke(
        app,
        [
            "issue-sealed-authorization",
            "--registry",
            str(registry_path),
            "--ledger",
            str(ledger_path),
            "--dataset",
            entry["dataset_id"],
            "--split",
            entry["splits"][0]["split_id"],
            "--survivor",
            "candidate-m1",
            "--preregistration",
            str(sources["preregistration"]),
            "--selector",
            selector_path,
            "--checkpoint",
            str(sources["checkpoint"]),
            "--config",
            str(sources["config"]),
            "--calibrator",
            str(sources["calibrator"]),
            "--output",
            str(tmp_path / "cli-authorization"),
        ],
    )
    assert authorization_result.exit_code == 0, authorization_result.output
    authorization_path = json.loads(authorization_result.output)["authorization"]
    verified, _ = verify_sealed_target_authorization(
        authorization_path,
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
    )
    assert verified.git_commit == "a" * 40
