from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from jepa4d.tests.test_validation_registry import (
    approve_test_sealed_authority,
    dataset_entry,
    registry_value,
)
from jepa4d.validation._content import write_content_addressed_json
from jepa4d.validation.access import (
    DatasetAccessController,
    write_sealed_target_authorization,
    write_sealed_target_selector_receipt,
)
from jepa4d.validation.ledger import (
    ConsumedTestLedger,
    ConsumptionEvent,
    FirstOpenLineage,
    FutureUse,
    append_first_open,
    effective_targets,
    load_events,
)
from jepa4d.validation.registry import DatasetRegistry


def ledger_value(*targets: dict, externally_append_only: bool = False) -> dict:
    return {
        "schema_version": "jepa4d-consumed-test-ledger-v1",
        "ledger_version": "test-v1",
        "event_store": {
            "root_env": "JEPA4D_TEST_EVENT_ROOT",
            "relative_path": "events/test-v1",
            "durability": "external-append-only" if externally_append_only else "local-filesystem-best-effort",
            "externally_append_only": externally_append_only,
            **(
                {}
                if externally_append_only
                else {"deployment_blocker": "Test-only local store; no external append-only durability."}
            ),
        },
        "targets": list(targets),
    }


@pytest.fixture(autouse=True)
def _canonical_event_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEPA4D_TEST_EVENT_ROOT", str(tmp_path / "validation-state"))


def unopened_target(entry: dict, *, state: str = "available-unopened") -> dict:
    return {
        "dataset_id": entry["dataset_id"],
        "split_id": entry["splits"][0]["split_id"],
        "state": state,
        **(
            {"seal_evidence": "Frozen opacity receipt states target_opened=false."}
            if state == "sealed-unopened"
            else {}
        ),
        "notes": "Ledger fixture.",
    }


def lineage(registry: DatasetRegistry) -> FirstOpenLineage:
    return FirstOpenLineage(
        opened_at="2026-06-30T08:00:00Z",
        time_precision="exact",
        experiment_id="test-first-open",
        git_commit="abcdef1",
        source_record="outputs/test/receipt.json",
        source_record_sha256="e" * 64,
        registry_sha256=registry.sha256,
        notes="Target bytes and labels first opened in formal evaluation.",
    )


def test_ledger_requires_every_registered_held_out_target() -> None:
    first = dataset_entry(dataset_id="fixture.first", split_id="fixture.first.test")
    second = dataset_entry(dataset_id="fixture.second", split_id="fixture.second.test")
    second["splits"][0]["id_manifest_sha256"] = "c" * 64
    registry = DatasetRegistry.model_validate(registry_value(first, second))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(first)))
    with pytest.raises(ValueError, match="missing held-out target rows"):
        ledger.validate_against(registry)


def test_first_open_is_content_addressed_single_use_and_reuse_is_explicit(tmp_path: Path) -> None:
    entry = dataset_entry(operations=["development-evaluation", "regression", "reporting"])
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    ledger.validate_against(registry)
    artifact = append_first_open(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        operation="development-evaluation",
        lineage=lineage(registry),
        permitted_future_uses=frozenset({FutureUse.REGRESSION, FutureUse.REPORTING}),
    )
    assert artifact.path.name == f"consumption-{artifact.sha256}.json"
    events = load_events(registry=registry, ledger=ledger)
    assert events[0].first_open.experiment_id == "test-first-open"
    assert (
        DatasetAccessController(registry=registry, ledger=ledger)
        .authorize(entry["dataset_id"], entry["splits"][0]["split_id"], "regression")
        .ledger_state.value
        == "consumed"
    )
    with pytest.raises(PermissionError, match="not permitted"):
        DatasetAccessController(registry=registry, ledger=ledger).authorize(
            entry["dataset_id"], entry["splits"][0]["split_id"], "mechanism-diagnostic"
        )
    with pytest.raises(ValueError, match="already consumed"):
        append_first_open(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            operation="development-evaluation",
            lineage=lineage(registry),
            permitted_future_uses=frozenset({FutureUse.REGRESSION}),
        )


@pytest.mark.parametrize("operation", ["metadata-audit", "reporting"])
def test_nonopening_operation_cannot_create_false_consumption(tmp_path: Path, operation: str) -> None:
    entry = dataset_entry(operations=[operation, "development-evaluation"])
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    with pytest.raises(PermissionError, match="does not open target data"):
        append_first_open(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            operation=operation,
            lineage=lineage(registry),
            permitted_future_uses=frozenset({FutureUse.REPORTING}),
        )
    assert load_events(registry=registry, ledger=ledger) == ()


def test_sealed_target_needs_hash_bound_authorization(tmp_path: Path) -> None:
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
    local_ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry, state="sealed-unopened")))
    with pytest.raises(PermissionError, match="authenticated content-addressed authorization"):
        append_first_open(
            registry=registry,
            ledger=ConsumedTestLedger.model_validate(
                ledger_value(unopened_target(entry, state="sealed-unopened"), externally_append_only=True)
            ),
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            operation="external-evaluation",
            lineage=lineage(registry),
            permitted_future_uses=frozenset({FutureUse.NO_FUTURE_USE}),
        )
    with pytest.raises(PermissionError, match="externally append-only event store"):
        append_first_open(
            registry=registry,
            ledger=local_ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            operation="external-evaluation",
            lineage=lineage(registry),
            permitted_future_uses=frozenset({FutureUse.NO_FUTURE_USE}),
            sealed_authorization=tmp_path / "not-opened.json",
        )
    ledger = ConsumedTestLedger.model_validate(
        ledger_value(unopened_target(entry, state="sealed-unopened"), externally_append_only=True)
    )
    sources = {}
    for name in ("preregistration", "checkpoint", "config", "calibrator"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"frozen-{name}".encode())
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
    authorization = write_sealed_target_authorization(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        survivor="candidate-m1",
        output_dir=tmp_path / "authorizations",
        **sources,
    )
    artifact = append_first_open(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        operation="external-evaluation",
        lineage=lineage(registry),
        permitted_future_uses=frozenset({FutureUse.NO_FUTURE_USE}),
        sealed_authorization=authorization.path,
    )
    assert json.loads(artifact.path.read_text())["sealed_authorization_sha256"] == authorization.sha256
    controller = DatasetAccessController(registry=registry, ledger=ledger)
    with pytest.raises(PermissionError, match="consumed target denies external-evaluation"):
        controller.authorize(
            entry["dataset_id"],
            entry["splits"][0]["split_id"],
            "external-evaluation",
        )
    with pytest.raises(PermissionError, match="cannot be reused after target consumption"):
        controller.authorize(
            entry["dataset_id"],
            entry["splits"][0]["split_id"],
            "external-evaluation",
            sealed_authorization=authorization.path,
        )


def test_concurrent_first_open_has_exactly_one_winner() -> None:
    entry = dataset_entry(operations=["development-evaluation", "reporting"])
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))

    def consume_once(index: int) -> str:
        artifact = append_first_open(
            registry=registry,
            ledger=ledger,
            dataset_id=entry["dataset_id"],
            split_id=entry["splits"][0]["split_id"],
            operation="development-evaluation",
            lineage=lineage(registry).model_copy(update={"experiment_id": f"concurrent-{index}"}),
            permitted_future_uses=frozenset({FutureUse.REPORTING}),
        )
        return artifact.sha256

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(consume_once, index) for index in range(2)]
    outcomes: list[str] = []
    errors: list[BaseException] = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except BaseException as error:  # noqa: BLE001 - both task outcomes are asserted below.
            errors.append(error)
    assert len(outcomes) == 1
    assert len(errors) == 1
    assert "already consumed" in str(errors[0])
    assert len(load_events(registry=registry, ledger=ledger)) == 1


def test_alternate_event_directory_and_replayed_root_cannot_override_canonical_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = dataset_entry()
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    event = ConsumptionEvent(
        schema_version="jepa4d-consumed-test-event-v1",
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        prior_state="available-unopened",
        open_operation="development-evaluation",
        first_open=lineage(registry),
        permitted_future_uses=frozenset({FutureUse.REPORTING}),
        registry_sha256=registry.sha256,
        base_ledger_sha256=ledger.sha256,
        event_store_sha256=ledger.event_store.sha256,
        resolved_instance_sha256=ledger.event_store.resolved_instance_sha256,
    )
    alternate = tmp_path / "attacker-selected-events"
    write_content_addressed_json(event, alternate, prefix="consumption")
    assert load_events(registry=registry, ledger=ledger) == ()
    decision = DatasetAccessController(registry=registry, ledger=ledger).authorize(
        entry["dataset_id"], entry["splits"][0]["split_id"], "development-evaluation"
    )
    assert decision.ledger_state is not None
    assert decision.ledger_state.value == "available-unopened"

    replay_root = tmp_path / "fresh-event-root"
    replay_directory = replay_root / ledger.event_store.relative_path
    write_content_addressed_json(event, replay_directory, prefix="consumption")
    monkeypatch.setenv("JEPA4D_TEST_EVENT_ROOT", str(replay_root))
    with pytest.raises(ValueError, match="resolved_instance_sha256 mismatch"):
        load_events(registry=registry, ledger=ledger)


def test_event_tamper_and_incoherent_historical_record_are_rejected(tmp_path: Path) -> None:
    entry = dataset_entry()
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    artifact = append_first_open(
        registry=registry,
        ledger=ledger,
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        operation="development-evaluation",
        lineage=lineage(registry),
        permitted_future_uses=frozenset({FutureUse.REPORTING}),
    )
    payload = json.loads(artifact.path.read_text())
    payload["first_open"]["experiment_id"] = "tampered"
    artifact.path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="content digest mismatch"):
        load_events(registry=registry, ledger=ledger)

    invalid = unopened_target(entry)
    invalid.update(state="consumed", permitted_future_uses=["reporting"])
    with pytest.raises(ValidationError, match="first_open lineage"):
        ConsumedTestLedger.model_validate(ledger_value(invalid))


@pytest.mark.parametrize("mismatch", ["registry", "ledger", "event_store", "resolved_instance"])
def test_event_lineage_hash_mismatch_is_rejected(tmp_path: Path, mismatch: str) -> None:
    entry = dataset_entry()
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(ledger_value(unopened_target(entry)))
    event_registry_sha256 = "0" * 64 if mismatch == "registry" else registry.sha256
    event = ConsumptionEvent(
        schema_version="jepa4d-consumed-test-event-v1",
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        prior_state="available-unopened",
        open_operation="development-evaluation",
        first_open=lineage(registry).model_copy(update={"registry_sha256": event_registry_sha256}),
        permitted_future_uses=frozenset({FutureUse.REPORTING}),
        registry_sha256=event_registry_sha256,
        base_ledger_sha256="1" * 64 if mismatch == "ledger" else ledger.sha256,
        event_store_sha256="2" * 64 if mismatch == "event_store" else ledger.event_store.sha256,
        resolved_instance_sha256=(
            "3" * 64 if mismatch == "resolved_instance" else ledger.event_store.resolved_instance_sha256
        ),
    )
    event_dir = ledger.event_store.resolve()
    write_content_addressed_json(event, event_dir, prefix="consumption")
    mismatch_field = {
        "registry": "registry",
        "ledger": "base_ledger",
        "event_store": "event_store",
        "resolved_instance": "resolved_instance_sha256",
    }[mismatch]
    with pytest.raises(ValueError, match=f"event {mismatch_field}.*mismatch"):
        load_events(registry=registry, ledger=ledger)
    with pytest.raises(ValueError, match=f"event {mismatch_field}.*mismatch"):
        effective_targets(registry, ledger, (event,))


def test_checked_in_ledger_covers_historical_consumption_and_sealed_diode() -> None:
    root = Path(__file__).resolve().parents[2]
    registry = DatasetRegistry.load(root / "configs/validation/dataset_registry.yaml")
    ledger = ConsumedTestLedger.load(root / "configs/validation/consumed_test_ledger.yaml")
    ledger.validate_against(registry)
    assert ledger.target("diode.geometry-external", "diode.validation-external").state.value == "sealed-unopened"
    consumed = {(target.dataset_id, target.split_id) for target in ledger.targets if target.state.value == "consumed"}
    assert {
        ("sun-rgbd.geometry-development", "sun-rgbd.phase2e-kv2-test"),
        ("sun-rgbd.geometry-development", "sun-rgbd.phase2f-four-family-development"),
        ("tum-rgbd.geometry-regression", "tum-rgbd.phase2c-freiburg3-test"),
        ("davis2017.identity-development", "davis2017.dogs-scale-exploratory"),
    } <= consumed
    schema = json.loads((root / "configs/validation/schemas/consumed-test-ledger.schema.json").read_text())
    assert schema == ConsumedTestLedger.model_json_schema()
