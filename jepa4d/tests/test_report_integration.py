from __future__ import annotations

import pytest

from jepa4d.evaluation.failure_taxonomy import (
    EvidenceLevel,
    ExecutionStatus,
    ProtocolStatus,
    ScientificStatus,
    ValidationStage,
    ValidationStatus,
)
from jepa4d.tests.test_consumed_test_ledger import unopened_target
from jepa4d.tests.test_validation_registry import dataset_entry, registry_value
from jepa4d.validation.ledger import (
    ConsumedTestLedger,
    ConsumptionEvent,
    FirstOpenLineage,
    FutureUse,
    LedgerState,
)
from jepa4d.validation.registry import AccessOperation, DataRole, DatasetRegistry
from jepa4d.validation.report_integration import (
    ReportDatasetRef,
    build_governed_validation_report,
    dashboard_role_from_registry,
)
from jepa4d.visualization.validation_dashboard import (
    ClaimBoundary,
    DatasetIdentityKind,
    DatasetRole,
    GateOutcome,
    TargetAccess,
)


@pytest.fixture(autouse=True)
def _event_store_root(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEPA4D_EVENT_ROOT", str(tmp_path / "validation-state"))


def _lineage(registry: DatasetRegistry) -> FirstOpenLineage:
    return FirstOpenLineage(
        opened_at="2026-06-30T18:00:00Z",
        time_precision="exact",
        experiment_id="governed-report-test",
        git_commit="abcdef1",
        source_record="outputs/governed/final.json",
        source_record_sha256="e" * 64,
        registry_sha256=registry.sha256,
        notes="Formal evaluation target opening.",
    )


def _ledger_value(*targets: dict) -> dict:
    return {
        "schema_version": "jepa4d-consumed-test-ledger-v1",
        "ledger_version": "report-integration-test-v1",
        "event_store": {
            "root_env": "JEPA4D_EVENT_ROOT",
            "relative_path": "validation/events",
            "durability": "local-filesystem-best-effort",
            "externally_append_only": False,
            "deployment_blocker": "Test fixture is not a production append-only event store.",
        },
        "targets": list(targets),
    }


def _status(evidence_level: EvidenceLevel = EvidenceLevel.DEVELOPMENT_BENCHMARK) -> ValidationStatus:
    return ValidationStatus(
        experiment_id="governed-report-test",
        stage=ValidationStage.REPRESENTATION,
        protocol=ProtocolStatus.FROZEN,
        execution=ExecutionStatus.COMPLETE,
        scientific=ScientificStatus.PASS,
        evidence_level=evidence_level,
        expected_cells=2,
        successful_cells=2,
        metric_gate_receipt_sha256=("9" * 64 if evidence_level is EvidenceLevel.EXTERNAL_CONFIRMATION else None),
    )


def _claims() -> ClaimBoundary:
    return ClaimBoundary(
        supported=("Frozen governed evaluation result.",),
        prohibited=("Any claim beyond the registered dataset role.",),
    )


@pytest.mark.parametrize(
    ("role", "dashboard_role"),
    [
        (DataRole.A1, DatasetRole.PRIMARY_DEVELOPMENT),
        (DataRole.A2, DatasetRole.COMPLEMENTARY_DEVELOPMENT),
        (DataRole.B, DatasetRole.EXTERNAL_TRANSFER),
        (DataRole.C, DatasetRole.STRESS_SAFETY),
        (DataRole.CONTRACT_ONLY, DatasetRole.CONTRACT_FIXTURE),
    ],
)
def test_registry_roles_have_one_canonical_dashboard_mapping(role: DataRole, dashboard_role: DatasetRole) -> None:
    assert dashboard_role_from_registry(role) is dashboard_role


def test_factory_derives_role_access_evidence_completeness_and_governance() -> None:
    entry = dataset_entry()
    registry = DatasetRegistry.model_validate(registry_value(entry))
    consumed = unopened_target(entry)
    consumed.update(
        state="consumed",
        first_open=_lineage(registry).model_dump(mode="json", exclude_none=True),
        permitted_future_uses=["regression", "reporting"],
    )
    ledger = ConsumedTestLedger.model_validate(_ledger_value(consumed))
    status = _status()

    report = build_governed_validation_report(
        registry=registry,
        ledger=ledger,
        events=(),
        status=status,
        dataset_splits=(ReportDatasetRef(entry["dataset_id"], entry["splits"][0]["split_id"]),),
        report_id="governed-report",
        title="Governed report",
        gate_name="quality-gate",
        gate_decision="Promote under the frozen development protocol.",
        claim_boundary=_claims(),
        timestamp="2026-06-30T18:30:00Z",
    )

    assert report.stage == "representation"
    assert report.evidence_level is EvidenceLevel.DEVELOPMENT_BENCHMARK
    assert report.gate.outcome is GateOutcome.PASS
    assert report.completeness.expected_cells == status.expected_cells
    assert report.datasets[0].role is DatasetRole.PRIMARY_DEVELOPMENT
    assert report.datasets[0].target_access is TargetAccess.CONSUMED
    assert report.datasets[0].identity_kind is DatasetIdentityKind.ID_MANIFEST
    assert report.datasets[0].id_manifest_sha256 == entry["splits"][0]["id_manifest_sha256"]
    assert report.governance is not None
    assert report.governance.registry_sha256 == registry.sha256
    assert report.wandb_summary_payload()["validation/governance/registry_sha256"] == registry.sha256


def _external_fixture() -> tuple[dict, DatasetRegistry, ConsumedTestLedger, ConsumptionEvent]:
    entry = dataset_entry(
        dataset_id="fixture.external",
        split_id="fixture.external.test",
        role="B",
        purpose="external-test",
        target_state="sealed",
        operations=["metadata-audit", "external-evaluation", "reporting"],
    )
    entry["splits"][0]["id_manifest"] = None
    entry["splits"][0]["id_manifest_sha256"] = None
    entry["sealed_authority"] = {
        "status": "pending",
        "blocker": "Test authority is intentionally not provisioned.",
    }
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(_ledger_value(unopened_target(entry, state="sealed-unopened")))
    event = ConsumptionEvent(
        schema_version="jepa4d-consumed-test-event-v1",
        dataset_id=entry["dataset_id"],
        split_id=entry["splits"][0]["split_id"],
        prior_state=LedgerState.SEALED_UNOPENED,
        open_operation=AccessOperation.EXTERNAL_EVALUATION,
        first_open=_lineage(registry),
        permitted_future_uses=frozenset({FutureUse.NO_FUTURE_USE}),
        registry_sha256=registry.sha256,
        base_ledger_sha256=ledger.sha256,
        event_store_sha256=ledger.event_store.sha256,
        resolved_instance_sha256=ledger.event_store.resolved_instance_sha256,
        sealed_authorization_sha256="f" * 64,
    )
    return entry, registry, ledger, event


def test_external_confirmation_requires_consumed_b_target_and_actual_external_event() -> None:
    entry, registry, unopened_ledger, event = _external_fixture()
    reference = (ReportDatasetRef(entry["dataset_id"], entry["splits"][0]["split_id"]),)
    status = _status(EvidenceLevel.EXTERNAL_CONFIRMATION)

    with pytest.raises(ValueError, match="consumed target state"):
        build_governed_validation_report(
            registry=registry,
            ledger=unopened_ledger,
            events=(),
            status=status,
            dataset_splits=reference,
            report_id="external-report",
            title="External report",
            gate_name="external-gate",
            gate_decision="Confirm transfer.",
            claim_boundary=_claims(),
        )

    report = build_governed_validation_report(
        registry=registry,
        ledger=unopened_ledger,
        events=(event,),
        status=status,
        dataset_splits=reference,
        report_id="external-report",
        title="External report",
        gate_name="external-gate",
        gate_decision="Confirm transfer.",
        claim_boundary=_claims(),
    )
    assert report.evidence_level is EvidenceLevel.EXTERNAL_CONFIRMATION
    assert report.datasets[0].role is DatasetRole.EXTERNAL_TRANSFER
    assert report.datasets[0].target_access is TargetAccess.CONSUMED
    assert report.datasets[0].identity_kind is DatasetIdentityKind.SPLIT_GOVERNANCE
    assert report.datasets[0].id_manifest_sha256 is None
    assert report.governance is not None
    assert report.governance.metric_gate_receipt_sha256 == "9" * 64
    assert report.wandb_summary_payload()["validation/governance/metric_gate_receipt_sha256"] == "9" * 64


def test_external_confirmation_rejects_consumed_b_base_without_first_open_event() -> None:
    entry, registry, _, _ = _external_fixture()
    consumed = unopened_target(entry, state="sealed-unopened")
    consumed.pop("seal_evidence")
    consumed.update(
        state="consumed",
        first_open=_lineage(registry).model_dump(mode="json", exclude_none=True),
        permitted_future_uses=["no-future-use"],
    )
    ledger = ConsumedTestLedger.model_validate(_ledger_value(consumed))
    with pytest.raises(ValueError, match="first-open event"):
        build_governed_validation_report(
            registry=registry,
            ledger=ledger,
            events=(),
            status=_status(EvidenceLevel.EXTERNAL_CONFIRMATION),
            dataset_splits=(ReportDatasetRef(entry["dataset_id"], entry["splits"][0]["split_id"]),),
            report_id="external-report",
            title="External report",
            gate_name="external-gate",
            gate_decision="Confirm transfer.",
            claim_boundary=_claims(),
        )


def test_terminal_report_cannot_claim_an_unconsumed_development_target() -> None:
    entry = dataset_entry()
    registry = DatasetRegistry.model_validate(registry_value(entry))
    ledger = ConsumedTestLedger.model_validate(_ledger_value(unopened_target(entry)))
    with pytest.raises(ValueError, match="requires consumed target state"):
        build_governed_validation_report(
            registry=registry,
            ledger=ledger,
            events=(),
            status=_status(),
            dataset_splits=(ReportDatasetRef(entry["dataset_id"], entry["splits"][0]["split_id"]),),
            report_id="development-report",
            title="Development report",
            gate_name="development-gate",
            gate_decision="Promote.",
            claim_boundary=_claims(),
        )
