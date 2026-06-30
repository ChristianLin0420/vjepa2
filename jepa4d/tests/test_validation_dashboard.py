from __future__ import annotations

import builtins
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from jepa4d.evaluation.failure_taxonomy import EvidenceLevel as SharedEvidenceLevel
from jepa4d.visualization.validation_dashboard import (
    HTML_FILENAME,
    IMMUTABLE_DIRECTORY_PREFIX,
    JSON_FILENAME,
    RECEIPT_FILENAME,
    SCHEMA_VERSION,
    STANDARD_VISUALIZATIONS,
    ClaimBoundary,
    Completeness,
    CompletenessStatus,
    DatasetEvidence,
    DatasetIdentityKind,
    DatasetRole,
    EvidenceLevel,
    GateCondition,
    GateDecision,
    GateDomain,
    GateOutcome,
    MetricDomain,
    MetricRecord,
    ResourcePolicy,
    TargetAccess,
    ValidationReport,
    VisualizationDeclaration,
    verify_immutable_validation_dashboard,
    verify_validation_dashboard,
    write_immutable_validation_dashboard,
    write_validation_dashboard,
)


def _dataset(
    *,
    role: DatasetRole = DatasetRole.PRIMARY_DEVELOPMENT,
    target_access: TargetAccess = TargetAccess.DEVELOPMENT_OPEN,
) -> DatasetEvidence:
    return DatasetEvidence(
        dataset_id="fixture-scenes",
        version="1.0",
        split="development-test",
        role=role,
        target_access=target_access,
        identity_kind=DatasetIdentityKind.SPLIT_GOVERNANCE,
        identity_sha256="a" * 64,
    )


def _report(**overrides: object) -> ValidationReport:
    values: dict[str, object] = {
        "report_id": "phase2-quality-001",
        "title": "Quality-first validation",
        "stage": "geometry",
        "evidence_level": EvidenceLevel.DEVELOPMENT_BENCHMARK,
        "datasets": (_dataset(),),
        "gate": GateDecision(
            name="development-survivor",
            outcome=GateOutcome.PASS,
            decision="Promote the frozen quality survivor.",
            conditions=(GateCondition("quality_noninferior", GateDomain.QUALITY, True, "Frozen threshold passed."),),
        ),
        "completeness": Completeness(expected_cells=4, succeeded_cells=4),
        "claim_boundary": ClaimBoundary(
            supported=("Development-set geometry quality under the frozen protocol.",),
            prohibited=("Cross-dataset transfer or deployment readiness.",),
        ),
        "metrics": (
            MetricRecord("raw_abs_rel", 0.123, "ratio", MetricDomain.QUALITY, "development-test"),
            MetricRecord("latency_p50", 8.5, "ms", MetricDomain.RESOURCE, "development-test"),
        ),
        "timestamp": "2026-06-30T12:00:00+00:00",
    }
    values.update(overrides)
    return ValidationReport(**values)  # type: ignore[arg-type]


def test_dashboard_writes_canonical_json_and_self_contained_html(tmp_path) -> None:
    report = _report(title="Quality <script>alert('x')</script>")
    json_path, html_path = write_validation_dashboard(report, tmp_path)

    assert json_path == tmp_path / JSON_FILENAME
    assert html_path == tmp_path / HTML_FILENAME
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["datasets"][0]["role"] == "A1-primary-development"
    assert payload["completeness"] == {
        "accounted_cells": 4,
        "accounted_fraction": 1.0,
        "expected_cells": 4,
        "expected_failure_cells": 0,
        "failed_cells": 0,
        "legal_skips": 0,
        "missing_cells": 0,
        "status": "complete",
        "succeeded_cells": 4,
    }

    document = html_path.read_text()
    assert "<script>alert" not in document
    assert "&lt;script&gt;alert" in document
    assert "Scientific quality" in document
    assert "Resource diagnostics" in document
    assert "split-governance" in document
    assert "ID-manifest SHA-256" in document
    assert "Manifest SHA-256" not in document
    assert "Cross-dataset transfer or deployment readiness." in document
    assert "src=" not in document
    for declaration in STANDARD_VISUALIZATIONS:
        assert document.count(f"data-panel-id='{declaration.panel_id}'") == 1

    receipt = verify_validation_dashboard(tmp_path)
    assert receipt["report_id"] == report.report_id
    assert (tmp_path / RECEIPT_FILENAME).is_file()
    for name in (JSON_FILENAME, HTML_FILENAME):
        artifact = tmp_path / name
        assert receipt["files"][name]["bytes"] == artifact.stat().st_size
        assert receipt["files"][name]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_dashboard_bundle_receipt_detects_cross_generation_or_tampered_file(tmp_path) -> None:
    write_validation_dashboard(_report(), tmp_path)
    receipt = verify_validation_dashboard(tmp_path)
    assert len(receipt["generation_id"]) == 64
    (tmp_path / HTML_FILENAME).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match receipt"):
        verify_validation_dashboard(tmp_path)


def test_immutable_dashboard_is_content_addressed_idempotent_and_no_clobber(tmp_path) -> None:
    first = write_immutable_validation_dashboard(_report(), tmp_path)
    assert first.directory.name == f"{IMMUTABLE_DIRECTORY_PREFIX}{first.generation_id}"
    assert verify_immutable_validation_dashboard(first.directory)["generation_id"] == first.generation_id
    before = first.json_path.stat().st_mtime_ns

    same = write_immutable_validation_dashboard(_report(), tmp_path)
    assert same == first
    assert same.json_path.stat().st_mtime_ns == before

    second = write_immutable_validation_dashboard(_report(report_id="another-generation"), tmp_path)
    assert second.directory != first.directory
    assert first.json_path.is_file()


def test_immutable_dashboard_never_repairs_or_overwrites_tampered_generation(tmp_path) -> None:
    bundle = write_immutable_validation_dashboard(_report(), tmp_path)
    bundle.html_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match receipt"):
        write_immutable_validation_dashboard(_report(), tmp_path)
    assert bundle.html_path.read_text(encoding="utf-8") == "tampered"


def test_concurrent_dashboard_writers_publish_one_receipt_bound_pair(tmp_path) -> None:
    reports = tuple(_report(report_id=f"concurrent-{index}", title=f"Concurrent {index}") for index in range(4))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda report: write_validation_dashboard(report, tmp_path), reports))

    receipt = verify_validation_dashboard(tmp_path)
    payload = json.loads((tmp_path / JSON_FILENAME).read_text())
    assert receipt["report_id"] == payload["report_id"]
    assert payload["report_id"] in {report.report_id for report in reports}


def test_wandb_payload_is_flat_domain_separated_and_offline(monkeypatch) -> None:
    real_import = builtins.__import__

    def reject_wandb(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("validation report payload must not import W&B")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_wandb)
    payload = _report().wandb_summary_payload()

    assert payload["validation/evidence_level"] == "development-benchmark"
    assert payload["validation/gate/outcome"] == "pass"
    assert payload["validation/completeness/status"] == "complete"
    assert payload["validation/quality/raw_abs_rel/development-test"] == pytest.approx(0.123)
    assert payload["validation/resource/latency_p50/development-test"] == pytest.approx(8.5)
    assert json.loads(str(payload["validation/datasets/roles_json"])) == {"fixture-scenes": "A1-primary-development"}


@pytest.mark.parametrize("evidence_level", tuple(SharedEvidenceLevel))
def test_shared_evidence_vocabulary_round_trips_through_dashboard(evidence_level: SharedEvidenceLevel) -> None:
    assert EvidenceLevel is SharedEvidenceLevel
    report = _report(evidence_level=evidence_level)
    payload = report.to_serializable()
    assert SharedEvidenceLevel(payload["evidence_level"]) is evidence_level
    assert report.wandb_summary_payload()["validation/evidence_level"] == evidence_level.value


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("contract_only", SharedEvidenceLevel.CONTRACT_ONLY),
        ("integration_only", SharedEvidenceLevel.INTEGRATION),
        ("development", SharedEvidenceLevel.DEVELOPMENT_BENCHMARK),
        ("held_out", SharedEvidenceLevel.DEVELOPMENT_BENCHMARK),
        ("external_confirmation", SharedEvidenceLevel.EXTERNAL_CONFIRMATION),
    ],
)
def test_legacy_evidence_values_parse_to_canonical_vocabulary(legacy: str, canonical: SharedEvidenceLevel) -> None:
    assert SharedEvidenceLevel(legacy) is canonical


def test_completeness_derives_accounting_status_and_final_gate_requires_complete_matrix() -> None:
    partial = Completeness(expected_cells=8, succeeded_cells=5, failed_cells=1)
    assert partial.status == CompletenessStatus.PARTIAL
    assert partial.missing_cells == 2
    with pytest.raises(ValueError, match="fully accounted"):
        _report(completeness=partial)

    blocked = _report(
        completeness=partial,
        gate=GateDecision("access-audit", GateOutcome.BLOCKED, "Dataset terms require review."),
    )
    assert blocked.completeness.status == CompletenessStatus.PARTIAL


def test_passing_gate_rejects_fully_accounted_matrix_with_failed_cells() -> None:
    accounted_with_failure = Completeness(expected_cells=4, succeeded_cells=3, failed_cells=1)
    assert accounted_with_failure.status == CompletenessStatus.COMPLETE
    with pytest.raises(ValueError, match="zero failed experiment cells"):
        _report(completeness=accounted_with_failure)


def test_diagnostic_resource_policy_cannot_silently_bind_promotion() -> None:
    resource_gate = GateDecision(
        "latency",
        GateOutcome.PASS,
        "Promote under an explicitly binding resource envelope.",
        (GateCondition("latency_under_budget", GateDomain.RESOURCE, True, "p95 is below the frozen limit."),),
    )
    with pytest.raises(ValueError, match="cannot bind"):
        _report(gate=resource_gate)

    report = _report(gate=resource_gate, resource_policy=ResourcePolicy.BINDING_GATE)
    assert report.resource_policy == ResourcePolicy.BINDING_GATE


def test_external_transfer_targets_cannot_be_development_open() -> None:
    with pytest.raises(ValueError, match="external-transfer"):
        _dataset(role=DatasetRole.EXTERNAL_TRANSFER, target_access=TargetAccess.DEVELOPMENT_OPEN)

    sealed = _dataset(role=DatasetRole.EXTERNAL_TRANSFER, target_access=TargetAccess.OPAQUE)
    assert sealed.target_access == TargetAccess.OPAQUE


def test_contract_rejects_nonfinite_metrics_gate_drift_and_panel_drift() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricRecord("loss", float("nan"), "nats", MetricDomain.QUALITY, "validation")
    with pytest.raises(ValueError, match="all conditions must pass"):
        GateDecision(
            "quality",
            GateOutcome.PASS,
            "Invalid passing gate.",
            (GateCondition("quality", GateDomain.QUALITY, False, "Threshold failed."),),
        )
    changed_panels = (*STANDARD_VISUALIZATIONS[:-1], VisualizationDeclaration("custom", "plot", "Ad hoc panel."))
    with pytest.raises(ValueError, match="STANDARD_VISUALIZATIONS"):
        _report(visualizations=changed_panels)

    with pytest.raises(ValueError, match="id-manifest identity"):
        DatasetEvidence(
            dataset_id="fixture",
            version="v1",
            split="test",
            role=DatasetRole.PRIMARY_DEVELOPMENT,
            target_access=TargetAccess.CONSUMED,
            identity_kind=DatasetIdentityKind.ID_MANIFEST,
            identity_sha256="a" * 64,
        )
