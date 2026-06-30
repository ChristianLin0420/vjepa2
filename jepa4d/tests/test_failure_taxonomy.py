import pytest

from jepa4d.evaluation.failure_taxonomy import (
    FAILURE_TAXONOMY_SCHEMA_VERSION,
    VALIDATION_STATUS_SCHEMA_VERSION,
    EvidenceLevel,
    ExecutionStatus,
    FailureCategory,
    FailureEvent,
    FailureSeverity,
    ProtocolStatus,
    RetryDisposition,
    ScientificStatus,
    ValidationStage,
    ValidationStatus,
)


def test_failure_event_is_versioned_typed_and_serializable() -> None:
    event = FailureEvent(
        event_id="failure-0001",
        experiment_id="phase2g-pilot",
        logical_cell_id="m1/seed-0/scene-a",
        stage=ValidationStage.GEOMETRY,
        category=FailureCategory.NUMERICAL,
        message="non-finite validation loss",
        severity=FailureSeverity.CELL_FAILURE,
        disposition=RetryDisposition.NEW_EXPERIMENT_REQUIRED,
        sample_id="scene-a/frame-17",
        contributing=(FailureCategory.CALIBRATION,),
        tags={"metric": "nll"},
    )
    payload = event.to_serializable()
    assert payload["schema_version"] == FAILURE_TAXONOMY_SCHEMA_VERSION
    assert payload["stage"] == "geometry"
    assert payload["category"] == "numerical"
    assert payload["contributing"] == (FailureCategory.CALIBRATION,)


def test_failure_artifact_serialization_redacts_credentials_and_pseudonymizes_sample() -> None:
    event = FailureEvent(
        event_id="failure-safe-1",
        experiment_id="phase3-formal",
        logical_cell_id="cell-1",
        stage=ValidationStage.OBJECT_GROUNDING,
        category=FailureCategory.NETWORK,
        message=(
            "upload failed with token=not-a-real-secret-value; target=coffee-mug at "
            "/restricted/source/video-007/frame-12.png"
        ),
        dataset_id="restricted.dataset",
        sample_id="restricted/source/video-007/frame-12",
        tags={"endpoint": "wandb-upload"},
    )
    raw = event.to_serializable()
    artifact = event.to_artifact_serializable()
    assert "not-a-real-secret-value" in raw["message"]
    assert "not-a-real-secret-value" not in artifact["message"]
    assert "coffee-mug" not in artifact["message"]
    assert "/restricted/source" not in artifact["message"]
    assert artifact["message"] == "[REDACTED_TARGET_CONTEXT]"
    assert artifact["sample_id"].startswith("sha256:")
    assert "video-007" not in artifact["sample_id"]
    self_issued_authorization = {
        "event_id": event.event_id,
        "experiment_id": event.experiment_id,
        "dataset_id": event.dataset_id,
        "artifact_policy_sha256": "a" * 64,
        "approved_by": "dataset-policy-reviewer",
    }
    with pytest.raises(ValueError, match="disclosure is disabled"):
        event.to_artifact_serializable(raw_sample_id_authorization=self_issued_authorization)


def test_failure_artifact_rejects_every_raw_sample_id_authorization_shape() -> None:
    event = FailureEvent(
        event_id="failure-safe-raw-denied",
        experiment_id="phase3-formal",
        logical_cell_id="cell-1",
        stage=ValidationStage.OBJECT_GROUNDING,
        category=FailureCategory.METRIC,
        message="metric failed",
        dataset_id="restricted.dataset",
        sample_id="restricted/source/video-007/frame-12",
    )

    for attempted_authorization in (True, object(), {"signed": True}, "authority-token"):
        with pytest.raises(ValueError, match="cryptographically verified policy-authority"):
            event.to_artifact_serializable(raw_sample_id_authorization=attempted_authorization)

    artifact = event.to_artifact_serializable()
    assert artifact["sample_id"].startswith("sha256:")
    assert artifact["sample_id_disclosure"] == {"mode": "pseudonymous"}
    assert event.sample_id is not None
    assert event.sample_id not in str(artifact)


@pytest.mark.parametrize(
    "message",
    [
        "ground truth is dog",
        "class dog",
        "groundTruth is dog",
        "targetLabel dog",
        "ground_truth category dog",
    ],
)
def test_failure_artifact_redacts_free_text_target_context_bypasses(message: str) -> None:
    event = FailureEvent(
        event_id="failure-target-context",
        experiment_id="phase3-formal",
        logical_cell_id="cell-1",
        stage=ValidationStage.OBJECT_GROUNDING,
        category=FailureCategory.METRIC,
        message=message,
    )

    assert event.to_serializable()["message"] == message
    assert event.to_artifact_serializable()["message"] == "[REDACTED_TARGET_CONTEXT]"


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        ({"api_key": "value"}, "unsafe failure tag key"),
        ({"note": "".join(("hf", "_", "abcdefghijklmnopqrstuvwxyz123456"))}, "unsafe failure tag value"),
        ({"Bad Key": "value"}, "unsafe failure tag key"),
        ({"target_label": "mug"}, "unsafe failure tag key"),
        ({"detail": "/restricted/source/frame.png"}, "unsafe failure tag value"),
        ({"detail": "label=private-class"}, "unsafe failure tag value"),
        ({"detail": "ground truth is dog"}, "unsafe failure tag value"),
        ({"detail": "class dog"}, "unsafe failure tag value"),
        ({"detail": "groundTruth is dog"}, "unsafe failure tag value"),
        ({"detail": "targetLabel dog"}, "unsafe failure tag value"),
    ],
)
def test_failure_tags_reject_sensitive_or_unbounded_artifact_metadata(tags, match) -> None:
    with pytest.raises(ValueError, match=match):
        FailureEvent(
            event_id="failure-safe-2",
            experiment_id="phase3-formal",
            logical_cell_id="cell-1",
            stage=ValidationStage.OBJECT_GROUNDING,
            category=FailureCategory.NETWORK,
            message="upload failed",
            tags=tags,
        )


def test_failure_event_rejects_untyped_or_ambiguous_categories() -> None:
    with pytest.raises(TypeError, match="FailureCategory"):
        FailureEvent(
            event_id="failure-1",
            experiment_id="experiment",
            logical_cell_id="cell",
            stage=ValidationStage.SYSTEM,
            category="safety",  # type: ignore[arg-type]
            message="unsafe contact",
        )
    with pytest.raises(ValueError, match="unique"):
        FailureEvent(
            event_id="failure-1",
            experiment_id="experiment",
            logical_cell_id="cell",
            stage=ValidationStage.SYSTEM,
            category=FailureCategory.SAFETY,
            message="unsafe contact",
            contributing=(FailureCategory.SAFETY,),
        )


def test_validation_status_keeps_protocol_execution_and_science_orthogonal() -> None:
    with pytest.raises(ValueError, match="metric/gate evidence receipt"):
        ValidationStatus(
            experiment_id="external-no-receipt",
            stage=ValidationStage.GEOMETRY,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.COMPLETE,
            scientific=ScientificStatus.PASS,
            evidence_level=EvidenceLevel.EXTERNAL_CONFIRMATION,
            expected_cells=1,
            successful_cells=1,
        )

    with pytest.raises(ValueError, match="successful evaluated cell"):
        ValidationStatus(
            experiment_id="external-no-evaluation",
            stage=ValidationStage.GEOMETRY,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.COMPLETE,
            scientific=ScientificStatus.PASS,
            evidence_level=EvidenceLevel.EXTERNAL_CONFIRMATION,
            expected_cells=1,
            skipped_cells=1,
            legal_skip_reasons=("Prespecified legal skip.",),
            metric_gate_receipt_sha256="b" * 64,
        )

    status = ValidationStatus(
        experiment_id="phase3-formal",
        stage=ValidationStage.OBJECT_GROUNDING,
        protocol=ProtocolStatus.FROZEN,
        execution=ExecutionStatus.COMPLETE,
        scientific=ScientificStatus.FAIL,
        evidence_level=EvidenceLevel.DEVELOPMENT,
        expected_cells=4,
        successful_cells=3,
        expected_failure_cells=1,
        failure_event_ids=("failure-expected-1",),
    )
    assert status.matrix_complete is True
    assert status.terminal_cells == 4
    payload = status.to_serializable()
    assert payload["schema_version"] == VALIDATION_STATUS_SCHEMA_VERSION
    assert payload["protocol"] == "frozen"
    assert payload["execution"] == "complete"
    assert payload["scientific"] == "fail"


def test_validation_status_rejects_premature_scientific_decision() -> None:
    with pytest.raises(ValueError, match="complete execution"):
        ValidationStatus(
            experiment_id="phase4-running",
            stage=ValidationStage.MEMORY,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.RUNNING,
            scientific=ScientificStatus.PASS,
            evidence_level=EvidenceLevel.DEVELOPMENT,
            expected_cells=4,
            successful_cells=2,
        )


def test_validation_status_requires_complete_accounting_and_legal_skip_reasons() -> None:
    with pytest.raises(ValueError, match="every expected cell"):
        ValidationStatus(
            experiment_id="phase1-incomplete",
            stage=ValidationStage.REPRESENTATION,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.COMPLETE,
            scientific=ScientificStatus.NOT_EVALUATED,
            evidence_level=EvidenceLevel.INTEGRATION_ONLY,
            expected_cells=3,
            successful_cells=2,
        )
    with pytest.raises(ValueError, match="legal skip reason"):
        ValidationStatus(
            experiment_id="phase5-skip",
            stage=ValidationStage.PLANNING,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.COMPLETE,
            scientific=ScientificStatus.INCONCLUSIVE,
            evidence_level=EvidenceLevel.DEVELOPMENT,
            expected_cells=2,
            successful_cells=1,
            skipped_cells=1,
        )


def test_failed_execution_requires_unexpected_failure_and_event() -> None:
    with pytest.raises(ValueError, match="unexpected failed cell"):
        ValidationStatus(
            experiment_id="phase6-failed",
            stage=ValidationStage.SYSTEM,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.FAILED,
            scientific=ScientificStatus.NOT_EVALUATED,
            evidence_level=EvidenceLevel.INTEGRATION_ONLY,
            expected_cells=2,
            successful_cells=1,
        )
    with pytest.raises(ValueError, match="failure event ID"):
        ValidationStatus(
            experiment_id="phase6-failed",
            stage=ValidationStage.SYSTEM,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.FAILED,
            scientific=ScientificStatus.NOT_EVALUATED,
            evidence_level=EvidenceLevel.INTEGRATION_ONLY,
            expected_cells=2,
            successful_cells=1,
            failed_cells=1,
        )


def test_external_confirmation_requires_completed_terminal_scientific_evidence() -> None:
    with pytest.raises(ValueError, match="complete execution"):
        ValidationStatus(
            experiment_id="external-not-run",
            stage=ValidationStage.GEOMETRY,
            protocol=ProtocolStatus.FROZEN,
            execution=ExecutionStatus.NOT_STARTED,
            scientific=ScientificStatus.NOT_EVALUATED,
            evidence_level=EvidenceLevel.EXTERNAL_CONFIRMATION,
            expected_cells=1,
        )

    status = ValidationStatus(
        experiment_id="external-complete",
        stage=ValidationStage.GEOMETRY,
        protocol=ProtocolStatus.FROZEN,
        execution=ExecutionStatus.COMPLETE,
        scientific=ScientificStatus.PASS,
        evidence_level=EvidenceLevel.EXTERNAL_CONFIRMATION,
        expected_cells=1,
        successful_cells=1,
        metric_gate_receipt_sha256="b" * 64,
    )
    assert status.evidence_level.value == "external-confirmation"
    assert status.metric_gate_receipt_sha256 == "b" * 64
