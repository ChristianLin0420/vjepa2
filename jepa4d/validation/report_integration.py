"""Derive validation dashboards from registry, ledger, and status authorities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from jepa4d.evaluation.failure_taxonomy import (
    EvidenceLevel,
    ScientificStatus,
    ValidationStage,
    ValidationStatus,
)
from jepa4d.validation._content import sha256_value
from jepa4d.validation.ledger import (
    ConsumedTestLedger,
    ConsumptionEvent,
    LedgerState,
    effective_targets,
)
from jepa4d.validation.registry import TARGET_SPLITS, AccessOperation, DataRole, DatasetRegistry, TargetState
from jepa4d.visualization.validation_dashboard import (
    ClaimBoundary,
    Completeness,
    DatasetEvidence,
    DatasetIdentityKind,
    DatasetRole,
    GateCondition,
    GateDecision,
    GateDomain,
    GateOutcome,
    GovernanceBinding,
    MetricRecord,
    ResourcePolicy,
    TargetAccess,
    ValidationReport,
)

REGISTRY_DASHBOARD_ROLE: Mapping[DataRole, DatasetRole] = MappingProxyType(
    {
        DataRole.A1: DatasetRole.PRIMARY_DEVELOPMENT,
        DataRole.A2: DatasetRole.COMPLEMENTARY_DEVELOPMENT,
        DataRole.B: DatasetRole.EXTERNAL_TRANSFER,
        DataRole.C: DatasetRole.STRESS_SAFETY,
        DataRole.CONTRACT_ONLY: DatasetRole.CONTRACT_FIXTURE,
    }
)

_STAGE_PHASE: dict[ValidationStage, str] = {
    ValidationStage.INFRASTRUCTURE: "phase0",
    ValidationStage.REPRESENTATION: "phase1",
    ValidationStage.GEOMETRY: "phase2",
    ValidationStage.OBJECT_GROUNDING: "phase3",
    ValidationStage.IDENTITY_TRACKING: "phase3",
    ValidationStage.MEMORY: "phase4",
    ValidationStage.DYNAMICS: "phase5",
    ValidationStage.PLANNING: "phase5",
    ValidationStage.SYSTEM: "phase6",
}

_SCIENTIFIC_GATE_OUTCOME: dict[ScientificStatus, GateOutcome] = {
    ScientificStatus.PASS: GateOutcome.PASS,
    ScientificStatus.FAIL: GateOutcome.FAIL,
    ScientificStatus.NO_SURVIVOR: GateOutcome.NO_SURVIVOR,
    ScientificStatus.INCONCLUSIVE: GateOutcome.INCONCLUSIVE,
    ScientificStatus.NOT_EVALUATED: GateOutcome.NOT_EVALUATED,
    ScientificStatus.NOT_APPLICABLE: GateOutcome.NOT_APPLICABLE,
}


@dataclass(frozen=True, slots=True)
class ReportDatasetRef:
    dataset_id: str
    split_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(self.split_id, str) or not self.split_id.strip():
            raise ValueError("split_id must be a non-empty string")


def dashboard_role_from_registry(value: DataRole | str) -> DatasetRole:
    """Map the authoritative compact registry role to its display role."""

    try:
        role = DataRole(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown registry data role: {value!r}") from error
    return REGISTRY_DASHBOARD_ROLE[role]


def _target_access(
    *,
    ledger_state: LedgerState | None,
    target_state: TargetState,
) -> TargetAccess:
    if ledger_state is LedgerState.CONSUMED:
        return TargetAccess.CONSUMED
    if ledger_state is not None:
        return TargetAccess.OPAQUE
    if target_state is TargetState.NOT_APPLICABLE:
        return TargetAccess.NOT_APPLICABLE
    if target_state is TargetState.OPEN:
        return TargetAccess.DEVELOPMENT_OPEN
    return TargetAccess.OPAQUE


def _split_governance_identity(registry: DatasetRegistry, dataset_id: str, split_id: str) -> str:
    entry, split = registry.split(dataset_id, split_id)
    return sha256_value(
        {
            "registry_sha256": registry.sha256,
            "dataset_id": dataset_id,
            "dataset_version": entry.source.version,
            "split": split.model_dump(mode="python", exclude_none=False),
        }
    )


def _status_condition(status: ValidationStatus, outcome: GateOutcome) -> GateCondition | None:
    if outcome == GateOutcome.PASS:
        return GateCondition(
            "governed_scientific_status",
            GateDomain.QUALITY,
            True,
            "The frozen ValidationStatus records scientific=pass.",
        )
    if outcome in {GateOutcome.FAIL, GateOutcome.NO_SURVIVOR}:
        return GateCondition(
            "governed_scientific_status",
            GateDomain.QUALITY,
            False,
            f"The frozen ValidationStatus records scientific={status.scientific.value}.",
        )
    return None


def build_governed_validation_report(
    *,
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    events: Sequence[ConsumptionEvent],
    status: ValidationStatus,
    dataset_splits: Sequence[ReportDatasetRef],
    report_id: str,
    title: str,
    gate_name: str,
    gate_decision: str,
    claim_boundary: ClaimBoundary,
    metrics: Sequence[MetricRecord] = (),
    gate_conditions: Sequence[GateCondition] = (),
    resource_policy: ResourcePolicy = ResourcePolicy.DIAGNOSTIC_ONLY,
    timestamp: str | None = None,
    wandb_url: str | None = None,
) -> ValidationReport:
    """Build a report whose scientific/governance fields cannot be caller-overridden."""

    if not isinstance(status, ValidationStatus):
        raise TypeError("status must be a ValidationStatus")
    if not dataset_splits:
        raise ValueError("at least one governed dataset split is required")
    if any(not isinstance(reference, ReportDatasetRef) for reference in dataset_splits):
        raise TypeError("dataset_splits must contain ReportDatasetRef values")
    references = [(reference.dataset_id, reference.split_id) for reference in dataset_splits]
    if len(references) != len(set(references)):
        raise ValueError("governed dataset split references must be unique")
    event_values = tuple(events)
    if any(not isinstance(event, ConsumptionEvent) for event in event_values):
        raise TypeError("events must contain ConsumptionEvent values")

    ledger.validate_against(registry)
    effective = effective_targets(registry, ledger, event_values)
    expected_phase = _STAGE_PHASE.get(status.stage)
    evidence: list[DatasetEvidence] = []
    selected_b_keys: set[tuple[str, str]] = set()
    terminal_scientific = status.scientific not in {
        ScientificStatus.NOT_EVALUATED,
        ScientificStatus.NOT_APPLICABLE,
    }
    for reference in dataset_splits:
        entry, split = registry.split(reference.dataset_id, reference.split_id)
        if expected_phase is not None and expected_phase not in entry.stages:
            raise ValueError(f"dataset {entry.dataset_id!r} is not registered for status stage {status.stage.value!r}")
        key = (reference.dataset_id, reference.split_id)
        ledger_target = effective.get(key)
        if (
            split.purpose in TARGET_SPLITS
            and terminal_scientific
            and (ledger_target is None or ledger_target.state is not LedgerState.CONSUMED)
        ):
            raise ValueError(
                f"terminal scientific report requires consumed target state for {entry.dataset_id}/{split.split_id}"
            )
        role = dashboard_role_from_registry(entry.role)
        if entry.role is DataRole.B:
            selected_b_keys.add(key)
        evidence.append(
            DatasetEvidence(
                dataset_id=entry.dataset_id,
                version=entry.source.version,
                split=split.split_id,
                role=role,
                target_access=_target_access(
                    ledger_state=None if ledger_target is None else ledger_target.state,
                    target_state=split.target_state,
                ),
                identity_kind=(
                    DatasetIdentityKind.ID_MANIFEST
                    if split.id_manifest_sha256 is not None
                    else DatasetIdentityKind.SPLIT_GOVERNANCE
                ),
                identity_sha256=(
                    split.id_manifest_sha256
                    if split.id_manifest_sha256 is not None
                    else _split_governance_identity(registry, entry.dataset_id, split.split_id)
                ),
                id_manifest_sha256=split.id_manifest_sha256,
            )
        )

    if status.evidence_level is EvidenceLevel.EXTERNAL_CONFIRMATION:
        external_event_keys = {
            (event.dataset_id, event.split_id)
            for event in event_values
            if event.open_operation is AccessOperation.EXTERNAL_EVALUATION
        }
        consumed_b_with_event = {
            key
            for key in selected_b_keys & external_event_keys
            if effective.get(key) is not None and effective[key].state is LedgerState.CONSUMED
        }
        if not consumed_b_with_event:
            raise ValueError(
                "external-confirmation report requires a selected, consumed B-role target with a first-open event"
            )

    outcome = _SCIENTIFIC_GATE_OUTCOME[status.scientific]
    derived_condition = _status_condition(status, outcome)
    conditions = tuple(gate_conditions)
    if derived_condition is not None:
        conditions = (derived_condition, *conditions)
    gate = GateDecision(gate_name, outcome, gate_decision, conditions)
    completeness = Completeness(
        expected_cells=status.expected_cells,
        succeeded_cells=status.successful_cells,
        expected_failure_cells=status.expected_failure_cells,
        failed_cells=status.failed_cells,
        legal_skips=status.skipped_cells,
    )
    effective_ledger_sha256 = sha256_value(
        {
            "base_ledger_sha256": ledger.sha256,
            "events": sorted(sha256_value(event) for event in event_values),
        }
    )
    governance = GovernanceBinding(
        registry_sha256=registry.sha256,
        base_ledger_sha256=ledger.sha256,
        effective_ledger_sha256=effective_ledger_sha256,
        validation_status_sha256=sha256_value(status.to_serializable()),
        metric_gate_receipt_sha256=status.metric_gate_receipt_sha256,
    )
    optional: dict[str, object] = {}
    if timestamp is not None:
        optional["timestamp"] = timestamp
    return ValidationReport(
        report_id=report_id,
        title=title,
        stage=status.stage.value,
        evidence_level=status.evidence_level,
        datasets=tuple(evidence),
        gate=gate,
        completeness=completeness,
        claim_boundary=claim_boundary,
        metrics=tuple(metrics),
        resource_policy=resource_policy,
        wandb_url=wandb_url,
        governance=governance,
        **optional,  # type: ignore[arg-type]
    )
