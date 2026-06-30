"""Stage-agnostic validation report contract and portable HTML dashboard.

The contract keeps scientific quality, resource diagnostics, dataset role, and
claim scope explicit.  Rendering and W&B payload generation are deliberately
offline: callers decide whether and when to send the returned payload to W&B.
"""

from __future__ import annotations

import fcntl
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from jepa4d.evaluation.failure_taxonomy import EvidenceLevel

SCHEMA_VERSION = "jepa4d-validation-dashboard-v1"
BUNDLE_SCHEMA_VERSION = "jepa4d-validation-dashboard-bundle-v1"
JSON_FILENAME = "validation_report.json"
HTML_FILENAME = "validation_dashboard.html"
RECEIPT_FILENAME = "validation_dashboard.receipt.json"
IMMUTABLE_DIRECTORY_PREFIX = "validation-dashboard-"


class DatasetRole(StrEnum):
    """Role a dataset is allowed to play in a scientific claim."""

    PRIMARY_DEVELOPMENT = "A1-primary-development"
    COMPLEMENTARY_DEVELOPMENT = "A2-complementary-development"
    EXTERNAL_TRANSFER = "B-external-transfer"
    STRESS_SAFETY = "C-stress-safety"
    CONSUMED_REGRESSION = "consumed-regression"
    CONTRACT_FIXTURE = "contract-fixture"


class DatasetIdentityKind(StrEnum):
    """What content identity is shown for one governed dataset split."""

    ID_MANIFEST = "id-manifest"
    SPLIT_GOVERNANCE = "split-governance"


class TargetAccess(StrEnum):
    """Whether labels/targets may have influenced development decisions."""

    DEVELOPMENT_OPEN = "development-open"
    OPAQUE = "opaque"
    OPENED_ONCE = "opened-once"
    CONSUMED = "consumed"
    NOT_APPLICABLE = "not-applicable"


class GateOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NO_SURVIVOR = "no-survivor"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"
    NOT_EVALUATED = "not-evaluated"
    NOT_APPLICABLE = "not-applicable"


class MetricDomain(StrEnum):
    QUALITY = "quality"
    RESOURCE = "resource"


class GateDomain(StrEnum):
    QUALITY = "quality"
    INTEGRITY = "integrity"
    RESOURCE = "resource"


class ResourcePolicy(StrEnum):
    """Whether resource numbers are descriptive or part of promotion."""

    DIAGNOSTIC_ONLY = "diagnostic-only"
    BINDING_GATE = "binding-gate"


class CompletenessStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class VisualizationDeclaration:
    panel_id: str
    kind: str
    purpose: str


STANDARD_VISUALIZATIONS: tuple[VisualizationDeclaration, ...] = (
    VisualizationDeclaration(
        "dataset-roles",
        "table",
        "Expose each dataset split, manifest identity, target access, and permitted scientific role.",
    ),
    VisualizationDeclaration(
        "evidence-gate-completeness",
        "status-cards-and-table",
        "Keep evidence strength, gate decision, and matrix accounting visible together.",
    ),
    VisualizationDeclaration(
        "quality-metrics",
        "metric-cards",
        "Show scientific quality outcomes without mixing in speed or memory measurements.",
    ),
    VisualizationDeclaration(
        "resource-diagnostics",
        "metric-cards",
        "Show latency, throughput, memory, and compute as a separately labeled diagnostic domain.",
    ),
    VisualizationDeclaration(
        "claim-boundary",
        "supported-prohibited-callout",
        "State what this evidence does and does not support before downstream interpretation.",
    ),
)


_METRIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _enum_member(value: object, enum_type: type[StrEnum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__} member")


@dataclass(frozen=True, slots=True)
class DatasetEvidence:
    dataset_id: str
    version: str
    split: str
    role: DatasetRole
    target_access: TargetAccess
    identity_kind: DatasetIdentityKind
    identity_sha256: str
    id_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_id", "version", "split"):
            _nonempty(getattr(self, name), name)
        _enum_member(self.role, DatasetRole, "role")
        _enum_member(self.target_access, TargetAccess, "target_access")
        _enum_member(self.identity_kind, DatasetIdentityKind, "identity_kind")
        if not _SHA256.fullmatch(self.identity_sha256):
            raise ValueError("identity_sha256 must be a lowercase 64-character SHA-256 digest")
        if self.id_manifest_sha256 is not None and not _SHA256.fullmatch(self.id_manifest_sha256):
            raise ValueError("id_manifest_sha256 must be a lowercase 64-character SHA-256 digest")
        if self.identity_kind is DatasetIdentityKind.ID_MANIFEST:
            if self.id_manifest_sha256 is None or self.identity_sha256 != self.id_manifest_sha256:
                raise ValueError("id-manifest identity must equal the registered id_manifest_sha256")
        elif self.id_manifest_sha256 is not None:
            raise ValueError("split-governance fallback cannot claim a registered id_manifest_sha256")
        if self.role == DatasetRole.EXTERNAL_TRANSFER and self.target_access == TargetAccess.DEVELOPMENT_OPEN:
            raise ValueError("an external-transfer dataset cannot have development-open targets")


@dataclass(frozen=True, slots=True)
class Completeness:
    expected_cells: int
    succeeded_cells: int
    expected_failure_cells: int = 0
    failed_cells: int = 0
    legal_skips: int = 0

    def __post_init__(self) -> None:
        values = (
            self.expected_cells,
            self.succeeded_cells,
            self.expected_failure_cells,
            self.failed_cells,
            self.legal_skips,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("completeness counts must be integers")
        if self.expected_cells <= 0:
            raise ValueError("expected_cells must be positive")
        if min(values[1:]) < 0:
            raise ValueError("completeness counts cannot be negative")
        if self.accounted_cells > self.expected_cells:
            raise ValueError("terminal cell counts exceed expected_cells")

    @property
    def accounted_cells(self) -> int:
        return self.succeeded_cells + self.expected_failure_cells + self.failed_cells + self.legal_skips

    @property
    def missing_cells(self) -> int:
        return self.expected_cells - self.accounted_cells

    @property
    def accounted_fraction(self) -> float:
        return self.accounted_cells / self.expected_cells

    @property
    def status(self) -> CompletenessStatus:
        if self.accounted_cells == self.expected_cells:
            return CompletenessStatus.COMPLETE
        if self.accounted_cells:
            return CompletenessStatus.PARTIAL
        return CompletenessStatus.EMPTY


@dataclass(frozen=True, slots=True)
class MetricRecord:
    name: str
    value: float
    unit: str
    domain: MetricDomain
    split: str

    def __post_init__(self) -> None:
        if not _METRIC_NAME.fullmatch(self.name):
            raise ValueError(
                "metric name must start with an alphanumeric character and contain only letters, numbers, '.', '-', '_'"
            )
        for name in ("unit", "split"):
            _nonempty(getattr(self, name), name)
        _enum_member(self.domain, MetricDomain, "domain")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("metric value must be numeric")
        if not math.isfinite(float(self.value)):
            raise ValueError("metric value must be finite")


@dataclass(frozen=True, slots=True)
class GateCondition:
    name: str
    domain: GateDomain
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        if not _METRIC_NAME.fullmatch(self.name):
            raise ValueError("gate condition name must be a path-safe metric name")
        _nonempty(self.detail, "detail")
        _enum_member(self.domain, GateDomain, "domain")
        if not isinstance(self.passed, bool):
            raise TypeError("gate condition passed must be Boolean")


@dataclass(frozen=True, slots=True)
class GateDecision:
    name: str
    outcome: GateOutcome
    decision: str
    conditions: tuple[GateCondition, ...] = ()

    def __post_init__(self) -> None:
        for name in ("name", "decision"):
            _nonempty(getattr(self, name), name)
        _enum_member(self.outcome, GateOutcome, "outcome")
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("gate condition names must be unique")
        if self.outcome == GateOutcome.PASS and (not self.conditions or not all(c.passed for c in self.conditions)):
            raise ValueError("a passing gate requires at least one condition and all conditions must pass")
        if self.outcome in {GateOutcome.FAIL, GateOutcome.NO_SURVIVOR} and (
            not self.conditions or all(c.passed for c in self.conditions)
        ):
            raise ValueError("a failed/no-survivor gate requires at least one failed condition")


@dataclass(frozen=True, slots=True)
class ClaimBoundary:
    supported: tuple[str, ...]
    prohibited: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.supported or not self.prohibited:
            raise ValueError("claim boundary requires at least one supported and one prohibited claim")
        for value in (*self.supported, *self.prohibited):
            _nonempty(value, "claim boundary entry")


@dataclass(frozen=True, slots=True)
class GovernanceBinding:
    """Content identities used to derive a governed validation report."""

    registry_sha256: str
    base_ledger_sha256: str
    effective_ledger_sha256: str
    validation_status_sha256: str
    metric_gate_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "registry_sha256",
            "base_ledger_sha256",
            "effective_ledger_sha256",
            "validation_status_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.metric_gate_receipt_sha256 is not None and not _SHA256.fullmatch(self.metric_gate_receipt_sha256):
            raise ValueError("metric_gate_receipt_sha256 must be a lowercase SHA-256 digest")


def wandb_summary_from_serializable(report: Mapping[str, Any]) -> dict[str, str | int | float]:
    """Rebuild the exact aggregate W&B summary from a serialized report."""

    datasets = report.get("datasets")
    gate = report.get("gate")
    completeness = report.get("completeness")
    claim_boundary = report.get("claim_boundary")
    visualizations = report.get("visualizations")
    metrics = report.get("metrics")
    if (
        not isinstance(datasets, (list, tuple))
        or not isinstance(gate, Mapping)
        or not isinstance(completeness, Mapping)
        or not isinstance(claim_boundary, Mapping)
        or not isinstance(visualizations, (list, tuple))
        or not isinstance(metrics, (list, tuple))
    ):
        raise ValueError("serialized validation report is incomplete for W&B summary reconstruction")
    conditions = gate.get("conditions")
    if not isinstance(conditions, (list, tuple)):
        raise ValueError("serialized validation report lacks gate conditions")
    governance = report.get("governance")
    if governance is not None and not isinstance(governance, Mapping):
        raise ValueError("serialized validation report has invalid governance")

    payload: dict[str, str | int | float] = {
        "validation/schema_version": str(report["schema_version"]),
        "validation/report_id": str(report["report_id"]),
        "validation/stage": str(report["stage"]),
        "validation/evidence_level": str(report["evidence_level"]),
        "validation/gate/name": str(gate["name"]),
        "validation/gate/outcome": str(gate["outcome"]),
        "validation/completeness/status": str(completeness["status"]),
        "validation/completeness/accounted_fraction": float(completeness["accounted_fraction"]),
        "validation/completeness/missing_cells": int(completeness["missing_cells"]),
        "validation/resource_policy": str(report["resource_policy"]),
        "validation/datasets/roles_json": json.dumps(
            {str(dataset["dataset_id"]): str(dataset["role"]) for dataset in datasets}, sort_keys=True
        ),
        "validation/datasets/identities_json": json.dumps(
            {
                f"{dataset['dataset_id']}/{dataset['split']}": {
                    "kind": dataset["identity_kind"],
                    "sha256": dataset["identity_sha256"],
                    "id_manifest_sha256": dataset["id_manifest_sha256"],
                }
                for dataset in datasets
            },
            sort_keys=True,
        ),
        "validation/claim/supported_json": json.dumps(claim_boundary["supported"]),
        "validation/claim/prohibited_json": json.dumps(claim_boundary["prohibited"]),
        "validation/dashboard/panels_json": json.dumps([declaration["panel_id"] for declaration in visualizations]),
    }
    if governance is not None:
        payload.update(
            {
                "validation/governance/registry_sha256": str(governance["registry_sha256"]),
                "validation/governance/base_ledger_sha256": str(governance["base_ledger_sha256"]),
                "validation/governance/effective_ledger_sha256": str(governance["effective_ledger_sha256"]),
                "validation/governance/status_sha256": str(governance["validation_status_sha256"]),
            }
        )
        if governance.get("metric_gate_receipt_sha256") is not None:
            payload["validation/governance/metric_gate_receipt_sha256"] = str(governance["metric_gate_receipt_sha256"])
    for condition in conditions:
        payload[f"validation/gate/{condition['domain']}/{condition['name']}"] = int(condition["passed"])
    for metric in metrics:
        payload[f"validation/{metric['domain']}/{metric['name']}/{metric['split']}"] = float(metric["value"])
    return payload


@dataclass(frozen=True, slots=True)
class ValidationReport:
    report_id: str
    title: str
    stage: str
    evidence_level: EvidenceLevel
    datasets: tuple[DatasetEvidence, ...]
    gate: GateDecision
    completeness: Completeness
    claim_boundary: ClaimBoundary
    metrics: tuple[MetricRecord, ...] = ()
    resource_policy: ResourcePolicy = ResourcePolicy.DIAGNOSTIC_ONLY
    visualizations: tuple[VisualizationDeclaration, ...] = field(default_factory=lambda: STANDARD_VISUALIZATIONS)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    wandb_url: str | None = None
    governance: GovernanceBinding | None = None

    def __post_init__(self) -> None:
        for name in ("report_id", "title", "stage", "timestamp"):
            _nonempty(getattr(self, name), name)
        _enum_member(self.evidence_level, EvidenceLevel, "evidence_level")
        _enum_member(self.resource_policy, ResourcePolicy, "resource_policy")
        if self.wandb_url is not None and not self.wandb_url.startswith(("https://", "http://")):
            raise ValueError("wandb_url must use http or https")
        if not self.datasets:
            raise ValueError("at least one dataset evidence record is required")
        dataset_keys = [(dataset.dataset_id, dataset.version, dataset.split) for dataset in self.datasets]
        if len(dataset_keys) != len(set(dataset_keys)):
            raise ValueError("dataset/version/split records must be unique")
        metric_keys = [(metric.domain, metric.name, metric.split) for metric in self.metrics]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("metric domain/name/split records must be unique")
        if self.visualizations != STANDARD_VISUALIZATIONS:
            raise ValueError("visualization declarations must exactly match STANDARD_VISUALIZATIONS")
        if self.resource_policy == ResourcePolicy.DIAGNOSTIC_ONLY and any(
            condition.domain == GateDomain.RESOURCE for condition in self.gate.conditions
        ):
            raise ValueError("resource conditions cannot bind a gate under the diagnostic-only resource policy")
        if (
            self.gate.outcome in {GateOutcome.PASS, GateOutcome.FAIL, GateOutcome.NO_SURVIVOR}
            and self.completeness.status != CompletenessStatus.COMPLETE
        ):
            raise ValueError("a final gate outcome requires a fully accounted experiment matrix")
        if self.gate.outcome == GateOutcome.PASS and self.completeness.failed_cells:
            raise ValueError("a passing gate requires zero failed experiment cells")

    def to_serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        payload["completeness"].update(
            {
                "accounted_cells": self.completeness.accounted_cells,
                "missing_cells": self.completeness.missing_cells,
                "accounted_fraction": self.completeness.accounted_fraction,
                "status": self.completeness.status,
            }
        )
        return payload

    def wandb_summary_payload(self) -> dict[str, str | int | float]:
        """Return stable summary keys without importing or contacting W&B."""
        return wandb_summary_from_serializable(self.to_serializable())


@dataclass(frozen=True, slots=True)
class ImmutableDashboardBundle:
    """Paths and content identity for one no-clobber dashboard generation."""

    directory: Path
    json_path: Path
    html_path: Path
    receipt_path: Path
    generation_id: str


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _metric_cards(metrics: list[MetricRecord], empty_message: str) -> str:
    if not metrics:
        return f"<p class='empty'>{_escape(empty_message)}</p>"
    return (
        "<div class='metric-grid'>"
        + "".join(
            "<article class='metric-card'>"
            f"<div class='metric-name'>{_escape(metric.name)}</div>"
            f"<div class='metric-value'>{_escape(f'{float(metric.value):.6g}')} "
            f"<span>{_escape(metric.unit)}</span></div>"
            f"<div class='metric-split'>split: {_escape(metric.split)}</div>"
            "</article>"
            for metric in metrics
        )
        + "</div>"
    )


def render_validation_dashboard(report: ValidationReport) -> str:
    """Render a dependency-free, self-contained dashboard."""
    dataset_rows = "".join(
        "<tr>"
        f"<td>{_escape(dataset.dataset_id)}</td><td>{_escape(dataset.version)}</td>"
        f"<td>{_escape(dataset.split)}</td><td><code>{_escape(dataset.role)}</code></td>"
        f"<td>{_escape(dataset.target_access)}</td><td>{_escape(dataset.identity_kind)}</td>"
        f"<td><code>{_escape(dataset.identity_sha256[:12])}&hellip;</code></td>"
        f"<td>{'<code>' + _escape(dataset.id_manifest_sha256[:12]) + '&hellip;</code>' if dataset.id_manifest_sha256 else 'not registered'}</td>"
        "</tr>"
        for dataset in report.datasets
    )
    condition_rows = (
        "".join(
            "<tr>"
            f"<td>{_escape(condition.name)}</td><td>{_escape(condition.domain)}</td>"
            f"<td><span class='badge {'pass' if condition.passed else 'fail'}'>"
            f"{'PASS' if condition.passed else 'FAIL'}</span></td><td>{_escape(condition.detail)}</td>"
            "</tr>"
            for condition in report.gate.conditions
        )
        or "<tr><td colspan='4' class='empty'>No gate conditions evaluated.</td></tr>"
    )
    quality = [metric for metric in report.metrics if metric.domain == MetricDomain.QUALITY]
    resources = [metric for metric in report.metrics if metric.domain == MetricDomain.RESOURCE]
    accounted_percent = 100.0 * report.completeness.accounted_fraction
    supported = "".join(f"<li>{_escape(value)}</li>" for value in report.claim_boundary.supported)
    prohibited = "".join(f"<li>{_escape(value)}</li>" for value in report.claim_boundary.prohibited)
    gate_class = (
        "pass"
        if report.gate.outcome == GateOutcome.PASS
        else "fail"
        if report.gate.outcome in {GateOutcome.FAIL, GateOutcome.NO_SURVIVOR}
        else "warn"
    )
    resource_interpretation = (
        "These measurements are descriptive and cannot determine promotion."
        if report.resource_policy == ResourcePolicy.DIAGNOSTIC_ONLY
        else "A preregistered resource constraint is explicitly part of this gate."
    )
    wandb_link = (
        ""
        if report.wandb_url is None
        else f" · <a href='{_escape(report.wandb_url)}' rel='noreferrer'>online W&amp;B run</a>"
    )
    governance_html = (
        "<p class='policy'><b>Governance-bound report.</b> "
        f"Registry <code>{_escape(report.governance.registry_sha256[:12])}&hellip;</code>; "
        f"effective ledger <code>{_escape(report.governance.effective_ledger_sha256[:12])}&hellip;</code>; "
        f"status <code>{_escape(report.governance.validation_status_sha256[:12])}&hellip;</code>"
        f"{'; metric/gate receipt <code>' + _escape(report.governance.metric_gate_receipt_sha256[:12]) + '&hellip;</code>' if report.governance.metric_gate_receipt_sha256 else ''}.</p>"
        if report.governance is not None
        else "<p class='policy'><b>Unbound report.</b> No registry/ledger/status derivation is asserted.</p>"
    )
    document = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{_escape(report.title)}</title><style>
:root{{--ink:#182230;--muted:#5f6b7a;--line:#d8e0e8;--paper:#fff;--canvas:#f4f7fa;--blue:#175cd3;
--green:#067647;--green-bg:#ecfdf3;--red:#b42318;--red-bg:#fef3f2;--amber:#b54708;--amber-bg:#fffaeb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 20px 64px}}h1{{margin:.2rem 0;font-size:2rem}}h2{{margin:0 0 1rem}}
.eyebrow{{color:var(--blue);font-weight:750;text-transform:uppercase;letter-spacing:.08em}}.subtitle{{color:var(--muted)}}
.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin-top:18px;padding:22px;box-shadow:0 3px 12px #1822300d}}
.status-grid,.metric-grid,.claim-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.status-card,.metric-card,.claim{{border:1px solid var(--line);border-radius:10px;padding:14px;background:#fbfcfd}}
.status-label,.metric-name{{color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}}
.status-value{{font-size:1.08rem;font-weight:750;margin-top:4px}}.metric-value{{font-size:1.65rem;font-weight:800;margin:.2rem 0}}
.metric-value span,.metric-split{{font-size:.82rem;color:var(--muted);font-weight:500}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}}th{{background:#f8fafc;font-size:.82rem}}
code{{font-size:.85em}}.badge{{display:inline-block;border-radius:999px;padding:3px 9px;font-size:.78rem;font-weight:800}}
.badge.pass{{background:var(--green-bg);color:var(--green)}}.badge.fail{{background:var(--red-bg);color:var(--red)}}
.badge.warn{{background:var(--amber-bg);color:var(--amber)}}.progress{{height:10px;background:#e8edf2;border-radius:99px;overflow:hidden;margin:10px 0}}
.progress>span{{display:block;height:100%;background:var(--blue);width:{accounted_percent:.6f}%}}.policy{{border-left:4px solid var(--blue);padding:8px 12px;background:#eff6ff}}
.claim.supported{{background:var(--green-bg);border-color:#abefc6}}.claim.prohibited{{background:var(--red-bg);border-color:#fecdca}}
.claim h3{{margin-top:0}}.empty{{color:var(--muted);font-style:italic}}.footer{{margin-top:20px;color:var(--muted);font-size:.82rem}}
@media(max-width:720px){{.panel{{padding:16px;overflow-x:auto}}h1{{font-size:1.6rem}}}}
</style></head><body><main>
<div class='eyebrow'>JEPA-4D systematic validation</div><h1>{_escape(report.title)}</h1>
<p class='subtitle'>Report <code>{_escape(report.report_id)}</code> · stage <b>{_escape(report.stage)}</b> · {_escape(report.timestamp)}</p>
<section class='panel' data-panel-id='dataset-roles'><h2>Dataset role ledger</h2>
<table><thead><tr><th>Dataset</th><th>Version</th><th>Split</th><th>Role</th><th>Target access</th><th>Identity kind</th><th>Identity SHA-256</th><th>ID-manifest SHA-256</th></tr></thead>
<tbody>{dataset_rows}</tbody></table></section>
<section class='panel' data-panel-id='evidence-gate-completeness'><h2>Evidence, gate, and completeness</h2>
<div class='status-grid'><article class='status-card'><div class='status-label'>Evidence level</div><div class='status-value'>{_escape(report.evidence_level)}</div></article>
<article class='status-card'><div class='status-label'>Gate outcome</div><div class='status-value'><span class='badge {gate_class}'>{_escape(report.gate.outcome).upper()}</span></div></article>
<article class='status-card'><div class='status-label'>Completeness</div><div class='status-value'>{_escape(report.completeness.status)}</div></article>
<article class='status-card'><div class='status-label'>Resource policy</div><div class='status-value'>{_escape(report.resource_policy)}</div></article></div>
{governance_html}
<div class='progress' title='{accounted_percent:.1f}% accounted'><span></span></div>
<p>{report.completeness.accounted_cells}/{report.completeness.expected_cells} cells accounted: {report.completeness.succeeded_cells} succeeded, {report.completeness.expected_failure_cells} expected failures, {report.completeness.failed_cells} unexpected failures, {report.completeness.legal_skips} legal skips, {report.completeness.missing_cells} missing.</p>
<p><b>{_escape(report.gate.name)}:</b> {_escape(report.gate.decision)}</p>
<table><thead><tr><th>Condition</th><th>Domain</th><th>Outcome</th><th>Interpretation</th></tr></thead><tbody>{condition_rows}</tbody></table></section>
<section class='panel' data-panel-id='quality-metrics'><h2>Scientific quality</h2>
{_metric_cards(quality, "No quality metric is reported at this evidence level.")}</section>
<section class='panel' data-panel-id='resource-diagnostics'><h2>Resource diagnostics</h2>
<p class='policy'><b>{_escape(report.resource_policy)}</b>: {_escape(resource_interpretation)} Resource measurements are rendered separately from scientific quality.</p>
{_metric_cards(resources, "No resource diagnostic is reported.")}</section>
<section class='panel' data-panel-id='claim-boundary'><h2>Claim boundary</h2><div class='claim-grid'>
<article class='claim supported'><h3>Supported</h3><ul>{supported}</ul></article>
<article class='claim prohibited'><h3>Not supported</h3><ul>{prohibited}</ul></article></div></section>
<p class='footer'>Schema <code>{SCHEMA_VERSION}</code> · portable report with no external scripts, fonts, or network calls{wandb_link}.</p>
</main></body></html>"""
    for declaration in STANDARD_VISUALIZATIONS:
        marker = f"data-panel-id='{declaration.panel_id}'"
        if document.count(marker) != 1:
            raise RuntimeError(f"rendered dashboard must contain exactly one declared panel {declaration.panel_id!r}")
    return document


def write_validation_dashboard(report: ValidationReport, output: str | Path) -> tuple[Path, Path]:
    """Publish a concurrency-safe JSON/HTML bundle with an atomic commit receipt."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_FILENAME
    html_path = directory / HTML_FILENAME
    receipt_path = directory / RECEIPT_FILENAME
    documents = {
        JSON_FILENAME: (json.dumps(report.to_serializable(), indent=2, sort_keys=True) + "\n").encode("utf-8"),
        HTML_FILENAME: render_validation_dashboard(report).encode("utf-8"),
    }
    files = {
        name: {"sha256": hashlib.sha256(document).hexdigest(), "bytes": len(document)}
        for name, document in documents.items()
    }
    generation_basis = {"report_id": report.report_id, "files": files}
    generation_id = hashlib.sha256(
        json.dumps(generation_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generation_id": generation_id,
        **generation_basis,
    }
    receipt_document = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")

    temporary_paths: dict[str, Path] = {}
    try:
        for name, document in (*documents.items(), (RECEIPT_FILENAME, receipt_document)):
            descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
            temporary_path = Path(temporary)
            temporary_paths[name] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())

        lock_path = directory / ".validation-dashboard.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            os.replace(temporary_paths[JSON_FILENAME], json_path)
            os.replace(temporary_paths[HTML_FILENAME], html_path)
            # The receipt is the commit marker and is always published last.
            os.replace(temporary_paths[RECEIPT_FILENAME], receipt_path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
    return json_path, html_path


def verify_validation_dashboard(output: str | Path) -> dict[str, Any]:
    """Verify the atomic receipt and both files of a published dashboard bundle."""

    directory = Path(output)
    receipt_path = directory / RECEIPT_FILENAME
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid validation dashboard receipt: {error}") from error
    if not isinstance(receipt, dict):
        raise ValueError("validation dashboard receipt must be a JSON object")
    if receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unexpected validation dashboard bundle schema")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != {JSON_FILENAME, HTML_FILENAME}:
        raise ValueError("validation dashboard receipt must bind the canonical JSON and HTML files")
    for name in (JSON_FILENAME, HTML_FILENAME):
        path = directory / name
        try:
            document = path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing validation dashboard artifact {name}: {error}") from error
        expected = files[name]
        if not isinstance(expected, dict):
            raise ValueError(f"invalid receipt entry for {name}")
        if expected.get("bytes") != len(document) or expected.get("sha256") != hashlib.sha256(document).hexdigest():
            raise ValueError(f"validation dashboard artifact does not match receipt: {name}")
    generation_basis = {"report_id": receipt.get("report_id"), "files": files}
    expected_generation = hashlib.sha256(
        json.dumps(generation_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if receipt.get("generation_id") != expected_generation:
        raise ValueError("validation dashboard generation ID does not bind the receipt contents")
    return receipt


def verify_immutable_validation_dashboard(output: str | Path) -> dict[str, Any]:
    """Verify a content-addressed dashboard directory and its canonical bundle."""

    directory = Path(output)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("immutable validation dashboard must be a real directory")
    receipt = verify_validation_dashboard(directory)
    expected_name = f"{IMMUTABLE_DIRECTORY_PREFIX}{receipt['generation_id']}"
    if directory.name != expected_name:
        raise ValueError("immutable validation dashboard directory does not match its generation ID")
    actual_names = {path.name for path in directory.iterdir()}
    expected_names = {JSON_FILENAME, HTML_FILENAME, RECEIPT_FILENAME}
    if actual_names != expected_names:
        raise ValueError("immutable validation dashboard directory contains unbound files")
    return receipt


def write_immutable_validation_dashboard(
    report: ValidationReport,
    output_root: str | Path,
) -> ImmutableDashboardBundle:
    """Publish one content-addressed dashboard generation without replacing files."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("immutable dashboard output root must be a real directory")
    temporary = Path(tempfile.mkdtemp(prefix=".validation-dashboard-", dir=root))
    target: Path | None = None
    try:
        write_validation_dashboard(report, temporary)
        (temporary / ".validation-dashboard.lock").unlink(missing_ok=True)
        receipt = verify_validation_dashboard(temporary)
        generation_id = str(receipt["generation_id"])
        target = root / f"{IMMUTABLE_DIRECTORY_PREFIX}{generation_id}"
        try:
            os.rename(temporary, target)
        except OSError:
            if not target.exists():
                raise
            verify_immutable_validation_dashboard(target)
        else:
            root_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
        verified = verify_immutable_validation_dashboard(target)
        if verified["generation_id"] != generation_id:
            raise RuntimeError("published dashboard generation changed during commit")
        return ImmutableDashboardBundle(
            directory=target,
            json_path=target / JSON_FILENAME,
            html_path=target / HTML_FILENAME,
            receipt_path=target / RECEIPT_FILENAME,
            generation_id=generation_id,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
