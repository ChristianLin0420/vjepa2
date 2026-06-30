"""Versioned cross-stage failure and validation-status contracts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

FAILURE_TAXONOMY_SCHEMA_VERSION = "jepa4d.failure-taxonomy.v1"
VALIDATION_STATUS_SCHEMA_VERSION = "jepa4d.validation-status.v1"


class ValidationStage(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    REPRESENTATION = "representation"
    GEOMETRY = "geometry"
    OBJECT_GROUNDING = "object_grounding"
    IDENTITY_TRACKING = "identity_tracking"
    MEMORY = "memory"
    DYNAMICS = "dynamics"
    PLANNING = "planning"
    SYSTEM = "system"
    CROSS_STAGE = "cross_stage"


class FailureCategory(StrEnum):
    # Data governance and ingestion.
    DATA_ACCESS = "data_access"
    LICENSE_OR_TERMS = "license_or_terms"
    MANIFEST_MISMATCH = "manifest_mismatch"
    DATA_INTEGRITY = "data_integrity"
    SPLIT_LEAKAGE = "split_leakage"
    TARGET_ISOLATION = "target_isolation"
    INPUT_DECODE = "input_decode"
    SCHEMA_MISMATCH = "schema_mismatch"
    # Model and stage behavior.
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    REPRESENTATION = "representation"
    GEOMETRY_DEPTH = "geometry_depth"
    GEOMETRY_POSE = "geometry_pose"
    GEOMETRY_SCALE = "geometry_scale"
    CALIBRATION = "calibration"
    OBJECT_GROUNDING = "object_grounding"
    IDENTITY_TRACKING = "identity_tracking"
    MEMORY_INSERT = "memory_insert"
    MEMORY_RETRIEVAL = "memory_retrieval"
    STALE_BELIEF = "stale_belief"
    DYNAMICS = "dynamics"
    PLANNING_GROUNDING = "planning_grounding"
    VERIFICATION = "verification"
    CONTROL = "control"
    COLLISION = "collision"
    SAFETY = "safety"
    # Evaluation and operations.
    METRIC = "metric"
    PAIRING = "pairing"
    RESAMPLING_UNIT = "resampling_unit"
    NUMERICAL = "numerical"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    PREEMPTION = "preemption"
    NODE_OR_GPU = "node_or_gpu"
    NETWORK = "network"
    WANDB_UPLOAD = "wandb_upload"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    INCOMPLETE_MATRIX = "incomplete_matrix"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


class FailureSeverity(StrEnum):
    DIAGNOSTIC = "diagnostic"
    CELL_FAILURE = "cell_failure"
    EXPERIMENT_FAILURE = "experiment_failure"
    SAFETY_CRITICAL = "safety_critical"


class RetryDisposition(StrEnum):
    NOT_RETRYABLE = "not_retryable"
    SAME_ID_REQUEUE = "same_id_requeue"
    NEW_EXPERIMENT_REQUIRED = "new_experiment_required"


class ProtocolStatus(StrEnum):
    DRAFT = "draft"
    PREREGISTERED = "preregistered"
    FROZEN = "frozen"
    SUPERSEDED = "superseded"


class ExecutionStatus(StrEnum):
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScientificStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    PASS = "pass"
    FAIL = "fail"
    NO_SURVIVOR = "no_survivor"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


class EvidenceLevel(StrEnum):
    """Canonical evidence vocabulary shared by status records and reports."""

    CONTRACT_ONLY = "contract-only"
    INTEGRATION = "integration"
    OFFICIAL_SMOKE = "official-smoke"
    OFFICIAL_MINI_SUBSET = "official mini subset"
    SEQUENCE_LEVEL = "sequence-level"
    TRAINING = "training"
    BENCHMARK = "benchmark"
    DEVELOPMENT_BENCHMARK = "development-benchmark"
    MECHANISM_DIAGNOSTIC = "mechanism-diagnostic"
    CLOSED_LOOP = "closed-loop"
    EXTERNAL_CONFIRMATION = "external-confirmation"

    # Source-compatible names for early Wave-A callers. These are aliases, not
    # additional serialized vocabulary values.
    INTEGRATION_ONLY = "integration"
    DEVELOPMENT = "development-benchmark"
    HELD_OUT = "development-benchmark"

    @classmethod
    def _missing_(cls, value: object) -> EvidenceLevel | None:
        legacy = {
            "contract_only": cls.CONTRACT_ONLY,
            "integration_only": cls.INTEGRATION,
            "development": cls.DEVELOPMENT_BENCHMARK,
            "held_out": cls.DEVELOPMENT_BENCHMARK,
            "external_confirmation": cls.EXTERNAL_CONFIRMATION,
        }
        return legacy.get(value) if isinstance(value, str) else None


_SAFE_TAG_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SENSITIVE_TAG_TOKENS = frozenset(
    {"api", "apikey", "authorization", "cookie", "credential", "key", "password", "secret", "token"}
)
_TARGET_OR_PATH_TAG_TOKENS = frozenset(
    {
        "annotation",
        "answer",
        "bbox",
        "box",
        "category",
        "class",
        "depth",
        "file",
        "filename",
        "groundtruth",
        "gt",
        "label",
        "mask",
        "path",
        "pose",
        "reward",
        "sample",
        "target",
    }
)
_OBVIOUS_CREDENTIAL_PATTERNS = (
    re.compile(r"wandb_v1_[A-Za-z0-9_-]+"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|credential|password|secret|token)\b\s*[:=]\s*[^\s,;]+"),
)
_TARGET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(label|target|ground[_ -]?truth|gt|class|category|bbox|box|mask|depth|pose|reward|answer)\b"
    r"\s*[:=]\s*(?:\[[^\]\r\n]*\]|\{[^}\r\n]*\}|\([^\)\r\n]*\)|[^\s,;]+)"
)
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TARGET_CONTEXT_TOKENS = frozenset(
    {
        "annotation",
        "answer",
        "bbox",
        "box",
        "category",
        "class",
        "depth",
        "groundtruth",
        "gt",
        "label",
        "mask",
        "pose",
        "reward",
        "target",
    }
)
_PATH_PATTERNS = (
    re.compile(r"(?i)\b(?:https?|s3|file)://[^\s,;]+"),
    re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/)[^\s,;]+"),
    re.compile(r"(?<![A-Za-z0-9])(?:\.{0,2}/)?(?:[A-Za-z0-9_.-]+/){2,}[A-Za-z0-9_.-]+"),
)


def _tag_key_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[_.-]+", value.casefold()) if token)


def _redact_credentials(value: str) -> str:
    redacted = value
    for pattern in _OBVIOUS_CREDENTIAL_PATTERNS:
        redacted = pattern.sub("[REDACTED_CREDENTIAL]", redacted)
    return redacted


def _has_target_context(value: str) -> bool:
    normalized = _CAMEL_CASE_BOUNDARY.sub(" ", value)
    tokens = tuple(re.findall(r"[A-Za-z0-9]+", normalized.casefold()))
    if set(tokens) & _TARGET_CONTEXT_TOKENS:
        return True
    return any(left == "ground" and right == "truth" for left, right in zip(tokens, tokens[1:], strict=False))


def _redact_artifact_text(value: str) -> str:
    redacted = _redact_credentials(value)
    assignment_redacted = _TARGET_ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED_TARGET]", redacted)
    if assignment_redacted != redacted or _has_target_context(redacted):
        return "[REDACTED_TARGET_CONTEXT]"
    redacted = assignment_redacted
    for pattern in _PATH_PATTERNS:
        redacted = pattern.sub("[REDACTED_PATH]", redacted)
    return redacted


def _safe_tag_value(value: str) -> bool:
    return (
        len(value) <= 256
        and all(character >= " " and character != "\x7f" for character in value)
        and _redact_artifact_text(value) == value
    )


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_enum(value: object, enum_type: type[StrEnum], name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__} value")


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """One attributable failure without embedding credentials or raw targets."""

    event_id: str
    experiment_id: str
    logical_cell_id: str
    stage: ValidationStage
    category: FailureCategory
    message: str
    dataset_id: str | None = None
    severity: FailureSeverity = FailureSeverity.CELL_FAILURE
    disposition: RetryDisposition = RetryDisposition.NOT_RETRYABLE
    sample_id: str | None = None
    expected: bool = False
    contributing: tuple[FailureCategory, ...] = ()
    tags: dict[str, str] = field(default_factory=dict)
    schema_version: str = field(default=FAILURE_TAXONOMY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.event_id, "event_id")
        _require_identifier(self.experiment_id, "experiment_id")
        _require_identifier(self.logical_cell_id, "logical_cell_id")
        _require_identifier(self.message, "message")
        _require_enum(self.stage, ValidationStage, "stage")
        _require_enum(self.category, FailureCategory, "category")
        _require_enum(self.severity, FailureSeverity, "severity")
        _require_enum(self.disposition, RetryDisposition, "disposition")
        if self.dataset_id is not None:
            _require_identifier(self.dataset_id, "dataset_id")
        if self.sample_id is not None:
            _require_identifier(self.sample_id, "sample_id")
        if not isinstance(self.expected, bool):
            raise TypeError("expected must be boolean")
        if any(not isinstance(value, FailureCategory) for value in self.contributing):
            raise TypeError("contributing values must be FailureCategory members")
        if self.category in self.contributing or len(set(self.contributing)) != len(self.contributing):
            raise ValueError("contributing categories must be unique and exclude the primary category")
        if not isinstance(self.tags, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) for key, value in self.tags.items()
        ):
            raise TypeError("tags must be a mapping of non-empty string keys to string values")
        for key, value in self.tags.items():
            tokens = set(_tag_key_tokens(key))
            if (
                not _SAFE_TAG_KEY.fullmatch(key)
                or tokens & _SENSITIVE_TAG_TOKENS
                or tokens & _TARGET_OR_PATH_TAG_TOKENS
            ):
                raise ValueError(f"unsafe failure tag key: {key!r}")
            if not _safe_tag_value(value):
                raise ValueError(f"unsafe failure tag value for {key!r}")

    def to_serializable(self) -> dict[str, Any]:
        """Return the internal record; use ``to_artifact_serializable`` for formal artifacts."""
        return asdict(self)

    def to_artifact_serializable(
        self,
        *,
        raw_sample_id_authorization: object | None = None,
        sample_id_salt: str = "jepa4d-failure-artifact-v1",
    ) -> dict[str, Any]:
        """Return a credential-redacted artifact record with pseudonymous sample identity.

        Raw sample-ID disclosure is deliberately disabled until an authorization
        can be verified against a cryptographic policy authority. The reserved
        authorization argument rejects every non-``None`` value so legacy callers
        fail closed rather than mistaking a self-issued record for authority. The
        pseudonymous digest is stable within an experiment namespace.
        """

        _require_identifier(sample_id_salt, "sample_id_salt")
        if raw_sample_id_authorization is not None:
            raise ValueError(
                "raw sample-ID disclosure is disabled until cryptographically verified "
                "policy-authority integration exists"
            )
        payload = asdict(self)
        for name in ("event_id", "experiment_id", "logical_cell_id"):
            payload[name] = _redact_credentials(payload[name])
        payload["message"] = _redact_artifact_text(self.message)
        payload["tags"] = {key: _redact_artifact_text(value) for key, value in self.tags.items()}
        if self.sample_id is not None:
            digest = hashlib.sha256(f"{sample_id_salt}\0{self.experiment_id}\0{self.sample_id}".encode()).hexdigest()
            payload["sample_id"] = f"sha256:{digest}"
            payload["sample_id_disclosure"] = {"mode": "pseudonymous"}
        else:
            payload["sample_id_disclosure"] = {"mode": "absent"}
        return payload


@dataclass(frozen=True, slots=True)
class ValidationStatus:
    """Orthogonal protocol, execution, and scientific status for one experiment."""

    experiment_id: str
    stage: ValidationStage
    protocol: ProtocolStatus
    execution: ExecutionStatus
    scientific: ScientificStatus
    evidence_level: EvidenceLevel
    expected_cells: int
    successful_cells: int = 0
    expected_failure_cells: int = 0
    failed_cells: int = 0
    skipped_cells: int = 0
    legal_skip_reasons: tuple[str, ...] = ()
    failure_event_ids: tuple[str, ...] = ()
    metric_gate_receipt_sha256: str | None = None
    schema_version: str = field(default=VALIDATION_STATUS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.experiment_id, "experiment_id")
        _require_enum(self.stage, ValidationStage, "stage")
        _require_enum(self.protocol, ProtocolStatus, "protocol")
        _require_enum(self.execution, ExecutionStatus, "execution")
        _require_enum(self.scientific, ScientificStatus, "scientific")
        _require_enum(self.evidence_level, EvidenceLevel, "evidence_level")
        counts = (
            self.expected_cells,
            self.successful_cells,
            self.expected_failure_cells,
            self.failed_cells,
            self.skipped_cells,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("cell counts must be integers")
        if self.expected_cells < 1 or any(value < 0 for value in counts[1:]):
            raise ValueError("expected_cells must be positive and terminal cell counts non-negative")
        terminal_cells = sum(counts[1:])
        if terminal_cells > self.expected_cells:
            raise ValueError("terminal cell counts cannot exceed expected_cells")
        if self.execution is ExecutionStatus.NOT_STARTED and terminal_cells:
            raise ValueError("a not-started experiment cannot contain terminal cells")
        if self.execution is ExecutionStatus.COMPLETE:
            if terminal_cells != self.expected_cells:
                raise ValueError("a complete experiment must account for every expected cell")
            if self.failed_cells:
                raise ValueError("unexpected failed cells require execution=FAILED")
        if self.execution is ExecutionStatus.FAILED and not self.failed_cells:
            raise ValueError("execution=FAILED requires at least one unexpected failed cell")
        if self.scientific not in (ScientificStatus.NOT_EVALUATED, ScientificStatus.NOT_APPLICABLE):
            if self.execution is not ExecutionStatus.COMPLETE:
                raise ValueError("a scientific decision requires complete execution")
            if self.protocol is not ProtocolStatus.FROZEN:
                raise ValueError("a scientific decision requires a frozen protocol")
        if self.skipped_cells != len(self.legal_skip_reasons):
            raise ValueError("every skipped cell requires exactly one legal skip reason")
        if any(not isinstance(reason, str) or not reason.strip() for reason in self.legal_skip_reasons):
            raise ValueError("legal skip reasons must be non-empty strings")
        if any(not isinstance(event_id, str) or not event_id.strip() for event_id in self.failure_event_ids):
            raise ValueError("failure_event_ids must be non-empty strings")
        if len(set(self.failure_event_ids)) != len(self.failure_event_ids):
            raise ValueError("failure_event_ids must be unique")
        if self.metric_gate_receipt_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.metric_gate_receipt_sha256
        ):
            raise ValueError("metric_gate_receipt_sha256 must be a lowercase SHA-256 digest")
        if (self.failed_cells or self.expected_failure_cells) and not self.failure_event_ids:
            raise ValueError("failure cells require at least one failure event ID")
        if self.evidence_level is EvidenceLevel.EXTERNAL_CONFIRMATION:
            terminal_decisions = {
                ScientificStatus.PASS,
                ScientificStatus.FAIL,
                ScientificStatus.NO_SURVIVOR,
                ScientificStatus.INCONCLUSIVE,
            }
            if self.protocol is not ProtocolStatus.FROZEN:
                raise ValueError("external-confirmation evidence requires a frozen protocol")
            if self.execution is not ExecutionStatus.COMPLETE:
                raise ValueError("external-confirmation evidence requires complete execution")
            if self.scientific not in terminal_decisions:
                raise ValueError("external-confirmation evidence requires a terminal scientific decision")
            if self.scientific is ScientificStatus.PASS:
                if self.successful_cells < 1:
                    raise ValueError("external-confirmation pass requires at least one successful evaluated cell")
                if self.metric_gate_receipt_sha256 is None:
                    raise ValueError("external-confirmation pass requires a metric/gate evidence receipt SHA-256")

    @property
    def terminal_cells(self) -> int:
        return self.successful_cells + self.expected_failure_cells + self.failed_cells + self.skipped_cells

    @property
    def matrix_complete(self) -> bool:
        return self.terminal_cells == self.expected_cells

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)
