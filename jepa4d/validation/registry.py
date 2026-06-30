"""Machine-readable dataset portfolio and target-access enforcement."""

from __future__ import annotations

import base64
import binascii
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jepa4d.validation._content import (
    ContentAddress,
    load_yaml_unique,
    sha256_file,
    sha256_value,
    write_content_addressed_json,
)

REGISTRY_SCHEMA_VERSION = "jepa4d-validation-registry-v1"
SNAPSHOT_SCHEMA_VERSION = "jepa4d-validation-registry-snapshot-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ENV_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, populate_by_name=True)


class DataRole(StrEnum):
    A1 = "A1"
    A2 = "A2"
    B = "B"
    C = "C"
    CONTRACT_ONLY = "contract-only"


class AuditStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"


class DatasetStatus(StrEnum):
    PLANNED = "planned"
    ACTIVE_DEVELOPMENT = "active-development"
    HISTORICAL_CONSUMED = "historical-consumed"
    EXTERNAL_SEALED = "external-sealed"
    CONTRACT_ONLY = "contract-only"


class SplitPurpose(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    DEVELOPMENT_TEST = "development-test"
    TRANSFER_TEST = "transfer-test"
    EXTERNAL_TEST = "external-test"
    STRESS_TEST = "stress-test"
    REGRESSION = "regression"
    CONTRACT = "contract"


class IndependentUnit(StrEnum):
    SCENE = "scene"
    VIDEO = "video"
    SUBJECT = "subject"
    EPISODE = "episode"
    RECORDING = "recording"
    SEQUENCE = "sequence"
    TRAJECTORY = "trajectory"
    IMAGE = "image"
    ENVIRONMENT = "environment"
    GENERATED_CASE = "generated-case"


class TargetState(StrEnum):
    OPEN = "open"
    SEALED = "sealed"
    SERVER_ONLY = "server-only"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not-applicable"


class AccessOperation(StrEnum):
    DOWNLOAD = "download"
    DECODE_SMOKE = "decode-smoke"
    TRAINING = "training"
    TUNING = "tuning"
    CALIBRATION = "calibration"
    CHECKPOINT_SELECTION = "checkpoint-selection"
    DEVELOPMENT_EVALUATION = "development-evaluation"
    EXTERNAL_EVALUATION = "external-evaluation"
    STRESS_EVALUATION = "stress-evaluation"
    REGRESSION = "regression"
    MECHANISM_DIAGNOSTIC = "mechanism-diagnostic"
    REPORTING = "reporting"
    METADATA_AUDIT = "metadata-audit"


SELECTION_OPERATIONS = frozenset(
    {
        AccessOperation.TRAINING,
        AccessOperation.TUNING,
        AccessOperation.CALIBRATION,
        AccessOperation.CHECKPOINT_SELECTION,
    }
)
TARGET_SPLITS = frozenset(
    {
        SplitPurpose.DEVELOPMENT_TEST,
        SplitPurpose.TRANSFER_TEST,
        SplitPurpose.EXTERNAL_TEST,
        SplitPurpose.STRESS_TEST,
    }
)


class SourceSpec(StrictModel):
    official_url: str
    version: str
    citation: str
    paper_url: str | None = None

    @field_validator("official_url", "paper_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "generated"} or (parsed.scheme == "https" and not parsed.netloc):
            raise ValueError("source URLs must use https:// or generated://")
        return value

    @property
    def canonical_url(self) -> str:
        parsed = urlsplit(self.official_url)
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


class LicenseSpec(StrictModel):
    status: AuditStatus
    name: str | None = None
    terms_url: str | None = None
    redistribution: Literal["prohibited", "metadata-only", "derived-only", "allowed", "pending"]
    privacy_notes: str
    blocker: str | None = None

    @model_validator(mode="after")
    def approved_is_complete(self) -> LicenseSpec:
        if self.status is AuditStatus.APPROVED and (not self.name or not self.terms_url):
            raise ValueError("approved licenses require name and terms_url")
        if self.status is not AuditStatus.APPROVED and not self.blocker:
            raise ValueError("pending/blocked licenses require an explicit blocker")
        return self


class AccessSpec(StrictModel):
    status: AuditStatus
    method: Literal["public", "click-through", "account", "request", "evaluation-server", "generated"]
    credentials_env: tuple[str, ...] = ()
    reviewed_at: str | None = None
    reviewer: str | None = None
    blocker: str | None = None

    @field_validator("credentials_env")
    @classmethod
    def validate_env_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not ENV_PATTERN.fullmatch(name) for name in value):
            raise ValueError("credentials_env must contain unique environment-variable names, never secret values")
        return value

    @model_validator(mode="after")
    def approved_is_complete(self) -> AccessSpec:
        if self.status is AuditStatus.APPROVED and (not self.reviewed_at or not self.reviewer):
            raise ValueError("approved access requires reviewed_at and reviewer")
        if self.status is not AuditStatus.APPROVED and not self.blocker:
            raise ValueError("pending/blocked access requires an explicit blocker")
        return self


class SealedAuthoritySpec(StrictModel):
    """Registry-governed Ed25519 authority for one-shot selector decisions."""

    status: Literal["pending", "approved", "blocked", "revoked"]
    key_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{2,127}$")
    public_key_ed25519_base64: str | None = None
    approved_at: str | None = None
    approved_by: str | None = None
    blocker: str | None = None

    @model_validator(mode="after")
    def validate_authority(self) -> SealedAuthoritySpec:
        if self.status == "approved":
            if not all((self.key_id, self.public_key_ed25519_base64, self.approved_at, self.approved_by)):
                raise ValueError("approved sealed authority requires key_id, public key, approved_at, and approved_by")
            encoded_public_key = self.public_key_ed25519_base64
            if encoded_public_key is None:  # Defensive narrowing for static analysis after the completeness check.
                raise ValueError("approved sealed authority requires a public key")
            try:
                public_key = base64.b64decode(encoded_public_key, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("sealed authority public key must be canonical base64") from error
            if base64.b64encode(public_key).decode("ascii") != encoded_public_key:
                raise ValueError("sealed authority public key must be canonical base64")
            if len(public_key) != 32:
                raise ValueError("Ed25519 public key must decode to exactly 32 bytes")
        elif not self.blocker:
            raise ValueError("non-approved sealed authority requires an explicit blocker")
        return self


class StorageSpec(StrictModel):
    status: AuditStatus
    expected_bytes: int | None = Field(default=None, ge=0)
    raw_root_env: str
    cache_root_env: str
    retention: str
    blocker: str | None = None

    @field_validator("raw_root_env", "cache_root_env")
    @classmethod
    def validate_env_name(cls, value: str) -> str:
        if not ENV_PATTERN.fullmatch(value):
            raise ValueError("storage roots must be environment-variable names, never machine-specific paths")
        return value

    @model_validator(mode="after")
    def pending_has_blocker(self) -> StorageSpec:
        if self.status is not AuditStatus.APPROVED and not self.blocker:
            raise ValueError("pending/blocked storage requires an explicit blocker")
        return self


class HashSpec(StrictModel):
    name: str
    kind: Literal["archive", "metadata", "id-manifest", "split-manifest", "fixture"]
    status: Literal["pending", "published", "verified"]
    sha256: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    provenance: str

    @model_validator(mode="after")
    def digest_matches_status(self) -> HashSpec:
        if self.status == "pending" and self.sha256 is not None:
            raise ValueError("pending hash must not claim a digest")
        if self.status != "pending" and (self.sha256 is None or not SHA256_PATTERN.fullmatch(self.sha256)):
            raise ValueError("published/verified hashes require a lowercase SHA-256 digest")
        return self


class ArtifactRule(StrictModel):
    artifact: Literal[
        "raw-data",
        "raw-targets",
        "sample-identifiers",
        "derived-features",
        "predictions",
        "aggregate-metrics",
        "qualitative-preview",
    ]
    local: Literal["deny", "restricted", "allowed"]
    wandb: Literal["deny", "aggregate-only", "preapproved-only", "allowed"]
    repository: Literal["deny", "metadata-only", "allowed"]
    notes: str

    @model_validator(mode="after")
    def deny_raw_remote_artifacts(self) -> ArtifactRule:
        if self.artifact in {"raw-data", "raw-targets"} and self.wandb == "allowed":
            raise ValueError("raw data/targets may not be unconditionally uploaded to W&B")
        return self


class SplitSpec(StrictModel):
    split_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    official_name: str
    purpose: SplitPurpose
    independent_unit: IndependentUnit
    target_state: TargetState
    targets_present: bool
    selection_rule: str
    id_manifest: str | None = None
    id_manifest_sha256: str | None = None
    expected_units: int | None = Field(default=None, ge=1)
    allowed_operations: frozenset[AccessOperation]
    seal_condition: str | None = None

    @field_validator("id_manifest_sha256")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_PATTERN.fullmatch(value):
            raise ValueError("id_manifest_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def enforce_target_opacity(self) -> SplitSpec:
        if self.purpose in TARGET_SPLITS and self.allowed_operations & SELECTION_OPERATIONS:
            forbidden = sorted(operation.value for operation in self.allowed_operations & SELECTION_OPERATIONS)
            raise ValueError(f"held-out target split permits selection operations: {forbidden}")
        if self.target_state is TargetState.SEALED and not self.seal_condition:
            raise ValueError("sealed splits require a seal_condition")
        if self.target_state in {TargetState.SEALED, TargetState.SERVER_ONLY, TargetState.UNAVAILABLE}:
            legal = {AccessOperation.METADATA_AUDIT, AccessOperation.REPORTING}
            if self.target_state is TargetState.SEALED:
                # This declares the only operation a future hash-bound unsealing receipt may authorize.
                legal.add(AccessOperation.EXTERNAL_EVALUATION)
            data_operations = self.allowed_operations - legal
            if data_operations:
                raise ValueError(
                    f"non-open split cannot grant data operations: {sorted(x.value for x in data_operations)}"
                )
        if self.id_manifest_sha256 and not self.id_manifest:
            raise ValueError("id_manifest_sha256 requires id_manifest")
        return self


class DatasetEntry(StrictModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    display_name: str
    stages: tuple[Annotated[str, Field(pattern=r"^phase[0-6]$")], ...] = Field(min_length=1)
    role: DataRole
    status: DatasetStatus
    claim_use: str
    source: SourceSpec
    license_info: LicenseSpec = Field(alias="license")
    access: AccessSpec
    sealed_authority: SealedAuthoritySpec | None = None
    storage: StorageSpec
    hashes: tuple[HashSpec, ...] = Field(min_length=1)
    splits: tuple[SplitSpec, ...] = Field(min_length=1)
    artifact_rules: tuple[ArtifactRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_role_and_splits(self) -> DatasetEntry:
        if len(self.stages) != len(set(self.stages)):
            raise ValueError(f"duplicate stage within {self.dataset_id}")
        split_ids = [split.split_id for split in self.splits]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"duplicate split_id within {self.dataset_id}")
        if self.role in {DataRole.B, DataRole.C}:
            for split in self.splits:
                forbidden = split.allowed_operations & SELECTION_OPERATIONS
                if forbidden:
                    raise ValueError(
                        f"role {self.role.value} cannot permit target selection: {sorted(x.value for x in forbidden)}"
                    )
        if self.role is DataRole.CONTRACT_ONLY:
            legal = {AccessOperation.DECODE_SMOKE, AccessOperation.METADATA_AUDIT, AccessOperation.REPORTING}
            for split in self.splits:
                if split.allowed_operations - legal:
                    raise ValueError("contract-only data cannot authorize fitting or quality evaluation")
        if any(split.target_state is TargetState.SEALED for split in self.splits) and self.sealed_authority is None:
            raise ValueError("datasets with sealed splits require a registry-governed sealed_authority")
        artifact_types = [rule.artifact for rule in self.artifact_rules]
        if len(artifact_types) != len(set(artifact_types)):
            raise ValueError(f"duplicate artifact rule within {self.dataset_id}")
        missing_raw_denials = {"raw-data", "raw-targets"} - set(artifact_types)
        if missing_raw_denials:
            raise ValueError(f"missing mandatory raw-artifact denial rules: {sorted(missing_raw_denials)}")
        if self.role is not DataRole.CONTRACT_ONLY:
            for rule in self.artifact_rules:
                if rule.artifact in {"raw-data", "raw-targets"} and (
                    rule.wandb != "deny" or rule.repository != "deny"
                ):
                    raise ValueError("non-contract raw data/targets must be denied from W&B and repository artifacts")
        return self

    @property
    def readiness_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        for name, value in (("license", self.license_info), ("access", self.access), ("storage", self.storage)):
            if value.status is not AuditStatus.APPROVED:
                blockers.append(f"{name}:{value.status.value}:{value.blocker}")
        if not any(value.status in {"published", "verified"} for value in self.hashes):
            blockers.append("hashes:no published or locally verified identity")
        if self.sealed_authority is not None and self.sealed_authority.status != "approved":
            blockers.append(f"sealed_authority:{self.sealed_authority.status}:{self.sealed_authority.blocker}")
        return tuple(blockers)


class AccessDecision(StrictModel):
    dataset_id: str
    split_id: str
    operation: AccessOperation
    authorized: Literal[True]
    grants_data_access: bool
    registry_sha256: str


class AccessDenied(PermissionError):
    """Raised when a registered data role cannot be used for an operation."""


class DatasetRegistry(StrictModel):
    schema_version: Literal["jepa4d-validation-registry-v1"]
    portfolio_version: str
    datasets: tuple[DatasetEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_identities(self) -> DatasetRegistry:
        dataset_ids: set[str] = set()
        split_ids: set[str] = set()
        manifest_digests: dict[str, str] = {}
        physical_identities: dict[tuple[object, ...], str] = {}
        for entry in self.datasets:
            if entry.dataset_id in dataset_ids:
                raise ValueError(f"duplicate dataset_id: {entry.dataset_id}")
            dataset_ids.add(entry.dataset_id)
            for split in entry.splits:
                if split.split_id in split_ids:
                    raise ValueError(f"duplicate global split_id: {split.split_id}")
                split_ids.add(split.split_id)
                if split.id_manifest_sha256:
                    prior = manifest_digests.get(split.id_manifest_sha256)
                    if prior is not None:
                        raise ValueError(
                            f"duplicate split ID manifest {split.id_manifest_sha256}: {prior} and {split.split_id}"
                        )
                    manifest_digests[split.id_manifest_sha256] = split.split_id
                identity = (
                    entry.source.canonical_url,
                    entry.source.version.casefold(),
                    split.official_name.casefold(),
                    split.selection_rule.casefold(),
                    split.id_manifest_sha256,
                )
                prior = physical_identities.get(identity)
                if prior is not None:
                    raise ValueError(f"duplicate physical split identity: {prior} and {split.split_id}")
                physical_identities[identity] = split.split_id
        return self

    @classmethod
    def load(cls, path: str | Path) -> DatasetRegistry:
        return cls.model_validate(load_yaml_unique(path))

    @property
    def sha256(self) -> str:
        return sha256_value(self)

    def dataset(self, dataset_id: str) -> DatasetEntry:
        for entry in self.datasets:
            if entry.dataset_id == dataset_id:
                return entry
        raise KeyError(f"unknown dataset_id {dataset_id!r}")

    def split(self, dataset_id: str, split_id: str) -> tuple[DatasetEntry, SplitSpec]:
        entry = self.dataset(dataset_id)
        for split in entry.splits:
            if split.split_id == split_id:
                return entry, split
        raise KeyError(f"unknown split_id {split_id!r} for dataset {dataset_id!r}")

    def _authorize_registered_operation(
        self,
        dataset_id: str,
        split_id: str,
        operation: AccessOperation | str,
        *,
        sealed_authorization_verified: bool = False,
        consumed_future_use: bool = False,
    ) -> AccessDecision:
        """Validate registry policy; public callers must use DatasetAccessController."""
        operation = AccessOperation(operation)
        entry, split = self.split(dataset_id, split_id)
        if operation in SELECTION_OPERATIONS and (entry.role is DataRole.B or split.purpose in TARGET_SPLITS):
            raise AccessDenied(f"{operation.value} is forbidden for transfer/external target {dataset_id}/{split_id}")
        if operation not in split.allowed_operations:
            raise AccessDenied(f"{operation.value} is not registered for {dataset_id}/{split_id}")
        non_data_operation = operation in {AccessOperation.METADATA_AUDIT, AccessOperation.REPORTING}
        if not non_data_operation and entry.readiness_blockers:
            raise AccessDenied(f"dataset audit is incomplete: {list(entry.readiness_blockers)}")
        if split.target_state is TargetState.SEALED:
            if non_data_operation:
                return AccessDecision(
                    dataset_id=dataset_id,
                    split_id=split_id,
                    operation=operation,
                    authorized=True,
                    grants_data_access=False,
                    registry_sha256=self.sha256,
                )
            if not sealed_authorization_verified:
                raise AccessDenied("sealed target requires a verified typed content-addressed authorization artifact")
        if split.target_state in {TargetState.SERVER_ONLY, TargetState.UNAVAILABLE}:
            if non_data_operation:
                return AccessDecision(
                    dataset_id=dataset_id,
                    split_id=split_id,
                    operation=operation,
                    authorized=True,
                    grants_data_access=False,
                    registry_sha256=self.sha256,
                )
            raise AccessDenied(f"split state {split.target_state.value} does not grant local target access")
        if consumed_future_use and operation not in {
            AccessOperation.REGRESSION,
            AccessOperation.MECHANISM_DIAGNOSTIC,
            AccessOperation.REPORTING,
        }:
            raise AccessDenied(f"operation {operation.value} is not a legal consumed-target future use")
        return AccessDecision(
            dataset_id=dataset_id,
            split_id=split_id,
            operation=operation,
            authorized=True,
            grants_data_access=operation not in {AccessOperation.METADATA_AUDIT, AccessOperation.REPORTING},
            registry_sha256=self.sha256,
        )


def freeze_registry(registry_path: str | Path, output_dir: str | Path) -> tuple[ContentAddress, ContentAddress]:
    registry = DatasetRegistry.load(registry_path)
    normalized = registry.model_dump(mode="json", by_alias=True, exclude_none=False)
    snapshot_payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "registry_sha256": registry.sha256,
        "registry": normalized,
    }
    snapshot = write_content_addressed_json(snapshot_payload, output_dir, prefix="dataset-registry")
    receipt_payload = {
        "schema_version": "jepa4d-validation-registry-freeze-receipt-v1",
        "source_sha256": sha256_file(registry_path),
        "registry_sha256": registry.sha256,
        "snapshot": {"name": snapshot.path.name, "sha256": snapshot.sha256, "bytes": snapshot.bytes},
    }
    receipt = write_content_addressed_json(receipt_payload, output_dir, prefix="dataset-registry-receipt")
    return snapshot, receipt
