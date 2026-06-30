"""Immutable split-decision manifests for validation datasets.

The manifest records *how* source units were selected and rejects obvious
target-bearing fields or text.  This screening plus a reviewer attestation is
defense in depth, not proof of target isolation: the selector must still run in
an execution environment where target paths are denied, and the denial must be
verified independently.  Rows are grouped by an explicit independent cluster
so that a scene, subject, video, or episode cannot be split between selected
and rejected populations accidentally.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field, StrictBool, field_validator, model_validator

from jepa4d.validation._content import (
    ContentAddress,
    load_yaml_unique,
    sha256_value,
    verify_content_addressed_json,
    write_content_addressed_json,
)
from jepa4d.validation.registry import SHA256_PATTERN, DatasetRegistry, IndependentUnit, StrictModel

SPLIT_MANIFEST_SCHEMA_VERSION = "jepa4d-split-manifest-v1"
SPLIT_MANIFEST_SNAPSHOT_SCHEMA_VERSION = "jepa4d-split-manifest-snapshot-v1"
REGISTERED_SPLIT_MANIFEST_SNAPSHOT_SCHEMA_VERSION = "jepa4d-registered-split-manifest-snapshot-v1"

IDENTIFIER_PATTERN = r"^[^\s]+$"
VERSIONED_RULE_PATTERN = r"^[a-z0-9][a-z0-9._-]*-v[1-9][0-9]*$"

MetadataScalar: TypeAlias = str | int | float | bool | None

# Operational source metadata such as frame_count, duration_seconds, or codec
# is useful for deterministic eligibility checks.  Target-bearing fields are
# not.  This deny-list is deliberately conservative; stage-specific target
# metadata belongs in separately access-controlled evaluation artifacts.
_FORBIDDEN_METADATA_TOKENS = frozenset(
    {
        "accuracy",
        "action",
        "actions",
        "answer",
        "answers",
        "annotation",
        "annotations",
        "bbox",
        "bboxes",
        "box",
        "boxes",
        "category",
        "categories",
        "class",
        "classes",
        "depth",
        "failure",
        "failures",
        "groundtruth",
        "gt",
        "keypoint",
        "keypoints",
        "label",
        "labels",
        "loss",
        "mask",
        "masks",
        "metric",
        "metrics",
        "outcome",
        "outcomes",
        "pose",
        "reward",
        "rewards",
        "score",
        "scores",
        "segmentation",
        "success",
        "successes",
        "target",
        "targets",
    }
)


def _text_tokens(text: str) -> tuple[str, ...]:
    # Split common camelCase/PascalCase evasions before case folding, then
    # normalize punctuation and whitespace into token boundaries.
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.casefold()).strip("_")
    return tuple(token for token in normalized.split("_") if token)


def _contains_target_like_tokens(tokens: tuple[str, ...]) -> bool:
    compact_phrases = {
        "".join(tokens[start:stop])
        for start in range(len(tokens))
        for stop in range(start + 2, min(len(tokens), start + 3) + 1)
    }
    return any(token in _FORBIDDEN_METADATA_TOKENS for token in (*tokens, *compact_phrases))


def _validate_no_target_like_text(value: str, *, field_name: str) -> str:
    if _contains_target_like_tokens(_text_tokens(value)):
        raise ValueError(f"forbidden target-like token in {field_name}: {value!r}")
    return value


def _validate_no_target_like_mapping(
    value: dict[str, MetadataScalar], *, field_name: str
) -> dict[str, MetadataScalar]:
    for key, item in value.items():
        tokens = _text_tokens(key)
        if not tokens or _contains_target_like_tokens(tokens):
            raise ValueError(f"forbidden target-like metadata key in {field_name}: {key!r}")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"metadata values must be finite in {field_name}: {key!r}")
        if isinstance(item, str):
            _validate_no_target_like_text(item, field_name=f"{field_name} string value {key!r}")
    return value


def _validate_sha256(value: str, *, field_name: str) -> str:
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


class MechanicalSelectionRule(StrictModel):
    """Executable identity of the prespecified unit-selection procedure."""

    algorithm: str = Field(pattern=VERSIONED_RULE_PATTERN)
    implementation: str = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    implementation_sha256: str
    seed: int = Field(ge=0, le=2**63 - 1)
    eligibility_rules: tuple[str, ...] = Field(min_length=1)
    parameters: dict[str, MetadataScalar] = Field(default_factory=dict)

    @field_validator("implementation_sha256")
    @classmethod
    def implementation_digest_is_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="selection implementation_sha256")

    @field_validator("eligibility_rules")
    @classmethod
    def rules_are_explicit_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not rule.strip() for rule in value):
            raise ValueError("eligibility_rules cannot contain blank rules")
        if len(value) != len(set(value)):
            raise ValueError("eligibility_rules must be unique and ordered")
        for rule in value:
            _validate_no_target_like_text(rule, field_name="selection eligibility_rules")
        return value

    @field_validator("parameters")
    @classmethod
    def parameters_exclude_target_like_content(cls, value: dict[str, MetadataScalar]) -> dict[str, MetadataScalar]:
        return _validate_no_target_like_mapping(value, field_name="selection parameters")


class TargetIsolationAttestation(StrictModel):
    """Reviewer assertion backed by execution-generated path-denial evidence.

    A valid attestation does not prove isolation by itself.  The selector must
    still execute with targets physically inaccessible; the receipt digest
    binds this declaration to evidence produced by that isolation boundary.
    """

    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    targets_accessible: StrictBool
    selector_implementation_sha256: str
    path_denial_receipt_sha256: str

    @field_validator("reviewed_at")
    @classmethod
    def review_time_has_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("reviewed_at must be an ISO-8601 timestamp with timezone") from error
        if parsed.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @field_validator("targets_accessible")
    @classmethod
    def target_paths_must_be_inaccessible(cls, value: bool) -> bool:
        if value:
            raise ValueError("targets_accessible must be false")
        return value

    @field_validator("selector_implementation_sha256", "path_denial_receipt_sha256")
    @classmethod
    def evidence_digest_is_sha256(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)


class SourceAssetIdentity(StrictModel):
    """Content identity for a source asset used by at least one unit row."""

    asset_id: str = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    source_ref: str = Field(min_length=1)
    sha256: str
    bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="source asset identity")


class ClusterIdentity(StrictModel):
    """A unique independent cluster that may contain one or more source units."""

    cluster_id: str = Field(min_length=1, pattern=IDENTIFIER_PATTERN)


class UnitRow(StrictModel):
    """Source-unit identity without intentional target-bearing metadata."""

    unit_id: str = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    cluster_id: str = Field(min_length=1, pattern=IDENTIFIER_PATTERN)
    physical_unit_sha256: str
    source_asset_ids: tuple[str, ...] = Field(min_length=1)
    metadata: dict[str, MetadataScalar] = Field(default_factory=dict)

    @field_validator("source_asset_ids")
    @classmethod
    def asset_references_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("source_asset_ids must be unique within a unit")
        return value

    @field_validator("physical_unit_sha256")
    @classmethod
    def physical_identity_is_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="physical_unit_sha256")

    @field_validator("metadata")
    @classmethod
    def metadata_excludes_target_like_content(cls, value: dict[str, MetadataScalar]) -> dict[str, MetadataScalar]:
        return _validate_no_target_like_mapping(value, field_name="unit metadata")


class SelectedUnitRow(UnitRow):
    disposition: Literal["selected"] = "selected"


class RejectedUnitRow(UnitRow):
    disposition: Literal["rejected"] = "rejected"
    rejection_reason: str = Field(min_length=1)

    @field_validator("rejection_reason")
    @classmethod
    def rejection_reason_is_explicit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rejected units require a non-blank rejection_reason")
        return _validate_no_target_like_text(value, field_name="rejection_reason")


class SplitManifest(StrictModel):
    """A versioned, deterministic split decision over independent source units."""

    schema_version: Literal["jepa4d-split-manifest-v1"]
    manifest_version: str = Field(min_length=1)
    portfolio_version: str = Field(min_length=1)
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    dataset_version: str = Field(min_length=1)
    split_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    independent_unit: IndependentUnit
    selection: MechanicalSelectionRule
    target_isolation: TargetIsolationAttestation
    source_assets: tuple[SourceAssetIdentity, ...] = Field(min_length=1)
    clusters: tuple[ClusterIdentity, ...] = Field(min_length=1)
    selected_units: tuple[SelectedUnitRow, ...] = Field(min_length=1)
    rejected_units: tuple[RejectedUnitRow, ...]

    @model_validator(mode="after")
    def identities_and_references_are_coherent(self) -> SplitManifest:
        if self.target_isolation.selector_implementation_sha256 != self.selection.implementation_sha256:
            raise ValueError("target-isolation attestation names a different selector implementation SHA-256")

        asset_ids = [asset.asset_id for asset in self.source_assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("duplicate source asset_id")
        asset_digests = [asset.sha256 for asset in self.source_assets]
        if len(asset_digests) != len(set(asset_digests)):
            raise ValueError("duplicate source asset SHA-256 identity")

        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("duplicate cluster_id")

        all_units: tuple[UnitRow, ...] = (*self.selected_units, *self.rejected_units)
        unit_ids = [unit.unit_id for unit in all_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("duplicate unit_id across selected/rejected rows")
        physical_unit_ids = [unit.physical_unit_sha256 for unit in all_units]
        if len(physical_unit_ids) != len(set(physical_unit_ids)):
            raise ValueError("duplicate physical_unit_sha256 across selected/rejected rows")

        known_assets = set(asset_ids)
        known_clusters = set(cluster_ids)
        for unit in all_units:
            unknown_assets = sorted(set(unit.source_asset_ids) - known_assets)
            if unknown_assets:
                raise ValueError(f"unit {unit.unit_id!r} references unknown source assets: {unknown_assets}")
            if unit.cluster_id not in known_clusters:
                raise ValueError(f"unit {unit.unit_id!r} references unknown cluster_id {unit.cluster_id!r}")

        selected_clusters = {unit.cluster_id for unit in self.selected_units}
        rejected_clusters = {unit.cluster_id for unit in self.rejected_units}
        overlap = sorted(selected_clusters & rejected_clusters)
        if overlap:
            raise ValueError(f"independent clusters cannot mix selected and rejected units: {overlap}")

        referenced_assets = {asset_id for unit in all_units for asset_id in unit.source_asset_ids}
        unused_assets = sorted(known_assets - referenced_assets)
        if unused_assets:
            raise ValueError(f"unreferenced source assets in manifest: {unused_assets}")
        referenced_clusters = {unit.cluster_id for unit in all_units}
        unused_clusters = sorted(known_clusters - referenced_clusters)
        if unused_clusters:
            raise ValueError(f"unreferenced cluster identities in manifest: {unused_clusters}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> SplitManifest:
        """Load and validate YAML or JSON using the same strict schema."""

        return cls.model_validate(load_yaml_unique(path))

    @property
    def sha256(self) -> str:
        """Hash the canonical normalized representation, independent of file format."""

        return sha256_value(self)

    def validate_against_registry(self, registry: DatasetRegistry) -> None:
        """Validate one-way binding to the registered portfolio and split.

        The manifest binds to ``portfolio_version`` rather than the registry
        digest.  A registry may therefore register this manifest's SHA-256
        without creating a circular hash dependency.
        """

        if self.portfolio_version != registry.portfolio_version:
            raise ValueError(
                "manifest portfolio_version does not match registry: "
                f"{self.portfolio_version!r} != {registry.portfolio_version!r}"
            )
        try:
            dataset, split = registry.split(self.dataset_id, self.split_id)
        except KeyError as error:
            raise ValueError(f"manifest dataset/split is not registered: {self.dataset_id}/{self.split_id}") from error
        if self.dataset_version != dataset.source.version:
            raise ValueError(
                "manifest dataset_version does not match registered source version: "
                f"{self.dataset_version!r} != {dataset.source.version!r}"
            )
        if self.independent_unit is not split.independent_unit:
            raise ValueError(
                "manifest independent_unit does not match registry: "
                f"{self.independent_unit.value!r} != {split.independent_unit.value!r}"
            )
        if split.id_manifest_sha256 is not None and split.id_manifest_sha256 != self.sha256:
            raise ValueError(
                "manifest SHA-256 does not match registered id_manifest_sha256: "
                f"{self.sha256} != {split.id_manifest_sha256}"
            )


class FrozenSplitManifest(StrictModel):
    """Validated payload stored in a content-addressed freeze artifact."""

    schema_version: Literal["jepa4d-split-manifest-snapshot-v1"]
    manifest_sha256: str
    manifest: SplitManifest

    @field_validator("manifest_sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="manifest_sha256")

    @model_validator(mode="after")
    def embedded_hash_matches(self) -> FrozenSplitManifest:
        if self.manifest_sha256 != self.manifest.sha256:
            raise ValueError("embedded split manifest does not match manifest_sha256")
        return self


class FrozenRegisteredSplitManifest(StrictModel):
    """Content-addressed split snapshot bound to one registry digest."""

    schema_version: Literal["jepa4d-registered-split-manifest-snapshot-v1"]
    registry_sha256: str
    manifest_sha256: str
    manifest: SplitManifest

    @field_validator("registry_sha256", "manifest_sha256")
    @classmethod
    def digest_is_sha256(cls, value: str, info) -> str:
        return _validate_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def embedded_hash_matches(self) -> FrozenRegisteredSplitManifest:
        if self.manifest_sha256 != self.manifest.sha256:
            raise ValueError("embedded split manifest does not match manifest_sha256")
        return self

    def validate_against_registry(self, registry: DatasetRegistry) -> None:
        if self.registry_sha256 != registry.sha256:
            raise ValueError(
                "registered split snapshot names a different registry SHA-256: "
                f"{self.registry_sha256} != {registry.sha256}"
            )
        self.manifest.validate_against_registry(registry)


def verify_cross_manifest_disjointness(manifests: Sequence[SplitManifest]) -> None:
    """Reject shared selected units/clusters across splits of one dataset.

    Rejected candidate rows are deliberately excluded: two independently
    audited selection runs may reject the same candidate.  Actual selected
    membership must remain disjoint, including its independent clusters.
    """

    unit_owners: dict[tuple[str, str], str] = {}
    cluster_owners: dict[tuple[str, str], str] = {}
    physical_unit_owners: dict[tuple[str, str], str] = {}
    conflicts: list[str] = []
    for manifest in manifests:
        selected_unit_ids = {unit.unit_id for unit in manifest.selected_units}
        selected_cluster_ids = {unit.cluster_id for unit in manifest.selected_units}
        selected_physical_ids = {unit.physical_unit_sha256 for unit in manifest.selected_units}
        for unit_id in sorted(selected_unit_ids):
            key = (manifest.dataset_id, unit_id)
            prior = unit_owners.get(key)
            if prior is not None:
                conflicts.append(
                    f"dataset {manifest.dataset_id!r} unit {unit_id!r} is selected by {prior!r} and "
                    f"{manifest.split_id!r}"
                )
            else:
                unit_owners[key] = manifest.split_id
        for cluster_id in sorted(selected_cluster_ids):
            key = (manifest.dataset_id, cluster_id)
            prior = cluster_owners.get(key)
            if prior is not None:
                conflicts.append(
                    f"dataset {manifest.dataset_id!r} cluster {cluster_id!r} is selected by {prior!r} and "
                    f"{manifest.split_id!r}"
                )
            else:
                cluster_owners[key] = manifest.split_id
        for physical_unit_sha256 in sorted(selected_physical_ids):
            key = (manifest.dataset_id, physical_unit_sha256)
            prior = physical_unit_owners.get(key)
            if prior is not None:
                conflicts.append(
                    f"dataset {manifest.dataset_id!r} physical unit {physical_unit_sha256!r} is selected by "
                    f"{prior!r} and {manifest.split_id!r}"
                )
            else:
                physical_unit_owners[key] = manifest.split_id
    if conflicts:
        raise ValueError("cross-manifest disjointness violation: " + "; ".join(conflicts))


def freeze_split_manifest(
    manifest_or_path: SplitManifest | str | Path,
    output_dir: str | Path,
) -> ContentAddress:
    """Write one deterministic, normalized, content-addressed freeze artifact."""

    manifest = (
        manifest_or_path if isinstance(manifest_or_path, SplitManifest) else SplitManifest.load(manifest_or_path)
    )
    frozen = FrozenSplitManifest(
        schema_version="jepa4d-split-manifest-snapshot-v1",
        manifest_sha256=manifest.sha256,
        manifest=manifest,
    )
    return write_content_addressed_json(frozen, output_dir, prefix="split-manifest")


def freeze_registered_split_manifest(
    manifest_or_path: SplitManifest | str | Path,
    registry: DatasetRegistry,
    output_dir: str | Path,
) -> ContentAddress:
    """Freeze a manifest only after validating its one-way registry binding."""

    manifest = (
        manifest_or_path if isinstance(manifest_or_path, SplitManifest) else SplitManifest.load(manifest_or_path)
    )
    manifest.validate_against_registry(registry)
    frozen = FrozenRegisteredSplitManifest(
        schema_version="jepa4d-registered-split-manifest-snapshot-v1",
        registry_sha256=registry.sha256,
        manifest_sha256=manifest.sha256,
        manifest=manifest,
    )
    return write_content_addressed_json(frozen, output_dir, prefix="registered-split-manifest")


def verify_frozen_split_manifest(path: str | Path) -> FrozenSplitManifest:
    """Verify the filename digest, payload schema, and embedded manifest digest."""

    value = verify_content_addressed_json(path, prefix="split-manifest")
    return FrozenSplitManifest.model_validate(value)


def verify_frozen_registered_split_manifest(
    path: str | Path,
    registry: DatasetRegistry,
) -> FrozenRegisteredSplitManifest:
    """Verify content addressing and the exact registry used at freeze time."""

    value = verify_content_addressed_json(path, prefix="registered-split-manifest")
    frozen = FrozenRegisteredSplitManifest.model_validate(value)
    frozen.validate_against_registry(registry)
    return frozen
