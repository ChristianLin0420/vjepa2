"""Scoped execution-readiness contract for the governed Phase 2 geometry portfolio.

This module never resolves dataset roots or reads dataset, cache, prediction, or
target artifacts. It validates the registry, ledger, authorization records,
preregistration, and exact source/test files behind the partial consumed-regression
runtime. It is not an access-control bypass: dataset access continues to require
the registry/ledger controller and the separately guarded formal preflight.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from jepa4d.validation._content import load_yaml_unique, sha256_file
from jepa4d.validation.ledger import ConsumedTestLedger, LedgerState
from jepa4d.validation.registry import (
    SHA256_PATTERN,
    TARGET_SPLITS,
    AccessOperation,
    DatasetRegistry,
    StrictModel,
    TargetState,
)

GEOMETRY_READINESS_SCHEMA_VERSION = "jepa4d-phase2-geometry-readiness-v1"
_SAFE_PATH_PATTERN = r"^[^/\x00][^\x00]*$"
_RUNTIME_FILE_ROLES = {
    "jepa4d/validation/geometry_official_mini.py": "implementation",
    "jepa4d/validation/wandb.py": "implementation",
    "jepa4d/visualization/validation_dashboard.py": "implementation",
    "slurm/geometry_official_mini.sbatch": "implementation",
    "slurm/submit_geometry_official_mini.sh": "implementation",
    "slurm/validate_geometry_official_mini.py": "implementation",
    "jepa4d/tests/test_geometry_official_mini.py": "test",
    "jepa4d/tests/test_geometry_official_mini_postflight.py": "test",
    "jepa4d/tests/test_geometry_official_mini_slurm.py": "test",
    "jepa4d/tests/test_validation_wandb.py": "test",
    "jepa4d/tests/test_validation_dashboard.py": "test",
}
_PHASE2G_RUNTIME_PATHS = (
    "jepa4d/evaluation/phase2g_data.py",
    "jepa4d/evaluation/phase2g_metrics.py",
    "jepa4d/evaluation/phase2g_visualization.py",
    "jepa4d/training/phase2g_protocol.py",
    "jepa4d/training/phase2g_runtime.py",
    "jepa4d/training/phase2g_training.py",
    "jepa4d/tests/test_phase2g_formal_core.py",
    "jepa4d/tests/test_phase2g_formal_slurm.py",
    "jepa4d/tests/test_phase2g_formal_postflight.py",
    "scripts/build_phase2g_data_cache.py",
    "scripts/audit_phase2g_formal.py",
    "scripts/run_phase2g_tuning.py",
    "scripts/select_phase2g_learning_rates.py",
    "scripts/run_phase2g_formal_training.py",
    "scripts/evaluate_phase2g_heldout.py",
    "scripts/select_phase2g_survivor.py",
    "scripts/write_phase2g_dependency_graph.py",
    "slurm/phase2g_contract.py",
    "slurm/phase2g_preflight.py",
    "slurm/phase2g_stage_gate.py",
    "slurm/phase2g_test_runner.py",
    "slurm/phase2g_postflight.py",
    "slurm/submit_phase2g.sh",
    "slurm/phase2g_tests.sbatch",
    "slurm/phase2g_opacity.sbatch",
    "slurm/phase2g_cache.sbatch",
    "slurm/phase2g_audit.sbatch",
    "slurm/phase2g_array_dispatch.sbatch",
    "slurm/phase2g_tune.sbatch",
    "slurm/phase2g_lr_select.sbatch",
    "slurm/phase2g_train.sbatch",
    "slurm/phase2g_evaluate.sbatch",
    "slurm/phase2g_select.sbatch",
    "slurm/phase2g_external_guard.sbatch",
    "slurm/phase2g_postflight.sbatch",
    "scripts/materialize_phase2g_sun.py",
    "slurm/lib.sh",
    "scripts/check_cuda.py",
    "jepa4d/benchmarks/geometry/sun_rgbd.py",
    "jepa4d/data/camera_geometry.py",
    "jepa4d/data/rgb_input.py",
    "jepa4d/data/schemas.py",
    "jepa4d/data/transforms.py",
    "jepa4d/evaluation/phase2e_feature_cache.py",
    "jepa4d/evaluation/phase2f_camera_controls.py",
    "jepa4d/evaluation/phase2f_data_cache.py",
    "jepa4d/evaluation/phase2f_metrics.py",
    "jepa4d/models/geometry_student.py",
    "jepa4d/models/phase2f_scale_geometry.py",
    "jepa4d/models/vjepa21_adapter.py",
    "jepa4d/training/phase2f_losses.py",
    "jepa4d/training/phase2f_training.py",
    "jepa4d/validation/_content.py",
    "jepa4d/validation/access.py",
    "jepa4d/validation/geometry_readiness.py",
    "jepa4d/validation/ledger.py",
    "jepa4d/validation/registry.py",
)


def _validate_relative_repository_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("repository metadata paths must be safe non-empty relative paths")
    return path.as_posix()


class GeometryGateId(StrEnum):
    METADATA_AUDIT = "metadata-audit"
    CONSUMED_TUM_REGRESSION = "consumed-tum-regression"
    SUN_DEVELOPMENT = "sun-development"
    DIODE_EXTERNAL = "diode-external"


class GeometryGateStatus(StrEnum):
    AUDIT_READY = "audit-ready"
    PARTIAL_RUNTIME_IMPLEMENTED = "partial-runtime-implemented"
    POLICY_BLOCKED = "policy-blocked"
    SEALED_BLOCKED = "sealed-blocked"
    EXECUTION_READY = "execution-ready"


class BlockerCategory(StrEnum):
    LEGAL = "legal"
    GOVERNANCE = "governance"
    MANIFEST = "manifest"
    IMPLEMENTATION = "implementation"
    PREREGISTRATION = "preregistration"


class RepositoryState(StrEnum):
    TRACKED_LEGACY = "tracked-legacy"
    IGNORED_LOCAL_RECEIPT = "ignored-local-receipt"


class RegistryBinding(StrictModel):
    path: str = Field(pattern=_SAFE_PATH_PATTERN)
    file_sha256: str
    semantic_sha256: str
    schema_version: Literal["jepa4d-validation-registry-v1"]
    portfolio_version: str

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("file_sha256", "semantic_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("registry binding hashes must be lowercase SHA-256")
        return value


class LedgerBinding(StrictModel):
    path: str = Field(pattern=_SAFE_PATH_PATTERN)
    file_sha256: str
    semantic_sha256: str
    schema_version: Literal["jepa4d-consumed-test-ledger-v1"]
    ledger_version: str
    event_store_sha256: str
    durability: Literal["local-filesystem-best-effort", "external-append-only"]
    externally_append_only: bool

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("file_sha256", "semantic_sha256", "event_store_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("ledger binding hashes must be lowercase SHA-256")
        return value


class SpecificationBinding(StrictModel):
    path: str = Field(pattern=_SAFE_PATH_PATTERN)
    file_sha256: str
    status: Literal["baseline-with-authorized-preregistration"]

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("file_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("specification binding hash must be lowercase SHA-256")
        return value


class PreregistrationBinding(StrictModel):
    path: str = Field(pattern=_SAFE_PATH_PATTERN)
    file_sha256: str
    status: Literal["preregistered-authorized"]

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("file_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("preregistration binding hash must be lowercase SHA-256")
        return value


class RuntimeFileBinding(StrictModel):
    path: str = Field(pattern=_SAFE_PATH_PATTERN)
    file_sha256: str
    role: Literal["implementation", "test"]

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("file_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("runtime file binding must be lowercase SHA-256")
        return value


class RuntimeBinding(StrictModel):
    scope: Literal["consumed-phase2b-official-smoke"]
    terminal_receipt: Literal["not-bound"]
    files: tuple[RuntimeFileBinding, ...]

    @model_validator(mode="after")
    def file_set_is_exact(self) -> RuntimeBinding:
        observed = [(value.path, value.role) for value in self.files]
        if observed != list(_RUNTIME_FILE_ROLES.items()):
            raise ValueError("runtime binding must cover the exact canonical implementation and test files in order")
        return self


class Phase2gRuntimeFileBinding(StrictModel):
    path: str = Field(pattern=_SAFE_PATH_PATTERN)
    file_sha256: str

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("file_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("Phase 2g runtime file binding must be a lowercase SHA-256")
        return value


class Phase2gRuntimeBinding(StrictModel):
    """Complete formal Phase 2g implementation, test, and orchestration identity."""

    scope: Literal["phase2g-formal-sun-development"]
    status: Literal["hash-bound-complete"]
    final_hash_binding_required: Literal[False]
    files: tuple[Phase2gRuntimeFileBinding, ...]

    @model_validator(mode="after")
    def file_set_is_exact(self) -> Phase2gRuntimeBinding:
        observed = tuple(value.path for value in self.files)
        if observed != _PHASE2G_RUNTIME_PATHS:
            raise ValueError("Phase 2g runtime binding must cover the exact canonical file inventory in order")
        return self


class CoreBindings(StrictModel):
    registry: RegistryBinding
    ledger: LedgerBinding
    specification: SpecificationBinding
    preregistration: PreregistrationBinding
    runtime: RuntimeBinding
    phase2g_runtime: Phase2gRuntimeBinding


class GateBlocker(StrictModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    category: BlockerCategory
    description: str = Field(min_length=1)


class GateTarget(StrictModel):
    dataset_id: str
    split_id: str
    registry_target_state: TargetState
    ledger_state: LedgerState | None


class GeometryGate(StrictModel):
    gate_id: GeometryGateId
    status: GeometryGateStatus
    metadata_ready: bool
    execution_ready: bool
    pack_authorizes_data_access: bool
    sealed: bool
    targets: tuple[GateTarget, ...] = ()
    registered_operations: frozenset[AccessOperation] = frozenset()
    blockers: tuple[GateBlocker, ...] = ()
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def status_is_fail_closed(self) -> GeometryGate:
        if self.status is GeometryGateStatus.EXECUTION_READY:
            if (
                self.gate_id is not GeometryGateId.SUN_DEVELOPMENT
                or not self.metadata_ready
                or not self.execution_ready
                or not self.pack_authorizes_data_access
                or self.blockers
                or self.sealed
            ):
                raise ValueError("only a blocker-free SUN development gate may authorize execution and data access")
        elif self.execution_ready or self.pack_authorizes_data_access:
            raise ValueError("only the execution-ready SUN development gate may authorize execution or data access")
        elif self.status is GeometryGateStatus.AUDIT_READY:
            if not self.metadata_ready or self.blockers:
                raise ValueError("audit-ready gates require metadata_ready=true and no blockers")
        elif not self.blockers:
            raise ValueError("blocked geometry gates require explicit typed blockers")
        if self.gate_id is GeometryGateId.METADATA_AUDIT:
            if self.targets or self.registered_operations or self.sealed:
                raise ValueError("metadata-audit is a repository-only gate without dataset target authority")
        elif not self.targets or not self.registered_operations:
            raise ValueError("dataset gates require exact targets and registered operations")
        if self.sealed != (self.gate_id is GeometryGateId.DIODE_EXTERNAL):
            raise ValueError("only the DIODE external gate may be marked sealed")
        codes = [blocker.code for blocker in self.blockers]
        if len(codes) != len(set(codes)):
            raise ValueError("gate blocker codes must be unique")
        return self


class LegacyManifestGap(StrictModel):
    dataset_id: str
    split_id: str
    registered_path: str = Field(pattern=_SAFE_PATH_PATTERN)
    registered_sha256: str
    repository_state: RepositoryState
    clean_clone_available: bool
    legacy_schema: str = Field(min_length=1)
    scope_gap: str = Field(min_length=1)
    migration_required: Literal[True]
    blocks_current_authorization: Literal[False]

    @field_validator("registered_path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_relative_repository_path(value)

    @field_validator("registered_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("legacy registered_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def repository_state_is_coherent(self) -> LegacyManifestGap:
        expected_available = self.repository_state is RepositoryState.TRACKED_LEGACY
        if self.clean_clone_available != expected_available:
            raise ValueError("clean_clone_available conflicts with repository_state")
        return self


class SunFamilyOverlap(StrictModel):
    family: Literal["kv1", "kv2", "realsense", "xtion"]
    phase2e_selected_count: int = Field(ge=1)
    phase2f_selected_count: int = Field(ge=1)
    exact_overlap_count: int = Field(ge=1)
    relationship: Literal["all-phase2e-reused", "phase2f-subset-of-phase2e"]

    @model_validator(mode="after")
    def counts_match_relationship(self) -> SunFamilyOverlap:
        if self.exact_overlap_count != self.phase2f_selected_count:
            raise ValueError("Phase 2f SUN membership must be fully consumed overlap")
        if self.relationship == "all-phase2e-reused":
            if self.phase2e_selected_count != self.phase2f_selected_count:
                raise ValueError("all-phase2e-reused requires equal Phase 2e/2f counts")
        elif self.phase2f_selected_count >= self.phase2e_selected_count:
            raise ValueError("phase2f-subset-of-phase2e requires a strict subset")
        return self


class SunHistoricalOverlap(StrictModel):
    kind: Literal["sun-family-membership"]
    dataset_id: Literal["sun-rgbd.geometry-development"]
    phase2e_manifest_sha256: str
    phase2f_selection_sha256: str
    exactness: Literal["historical-assertion-selection-hash-not-clean-clone-verifiable"]
    families: tuple[SunFamilyOverlap, ...]
    supports_fresh_external_claim: Literal[False]
    limitation: str = Field(min_length=1)

    @field_validator("phase2e_manifest_sha256", "phase2f_selection_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("SUN overlap hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def all_families_are_exactly_recorded(self) -> SunHistoricalOverlap:
        family_ids = [value.family for value in self.families]
        if set(family_ids) != {"kv1", "kv2", "realsense", "xtion"} or len(family_ids) != 4:
            raise ValueError("SUN overlap must record each of the four families exactly once")
        return self


class TumPriorRoleOverlap(StrictModel):
    phase2b_role: Literal["train", "validation", "test"]
    frame_indices_reused_by_phase2c_train: tuple[int, ...] = Field(min_length=1)

    @field_validator("frame_indices_reused_by_phase2c_train")
    @classmethod
    def indices_are_sorted_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("TUM overlap frame indices must be sorted and unique")
        return value


class TumHistoricalOverlap(StrictModel):
    kind: Literal["tum-frame-index"]
    dataset_id: Literal["tum-rgbd.geometry-regression"]
    recording: Literal["freiburg1_xyz"]
    phase2b_manifest_sha256: str
    phase2c_manifest_sha256: str
    reused_by_phase2c_role: Literal["train"]
    prior_roles: tuple[TumPriorRoleOverlap, ...]
    exact_overlap_count: int = Field(ge=1)
    supports_fresh_external_claim: Literal[False]

    @field_validator("phase2b_manifest_sha256", "phase2c_manifest_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("TUM overlap hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def overlap_is_complete_and_disjoint(self) -> TumHistoricalOverlap:
        roles = [value.phase2b_role for value in self.prior_roles]
        if roles != ["train", "validation", "test"]:
            raise ValueError("TUM overlap must record train, validation, and test roles in order")
        indices = [index for value in self.prior_roles for index in value.frame_indices_reused_by_phase2c_train]
        if len(indices) != len(set(indices)) or len(indices) != self.exact_overlap_count:
            raise ValueError("TUM exact_overlap_count must equal the disjoint exact frame-index union")
        return self


HistoricalOverlap = Annotated[SunHistoricalOverlap | TumHistoricalOverlap, Field(discriminator="kind")]


class ClaimBoundary(StrictModel):
    supported: tuple[str, ...] = Field(min_length=1)
    prohibited: tuple[str, ...] = Field(min_length=1)


class GeometryReadinessPack(StrictModel):
    schema_version: Literal["jepa4d-phase2-geometry-readiness-v1"]
    pack_version: str
    scope: Literal["phase2g-sun-development-authorization"]
    bindings: CoreBindings
    gates: tuple[GeometryGate, ...]
    legacy_manifest_gaps: tuple[LegacyManifestGap, ...]
    historical_overlaps: tuple[HistoricalOverlap, ...]
    claim_boundary: ClaimBoundary

    @model_validator(mode="after")
    def portfolio_shape_is_exact(self) -> GeometryReadinessPack:
        gate_ids = [gate.gate_id for gate in self.gates]
        if gate_ids != list(GeometryGateId):
            raise ValueError("geometry gates must contain the four canonical gates in canonical order")
        authorized = [
            gate
            for gate in self.gates
            if gate.execution_ready
            or gate.pack_authorizes_data_access
            or gate.status is GeometryGateStatus.EXECUTION_READY
        ]
        if len(authorized) != 1 or authorized[0].gate_id is not GeometryGateId.SUN_DEVELOPMENT:
            raise ValueError("exactly the SUN development gate must carry Phase 2g execution authorization")
        gap_keys = [(gap.dataset_id, gap.split_id) for gap in self.legacy_manifest_gaps]
        expected_gap_keys = {
            ("sun-rgbd.geometry-development", "sun-rgbd.phase2e-kv2-test"),
            ("sun-rgbd.geometry-development", "sun-rgbd.phase2f-four-family-development"),
            ("tum-rgbd.geometry-regression", "tum-rgbd.phase2b-freiburg1-xyz-test"),
            ("tum-rgbd.geometry-regression", "tum-rgbd.phase2c-freiburg3-test"),
        }
        if set(gap_keys) != expected_gap_keys or len(gap_keys) != len(expected_gap_keys):
            raise ValueError("legacy manifest gaps must cover each registered SUN/TUM geometry split exactly once")
        overlap_kinds = [overlap.kind for overlap in self.historical_overlaps]
        if overlap_kinds != ["sun-family-membership", "tum-frame-index"]:
            raise ValueError("historical overlaps must contain canonical SUN then TUM records")
        return self

    @classmethod
    def load(cls, path: str | Path) -> GeometryReadinessPack:
        return cls.model_validate(load_yaml_unique(path))

    def validate_repository(self, repository_root: str | Path) -> None:
        """Verify checked-in authorization metadata and source without resolving data roots."""
        root = Path(repository_root).resolve(strict=True)
        registry_path = _resolve_repository_path(root, self.bindings.registry.path)
        ledger_path = _resolve_repository_path(root, self.bindings.ledger.path)
        specification_path = _resolve_repository_path(root, self.bindings.specification.path)
        preregistration_path = _resolve_repository_path(root, self.bindings.preregistration.path)
        _require_file_hash(registry_path, self.bindings.registry.file_sha256)
        _require_file_hash(ledger_path, self.bindings.ledger.file_sha256)
        _require_file_hash(specification_path, self.bindings.specification.file_sha256)
        _require_file_hash(preregistration_path, self.bindings.preregistration.file_sha256)

        registry = DatasetRegistry.load(registry_path)
        ledger = ConsumedTestLedger.load(ledger_path)
        ledger.validate_against(registry)
        if (
            registry.sha256 != self.bindings.registry.semantic_sha256
            or registry.schema_version != self.bindings.registry.schema_version
            or registry.portfolio_version != self.bindings.registry.portfolio_version
        ):
            raise ValueError("geometry readiness registry binding is stale")
        if (
            ledger.sha256 != self.bindings.ledger.semantic_sha256
            or ledger.schema_version != self.bindings.ledger.schema_version
            or ledger.ledger_version != self.bindings.ledger.ledger_version
            or ledger.event_store.sha256 != self.bindings.ledger.event_store_sha256
            or ledger.event_store.durability != self.bindings.ledger.durability
            or ledger.event_store.externally_append_only != self.bindings.ledger.externally_append_only
        ):
            raise ValueError("geometry readiness ledger binding is stale")

        tracked_paths = _git_tracked_paths(root)
        for path in (
            self.bindings.registry.path,
            self.bindings.ledger.path,
            self.bindings.specification.path,
        ):
            if path not in tracked_paths:
                raise ValueError(f"core geometry metadata is absent from a clean clone: {path}")
        for binding in self.bindings.runtime.files:
            if binding.path not in tracked_paths:
                raise ValueError(f"bound geometry runtime file is absent from a clean clone: {binding.path}")
            _require_file_hash(_resolve_repository_path(root, binding.path), binding.file_sha256)
        for phase2g_binding in self.bindings.phase2g_runtime.files:
            _require_file_hash(_resolve_repository_path(root, phase2g_binding.path), phase2g_binding.file_sha256)
        for gap in self.legacy_manifest_gaps:
            _, split = registry.split(gap.dataset_id, gap.split_id)
            if (split.id_manifest, split.id_manifest_sha256) != (gap.registered_path, gap.registered_sha256):
                raise ValueError(f"legacy manifest gap is stale for {gap.dataset_id}/{gap.split_id}")
            is_tracked = gap.registered_path in tracked_paths
            if gap.repository_state is RepositoryState.TRACKED_LEGACY:
                if not is_tracked:
                    raise ValueError(f"required legacy manifest is absent from clean clone: {gap.registered_path}")
                _require_file_hash(_resolve_repository_path(root, gap.registered_path), gap.registered_sha256)
            elif is_tracked:
                raise ValueError(
                    f"local-only receipt is now tracked; update its migration status: {gap.registered_path}"
                )

        gates = {gate.gate_id: gate for gate in self.gates}
        _validate_dataset_gate(
            registry,
            ledger,
            gates[GeometryGateId.CONSUMED_TUM_REGRESSION],
            expected_dataset="tum-rgbd.geometry-regression",
        )
        _validate_dataset_gate(
            registry,
            ledger,
            gates[GeometryGateId.SUN_DEVELOPMENT],
            expected_dataset="sun-rgbd.geometry-development",
        )
        _validate_dataset_gate(
            registry,
            ledger,
            gates[GeometryGateId.DIODE_EXTERNAL],
            expected_dataset="diode.geometry-external",
        )
        if registry.dataset("tum-rgbd.geometry-regression").readiness_blockers:
            raise ValueError("TUM consumed regression metadata unexpectedly has registry audit blockers")
        sun = registry.dataset("sun-rgbd.geometry-development")
        if sun.readiness_blockers:
            raise ValueError("SUN restricted-use registry audit unexpectedly retains blockers")
        sun_license = sun.license_info
        sun_authorization = sun_license.restricted_use_authorization
        if (
            sun_license.name is not None
            or sun_license.terms_url is not None
            or sun_license.redistribution != "prohibited"
            or sun_authorization is None
            or sun_authorization.scope != "internal-research-only"
            or sun_authorization.standard_license_claimed
            or not sun_authorization.official_citation_required
            or sun_authorization.raw_redistribution_allowed
        ):
            raise ValueError("SUN restricted-use approval boundary differs from the bound governance decision")
        diode_blockers = registry.dataset("diode.geometry-external").readiness_blockers
        if not any(value.startswith("sealed_authority:pending:") for value in diode_blockers):
            raise ValueError("DIODE signer blocker is missing from the bound registry")
        if ledger.event_store.externally_append_only:
            raise ValueError("this pack is stale: it records the current non-append-only DIODE blocker")
        self._validate_tum_overlap_from_tracked_manifests(root)

    def _validate_tum_overlap_from_tracked_manifests(self, root: Path) -> None:
        gaps = {(gap.dataset_id, gap.split_id): gap for gap in self.legacy_manifest_gaps}
        phase2b_gap = gaps[("tum-rgbd.geometry-regression", "tum-rgbd.phase2b-freiburg1-xyz-test")]
        phase2c_gap = gaps[("tum-rgbd.geometry-regression", "tum-rgbd.phase2c-freiburg3-test")]
        phase2b = load_yaml_unique(_resolve_repository_path(root, phase2b_gap.registered_path))
        phase2c = load_yaml_unique(_resolve_repository_path(root, phase2c_gap.registered_path))
        if not isinstance(phase2b, Mapping) or not isinstance(phase2c, Mapping):
            raise ValueError("tracked TUM legacy manifests must be mappings")
        sequences = phase2c.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError("Phase 2c legacy manifest lacks its sequence rows")
        source_rows = [
            value for value in sequences if isinstance(value, Mapping) and value.get("sequence_id") == "freiburg1_xyz"
        ]
        if len(source_rows) != 1 or not isinstance(source_rows[0].get("selected_indices"), list):
            raise ValueError("Phase 2c legacy manifest lacks one Freiburg1 XYZ selection")
        phase2c_train = {int(value) for value in source_rows[0]["selected_indices"]}
        overlap = next(value for value in self.historical_overlaps if isinstance(value, TumHistoricalOverlap))
        recomputed: dict[str, tuple[int, ...]] = {}
        for role in ("train", "validation", "test"):
            values = phase2b.get(f"{role}_indices")
            if not isinstance(values, list):
                raise ValueError(f"Phase 2b legacy manifest lacks {role}_indices")
            recomputed[role] = tuple(sorted(phase2c_train & {int(value) for value in values}))
        recorded = {value.phase2b_role: value.frame_indices_reused_by_phase2c_train for value in overlap.prior_roles}
        if recomputed != recorded:
            raise ValueError("recorded TUM historical overlap differs from the tracked hashed manifests")

    def status_by_gate(self) -> dict[str, str]:
        return {gate.gate_id.value: gate.status.value for gate in self.gates}


def _resolve_repository_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"repository metadata path escapes root: {relative_path}")
    return path


def _require_file_hash(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"required repository metadata file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"repository metadata SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def _git_tracked_paths(root: Path) -> frozenset[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"could not audit clean-clone artifacts with git ls-files: {message}")
    return frozenset(value.decode("utf-8") for value in result.stdout.split(b"\0") if value)


def _validate_dataset_gate(
    registry: DatasetRegistry,
    ledger: ConsumedTestLedger,
    gate: GeometryGate,
    *,
    expected_dataset: str,
) -> None:
    dataset_ids = {target.dataset_id for target in gate.targets}
    if dataset_ids != {expected_dataset}:
        raise ValueError(f"geometry gate {gate.gate_id.value} must bind only {expected_dataset}")
    entry = registry.dataset(expected_dataset)
    expected_splits = {split.split_id for split in entry.splits}
    observed_splits = {target.split_id for target in gate.targets}
    if observed_splits != expected_splits or len(gate.targets) != len(expected_splits):
        raise ValueError(f"geometry gate {gate.gate_id.value} does not bind every registered split exactly once")
    expected_operations: set[AccessOperation] = set()
    ledger_targets = {(target.dataset_id, target.split_id): target for target in ledger.targets}
    for target in gate.targets:
        _, split = registry.split(target.dataset_id, target.split_id)
        key = (target.dataset_id, target.split_id)
        if target.registry_target_state is not split.target_state:
            raise ValueError(f"geometry gate target state is stale for {target.dataset_id}/{target.split_id}")
        if split.purpose in TARGET_SPLITS:
            ledger_target = ledger_targets.get(key)
            if ledger_target is None or target.ledger_state is not ledger_target.state:
                raise ValueError(f"geometry gate ledger state is stale for {target.dataset_id}/{target.split_id}")
        elif key in ledger_targets or target.ledger_state is not None:
            raise ValueError(
                f"non-target split must remain absent from the ledger: {target.dataset_id}/{target.split_id}"
            )
        expected_operations.update(split.allowed_operations)
    if gate.registered_operations != frozenset(expected_operations):
        raise ValueError(f"geometry gate registered operations are stale for {expected_dataset}")


def load_and_validate_geometry_readiness(
    path: str | Path,
    repository_root: str | Path,
) -> GeometryReadinessPack:
    pack = GeometryReadinessPack.load(path)
    pack.validate_repository(repository_root)
    return pack


__all__ = [
    "GEOMETRY_READINESS_SCHEMA_VERSION",
    "GeometryGateId",
    "GeometryGateStatus",
    "GeometryReadinessPack",
    "SunHistoricalOverlap",
    "TumHistoricalOverlap",
    "load_and_validate_geometry_readiness",
]
