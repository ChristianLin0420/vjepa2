from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from jepa4d.validation._content import sha256_file
from jepa4d.validation.access import DatasetAccessController
from jepa4d.validation.geometry_readiness import (
    GeometryGateId,
    GeometryGateStatus,
    GeometryReadinessPack,
    RepositoryState,
    SunHistoricalOverlap,
    TumHistoricalOverlap,
    load_and_validate_geometry_readiness,
)
from jepa4d.validation.ledger import ConsumedTestLedger
from jepa4d.validation.registry import AccessOperation, DatasetRegistry

ROOT = Path(__file__).resolve().parents[2]
PACK_PATH = ROOT / "configs/validation/geometry/phase2_readiness_v1.yaml"


def _pack() -> GeometryReadinessPack:
    return GeometryReadinessPack.load(PACK_PATH)


def test_checked_in_pack_is_bound_and_fail_closed_without_dataset_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_data_root = tmp_path / "dataset-root-must-not-be-resolved"
    forbidden_cache_root = tmp_path / "cache-root-must-not-be-resolved"
    forbidden_diode_root = tmp_path / "diode-root-must-not-be-resolved"
    monkeypatch.setenv("JEPA4D_DATA_ROOT", str(forbidden_data_root))
    monkeypatch.setenv("JEPA4D_CACHE_ROOT", str(forbidden_cache_root))
    monkeypatch.setenv("JEPA4D_DIODE_SEALED_ROOT", str(forbidden_diode_root))

    pack = load_and_validate_geometry_readiness(PACK_PATH, ROOT)
    assert pack.status_by_gate() == {
        "metadata-audit": "audit-ready",
        "consumed-tum-regression": "partial-runtime-implemented",
        "sun-development": "execution-ready",
        "diode-external": "sealed-blocked",
    }
    assert pack.scope == "phase2g-sun-development-authorization"
    authorized = [gate.gate_id for gate in pack.gates if gate.execution_ready or gate.pack_authorizes_data_access]
    assert authorized == [GeometryGateId.SUN_DEVELOPMENT]
    assert not forbidden_data_root.exists()
    assert not forbidden_cache_root.exists()
    assert not forbidden_diode_root.exists()


def test_gate_targets_operations_and_diode_seal_are_exact() -> None:
    pack = _pack()
    gates = {gate.gate_id: gate for gate in pack.gates}
    metadata = gates[GeometryGateId.METADATA_AUDIT]
    assert metadata.status is GeometryGateStatus.AUDIT_READY
    assert metadata.targets == () and metadata.registered_operations == frozenset()

    tum = gates[GeometryGateId.CONSUMED_TUM_REGRESSION]
    assert tum.status is GeometryGateStatus.PARTIAL_RUNTIME_IMPLEMENTED
    assert all(target.ledger_state is not None for target in tum.targets)
    assert {target.ledger_state.value for target in tum.targets if target.ledger_state is not None} == {"consumed"}
    assert {operation.value for operation in tum.registered_operations} == {
        "regression",
        "mechanism-diagnostic",
        "reporting",
    }
    assert "TUM_GOVERNED_RUNTIME_COVERAGE_INCOMPLETE" in {blocker.code for blocker in tum.blockers}

    sun = gates[GeometryGateId.SUN_DEVELOPMENT]
    assert sun.status is GeometryGateStatus.EXECUTION_READY
    assert sun.execution_ready and sun.pack_authorizes_data_access
    assert sun.blockers == ()
    assert {
        target.split_id: None if target.ledger_state is None else target.ledger_state.value for target in sun.targets
    } == {
        "sun-rgbd.phase2e-kv2-test": "consumed",
        "sun-rgbd.phase2f-four-family-development": "consumed",
        "sun-rgbd.phase2g-four-family-development": None,
    }
    assert {operation.value for operation in sun.registered_operations} == {
        "decode-smoke",
        "training",
        "tuning",
        "calibration",
        "checkpoint-selection",
        "development-evaluation",
        "regression",
        "mechanism-diagnostic",
        "reporting",
    }

    registry = DatasetRegistry.load(ROOT / "configs/validation/dataset_registry.yaml")
    sun_entry = registry.dataset("sun-rgbd.geometry-development")
    authorization = sun_entry.license_info.restricted_use_authorization
    assert sun_entry.status.value == "active-development"
    assert sun_entry.readiness_blockers == ()
    assert authorization is not None
    assert authorization.reviewer == "User-authorized project owner, 2026-06-30"
    assert not authorization.standard_license_claimed
    assert not authorization.raw_redistribution_allowed
    _, phase2g = registry.split("sun-rgbd.geometry-development", "sun-rgbd.phase2g-four-family-development")
    assert phase2g.purpose.value == "train"
    assert phase2g.target_state.value == "open"
    assert phase2g.expected_units == 4096
    assert phase2g.id_manifest is None and phase2g.id_manifest_sha256 is None

    diode = gates[GeometryGateId.DIODE_EXTERNAL]
    assert diode.sealed and diode.status is GeometryGateStatus.SEALED_BLOCKED
    assert len(diode.targets) == 1
    assert diode.targets[0].registry_target_state.value == "sealed"
    assert diode.targets[0].ledger_state is not None
    assert diode.targets[0].ledger_state.value == "sealed-unopened"
    assert {blocker.code for blocker in diode.blockers} >= {
        "DIODE_SIGNER_PENDING",
        "APPEND_ONLY_EVENT_STORE_MISSING",
        "DIODE_EXTERNAL_PREREGISTRATION_MISSING",
    }
    assert not diode.execution_ready and not diode.pack_authorizes_data_access


def test_phase2g_non_target_split_is_absent_from_ledger_and_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    registry = DatasetRegistry.load(ROOT / "configs/validation/dataset_registry.yaml")
    ledger = ConsumedTestLedger.load(ROOT / "configs/validation/consumed_test_ledger.yaml")
    split_id = "sun-rgbd.phase2g-four-family-development"
    assert all(
        (target.dataset_id, target.split_id) != ("sun-rgbd.geometry-development", split_id)
        for target in ledger.targets
    )
    controller = DatasetAccessController(registry=registry, ledger=ledger)
    required = {
        AccessOperation.DECODE_SMOKE,
        AccessOperation.TRAINING,
        AccessOperation.TUNING,
        AccessOperation.CALIBRATION,
        AccessOperation.CHECKPOINT_SELECTION,
        AccessOperation.DEVELOPMENT_EVALUATION,
        AccessOperation.MECHANISM_DIAGNOSTIC,
        AccessOperation.REPORTING,
    }
    for operation in required:
        decision = controller.authorize("sun-rgbd.geometry-development", split_id, operation)
        assert decision.authorized and decision.ledger_state is None
        assert decision.grants_data_access is (operation is not AccessOperation.REPORTING)


def test_runtime_binding_is_exact_hash_bound_and_has_no_terminal_receipt() -> None:
    runtime = _pack().bindings.runtime
    assert runtime.scope == "consumed-phase2b-official-smoke"
    assert runtime.terminal_receipt == "not-bound"
    assert [(value.path, value.role) for value in runtime.files] == [
        ("jepa4d/validation/geometry_official_mini.py", "implementation"),
        ("jepa4d/validation/wandb.py", "implementation"),
        ("jepa4d/visualization/validation_dashboard.py", "implementation"),
        ("slurm/geometry_official_mini.sbatch", "implementation"),
        ("slurm/submit_geometry_official_mini.sh", "implementation"),
        ("slurm/validate_geometry_official_mini.py", "implementation"),
        ("jepa4d/tests/test_geometry_official_mini.py", "test"),
        ("jepa4d/tests/test_geometry_official_mini_postflight.py", "test"),
        ("jepa4d/tests/test_geometry_official_mini_slurm.py", "test"),
        ("jepa4d/tests/test_validation_wandb.py", "test"),
        ("jepa4d/tests/test_validation_dashboard.py", "test"),
    ]
    for runtime_binding in runtime.files:
        assert sha256_file(ROOT / runtime_binding.path) == runtime_binding.file_sha256

    phase2g = _pack().bindings.phase2g_runtime
    assert phase2g.scope == "phase2g-formal-sun-development"
    assert phase2g.status == "hash-bound-complete"
    assert not phase2g.final_hash_binding_required
    assert len(phase2g.files) == 57
    for phase2g_binding in phase2g.files:
        assert sha256_file(ROOT / phase2g_binding.path) == phase2g_binding.file_sha256


def test_authorized_preregistration_binding_is_exact() -> None:
    preregistration = _pack().bindings.preregistration
    assert preregistration.status == "preregistered-authorized"
    assert preregistration.path == "docs/experiments/2026-06-30-phase2g-quality-first-preregistered.md"
    assert sha256_file(ROOT / preregistration.path) == preregistration.file_sha256


def test_historical_overlap_is_typed_exact_and_never_fresh() -> None:
    pack = _pack()
    sun, tum = pack.historical_overlaps
    assert isinstance(sun, SunHistoricalOverlap)
    assert isinstance(tum, TumHistoricalOverlap)
    assert not sun.supports_fresh_external_claim
    assert {value.family: value.exact_overlap_count for value in sun.families} == {
        "kv1": 128,
        "kv2": 128,
        "realsense": 128,
        "xtion": 128,
    }
    assert sun.phase2f_selection_sha256 == "01a8c4577289034db86b63c4f6e9eaef9afd7aa636a9d952e9878dae3758bca6"

    assert not tum.supports_fresh_external_claim
    assert tum.exact_overlap_count == 8
    assert {value.phase2b_role: value.frame_indices_reused_by_phase2c_train for value in tum.prior_roles} == {
        "train": (35, 172, 248, 384, 483),
        "validation": (520, 631),
        "test": (779,),
    }


def test_clean_clone_audit_distinguishes_tracked_legacy_from_ignored_receipt() -> None:
    pack = _pack()
    gaps = {(gap.dataset_id, gap.split_id): gap for gap in pack.legacy_manifest_gaps}
    phase2f = gaps[("sun-rgbd.geometry-development", "sun-rgbd.phase2f-four-family-development")]
    assert phase2f.repository_state is RepositoryState.IGNORED_LOCAL_RECEIPT
    assert not phase2f.clean_clone_available
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", phase2f.registered_path],
        check=False,
        capture_output=True,
    )
    assert tracked.returncode != 0
    for key, gap in gaps.items():
        assert not gap.blocks_current_authorization
        if key == ("sun-rgbd.geometry-development", "sun-rgbd.phase2f-four-family-development"):
            continue
        assert gap.repository_state is RepositoryState.TRACKED_LEGACY
        assert gap.clean_clone_available
        assert (ROOT / gap.registered_path).is_file()


def test_stale_core_binding_fails_repository_validation() -> None:
    value = _pack().model_dump(mode="json")
    value["bindings"]["registry"]["file_sha256"] = "0" * 64
    stale = GeometryReadinessPack.model_validate(value)
    with pytest.raises(ValueError, match="repository metadata SHA-256 mismatch"):
        stale.validate_repository(ROOT)


def test_runtime_binding_cannot_omit_a_canonical_file() -> None:
    value = _pack().model_dump(mode="json")
    value["bindings"]["runtime"]["files"].pop()
    with pytest.raises(ValidationError, match="exact canonical implementation and test files in order"):
        GeometryReadinessPack.model_validate(value)


def test_stale_runtime_binding_fails_repository_validation() -> None:
    value = _pack().model_dump(mode="json")
    value["bindings"]["runtime"]["files"][0]["file_sha256"] = "0" * 64
    stale = GeometryReadinessPack.model_validate(value)
    with pytest.raises(ValueError, match="repository metadata SHA-256 mismatch"):
        stale.validate_repository(ROOT)


def test_pack_cannot_claim_execution_or_unsealed_diode() -> None:
    execution_value = _pack().model_dump(mode="json")
    execution_value["gates"][1]["execution_ready"] = True
    with pytest.raises(ValidationError, match="only the execution-ready SUN development gate"):
        GeometryReadinessPack.model_validate(execution_value)

    missing_sun_authority = _pack().model_dump(mode="json")
    missing_sun_authority["gates"][2]["execution_ready"] = False
    with pytest.raises(ValidationError, match="only a blocker-free SUN development gate"):
        GeometryReadinessPack.model_validate(missing_sun_authority)

    unsealed_value = _pack().model_dump(mode="json")
    unsealed_value["gates"][3]["sealed"] = False
    with pytest.raises(ValidationError, match="only the DIODE external gate may be marked sealed"):
        GeometryReadinessPack.model_validate(unsealed_value)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-readiness.yaml"
    duplicate.write_text(PACK_PATH.read_text(encoding="utf-8") + "pack_version: shadow-v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key 'pack_version'"):
        GeometryReadinessPack.load(duplicate)


def test_checked_in_json_schema_matches_runtime_model() -> None:
    schema_path = ROOT / "configs/validation/schemas/geometry-readiness.schema.json"
    assert json.loads(schema_path.read_text(encoding="utf-8")) == GeometryReadinessPack.model_json_schema()
