#!/usr/bin/env python3
"""Fail-closed metadata and authorization preflight for formal Phase 2g-A."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jepa4d.validation._content import load_yaml_unique
from jepa4d.validation.access import DatasetAccessController
from jepa4d.validation.geometry_readiness import load_and_validate_geometry_readiness
from jepa4d.validation.ledger import ConsumedTestLedger
from jepa4d.validation.registry import AccessOperation, DatasetRegistry
from slurm.phase2g_contract import PREFLIGHT_SCHEMA, atomic_json, file_identity, sha256_file

SUN_DATASET_ID = "sun-rgbd.geometry-development"
SUN_SPLIT_ID = "sun-rgbd.phase2g-four-family-development"
EXPECTED_ARCHIVE_SHA256 = "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c"
REQUIRED_RUNNERS = (
    "scripts/materialize_phase2g_sun.py",
    "scripts/build_phase2g_data_cache.py",
    "scripts/audit_phase2g_formal.py",
    "scripts/run_phase2g_tuning.py",
    "scripts/select_phase2g_learning_rates.py",
    "scripts/run_phase2g_formal_training.py",
    "scripts/evaluate_phase2g_heldout.py",
    "scripts/select_phase2g_survivor.py",
)
TRANSITIVE_RUNTIME_PATHS = (
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


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _require_clean_pushed(root: Path) -> dict[str, str]:
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    if not branch:
        raise ValueError("formal Phase 2g submission forbids detached HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    try:
        upstream = _git(root, "rev-parse", "@{u}")
        upstream_name = _git(root, "rev-parse", "--abbrev-ref", "@{u}")
    except subprocess.CalledProcessError as error:
        raise ValueError("formal Phase 2g requires a configured pushed upstream") from error
    if status:
        raise ValueError("formal Phase 2g requires a clean committed worktree")
    if commit != upstream:
        raise ValueError("formal Phase 2g HEAD must equal its pushed upstream")
    return {"commit": commit, "branch": branch, "upstream": upstream_name}


def _require_tracked(root: Path, path: Path, label: str) -> None:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the repository") from error
    result = subprocess.run(
        ("git", "-C", str(root), "ls-files", "--error-unmatch", relative.as_posix()),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} is not available in the pushed clean clone")


def _validate_preregistration(path: Path, root: Path) -> dict[str, Any]:
    _require_tracked(root, path, "Phase 2g preregistration")
    text = path.read_text(encoding="utf-8")
    lowered = text.casefold()
    if "phase 2g-a" not in lowered or "preregistered" not in lowered:
        raise ValueError("Phase 2g preregistration identity is missing")
    if "frozen and authorized for internal-research execution on sun rgb-d" not in lowered:
        raise ValueError("Phase 2g preregistration does not explicitly authorize this SUN-only execution")
    forbidden = ("proposed, not preregistered", "not authorized for execution")
    if any(marker in lowered for marker in forbidden):
        raise ValueError("Phase 2g preregistration still contains a non-authorization status")
    markers = (
        "1,024 eligible",
        "5e-4",
        "1e-3",
        "2e-3",
        "152 logical tasks",
        "external_final_authorized=false",
        EXPECTED_ARCHIVE_SHA256,
    )
    missing = [marker for marker in markers if marker.casefold() not in lowered]
    if missing:
        raise ValueError(f"Phase 2g preregistration lacks frozen markers: {missing}")
    return file_identity(path)


def _raw_gate(readiness: dict[str, Any], gate_id: str) -> dict[str, Any]:
    gates = readiness.get("gates")
    if not isinstance(gates, list):
        raise ValueError("geometry readiness lacks gates")
    matches = [gate for gate in gates if isinstance(gate, dict) and gate.get("gate_id") == gate_id]
    if len(matches) != 1:
        raise ValueError(f"geometry readiness lacks one {gate_id} gate")
    return matches[0]


def _validate_readiness(
    path: Path,
    root: Path,
    preregistration: Path,
    registry_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    _require_tracked(root, path, "geometry readiness pack")
    pack = load_and_validate_geometry_readiness(path, root)
    value = load_yaml_unique(path)
    if not isinstance(value, dict):
        raise ValueError("geometry readiness pack must be a mapping")
    if value.get("scope") != "phase2g-sun-development-authorization":
        raise ValueError("geometry readiness scope does not authorize Phase 2g SUN development")
    bindings = value.get("bindings")
    prereg_binding = bindings.get("preregistration") if isinstance(bindings, dict) else None
    registry_binding = bindings.get("registry") if isinstance(bindings, dict) else None
    ledger_binding = bindings.get("ledger") if isinstance(bindings, dict) else None
    runtime_binding = bindings.get("phase2g_runtime") if isinstance(bindings, dict) else None
    expected_relative = preregistration.resolve(strict=True).relative_to(root).as_posix()
    if (
        not isinstance(prereg_binding, dict)
        or prereg_binding.get("path") != expected_relative
        or prereg_binding.get("file_sha256") != sha256_file(preregistration)
        or prereg_binding.get("status") != "preregistered-authorized"
    ):
        raise ValueError("geometry readiness is not hash-bound to the authorized Phase 2g preregistration")
    if (
        not isinstance(registry_binding, dict)
        or registry_binding.get("path") != registry_path.resolve(strict=True).relative_to(root).as_posix()
        or registry_binding.get("file_sha256") != sha256_file(registry_path)
        or not isinstance(ledger_binding, dict)
        or ledger_binding.get("path") != ledger_path.resolve(strict=True).relative_to(root).as_posix()
        or ledger_binding.get("file_sha256") != sha256_file(ledger_path)
    ):
        raise ValueError("geometry readiness registry/ledger bindings differ from preflight inputs")
    runtime_files = runtime_binding.get("files") if isinstance(runtime_binding, dict) else None
    if (
        not isinstance(runtime_binding, dict)
        or runtime_binding.get("status") != "hash-bound-complete"
        or runtime_binding.get("final_hash_binding_required") is not False
        or not isinstance(runtime_files, list)
        or not runtime_files
    ):
        raise ValueError("formal Phase 2g runtime inventory is still pending final hash binding")
    bound_runtime_paths: set[str] = set()
    for row in runtime_files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("formal Phase 2g runtime binding contains an invalid file row")
        runtime_path = (root / str(row["path"])).resolve(strict=True)
        if runtime_path.relative_to(root).as_posix() != row["path"] or row.get("file_sha256") != sha256_file(
            runtime_path
        ):
            raise ValueError(f"formal Phase 2g runtime hash binding is stale: {row.get('path')}")
        _require_tracked(root, runtime_path, "formal Phase 2g runtime file")
        bound_runtime_paths.add(str(row["path"]))
    required_runtime_paths = {
        "slurm/phase2g_preflight.py",
        "slurm/phase2g_contract.py",
        "slurm/phase2g_postflight.py",
        "slurm/submit_phase2g.sh",
        "scripts/write_phase2g_dependency_graph.py",
        *REQUIRED_RUNNERS,
        *TRANSITIVE_RUNTIME_PATHS,
    }
    missing_runtime = sorted(required_runtime_paths - bound_runtime_paths)
    if missing_runtime:
        raise ValueError(f"formal Phase 2g runtime binding is incomplete: {missing_runtime}")
    sun = _raw_gate(value, "sun-development")
    blockers = sun.get("blockers")
    if (
        sun.get("execution_ready") is not True
        or sun.get("pack_authorizes_data_access") is not True
        or blockers not in ([], None)
        or sun.get("status") not in {"execution-ready", "authorized", "ready"}
    ):
        codes = [item.get("code") for item in blockers or [] if isinstance(item, dict)]
        raise ValueError(f"SUN readiness does not authorize Phase 2g-A execution; blockers={codes}")
    diode = _raw_gate(value, "diode-external")
    if (
        diode.get("sealed") is not True
        or diode.get("execution_ready") is not False
        or diode.get("pack_authorizes_data_access") is not False
    ):
        raise ValueError("DIODE readiness must remain sealed and unauthorized")
    return {
        **file_identity(path),
        "schema_version": pack.schema_version,
        "pack_version": pack.pack_version,
        "registry_semantic_sha256": pack.bindings.registry.semantic_sha256,
        "ledger_semantic_sha256": pack.bindings.ledger.semantic_sha256,
        "preregistration_sha256": pack.bindings.preregistration.file_sha256,
    }


def _validate_registry(
    path: Path,
    ledger_path: Path,
    root: Path,
    archive: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_tracked(root, path, "validation registry")
    _require_tracked(root, ledger_path, "consumed-test ledger")
    registry = DatasetRegistry.load(path)
    ledger = ConsumedTestLedger.load(ledger_path)
    entry = registry.dataset(SUN_DATASET_ID)
    if entry.readiness_blockers:
        raise ValueError(f"SUN registry audit is incomplete: {entry.readiness_blockers}")
    archive_rows = [value for value in entry.hashes if value.name == "SUNRGBD.zip" and value.kind == "archive"]
    if len(archive_rows) != 1 or archive_rows[0].status != "verified":
        raise ValueError("registry lacks one verified SUNRGBD.zip identity")
    expected = archive_rows[0]
    observed_sha256 = sha256_file(archive.resolve(strict=True))
    if expected.sha256 != EXPECTED_ARCHIVE_SHA256 or observed_sha256 != expected.sha256:
        raise ValueError("SUN archive SHA-256 differs from the authorized registry identity")
    if expected.bytes is not None and archive.stat().st_size != expected.bytes:
        raise ValueError("SUN archive byte count differs from the authorized registry identity")
    _, split = registry.split(SUN_DATASET_ID, SUN_SPLIT_ID)
    required_operations = (
        AccessOperation.DECODE_SMOKE,
        AccessOperation.TRAINING,
        AccessOperation.TUNING,
        AccessOperation.CALIBRATION,
        AccessOperation.CHECKPOINT_SELECTION,
        AccessOperation.DEVELOPMENT_EVALUATION,
        AccessOperation.MECHANISM_DIAGNOSTIC,
        AccessOperation.REPORTING,
    )
    if split.expected_units != 4096 or not set(required_operations).issubset(split.allowed_operations):
        raise ValueError("registered Phase 2g SUN split lacks 4,096 units or required operations")
    controller = DatasetAccessController(registry=registry, ledger=ledger)
    decisions = [
        controller.authorize(SUN_DATASET_ID, SUN_SPLIT_ID, operation).model_dump(mode="json")
        for operation in required_operations
    ]
    if any(decision.get("authorized") is not True for decision in decisions):
        raise ValueError("DatasetAccessController did not authorize every formal SUN operation")
    return (
        {**file_identity(path), "semantic_sha256": registry.sha256, "portfolio_version": registry.portfolio_version},
        decisions,
    )


def _reject_diode_environment() -> None:
    violations = []
    for key, value in os.environ.items():
        if "DIODE" in key.upper() or re.search(r"(?i)(?:^|[/\\])diode(?:[/\\]|$)", value):
            violations.append(key)
    if violations:
        raise ValueError(f"DIODE path/environment is forbidden for Phase 2g-A: {sorted(violations)}")


def validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve(strict=True)
    git = _require_clean_pushed(root)
    _reject_diode_environment()
    for path, label in (
        (args.vjepa_checkpoint, "V-JEPA checkpoint"),
        (args.vjepa_implementation, "V-JEPA implementation"),
    ):
        if not path.resolve(strict=True).is_dir():
            raise ValueError(f"{label} must be a directory")
    missing_runners = [name for name in REQUIRED_RUNNERS if not (root / name).is_file()]
    if missing_runners:
        raise ValueError(f"formal Phase 2g core runners are missing: {missing_runners}")
    preregistration = _validate_preregistration(args.preregistration.resolve(strict=True), root)
    registry, access_decisions = _validate_registry(
        args.registry.resolve(strict=True),
        args.ledger.resolve(strict=True),
        root,
        args.sun_archive,
    )
    readiness = _validate_readiness(
        args.readiness.resolve(strict=True),
        root,
        args.preregistration.resolve(strict=True),
        args.registry.resolve(strict=True),
        args.ledger.resolve(strict=True),
    )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": args.execution_id,
        "git": git,
        "preregistration": preregistration,
        "registry": registry,
        "ledger": file_identity(args.ledger.resolve(strict=True)),
        "dataset_access": {
            "dataset_id": SUN_DATASET_ID,
            "split_id": SUN_SPLIT_ID,
            "authorized": True,
            "decisions": access_decisions,
        },
        "readiness": readiness,
        "sun_archive": {
            "name": "SUNRGBD.zip",
            "bytes": args.sun_archive.stat().st_size,
            "sha256": EXPECTED_ARCHIVE_SHA256,
        },
        "core_runners": list(REQUIRED_RUNNERS),
        "manifest_contract": {
            "producer": "C",
            "validator": "Q",
            "selected_per_family": 1024,
            "families": ["kv1", "kv2", "realsense", "xtion"],
            "target_free_membership": True,
        },
        "diode": {
            "sealed": True,
            "archive_path_received": False,
            "archive_touched": False,
            "external_final_authorized": False,
        },
        "wandb": {"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--sun-archive", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_preflight(args)
    atomic_json(args.output, result)
    print(f"Phase 2g preflight PASS: {args.output}")


if __name__ == "__main__":
    main()
