from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.write_phase2f_dependency_graph import (
    ACCOUNT,
    PARTITIONS,
    expected_labels,
    parse_sbatch,
    validate_jobs,
)
from slurm.phase2f_asset_audit import audit_assets
from slurm.phase2f_contract import reject_secrets
from slurm.phase2f_final_guard import assert_registry_clear

ROOT = Path(__file__).resolve().parents[2]
SBATCH_FILES = sorted((ROOT / "slurm").glob("phase2f_*.sbatch"))


def _parent_map() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"T": [], "A": ["T"], "C": ["T"], "Q": ["C"]}
    result.update({f"L{index:02d}": ["Q"] for index in range(12)})
    result["LA"] = [f"L{index:02d}" for index in range(12)]
    result.update({f"P{index}": ["LA"] for index in range(4)})
    result["PG"] = [f"P{index}" for index in range(4)]
    formal = [
        f"F-{arm}-R{rotation}-S{seed}"
        for arm in ("M0", "M1", "M2", "M3")
        for rotation in range(4)
        for seed in range(3)
    ]
    result.update({label: ["PG"] for label in formal})
    result["S"] = formal
    result["E"] = ["S", "A"]
    result["Z"] = ["E"]
    return result


def test_exact_73_job_static_dag_and_only_t_is_root() -> None:
    labels = expected_labels()
    parents = _parent_map()
    assert len(labels) == 73
    assert set(parents) == labels
    assert [label for label, values in parents.items() if not values] == ["T"]
    assert len([label for label in labels if label.startswith("L") and label != "LA"]) == 12
    assert len([label for label in labels if label.startswith("F-")]) == 48


def test_sbatch_resources_are_frozen_and_gpu_is_requested_only_when_needed() -> None:
    expected_files = {
        "phase2f_tests.sbatch",
        "phase2f_asset_audit.sbatch",
        "phase2f_cache.sbatch",
        "phase2f_static_audit.sbatch",
        "phase2f_latency.sbatch",
        "phase2f_latency_aggregate.sbatch",
        "phase2f_train.sbatch",
        "phase2f_pilot_gate.sbatch",
        "phase2f_select.sbatch",
        "phase2f_final.sbatch",
        "phase2f_postflight.sbatch",
    }
    assert {path.name for path in SBATCH_FILES} == expected_files
    gpu = {
        "phase2f_tests.sbatch",
        "phase2f_cache.sbatch",
        "phase2f_latency.sbatch",
        "phase2f_train.sbatch",
        "phase2f_final.sbatch",
    }
    for path in SBATCH_FILES:
        resources = parse_sbatch(path)
        directives = resources["directives"]
        assert directives["account"] == ACCOUNT
        assert directives["partition"] == PARTITIONS
        assert resources["time_seconds"] <= 4 * 60 * 60
        assert resources["gpu_requested"] is (path.name in gpu)


def test_graph_validator_rejects_gpu_or_parent_drift() -> None:
    parents = _parent_map()
    gpu_labels = {"T", "C", "E", *(f"L{index:02d}" for index in range(12))}
    gpu_labels.update(f"P{index}" for index in range(4))
    gpu_labels.update(label for label in parents if label.startswith("F-"))
    jobs: dict[str, dict[str, Any]] = {
        label: {"parents": values, "resources": {"gpu_requested": label in gpu_labels}}
        for label, values in parents.items()
    }
    validate_jobs(jobs)
    jobs["A"]["resources"]["gpu_requested"] = True
    with pytest.raises(ValueError, match="GPU request mismatch"):
        validate_jobs(jobs)


def test_submitter_holds_every_job_writes_graph_then_releases_in_chunks() -> None:
    source = (ROOT / "slurm" / "submit_phase2f.sh").read_text(encoding="utf-8")
    assert "--hold" in source
    assert '--dependency "afterok:$dependency"' in source
    assert "scripts/write_phase2f_dependency_graph.py" in source
    assert "offset+=20" in source
    assert source.index("scripts/write_phase2f_dependency_graph.py") < source.index("scontrol release")
    assert "checkpoints/datasets/DIODE/devkit" in source
    assert 'phase2f_final_guard.py" registry-clear' in source


def test_pipeline_wrappers_do_not_precreate_cli_output_and_final_interface_is_exact() -> None:
    for name in (
        "phase2f_latency.sbatch",
        "phase2f_latency_aggregate.sbatch",
        "phase2f_train.sbatch",
        "phase2f_pilot_gate.sbatch",
        "phase2f_select.sbatch",
        "phase2f_final.sbatch",
    ):
        source = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'mkdir -p "$OUT"' not in source
        assert 'PROVENANCE="$JEPA4D_JOB_LOG_DIR/execution_provenance.json"' in source
    final = (ROOT / "slurm" / "phase2f_final.sbatch").read_text(encoding="utf-8")
    for flag in (
        "--archive",
        "--asset-seal",
        "--diode-meta",
        "--intrinsics",
        "--devkit-license",
        "--selector",
        "--sentinel",
        "--vjepa-checkpoint",
        "--vjepa-implementation",
        "--provenance",
        "--output",
    ):
        assert flag in final
    assert 'phase2f_final_guard.py" open' not in final
    assert 'phase2f_final_guard.py" no-survivor' not in final


def test_asset_audit_reads_compressed_bytes_only_and_checks_pinned_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "val.tar.gz"
    archive.write_bytes(b"opaque-compressed-target-bytes")
    devkit = tmp_path / "devkit"
    devkit.mkdir()
    files = {"diode_meta.json": b"{}", "intrinsics.txt": b"K", "LICENSE": b"MIT"}
    expected: dict[str, str] = {}
    for name, payload in files.items():
        (devkit / name).write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    subprocess.run(("git", "init", "-q", str(devkit)), check=True)
    subprocess.run(("git", "-C", str(devkit), "config", "user.email", "phase2f@test.invalid"), check=True)
    subprocess.run(("git", "-C", str(devkit), "config", "user.name", "Phase2f Test"), check=True)
    subprocess.run(("git", "-C", str(devkit), "add", "."), check=True)
    subprocess.run(("git", "-C", str(devkit), "commit", "-qm", "fixture"), check=True)
    commit = subprocess.check_output(("git", "-C", str(devkit), "rev-parse", "HEAD"), text=True).strip()
    result = audit_assets(
        archive,
        devkit,
        expected_bytes=archive.stat().st_size,
        expected_md5=hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest(),
        expected_commit=commit,
        expected_files=expected,
    )
    assert result["target_opacity"] == {
        "compressed_stream_only": True,
        "tar_listed": False,
        "tar_extracted": False,
        "target_array_loaded": False,
        "target_statistics_computed": False,
        "target_preview_generated": False,
    }
    source = (ROOT / "slurm" / "phase2f_asset_audit.py").read_text(encoding="utf-8")
    assert "import tarfile" not in source
    assert ".getmembers(" not in source


def test_final_open_registry_blocks_same_preregistration(tmp_path: Path) -> None:
    preregistration = "a" * 64
    assert_registry_clear(tmp_path, preregistration)
    sentinel = tmp_path / "old" / "final" / "FRESH_FINAL_OPENED.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text(
        json.dumps(
            {"schema_version": "jepa4d-phase2f-fresh-final-opened-v1", "preregistration_sha256": preregistration}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="already opened"):
        assert_registry_clear(tmp_path, preregistration)


def test_provenance_rejects_credential_like_fields() -> None:
    with pytest.raises(ValueError, match="credential-like"):
        reject_secrets({"execution_id": "x", "wandb_api_key": "forbidden"})
