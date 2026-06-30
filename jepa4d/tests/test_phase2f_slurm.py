from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.write_phase2f_dependency_graph import (
    ACCOUNT,
    PARTITIONS,
    _parse_job,
    expected_labels,
    parse_sbatch,
    validate_jobs,
)
from slurm import phase2f_contract
from slurm.phase2f_asset_audit import audit_assets
from slurm.phase2f_contract import (
    LOGICAL_JOB_ID_PATTERN,
    SUBMISSION_POLICY,
    _current_scheduler_ids,
    load_graph,
    reject_secrets,
    scheduler_completed,
)
from slurm.phase2f_final_guard import assert_registry_clear
from slurm.phase2f_postflight import _verify_hash_identities

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


def test_sbatch_resources_are_frozen_and_every_partition_job_requests_a_gpu() -> None:
    expected_files = {
        "phase2f_tests.sbatch",
        "phase2f_asset_audit.sbatch",
        "phase2f_cache.sbatch",
        "phase2f_static_audit.sbatch",
        "phase2f_array_dispatch.sbatch",
        "phase2f_latency.sbatch",
        "phase2f_latency_aggregate.sbatch",
        "phase2f_train.sbatch",
        "phase2f_pilot_gate.sbatch",
        "phase2f_select.sbatch",
        "phase2f_final.sbatch",
        "phase2f_postflight.sbatch",
    }
    assert {path.name for path in SBATCH_FILES} == expected_files
    for path in SBATCH_FILES:
        resources = parse_sbatch(path)
        directives = resources["directives"]
        assert directives["account"] == ACCOUNT
        assert directives["partition"] == PARTITIONS
        assert resources["time_seconds"] <= 4 * 60 * 60
        assert resources["gpu_requested"] is True


def test_graph_validator_rejects_gpu_or_parent_drift() -> None:
    parents = _parent_map()
    jobs: dict[str, dict[str, Any]] = {
        label: {"parents": values, "resources": {"gpu_requested": True}} for label, values in parents.items()
    }
    validate_jobs(jobs)
    jobs["A"]["resources"]["gpu_requested"] = False
    with pytest.raises(ValueError, match="require one GPU"):
        validate_jobs(jobs)


def test_submitter_uses_exact_12_held_p2f8_submissions_and_global_cap() -> None:
    submitter = ROOT / "slurm" / "submit_phase2f.sh"
    source = submitter.read_text(encoding="utf-8")
    assert os.access(submitter, os.X_OK)
    assert "--hold" in source
    assert '--dependency "afterok:$dependency"' in source
    submitted = re.findall(r'^([A-Z]+)="\$\(submit "\$name"', source, flags=re.MULTILINE)
    assert submitted == ["T", "A", "C", "Q", "L", "LA", "P", "PG", "F", "S", "E", "Z"]
    names = re.findall(r'^name="p2f8-([A-Z]+)-\$SHORT"$', source, flags=re.MULTILINE)
    assert names == submitted
    assert len(set(names)) == 12
    assert source.count("slurm/phase2f_array_dispatch.sbatch") == 3
    assert "--array=0-11%8" in source
    assert "--array=0-3%4" in source
    assert "--array=0-47%8" in source
    assert 'L="$(submit "$name" "$Q:$A"' in source
    assert 'LA="$(submit "$name" "$L"' in source
    assert 'P="$(submit "$name" "$LA"' in source
    assert 'PG="$(submit "$name" "$P"' in source
    assert 'F="$(submit "$name" "$PG"' in source
    assert 'S="$(submit "$name" "$F"' in source
    assert "[[ ${#submission_ids[@]} -eq 12 ]]" in source
    assert "[[ ${#graph_jobs[@]} -eq 73 ]]" in source
    assert "scripts/write_phase2f_dependency_graph.py" in source
    assert source.index("scripts/write_phase2f_dependency_graph.py") < source.index('scontrol release "$joined"')
    assert "offset+=20" not in source
    assert "checkpoints/datasets/DIODE/devkit" in source
    assert 'phase2f_final_guard.py" registry-clear' in source


def test_array_dispatch_wrapper_has_exact_bounds_and_logical_mapping() -> None:
    path = ROOT / "slurm" / "phase2f_array_dispatch.sbatch"
    source = path.read_text(encoding="utf-8")
    subprocess.run(("bash", "-n", str(path)), check=True)
    assert "#SBATCH --job-name=p2f8-array" in source
    assert "#SBATCH --output=slurm-%x-%A_%a.out" in source
    assert 'export JEPA4D_LOGICAL_JOB_ID="${ARRAY_ID}_${TASK_ID}"' in source
    assert "TASK_ID >= 0 && TASK_ID < 12" in source
    assert "TASK_ID >= 0 && TASK_ID < 4" in source
    assert "TASK_ID >= 0 && TASK_ID < 48" in source
    assert "arm_index=$((TASK_ID / 12))" in source
    assert "within_arm=$((TASK_ID % 12))" in source
    assert "rotation_index=$((within_arm / 3))" in source
    assert "seed=$((within_arm % 3))" in source
    assert 'JEPA4D_JOB_LABEL="F-$JEPA4D_ARM-$JEPA4D_ROTATION-S$seed"' in source
    assert 'exec bash "$ROOT/slurm/phase2f_latency.sbatch"' in source
    assert source.count('exec bash "$ROOT/slurm/phase2f_train.sbatch"') == 2
    assert "eval " not in source


def test_graph_parser_accepts_numeric_and_numeric_task_ids_with_exact_policy(tmp_path: Path) -> None:
    scalar = _parse_job(
        "T|12345|p2f8-T-deadbeef|-|slurm/phase2f_tests.sbatch|outputs/tests.json",
        ROOT,
    )
    array = _parse_job(
        "L07|23456_7|p2f8-L07-deadbeef|Q|slurm/phase2f_latency.sbatch|outputs/latency-07.json",
        ROOT,
    )
    assert scalar["job_id"] == "12345"
    assert array["job_id"] == "23456_7"
    assert LOGICAL_JOB_ID_PATTERN.fullmatch("12345")
    assert LOGICAL_JOB_ID_PATTERN.fullmatch("23456_7")
    for invalid in ("123_", "_7", "123_bad", "123_7_8", "N_TASK"):
        assert LOGICAL_JOB_ID_PATTERN.fullmatch(invalid) is None
        with pytest.raises(ValueError, match="invalid job specification"):
            _parse_job(
                f"L07|{invalid}|p2f8-L07-deadbeef|Q|slurm/phase2f_latency.sbatch|outputs/x.json",
                ROOT,
            )

    parents = _parent_map()
    jobs: dict[str, dict[str, Any]] = {}
    scalar_index = 10_000
    for label, parent_labels in parents.items():
        if label.startswith("L") and label != "LA":
            task_id = int(label[1:])
            job_id = f"20000_{task_id}"
        elif label.startswith("P") and label != "PG":
            task_id = int(label[1:])
            job_id = f"30000_{task_id}"
        elif label.startswith("F-"):
            job_id = f"40000_{len([name for name in jobs if name.startswith('F-')])}"
        else:
            job_id = str(scalar_index)
            scalar_index += 1
        jobs[label] = {"job_id": job_id, "parents": parent_labels, "resources": {"gpu_requested": True}}
    graph = {
        "schema_version": "jepa4d-phase2f-dependency-graph-v1",
        "git_clean": True,
        "submission_policy": SUBMISSION_POLICY,
        "jobs": jobs,
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    loaded, _ = load_graph(graph_path)
    assert len(loaded["jobs"]) == 73
    assert loaded["submission_policy"] == {
        "all_jobs_submitted_held": True,
        "dependency_type": "afterok",
        "only_root_without_dependency": "T",
        "release_after_atomic_graph_write": True,
        "logical_job_count": 73,
        "scheduler_submission_count": 12,
        "max_parallel_tasks": 8,
        "array_task_throttle": 8,
    }


def test_runtime_array_identity_uses_base_and_task_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    assert _current_scheduler_ids("12345") == ("12345", "12345")

    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    assert _current_scheduler_ids("98765") == ("98765", "98765")

    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "23456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert _current_scheduler_ids("23456_7") == ("23456_7", "98765")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    with pytest.raises(ValueError, match="array identity is incomplete"):
        _current_scheduler_ids("23456_7")


def test_scheduler_completion_matches_exact_array_element(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_run(argv: tuple[str, ...], **_kwargs: Any) -> Any:
        observed.append(argv)
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "23456_6|COMPLETED|0:0\n23456_7|COMPLETED|0:0\n"},
        )()

    monkeypatch.setattr(phase2f_contract, "shutil_which", lambda _name: "/usr/bin/sacct")
    monkeypatch.setattr(phase2f_contract.subprocess, "run", fake_run)
    assert scheduler_completed("23456_7") is True
    assert scheduler_completed("23456_8") is False
    assert observed[0][0:5] == ("sacct", "-n", "-X", "-j", "23456_7")


def test_postflight_batches_scheduler_state_audit() -> None:
    source = (ROOT / "slurm" / "phase2f_postflight.py").read_text(encoding="utf-8")
    assert "scheduler_completed_many(predecessor_ids)" in source
    assert "scheduler_completed(str(job" not in source


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


def test_postflight_distinguishes_wandb_snapshots_from_current_local_hashes(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.bin"
    canonical.write_bytes(b"current canonical bytes")
    uploaded_receipt = tmp_path / "parent-receipt.json"
    uploaded_receipt.write_text('{"status":"finalized-after-upload"}\n', encoding="utf-8")
    outer_receipt = tmp_path / "child-receipt.json"
    outer_receipt.write_text("{}\n", encoding="utf-8")
    receipt = {
        "canonical": {
            "path": str(canonical),
            "bytes": canonical.stat().st_size,
            "sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        },
        "parent": {
            "receipt": {
                "path": str(uploaded_receipt),
                "bytes": uploaded_receipt.stat().st_size,
                "sha256": hashlib.sha256(uploaded_receipt.read_bytes()).hexdigest(),
            },
            "wandb": {
                "schema_version": "jepa4d-phase2f-wandb-artifact-receipt-v1",
                "files": [
                    {
                        "path": str(uploaded_receipt),
                        "bytes": 3,
                        "sha256": hashlib.sha256(b"old").hexdigest(),
                    }
                ],
            },
        },
    }
    assert _verify_hash_identities(receipt, outer_receipt) == 2

    uploaded_receipt.write_text('{"status":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="postflight file hash mismatch"):
        _verify_hash_identities(receipt, outer_receipt)


def test_postflight_still_verifies_wandb_only_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "report.html"
    artifact.write_text("original report\n", encoding="utf-8")
    outer_receipt = tmp_path / "receipt.json"
    outer_receipt.write_text("{}\n", encoding="utf-8")
    receipt = {
        "wandb": {
            "schema_version": "jepa4d-phase2f-wandb-artifact-receipt-v1",
            "files": [
                {
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }
    }
    assert _verify_hash_identities(receipt, outer_receipt) == 1

    artifact.write_text("tampered report\n", encoding="utf-8")
    with pytest.raises(ValueError, match="postflight file hash mismatch"):
        _verify_hash_identities(receipt, outer_receipt)


def test_postflight_does_not_exempt_unscoped_files_lists(tmp_path: Path) -> None:
    artifact = tmp_path / "array.npz"
    artifact.write_bytes(b"current")
    outer_receipt = tmp_path / "receipt.json"
    outer_receipt.write_text("{}\n", encoding="utf-8")
    receipt = {
        "not_wandb": {
            "files": [
                {
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(b"stale").hexdigest(),
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="postflight file hash mismatch"):
        _verify_hash_identities(receipt, outer_receipt)


def test_postflight_hash_memo_does_not_hide_conflicting_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    outer_receipt = tmp_path / "receipt.json"
    outer_receipt.write_text("{}\n", encoding="utf-8")
    identity = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    memo: set[tuple[str, str, int | None]] = set()
    assert _verify_hash_identities({"checkpoint": identity}, outer_receipt, memo) == 1
    assert _verify_hash_identities({"checkpoint": identity}, outer_receipt, memo) == 1

    identity["sha256"] = hashlib.sha256(b"wrong checkpoint").hexdigest()
    with pytest.raises(ValueError, match="postflight file hash mismatch"):
        _verify_hash_identities({"checkpoint": identity}, outer_receipt, memo)
