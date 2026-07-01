from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from slurm.phase2g_contract import reject_diode_paths
from slurm.phase2g_postflight import verify_wandb_backend


class _Artifact:
    def __init__(self, receipt: dict[str, Any], source: Path) -> None:
        self.id = receipt["artifact_id"]
        self.version = receipt["artifact_version"]
        self.digest = receipt["artifact_digest"]
        self.state = "COMMITTED"
        self.source = source
        self.manifest = SimpleNamespace(
            entries={item["name"]: SimpleNamespace(ref=None, size=item["bytes"]) for item in receipt["files"]}
        )

    def download(self, *, root: str, **_kwargs: Any) -> str:
        destination = Path(root)
        for child in self.source.rglob("*"):
            if child.is_file():
                target = destination / child.relative_to(self.source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(child, target)
        return str(destination)

    def verify(self, _root: str) -> None:
        return None


class _Api:
    def __init__(self, receipt: dict[str, Any], source: Path) -> None:
        self.receipt = receipt
        self.run_value = SimpleNamespace(
            entity=receipt["entity"],
            project=receipt["project"],
            id=receipt["run_id"],
            name=receipt["run_name"],
            group=receipt["group"],
            job_type=receipt["job_type"],
            url=receipt["run_url"],
            state="finished",
        )
        self.artifact_value = _Artifact(receipt, source)

    def run(self, path: str) -> Any:
        assert path == f"{self.receipt['entity']}/{self.receipt['project']}/{self.receipt['run_id']}"
        return self.run_value

    def artifact(self, path: str) -> Any:
        assert path.endswith(f"{self.receipt['artifact_name']}:{self.receipt['artifact_version']}")
        return self.artifact_value


def _receipt(source: Path) -> dict[str, Any]:
    files = []
    for path in sorted(child for child in source.rglob("*") if child.is_file()):
        files.append(
            {
                "name": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": "jepa4d-phase2g-wandb-artifact-receipt-v1",
        "mode": "online",
        "status": "success",
        "entity": "test-entity",
        "project": "test-project",
        "group": "phase2g-quality-exec",
        "job_type": "formal",
        "run_name": "exec-f-m0-r0-s0",
        "run_id": "run123",
        "run_url": "https://wandb.invalid/test-entity/test-project/runs/run123",
        "artifact_name": "phase2g-exec-f-m0-r0-s0",
        "artifact_id": "test-entity/test-project/artifact:v0",
        "artifact_version": "v0",
        "artifact_digest": "backend-digest",
        "files": files,
    }


def test_backend_verifier_downloads_and_hashes_every_file(tmp_path: Path) -> None:
    source = tmp_path / "artifact"
    (source / "nested").mkdir(parents=True)
    (source / "metrics.json").write_text('{"status":"pass"}\n', encoding="utf-8")
    (source / "nested" / "checkpoint.pt").write_bytes(b"checkpoint")
    receipt = _receipt(source)
    evidence = verify_wandb_backend(receipt, api=_Api(receipt, source))
    assert evidence["status"] == "verified"
    assert evidence["artifact_digest"] == "backend-digest"
    assert [item["name"] for item in evidence["files"]] == ["metrics.json", "nested/checkpoint.pt"]
    assert len(evidence["files_sha256"]) == 64


def test_backend_verifier_rejects_downloaded_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "artifact"
    source.mkdir()
    artifact = source / "metrics.json"
    artifact.write_text("original", encoding="utf-8")
    receipt = _receipt(source)
    artifact.write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash differs"):
        verify_wandb_backend(receipt, api=_Api(receipt, source))


def test_external_path_guard_allows_policy_labels_but_rejects_paths() -> None:
    reject_diode_paths({"policy": "DIODE remains sealed", "external_final_authorized": False})
    with pytest.raises(ValueError, match="DIODE path-bearing field"):
        reject_diode_paths({"diode_archive_path": "/restricted/archive"})
    with pytest.raises(ValueError, match="external-final path"):
        reject_diode_paths({"source": "/datasets/DIODE/val.tar.gz"})


def test_postflight_source_requires_all_152_backend_artifacts_and_terminal_receipt() -> None:
    source = (Path(__file__).resolve().parents[2] / "slurm" / "phase2g_postflight.py").read_text(encoding="utf-8")
    assert 'verified_task_artifacts": 152' in source
    assert 'verify_wandb_backend(receipt["wandb"])' in source
    assert "write_content_addressed_json(terminal" in source
    assert source.index("terminal_address = write_content_addressed_json") < source.index(
        '(output / "SUCCESS").write_text'
    )
