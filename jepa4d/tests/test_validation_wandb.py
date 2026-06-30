from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jepa4d.validation.wandb import (
    SafeArtifactFile,
    finalize_safe_online_run,
    publish_safe_online_run,
    validate_safe_artifact_files,
    validate_safe_wandb_final_receipt,
    validate_safe_wandb_receipt,
)


class _LoggedArtifact:
    id = "entity/project/artifact:v0"
    version = "v0"
    digest = "artifact-digest"

    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _Artifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))


class _Run:
    offline = False
    entity = "test-entity"
    project = "test-project"
    id = "safe-run-id"
    url = "https://wandb.ai/test-entity/test-project/runs/safe-run-id"

    def __init__(self) -> None:
        self.summary: dict[str, object] = {}
        self.logged = _LoggedArtifact()
        self.artifact: _Artifact | None = None
        self.exit_codes: list[int] = []

    def log_artifact(self, artifact: _Artifact) -> _LoggedArtifact:
        self.artifact = artifact
        return self.logged

    def finish(self, *, exit_code: int) -> None:
        self.exit_codes.append(exit_code)


def _safe_files(root: Path) -> tuple[SafeArtifactFile, ...]:
    (root / "validation-report.json").write_text(
        json.dumps({"schema_version": "jepa4d-validation-dashboard-v1", "metrics": []}), encoding="utf-8"
    )
    (root / "validation-dashboard.html").write_text(
        "<!doctype html><title>Governed aggregate dashboard</title>", encoding="utf-8"
    )
    return (
        SafeArtifactFile(root / "validation-report.json", "dashboard-json"),
        SafeArtifactFile(root / "validation-dashboard.html", "dashboard-html"),
    )


def test_safe_online_publisher_waits_and_returns_path_free_receipt(tmp_path, monkeypatch) -> None:
    run = _Run()
    artifacts: list[_Artifact] = []

    def artifact(name: str, type: str) -> _Artifact:
        value = _Artifact(name, type)
        artifacts.append(value)
        return value

    fake = SimpleNamespace(init=lambda **kwargs: run, Artifact=artifact)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_MODE", "online")

    receipt = publish_safe_online_run(
        entity="test-entity",
        project="test-project",
        group="governed-mini",
        job_type="official-mini",
        run_name="unique-run",
        config={"registry_sha256": "a" * 64},
        summary={"validation/quality/abs_rel": 0.2},
        artifact_name="governed-mini-artifact",
        artifact_root=tmp_path,
        files=_safe_files(tmp_path),
    )

    assert run.logged.waited is True
    assert run.exit_codes == [0]
    assert artifacts[0].type == "governed-validation"
    assert {name for _, name in artifacts[0].files} == {
        "validation-report.json",
        "validation-dashboard.html",
    }
    assert all("path" not in item for item in receipt["files"])
    assert all(str(tmp_path) not in str(item) for item in receipt["files"])
    validate_safe_wandb_receipt(receipt)


def test_safe_publisher_rejects_offline_before_import_or_file_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_MODE=online"):
        publish_safe_online_run(
            entity=None,
            project="test-project",
            group="group",
            job_type="official-mini",
            run_name="run",
            config={},
            summary={},
            artifact_name="artifact",
            artifact_root=tmp_path,
            files=(),
        )


@pytest.mark.parametrize(
    ("name", "payload", "role", "match"),
    [
        ("samples.json", {"sample_id": "rgb-001"}, "aggregate-receipt", "unsafe field"),
        ("targets.json", {"targets": ["dog"]}, "aggregate-receipt", "unsafe field"),
        ("paths.json", {"note": "/lustre/restricted/data"}, "aggregate-receipt", "path content"),
        ("prediction.json", {"predictions": [0.1]}, "aggregate-receipt", "unsafe field"),
        ("depth.json", {"depth": [0.1]}, "aggregate-receipt", "unsafe field"),
        ("units.json", {"per_unit": [{"unit_id": "u", "value": 0.1}]}, "aggregate-receipt", "unsafe field"),
        ("secret.json", {"note": "hf_abcdefghijklmnopqrstuvwxyz123456"}, "aggregate-receipt", "credential"),
    ],
)
def test_safe_artifact_allowlist_rejects_raw_or_sensitive_content(
    tmp_path, name: str, payload: dict, role: str, match: str
) -> None:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        validate_safe_artifact_files((SafeArtifactFile(path, role),), artifact_root=tmp_path)


@pytest.mark.parametrize("name", ["prediction.npz", "cloud.ply", "preview.png"])
def test_safe_artifact_allowlist_rejects_prediction_media_and_binary_outputs(tmp_path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"not-safe")
    with pytest.raises(ValueError, match="JSON or self-contained HTML"):
        validate_safe_artifact_files(
            (SafeArtifactFile(path, "aggregate-receipt"),),
            artifact_root=tmp_path,
        )


def test_safe_publisher_rejects_wandb_offline_fallback(tmp_path, monkeypatch) -> None:
    run = _Run()
    run.offline = True
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Artifact=_Artifact))
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(RuntimeError, match="did not initialize online"):
        publish_safe_online_run(
            entity=None,
            project="test-project",
            group="group",
            job_type="official-mini",
            run_name="run",
            config={},
            summary={},
            artifact_name="artifact",
            artifact_root=tmp_path,
            files=_safe_files(tmp_path),
        )


def test_safe_publisher_detects_artifact_mutation_during_upload(tmp_path, monkeypatch) -> None:
    files = _safe_files(tmp_path)
    run = _Run()

    class _MutatingLoggedArtifact(_LoggedArtifact):
        def wait(self) -> None:
            files[0].path.write_text(
                json.dumps({"schema_version": "jepa4d-validation-dashboard-v1", "metrics": []}, indent=2),
                encoding="utf-8",
            )
            super().wait()

    run.logged = _MutatingLoggedArtifact()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Artifact=_Artifact))
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(RuntimeError, match="changed during upload"):
        publish_safe_online_run(
            entity=None,
            project="test-project",
            group="group",
            job_type="official-mini",
            run_name="run",
            config={},
            summary={},
            artifact_name="artifact",
            artifact_root=tmp_path,
            files=files,
        )
    assert run.exit_codes == [1]


def test_safe_wandb_receipt_rejects_duplicate_names_or_roles() -> None:
    receipt = {
        "schema_version": "jepa4d-safe-wandb-receipt-v1",
        "status": "uploaded-preliminary",
        "terminal_status": "pending-postflight",
        "mode": "online",
        "run_id": "run",
        "run_url": "https://wandb.ai/entity/project/runs/run",
        "artifact_id": "entity/project/artifact:v0",
        "artifact_version": "v0",
        "artifact_digest": "digest",
        "config_sha256": "c" * 64,
        "summary_sha256": "d" * 64,
        "files": [
            {"name": "first.json", "role": "aggregate-receipt", "bytes": 1, "sha256": "a" * 64},
            {"name": "second.json", "role": "aggregate-receipt", "bytes": 1, "sha256": "b" * 64},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_safe_wandb_receipt(receipt)


def test_safe_publisher_rejects_none_backend_identities(tmp_path, monkeypatch) -> None:
    run = _Run()
    run.url = None  # type: ignore[assignment]
    run.logged.id = None  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Artifact=_Artifact))
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(RuntimeError, match="complete online"):
        publish_safe_online_run(
            entity=None,
            project="test-project",
            group="group",
            job_type="official-mini",
            run_name="run",
            config={},
            summary={},
            artifact_name="artifact",
            artifact_root=tmp_path,
            files=_safe_files(tmp_path),
        )
    assert run.exit_codes == [1]


def test_terminal_finalizer_resumes_exact_run_and_uploads_terminal_receipts(tmp_path, monkeypatch) -> None:
    run = _Run()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Artifact=_Artifact))
    monkeypatch.setenv("WANDB_MODE", "online")
    preliminary = publish_safe_online_run(
        entity="test-entity",
        project="test-project",
        group="group",
        job_type="official-mini",
        run_name="run",
        config={},
        summary={},
        artifact_name="artifact",
        artifact_root=tmp_path,
        files=_safe_files(tmp_path),
    )
    execution = tmp_path / "execution.json"
    postflight = tmp_path / "postflight.json"
    execution.write_text(
        json.dumps({"schema_version": "jepa4d-geometry-official-mini-execution-v1", "status": "pass"}),
        encoding="utf-8",
    )
    postflight.write_text(
        json.dumps({"schema_version": "jepa4d-geometry-official-mini-postflight-v1", "status": "pass"}),
        encoding="utf-8",
    )
    finalized = finalize_safe_online_run(
        preliminary_receipt=preliminary,
        artifact_root=tmp_path,
        files=(
            SafeArtifactFile(execution, "execution-receipt"),
            SafeArtifactFile(postflight, "postflight-receipt"),
        ),
        summary={"validation/postflight_receipt_sha256": "a" * 64},
    )
    validate_safe_wandb_final_receipt(finalized)
    assert finalized["run_id"] == preliminary["run_id"]
    assert {item["role"] for item in finalized["files"]} == {"execution-receipt", "postflight-receipt"}
    assert run.exit_codes == [0, 0]
