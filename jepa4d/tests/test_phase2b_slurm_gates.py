from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import wandb

from scripts import run_phase2b_geometry_distillation as phase2b
from slurm.phase2b_gate import asset_inventory, repository_fingerprint, sha256


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value)
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _formal_output(root: Path) -> Path:
    output = root / "formal"
    normalizations = {}
    for name in ("rgb_probe", "vjepa_final", "vjepa_multilayer"):
        path = output / f"{name}-normalization.pt"
        _write(path, f"normalization-{name}".encode())
        normalizations[path.name] = sha256(path)

    variants = []
    common_metrics = {
        "metric_abs_rel": 0.2,
        "metric_rmse_m": 0.4,
        "metric_delta_1": 0.8,
        "aligned_abs_rel": 0.18,
    }
    common_runtime = {
        "encoder_ms_per_frame": 3.0,
        "head_ms_per_frame": 1.0,
        "total_ms_per_frame": 4.0,
    }
    variants.append(
        {
            "variant_id": "vggt_teacher",
            "seed": None,
            "metrics": common_metrics,
            "runtime": common_runtime,
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
    )
    for name in ("rgb_probe", "vjepa_final", "vjepa_multilayer"):
        for seed in (0, 1, 2):
            checkpoint = output / "checkpoints" / f"{name}-seed{seed}.pt"
            _write(checkpoint, f"checkpoint-{name}-{seed}".encode())
            variants.append(
                {
                    "variant_id": name,
                    "seed": seed,
                    "metrics": common_metrics,
                    "runtime": common_runtime,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )

    run_url = "https://wandb.ai/test/project/runs/phase2b"
    _write(
        output / "comparison.json",
        {
            "variants": variants,
            "failures": [],
            "aggregates": {},
            "artifacts": normalizations,
            "wandb_url": run_url,
        },
    )
    _write(output / "failures.json", [])
    _write(output / "completion_gate.json", {"status": "success"})
    _write(output / "geometry_student_report.html", "<html>phase2b</html>")
    _write(output / "diagnostics" / "debug.json", {"finite": True})
    _write(output / "artifact_manifest.json", phase2b._artifact_manifest(output))
    _write(
        output / "wandb_artifact_receipt.json",
        {
            "schema_version": "jepa4d-phase2b-wandb-artifact-v1",
            "status": "success",
            "mode": "online",
            "run_id": "phase2b",
            "run_url": run_url,
            "artifact_name": "phase2b-comparison",
            "artifact_version": "v0",
            "artifact_digest": "digest",
            "artifact_manifest_sha256": sha256(output / "artifact_manifest.json"),
        },
    )
    return output


def _validate(output: Path, report: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[2] / "slurm" / "validate_phase2b_output.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--report",
            str(report),
            "--require-wandb",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_formal_output_validator_accepts_exact_manifest_and_online_receipt(tmp_path: Path) -> None:
    result = _validate(_formal_output(tmp_path), tmp_path / "validation.json")
    assert result.returncode == 0, result.stdout + result.stderr


def test_formal_output_validator_rejects_stale_manifest(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    (output / "diagnostics" / "debug.json").write_text('{"mutated": true}\n')
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "artifact manifest" in result.stdout


def test_formal_output_validator_rejects_incomplete_wandb_receipt(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    receipt_path = output / "wandb_artifact_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_digest"] = ""
    _write(receipt_path, receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "receipt is incomplete" in result.stdout


def test_repository_and_asset_fingerprints_detect_content_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "phase2b@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Phase2b Test"], check=True)
    _write(repository / ".gitignore", "ignored/\n")
    _write(repository / "tracked.txt", "before\n")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    before = repository_fingerprint(repository)
    _write(repository / "ignored" / "runtime.log", "ignored\n")
    assert repository_fingerprint(repository) == before
    _write(repository / "tracked.txt", "after\n")
    assert repository_fingerprint(repository)["sha256"] != before["sha256"]

    asset = tmp_path / "asset"
    _write(asset / "weights.bin", b"weights-v1")
    first_asset = asset_inventory(asset)
    _write(asset / "weights.bin", b"weights-v2")
    assert asset_inventory(asset)["sha256"] != first_asset["sha256"]


def test_wandb_artifact_helper_waits_for_backend_receipt(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    _write(output / "result.json", {"ok": True})
    _write(output / "artifact_manifest.json", phase2b._artifact_manifest(output))

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type

        def add_dir(self, path: str, name: str) -> None:
            assert Path(path) == output
            assert name == "phase2b"

    class LoggedArtifact:
        name = "formal-artifact"
        version = "v7"
        digest = "server-digest"
        timeout: int | None = None

        def wait(self, timeout: int | None = None) -> None:
            self.timeout = timeout

    logged = LoggedArtifact()

    class FakeRun:
        id = "run-id"
        url = "https://wandb.ai/test/project/runs/run-id"
        path = "test/project/run-id"

        def log_artifact(self, artifact: FakeArtifact) -> LoggedArtifact:
            assert artifact.name.endswith("phase2b-comparison")
            return logged

    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    receipt = phase2b._upload_wandb_artifact(FakeRun(), output, "success")
    assert logged.timeout == 900
    assert receipt["artifact_digest"] == "server-digest"
    assert json.loads((output / "wandb_artifact_receipt.json").read_text()) == receipt
