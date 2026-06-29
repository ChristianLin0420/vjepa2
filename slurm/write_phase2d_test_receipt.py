"""Write a content-bound receipt after the Phase 2d/2e Slurm test gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import torch
import typer


def _command(*values: str, cwd: Path) -> str:
    return subprocess.check_output(values, cwd=cwd, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(
    repo_root: Annotated[Path, typer.Option("--repo-root")],
    cuda_report: Annotated[Path, typer.Option("--cuda-report")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    repo_root = repo_root.resolve(strict=True)
    cuda_report = cuda_report.resolve(strict=True)
    status = _command("git", "status", "--porcelain", cwd=repo_root)
    if status:
        raise RuntimeError("test receipt requires a clean Git worktree")
    cuda = json.loads(cuda_report.read_text())
    if cuda.get("schema_version") != "jepa4d-cuda-health-v2" or cuda.get("status") != "pass":
        raise RuntimeError("CUDA report is not successful")
    source_paths = [
        "jepa4d/evaluation/fusion_attribution.py",
        "jepa4d/evaluation/phase2d_calibration_audit.py",
        "jepa4d/visualization/fusion_attribution_report.py",
        "jepa4d/evaluation/phase2c_source.py",
        "jepa4d/evaluation/phase2e_feature_cache.py",
        "jepa4d/evaluation/phase2e_final.py",
        "jepa4d/models/factorized_geometry.py",
        "jepa4d/benchmarks/geometry/sun_rgbd.py",
        "scripts/run_phase2d_fusion_attribution.py",
        "scripts/run_phase2d_latency_confirmation.py",
        "scripts/log_phase2d_diagnostics.py",
        "scripts/aggregate_phase2d_latency.py",
        "scripts/aggregate_phase2d_diagnostics.py",
        "scripts/build_phase2e_sunrgbd_feature_cache.py",
        "scripts/run_phase2e_factorized_shard.py",
        "scripts/evaluate_phase2e_final.py",
        "scripts/write_phase2_dependency_graphs.py",
        "slurm/validate_phase2d_diagnostics.py",
        "slurm/validate_phase2e_cache.py",
        "slurm/validate_phase2e_shard.py",
        "slurm/validate_phase2e_final.py",
    ]
    source: dict[str, Any] = {}
    for relative in source_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt = {
        "schema_version": "jepa4d-phase2d-test-receipt-v1",
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _command("git", "rev-parse", "HEAD", cwd=repo_root),
        "git_branch": _command("git", "branch", "--show-current", cwd=repo_root),
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST")
        },
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_report": {"path": str(cuda_report), "sha256": _sha256(cuda_report)},
        "source": source,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    typer.run(main)
