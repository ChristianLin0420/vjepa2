"""Fail closed unless a Phase 2d/2e job matches its passing test receipt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Annotated

import torch
import typer

ALLOWED_PARTITIONS = {"polar4", "polar3", "polar", "batch_block1", "grizzly", "batch_block2", "batch_block3"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(repo_root: Path, receipt_path: Path) -> dict[str, object]:
    repo_root = repo_root.resolve(strict=True)
    receipt = json.loads(receipt_path.resolve(strict=True).read_text())
    if receipt.get("schema_version") != "jepa4d-phase2d-test-receipt-v1" or receipt.get("status") != "pass":
        raise RuntimeError("Phase 2d test receipt does not pass")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip()
    if commit != receipt.get("git_commit"):
        raise RuntimeError("current commit differs from the passing test receipt")
    if status:
        raise RuntimeError("current worktree is not clean")
    slurm = receipt.get("slurm")
    if not isinstance(slurm, dict) or any(
        not isinstance(slurm.get(key), str) or not slurm[key]
        for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST")
    ):
        raise RuntimeError("test receipt is not bound to a complete Slurm allocation")
    requested_partitions = [value.strip() for value in slurm["SLURM_JOB_PARTITION"].split(",")]
    if not requested_partitions or any(not value or value not in ALLOWED_PARTITIONS for value in requested_partitions):
        raise RuntimeError("test receipt requested a partition outside the approved Phase 2d/2e set")
    if receipt.get("torch") != torch.__version__ or receipt.get("cuda_build") != torch.version.cuda:
        raise RuntimeError("current PyTorch/CUDA build differs from the passing test environment")
    cuda_identity = receipt.get("cuda_report")
    if not isinstance(cuda_identity, dict) or not isinstance(cuda_identity.get("path"), str):
        raise RuntimeError("test receipt has no CUDA report identity")
    cuda_path = Path(cuda_identity["path"]).resolve(strict=True)
    if _sha256(cuda_path) != cuda_identity.get("sha256"):
        raise RuntimeError("passing CUDA report identity has changed")
    cuda = json.loads(cuda_path.read_text())
    stress = cuda.get("stress")
    if (
        cuda.get("schema_version") != "jepa4d-cuda-health-v2"
        or cuda.get("status") != "pass"
        or cuda.get("errors") != []
        or cuda.get("cuda_available") is not True
        or cuda.get("torch") != receipt.get("torch")
        or cuda.get("torch_cuda_build") != receipt.get("cuda_build")
        or not isinstance(cuda.get("selected_device"), dict)
        or not isinstance(stress, dict)
        or stress.get("finite") is not True
        or float(stress.get("requested_seconds", -1)) < 20.0
        or int(stress.get("allocation_mib", -1)) < 1024
        or cuda.get("slurm", {}).get("job_id") != slurm["SLURM_JOB_ID"]
        or cuda.get("slurm", {}).get("job_name") != slurm["SLURM_JOB_NAME"]
    ):
        raise RuntimeError("recorded CUDA health/stress report does not satisfy the frozen test gate")
    for relative, identity in receipt["source"].items():
        path = repo_root / relative
        if path.stat().st_size != int(identity["bytes"]) or _sha256(path) != identity["sha256"]:
            raise RuntimeError(f"source identity differs from passing receipt: {relative}")
    return receipt


def main(
    repo_root: Annotated[Path, typer.Option("--repo-root")],
    receipt_path: Annotated[Path, typer.Option("--receipt")],
) -> None:
    validate_receipt(repo_root, receipt_path)


if __name__ == "__main__":
    typer.run(main)
