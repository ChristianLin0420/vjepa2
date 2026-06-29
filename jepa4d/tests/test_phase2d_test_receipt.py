import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from slurm.validate_phase2d_test_receipt import validate_receipt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=repo, check=True)
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    cuda_path = tmp_path / "cuda.json"
    cuda = {
        "schema_version": "jepa4d-cuda-health-v2",
        "status": "pass",
        "errors": [],
        "cuda_available": True,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "selected_device": {"name": "fixture-gpu"},
        "stress": {"finite": True, "requested_seconds": 20.0, "allocation_mib": 1024},
        "slurm": {"job_id": "123", "job_name": "phase2d-tests"},
    }
    cuda_path.write_text(json.dumps(cuda) + "\n")
    receipt = {
        "schema_version": "jepa4d-phase2d-test-receipt-v1",
        "status": "pass",
        "git_commit": commit,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "slurm": {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NAME": "phase2d-tests",
            "SLURM_JOB_PARTITION": "polar4",
            "SLURM_JOB_NODELIST": "fixture-node",
        },
        "cuda_report": {"path": str(cuda_path), "sha256": _sha256(cuda_path)},
        "source": {"source.py": {"bytes": source.stat().st_size, "sha256": _sha256(source)}},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    return repo, receipt_path, cuda_path


def test_validate_receipt_checks_runtime_and_cuda_evidence(tmp_path: Path) -> None:
    repo, receipt, _ = _receipt_fixture(tmp_path)
    assert validate_receipt(repo, receipt)["status"] == "pass"


def test_validate_receipt_accepts_approved_partition_request_list(tmp_path: Path) -> None:
    repo, receipt_path, _ = _receipt_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["slurm"]["SLURM_JOB_PARTITION"] = "polar4,polar3,polar,batch_block1"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    assert validate_receipt(repo, receipt_path)["status"] == "pass"


@pytest.mark.parametrize("partitions", ["polar4,unapproved", "polar4,,polar3", ""])
def test_validate_receipt_rejects_unapproved_partition_request_list(tmp_path: Path, partitions: str) -> None:
    repo, receipt_path, _ = _receipt_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["slurm"]["SLURM_JOB_PARTITION"] = partitions
    receipt_path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(RuntimeError, match="partition outside|complete Slurm allocation"):
        validate_receipt(repo, receipt_path)


def test_validate_receipt_rejects_changed_cuda_report(tmp_path: Path) -> None:
    repo, receipt, cuda = _receipt_fixture(tmp_path)
    cuda.write_text("{}\n")
    with pytest.raises(RuntimeError, match="CUDA report identity has changed"):
        validate_receipt(repo, receipt)
