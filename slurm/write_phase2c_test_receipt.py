"""Write the content-bound passing-test receipt consumed by Phase 2c."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slurm.phase2b_gate import environment_fingerprint, repository_fingerprint, sha256  # noqa: E402
from slurm.phase2c_gate import atomic_json, protocol_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Phase 2c test receipts must be produced by Slurm")
    cuda_report = json.loads(args.cuda_report.read_text())
    if cuda_report.get("status") != "pass":
        raise RuntimeError("CUDA test report does not pass")
    report = {
        "schema_version": "jepa4d-phase2c-tests-v1",
        "status": "pass",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": job_id,
        "protocol": protocol_contract(),
        "repository": repository_fingerprint(args.repo_root),
        "environment": environment_fingerprint(),
        "cuda_report": {
            "path": str(args.cuda_report.resolve(strict=True)),
            "sha256": sha256(args.cuda_report),
            "summary": cuda_report,
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
