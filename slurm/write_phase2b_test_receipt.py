"""Write the durable receipt consumed by Phase 2b preflight and training."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from slurm.phase2b_gate import environment_fingerprint, repository_fingerprint, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cuda_report = json.loads(args.cuda_report.read_text())
    if cuda_report.get("status") != "pass":
        raise RuntimeError(f"CUDA test report is not passing: {cuda_report.get('status')}")
    report: dict[str, Any] = {
        "schema_version": "jepa4d-phase2b-tests-v1",
        "status": "pass",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "repository": repository_fingerprint(args.repo_root),
        "environment": environment_fingerprint(),
        "cuda_report": {
            "path": str(args.cuda_report.resolve()),
            "sha256": sha256(args.cuda_report),
            "summary": cuda_report,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
