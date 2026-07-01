#!/usr/bin/env python3
"""Run the formal Phase 2g-A clean-commit test and CUDA gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slurm.phase2g_contract import TEST_SCHEMA, atomic_json, file_identity, load_graph
from slurm.phase2g_stage_gate import publish_online_wandb


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _run(root: Path, logs: Path, name: str, argv: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    log = logs / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"formal Phase 2g test command {name} failed; see {log}")
    return {
        "name": name,
        "argv": argv,
        "exit_code": 0,
        "duration_seconds": time.perf_counter() - started,
        "log": file_identity(log),
        "summary_tail": (result.stdout + result.stderr).splitlines()[-3:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    graph, graph_sha256 = load_graph(args.graph)
    job = graph["jobs"]["T"]
    if os.environ.get("SLURM_JOB_ID", str(job["job_id"])) != str(job["job_id"]):
        raise ValueError("test allocation differs from the Phase 2g graph")
    commit = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", "@{u}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != graph["git_commit"] or upstream != commit or status:
        raise RuntimeError("formal Phase 2g tests require the graph's exact clean pushed commit")
    args.log_root.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("JEPA4D_PYTHON", str(root / ".conda-gpu/bin/python"))
    shell_files = sorted(str(path.relative_to(root)) for path in root.glob("slurm/phase2g_*.sbatch"))
    shell_files.append("slurm/submit_phase2g.sh")
    python_files = sorted(
        str(path.relative_to(root))
        for pattern in ("jepa4d/**/*phase2g*.py", "scripts/*phase2g*.py", "slurm/phase2g_*.py")
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    )
    commands = [
        ("bash_syntax", ["bash", "-n", *shell_files]),
        ("compileall", [python, "-m", "compileall", "-q", *python_files]),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", *python_files]),
        ("ruff_check", [python, "-m", "ruff", "check", *python_files]),
        ("mypy", [python, "-m", "mypy", *python_files]),
        ("pytest", [python, "-m", "pytest", "jepa4d/tests", "-ra", "-q"]),
        (
            "cuda_health",
            [
                python,
                "scripts/check_cuda.py",
                "--device",
                "0",
                "--stress-seconds",
                "20",
                "--matrix-size",
                "4096",
                "--allocation-mib",
                "1024",
                "--json-output",
                str(args.log_root / "cuda-health.json"),
            ],
        ),
    ]
    results = [_run(root, args.log_root, name, command) for name, command in commands]
    pytest_log = (args.log_root / "pytest.log").read_text(encoding="utf-8")
    pytest_summary = next(
        (line.strip() for line in reversed(pytest_log.splitlines()) if re.search(r"\bpassed\b", line)), "unknown"
    )
    provenance = {
        "schema_version": "jepa4d-phase2g-execution-provenance-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": graph["execution_id"],
        "job_label": "T",
        "git_commit": commit,
        "git_branch": graph["git_branch"],
        "git_clean": True,
        "git_pushed": True,
        "preregistration_sha256": graph["preregistration"]["sha256"],
        "preflight_sha256": graph["preflight"]["sha256"],
        "test_receipt_sha256": "self",
        "dependency_graph_sha256": graph_sha256,
        "slurm": {
            "job_id": str(job["job_id"]),
            "allocation_job_id": str(job["job_id"]),
            "job_name": os.environ.get("SLURM_JOB_NAME", job["job_name"]),
            "account": graph["account"],
            "partition_fallback": graph["partition_fallback"],
            "actual_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
            "resources": job["resources"],
        },
        "parents": [],
        "sources": {},
        "data_access_decision": {
            "preflight": graph["preflight"],
            "registry": graph["registry"],
            "readiness": graph["readiness"],
            "data_access_authorized": True,
            "sun_dataset_id": "sun-rgbd.geometry-development",
            "external_final_authorized": False,
        },
        "external_final_authorized": False,
    }
    summary_path = atomic_json(
        args.log_root / "test_summary.json",
        {
            "schema_version": "jepa4d-phase2g-test-summary-v1",
            "status": "pass",
            "pytest_summary": pytest_summary,
            "command_count": len(results),
        },
    )
    receipt: dict[str, Any] = {
        "schema_version": TEST_SCHEMA,
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_branch": graph["git_branch"],
        "git_clean": True,
        "git_pushed": True,
        "dependency_graph_sha256": graph_sha256,
        "preregistration_sha256": graph["preregistration"]["sha256"],
        "commands": results,
        "pytest_summary": pytest_summary,
        "cuda_report": file_identity(args.log_root / "cuda-health.json"),
        "execution_provenance": provenance,
    }
    receipt["wandb"] = publish_online_wandb(
        provenance=provenance,
        job_type="tests",
        artifact_files=(summary_path, args.log_root / "cuda-health.json"),
        summary={"pytest_pass": True, "command_count": len(results)},
    )
    atomic_json(args.output, receipt)
    atomic_json(args.output.parent / "wandb_receipt.json", receipt["wandb"])
    (args.output.parent / "SUCCESS").write_text("pass\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "pytest": pytest_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
