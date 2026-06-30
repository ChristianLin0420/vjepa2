#!/usr/bin/env python3
"""Run the exact Phase 2f quality/CUDA suite and write the clean-commit test receipt."""

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

from jepa4d.evaluation.phase2f_metrics import publish_online_wandb
from slurm.phase2f_contract import (
    GRAPH_SCHEMA,
    PROVENANCE_SCHEMA,
    TEST_SCHEMA,
    atomic_json,
    file_identity,
    load_graph,
    reject_secrets,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _run(root: Path, log_root: Path, name: str, argv: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)
    duration = time.perf_counter() - started
    log = log_root / f"{name}.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"test command {name} failed ({result.returncode}); see {log}")
    return {
        "name": name,
        "argv": argv,
        "exit_code": result.returncode,
        "duration_seconds": duration,
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
        raise ValueError("test job ID differs from canonical graph")
    if args.output.resolve() != Path(str(graph["test_receipt"])).resolve():
        raise ValueError("test receipt output differs from canonical graph")
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if commit != graph["git_commit"] or status:
        raise RuntimeError("Phase 2f tests require the graph's exact clean commit")
    args.log_root.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("JEPA4D_PYTHON", str(root / ".conda-gpu/bin/python"))
    phase2f_mypy = sorted(
        str(path.relative_to(root))
        for pattern in ("scripts/*phase2f*.py", "slurm/phase2f_*.py")
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    )
    shell_files = sorted(str(path.relative_to(root)) for path in root.glob("slurm/phase2f_*.sbatch"))
    shell_files.append("slurm/submit_phase2f.sh")
    commands = [
        ("bash_syntax", ["bash", "-n", *shell_files]),
        ("compileall", [python, "-m", "compileall", "-q", "jepa4d", "scripts", "slurm"]),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", "jepa4d", "scripts", "slurm"]),
        ("ruff_check", [python, "-m", "ruff", "check", "jepa4d", "scripts", "slurm"]),
        ("mypy", [python, "-m", "mypy", "jepa4d", *phase2f_mypy]),
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
    results = [_run(root, args.log_root, name, argv) for name, argv in commands]
    pytest_text = (args.log_root / "pytest.log").read_text(encoding="utf-8")
    pytest_summary = next(
        (line.strip() for line in reversed(pytest_text.splitlines()) if re.search(r"\bpassed\b", line)), "unknown"
    )
    slurm = {
        "job_id": str(job["job_id"]),
        "job_name": os.environ.get("SLURM_JOB_NAME", job["job_name"]),
        "requested_partition_fallback": graph["partition_fallback"],
        "environment_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST", "unknown"),
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "execution_id": graph["execution_id"],
        "job_label": "T",
        "git_commit": commit,
        "git_branch": graph["git_branch"],
        "git_clean": True,
        "preregistration": graph["preregistration"],
        "preregistration_sha256": graph["preregistration"]["sha256"],
        "dependency_graph": file_identity(args.graph, schema=GRAPH_SCHEMA),
        "dependency_graph_sha256": graph_sha256,
        "slurm": slurm,
        "parents": [],
        "sources": graph["sources"],
    }
    receipt: dict[str, Any] = {
        "schema_version": TEST_SCHEMA,
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_branch": graph["git_branch"],
        "git_clean": True,
        "git_status": status,
        "dependency_graph_sha256": graph_sha256,
        "preregistration_sha256": graph["preregistration"]["sha256"],
        "commands": results,
        "pytest_summary": pytest_summary,
        "cuda_report": file_identity(args.log_root / "cuda-health.json"),
        "slurm": slurm,
        "execution_provenance": provenance,
    }
    reject_secrets(receipt)
    output = atomic_json(args.output, receipt)
    execution_id = str(graph["execution_id"])
    wandb = publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type="tests",
        run_name=f"{execution_id}-tests-{job['job_id']}",
        config={"execution_id": execution_id, "git_commit": commit, "graph_sha256": graph_sha256},
        summary={"status": "pass", "pytest_summary": pytest_summary},
        artifact_name=f"phase2f-tests-{execution_id}",
        artifact_files=(output, args.log_root / "cuda-health.json"),
    )
    receipt["wandb"] = wandb
    atomic_json(output, receipt)
    atomic_json(output.parent / "wandb_receipt.json", wandb)
    (output.parent / "SUCCESS").write_text("pass\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "pytest": pytest_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
