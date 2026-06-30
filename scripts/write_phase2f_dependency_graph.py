#!/usr/bin/env python3
"""Write the immutable Phase 2f Slurm dependency graph before held jobs are released."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "jepa4d-phase2f-dependency-graph-v1"
ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
PARTITIONS = "polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3"
MAX_SECONDS = 4 * 60 * 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "relative_path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _duration_seconds(value: str) -> int:
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError(f"SBATCH time must be HH:MM:SS, got {value}")
    hours, minutes, seconds = (int(item) for item in fields)
    return hours * 3600 + minutes * 60 + seconds


def parse_sbatch(path: Path) -> dict[str, Any]:
    directives: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#SBATCH\s+--([^=\s]+)(?:=(\S+)|\s+(\S+))?$", line)
        if match:
            directives[match.group(1)] = match.group(2) or match.group(3) or "true"
    required = {"account", "partition", "nodes", "ntasks", "cpus-per-task", "mem", "time"}
    missing = sorted(required - directives.keys())
    if missing:
        raise ValueError(f"{path} lacks required SBATCH directives: {missing}")
    if directives["account"] != ACCOUNT:
        raise ValueError(f"{path} account drifted")
    if directives["partition"] != PARTITIONS:
        raise ValueError(f"{path} partition fallback drifted")
    seconds = _duration_seconds(directives["time"])
    if seconds > MAX_SECONDS:
        raise ValueError(f"{path} exceeds four-hour limit")
    return {"directives": directives, "time_seconds": seconds, "gpu_requested": "gres" in directives}


def _parse_job(value: str, root: Path) -> dict[str, Any]:
    fields = value.split("|")
    if len(fields) != 6:
        raise ValueError("--job must be LABEL|JOB_ID|JOB_NAME|PARENTS|-relative-sbatch|receipt")
    label, job_id, job_name, parents_raw, sbatch_raw, receipt_raw = fields
    if not re.fullmatch(r"[A-Z0-9-]+", label) or not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) or not job_name:
        raise ValueError(f"invalid job specification: {value}")
    parents = [] if parents_raw == "-" else parents_raw.split(",")
    sbatch_path = (root / sbatch_raw).resolve(strict=True)
    sbatch_identity = file_identity(sbatch_path, root)
    resources = parse_sbatch(sbatch_path)
    receipt = Path(receipt_raw)
    if not receipt.is_absolute():
        receipt = (root / receipt).resolve()
    return {
        "label": label,
        "job_id": job_id,
        "job_name": job_name,
        "parents": parents,
        "sbatch": sbatch_identity,
        "resources": resources,
        "expected_receipt": str(receipt),
        "expected_success": str(receipt.parent / "SUCCESS"),
    }


def expected_labels() -> set[str]:
    labels = {"T", "A", "C", "Q", "LA", "PG", "S", "E", "Z"}
    labels.update(f"L{index:02d}" for index in range(12))
    labels.update(f"P{index}" for index in range(4))
    labels.update(
        f"F-{arm}-R{rotation}-S{seed}"
        for arm in ("M0", "M1", "M2", "M3")
        for rotation in range(4)
        for seed in range(3)
    )
    return labels


def validate_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    expected = expected_labels()
    if set(jobs) != expected:
        raise ValueError(
            f"Phase 2f graph labels differ: missing={sorted(expected - set(jobs))}, extra={sorted(set(jobs) - expected)}"
        )
    for label, job in jobs.items():
        unknown = set(job["parents"]) - set(jobs)
        if unknown or label in job["parents"]:
            raise ValueError(f"invalid parents for {label}: {job['parents']}")
        if not bool(job["resources"]["gpu_requested"]):
            raise ValueError(f"approved Phase 2f partitions require one GPU for every job: {label}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(label: str) -> None:
        if label in visiting:
            raise ValueError("dependency graph contains a cycle")
        if label in visited:
            return
        visiting.add(label)
        for parent in jobs[label]["parents"]:
            visit(parent)
        visiting.remove(label)
        visited.add(label)

    for label in jobs:
        visit(label)


def discover_sources(root: Path) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    patterns = (
        "docs/experiments/*phase2f*preregistered.md",
        "docs/experiments/*phase2f*scheduler-amendment.md",
        "jepa4d/**/*phase2f*.py",
        "scripts/*phase2f*.py",
        "slurm/phase2f_*.py",
        "slurm/phase2f_*.sbatch",
        "slurm/submit_phase2f.sh",
    )
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file() and "__pycache__" not in path.parts)
    return [file_identity(path, root) for path in sorted(paths)]


def write_graph(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve(strict=True)
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Phase 2f graph requires a clean committed worktree")
    commit = _git(root, "rev-parse", "HEAD")
    jobs_list = [_parse_job(value, root) for value in args.job]
    jobs = {job["label"]: job for job in jobs_list}
    if len(jobs) != len(jobs_list) or len({job["job_id"] for job in jobs_list}) != len(jobs_list):
        raise ValueError("job labels and Slurm IDs must be unique")
    validate_jobs(jobs)
    preregistration = file_identity(args.preregistration.resolve(strict=True), root)
    graph = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": args.execution_id,
        "repository_root": str(root),
        "git_commit": commit,
        "git_branch": _git(root, "branch", "--show-current"),
        "git_clean": True,
        "account": ACCOUNT,
        "partition_fallback": PARTITIONS,
        "submission_policy": {
            "all_jobs_submitted_held": True,
            "release_after_atomic_graph_write": True,
            "only_root_without_dependency": "T",
            "dependency_type": "afterok",
            "logical_job_count": 73,
            "scheduler_submission_count": 12,
            "max_parallel_tasks": 8,
            "array_task_throttle": 8,
        },
        "preregistration": preregistration,
        "test_receipt": str(args.test_receipt.resolve()),
        "output_root": str(args.output_root.resolve()),
        "jobs": jobs,
        "sources": discover_sources(root),
    }
    roots = [label for label, job in jobs.items() if not job["parents"]]
    if roots != ["T"]:
        raise ValueError(f"only T may be runnable at release, found roots={roots}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(graph, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", action="append", default=[], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    value = write_graph(parse_args())
    print(json.dumps({"execution_id": value["execution_id"], "jobs": len(value["jobs"])}, sort_keys=True))
