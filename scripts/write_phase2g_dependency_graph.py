#!/usr/bin/env python3
"""Validate and atomically write the formal Phase 2g-A Slurm graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slurm.phase2g_contract import (
    ACCOUNT,
    GRAPH_SCHEMA,
    MAX_SECONDS,
    PARTITIONS,
    SUBMISSION_POLICY,
    file_identity,
    sha256_file,
    validate_jobs,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _duration_seconds(value: str) -> int:
    fields = value.split(":")
    if len(fields) != 3:
        raise ValueError(f"SBATCH time must be HH:MM:SS, got {value}")
    hours, minutes, seconds = (int(item) for item in fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid SBATCH time: {value}")
    return hours * 3600 + minutes * 60 + seconds


def parse_sbatch(path: Path) -> dict[str, Any]:
    directives: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#SBATCH\s+--([^=\s]+)(?:=(\S+)|\s+(\S+))?$", line)
        if match:
            directives[match.group(1)] = match.group(2) or match.group(3) or "true"
    required = {"account", "partition", "nodes", "ntasks", "gres", "cpus-per-task", "mem", "time"}
    missing = sorted(required - set(directives))
    if missing:
        raise ValueError(f"{path} lacks required SBATCH directives: {missing}")
    if directives["account"] != ACCOUNT or directives["partition"] != PARTITIONS:
        raise ValueError(f"{path} account/partition policy drifted")
    if directives["nodes"] != "1" or directives["ntasks"] != "1" or directives["gres"] != "gpu:1":
        raise ValueError(f"{path} must request one node, task, and GPU")
    seconds = _duration_seconds(directives["time"])
    if seconds > MAX_SECONDS:
        raise ValueError(f"{path} exceeds the four-hour maximum")
    return {"directives": directives, "time_seconds": seconds, "gpu_requested": True}


def _parse_job(value: str, root: Path) -> dict[str, Any]:
    fields = value.split("|")
    if len(fields) != 7:
        raise ValueError("--job must be LABEL|ID|NAME|PARENTS|SUBMISSION_SBATCH|ENTRYPOINT|RECEIPT")
    label, job_id, name, raw_parents, raw_sbatch, raw_entrypoint, raw_receipt = fields
    if re.fullmatch(r"[A-Z0-9-]+", label) is None or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None:
        raise ValueError(f"invalid job specification: {value}")
    submission = (root / raw_sbatch).resolve(strict=True)
    entrypoint = (root / raw_entrypoint).resolve(strict=True)
    receipt = Path(raw_receipt)
    if not receipt.is_absolute():
        receipt = (root / receipt).resolve()
    return {
        "label": label,
        "job_id": job_id,
        "job_name": name,
        "parents": [] if raw_parents == "-" else raw_parents.split(","),
        "submission_sbatch": file_identity(submission, root=root),
        "entrypoint": file_identity(entrypoint, root=root),
        "resources": parse_sbatch(submission),
        "expected_receipt": str(receipt),
        "expected_success": str(receipt.parent / "SUCCESS"),
    }


def _path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return file_identity(resolved)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"source must be a regular file or directory: {path}")
    digest = hashlib.sha256()
    files = 0
    total = 0
    for child in sorted(value for value in resolved.rglob("*") if value.is_file() and not value.is_symlink()):
        relative = child.relative_to(resolved).as_posix()
        child_digest = sha256_file(child)
        digest.update(relative.encode("utf-8") + b"\0" + child_digest.encode("ascii") + b"\n")
        files += 1
        total += child.stat().st_size
    if files == 0:
        raise ValueError(f"source directory is empty: {path}")
    return {
        "path": str(resolved),
        "kind": "directory",
        "files": files,
        "bytes": total,
        "content_sha256": digest.hexdigest(),
    }


def _parse_source(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError(f"invalid --source: {value}")
    return name, Path(raw_path)


def discover_sources(root: Path) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for pattern in (
        "docs/experiments/*phase2g*preregistered.md",
        "jepa4d/**/*phase2g*.py",
        "scripts/*phase2g*.py",
        "slurm/phase2g_*.py",
        "slurm/phase2g_*.sbatch",
        "slurm/submit_phase2g.sh",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file() and "__pycache__" not in path.parts)
    return [file_identity(path, root=root) for path in sorted(paths)]


def build_graph(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve(strict=True)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    commit = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", "@{u}")
    if status or commit != upstream:
        raise RuntimeError("Phase 2g graph requires the exact clean pushed commit")
    jobs_list = [_parse_job(value, root) for value in args.job]
    jobs = {str(job["label"]): job for job in jobs_list}
    if len(jobs) != len(jobs_list) or len({job["job_id"] for job in jobs_list}) != len(jobs_list):
        raise ValueError("Phase 2g job labels and logical scheduler IDs must be unique")
    validate_jobs(jobs)
    sources: dict[str, Any] = {}
    for raw in args.source:
        name, path = _parse_source(raw)
        if name in sources:
            raise ValueError(f"duplicate source name: {name}")
        sources[name] = _path_identity(path)
    preflight_path = args.preflight.resolve(strict=True)
    preflight_value = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        not isinstance(preflight_value, dict)
        or preflight_value.get("schema_version") != "jepa4d-phase2g-preflight-v1"
        or preflight_value.get("status") != "pass"
        or preflight_value.get("execution_id") != args.execution_id
        or not isinstance(preflight_value.get("git"), dict)
        or preflight_value["git"].get("commit") != commit
    ):
        raise ValueError("Phase 2g graph requires a passing preflight receipt")
    registry_identity = file_identity(args.registry.resolve(strict=True))
    registry_receipt = preflight_value.get("registry")
    if not isinstance(registry_receipt, dict) or registry_receipt.get("sha256") != registry_identity["sha256"]:
        raise ValueError("preflight registry identity differs from graph input")
    registry_identity["semantic_sha256"] = registry_receipt.get("semantic_sha256")
    preregistration_identity = file_identity(args.preregistration.resolve(strict=True))
    readiness_identity = file_identity(args.readiness.resolve(strict=True))
    if (
        preflight_value.get("preregistration", {}).get("sha256") != preregistration_identity["sha256"]
        or preflight_value.get("readiness", {}).get("sha256") != readiness_identity["sha256"]
    ):
        raise ValueError("preflight preregistration/readiness identity differs from graph input")
    graph = {
        "schema_version": GRAPH_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": args.execution_id,
        "repository_root": str(root),
        "git_commit": commit,
        "git_branch": _git(root, "branch", "--show-current"),
        "git_upstream": _git(root, "rev-parse", "--abbrev-ref", "@{u}"),
        "git_clean": True,
        "git_pushed": True,
        "account": ACCOUNT,
        "partition_fallback": PARTITIONS,
        "submission_policy": SUBMISSION_POLICY,
        "preregistration": preregistration_identity,
        "preflight": file_identity(preflight_path),
        "registry": registry_identity,
        "ledger": file_identity(args.ledger.resolve(strict=True)),
        "readiness": readiness_identity,
        "test_receipt": str(args.test_receipt.resolve()),
        "output_root": str(args.output_root.resolve()),
        "jobs": jobs,
        "sources": sources,
        "orchestration_sources": discover_sources(root),
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--test-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--job", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only == (args.output is not None):
        raise ValueError("choose exactly one of --validate-only or --output")
    graph = build_graph(args)
    if args.validate_only:
        print(json.dumps({"status": "pass", "logical_jobs": len(graph["jobs"]), "submissions": 11}, sort_keys=True))
        return
    assert args.output is not None
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(graph, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
