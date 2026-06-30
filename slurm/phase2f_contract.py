#!/usr/bin/env python3
"""Shared Phase 2f graph, parent, provenance, skip, and W&B receipt contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2f_metrics import publish_online_wandb, require_finite_tree

GRAPH_SCHEMA = "jepa4d-phase2f-dependency-graph-v1"
TEST_SCHEMA = "jepa4d-phase2f-test-receipt-v1"
PROVENANCE_SCHEMA = "jepa4d-phase2f-execution-provenance-v1"
WANDB_SCHEMA = "jepa4d-phase2f-wandb-artifact-receipt-v1"
ALLOWED_PARTITIONS = {"polar4", "polar3", "polar", "batch_block1", "grizzly", "batch_block2", "batch_block3"}
SUCCESS_STATUSES = {
    "success",
    "pass",
    "skipped_not_qualified",
    "skipped_not_latency_qualified",
    "skipped_no_survivor",
    "no_survivor",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    require_finite_tree(value, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def file_identity(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if schema is not None:
        value["schema"] = schema
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    require_finite_tree(value, str(path))
    return value


def reject_secrets(value: Any, location: str = "receipt") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(term in lowered for term in ("api_key", "token", "secret", "password", "netrc")):
                raise ValueError(f"credential-like field forbidden at {location}.{key}")
            reject_secrets(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_secrets(child, f"{location}[{index}]")


def load_graph(path: Path) -> tuple[dict[str, Any], str]:
    graph = load_json(path)
    if graph.get("schema_version") != GRAPH_SCHEMA:
        raise ValueError("unexpected Phase 2f dependency graph schema")
    jobs = graph.get("jobs")
    if not isinstance(jobs, dict) or len(jobs) != 73 or graph.get("git_clean") is not True:
        raise ValueError("Phase 2f dependency graph is incomplete")
    if graph.get("submission_policy") != {
        "all_jobs_submitted_held": True,
        "dependency_type": "afterok",
        "only_root_without_dependency": "T",
        "release_after_atomic_graph_write": True,
    }:
        raise ValueError("Phase 2f graph submission policy drifted")
    return graph, sha256_file(path.resolve(strict=True))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _actual_partition(job_id: str) -> str:
    if shutil_which("scontrol") and job_id.isdigit():
        result = subprocess.run(("scontrol", "show", "job", "-o", job_id), capture_output=True, text=True, check=False)
        match = re.search(r"(?:^|\s)Partition=(\S+)", result.stdout)
        if result.returncode == 0 and match:
            return match.group(1)
    raw = os.environ.get("SLURM_JOB_PARTITION", "unknown")
    return raw if "," not in raw else "unknown_from_fallback_environment"


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def scheduler_completed(job_id: str) -> bool:
    if not shutil_which("sacct") or not job_id.isdigit():
        return False
    result = subprocess.run(
        ("sacct", "-n", "-X", "-j", job_id, "--format=State,ExitCode", "--parsable2"),
        capture_output=True,
        text=True,
        check=False,
    )
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    return result.returncode == 0 and any(row[0].split("+")[0] == "COMPLETED" and row[1] == "0:0" for row in rows)


def validate_wandb(receipt: Mapping[str, Any]) -> None:
    value = receipt.get("wandb")
    if not isinstance(value, Mapping):
        raise ValueError("receipt lacks embedded W&B identity")
    required = ("run_id", "run_url", "artifact_name", "artifact_version", "artifact_id", "artifact_digest")
    if (
        value.get("schema_version") != WANDB_SCHEMA
        or value.get("mode") != "online"
        or value.get("status") != "success"
    ):
        raise ValueError("W&B receipt is not a successful online upload")
    if any(not value.get(key) for key in required):
        raise ValueError("W&B receipt identity is incomplete")


def validate_test_receipt(path: Path, graph: Mapping[str, Any], graph_sha256: str) -> dict[str, Any]:
    receipt = load_json(path)
    if receipt.get("schema_version") != TEST_SCHEMA or receipt.get("status") != "pass":
        raise ValueError("Phase 2f test receipt did not pass")
    if receipt.get("git_commit") != graph.get("git_commit") or receipt.get("dependency_graph_sha256") != graph_sha256:
        raise ValueError("test receipt commit/graph binding mismatch")
    if receipt.get("git_clean") is not True or str(path.resolve()) != str(Path(str(graph["test_receipt"])).resolve()):
        raise ValueError("test receipt clean/path binding mismatch")
    validate_wandb(receipt)
    return receipt


def validate_parent_receipt(
    *, graph: Mapping[str, Any], child_label: str, parent_label: str, path: Path
) -> dict[str, Any]:
    child = graph["jobs"][child_label]
    if parent_label not in child["parents"]:
        raise ValueError(f"{parent_label} is not a registered parent of {child_label}")
    parent = graph["jobs"][parent_label]
    if path.resolve() != Path(str(parent["expected_receipt"])).resolve():
        raise ValueError(f"parent receipt path differs for {parent_label}")
    if not Path(str(parent["expected_success"])).is_file():
        raise ValueError(f"parent {parent_label} lacks SUCCESS")
    if not scheduler_completed(str(parent["job_id"])):
        raise ValueError(f"parent {parent_label} is not scheduler COMPLETED 0:0")
    receipt = load_json(path)
    if receipt.get("status") not in SUCCESS_STATUSES:
        raise ValueError(f"parent {parent_label} status is not successful/registered skip")
    if parent_label != "T":
        provenance = receipt.get("execution_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"parent {parent_label} lacks execution_provenance")
        keys = (
            "execution_id",
            "git_commit",
            "preregistration_sha256",
            "test_receipt_sha256",
            "dependency_graph_sha256",
        )
        expected = {
            "execution_id": graph["execution_id"],
            "git_commit": graph["git_commit"],
            "preregistration_sha256": graph["preregistration"]["sha256"],
        }
        if any(provenance.get(key) != expected[key] for key in expected) or any(
            not provenance.get(key) for key in keys
        ):
            raise ValueError(f"parent {parent_label} execution identity mismatch")
    validate_wandb(receipt)
    return receipt


def build_provenance(
    graph_path: Path,
    label: str,
    test_receipt_path: Path,
    parents: Mapping[str, Path],
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    graph, graph_hash = load_graph(graph_path)
    jobs = graph["jobs"]
    if label not in jobs:
        raise ValueError(f"job label absent from graph: {label}")
    current_job_id = os.environ.get("SLURM_JOB_ID", str(jobs[label]["job_id"]))
    if str(jobs[label]["job_id"]) != current_job_id:
        raise ValueError(f"current Slurm job {current_job_id} != graph {jobs[label]['job_id']}")
    expected_parents = set(jobs[label]["parents"])
    if set(parents) != expected_parents:
        raise ValueError(f"parent set mismatch for {label}: {set(parents)} != {expected_parents}")
    root = Path(str(graph["repository_root"])).resolve(strict=True)
    if _git(root, "rev-parse", "HEAD") != graph["git_commit"]:
        raise ValueError("execution commit differs from graph")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Phase 2f execution requires a clean worktree")
    test_receipt = validate_test_receipt(test_receipt_path, graph, graph_hash)
    parent_identities = []
    for parent_label in jobs[label]["parents"]:
        receipt = validate_parent_receipt(
            graph=graph, child_label=label, parent_label=parent_label, path=parents[parent_label]
        )
        parent_identities.append(
            {
                "label": parent_label,
                "job_id": jobs[parent_label]["job_id"],
                "receipt": file_identity(parents[parent_label], schema=str(receipt.get("schema_version", "unknown"))),
                "status": receipt["status"],
                "wandb": dict(receipt["wandb"]),
            }
        )
    actual_partition = _actual_partition(current_job_id)
    if actual_partition not in ALLOWED_PARTITIONS and actual_partition != "unknown_from_fallback_environment":
        raise ValueError(f"job allocated on unregistered partition {actual_partition}")
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": graph["execution_id"],
        "job_label": label,
        "git_commit": graph["git_commit"],
        "git_branch": graph["git_branch"],
        "git_clean": True,
        "preregistration": dict(graph["preregistration"]),
        "preregistration_sha256": graph["preregistration"]["sha256"],
        "test_receipt": file_identity(test_receipt_path, schema=TEST_SCHEMA),
        "test_receipt_sha256": sha256_file(test_receipt_path),
        "test_job_id": graph["jobs"]["T"]["job_id"],
        "dependency_graph": file_identity(graph_path, schema=GRAPH_SCHEMA),
        "dependency_graph_sha256": graph_hash,
        "slurm": {
            "job_id": current_job_id,
            "job_name": os.environ.get("SLURM_JOB_NAME", jobs[label]["job_name"]),
            "account": graph["account"],
            "requested_partition_fallback": graph["partition_fallback"],
            "actual_partition": actual_partition,
            "node_list": os.environ.get("SLURM_JOB_NODELIST", "unknown"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", "unknown"),
            "job_resources": dict(jobs[label]["resources"]),
        },
        "parents": parent_identities,
        "sources": {name: file_identity(path) for name, path in sorted(sources.items())},
        "test_receipt_status": test_receipt["status"],
    }
    reject_secrets(provenance, "execution_provenance")
    require_finite_tree(provenance, "execution_provenance")
    return provenance


def _parse_pairs(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"invalid/duplicate {label}: {value}")
        result[name] = Path(raw_path)
    return result


def command_emit(args: argparse.Namespace) -> None:
    provenance = build_provenance(
        args.graph,
        args.label,
        args.test_receipt,
        _parse_pairs(args.parent, "parent"),
        _parse_pairs(args.source, "source"),
    )
    atomic_json(args.output, provenance)


def _publish_for_receipt(
    *, receipt_path: Path, receipt: dict[str, Any], provenance: Mapping[str, Any], job_type: str, files: Sequence[Path]
) -> dict[str, Any]:
    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    return publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type=job_type,
        run_name=f"{execution_id}-{job_type}-{job_id}",
        config={
            "execution_id": execution_id,
            "git_commit": provenance["git_commit"],
            "job_label": provenance["job_label"],
        },
        summary={"status": receipt["status"]},
        artifact_name=f"phase2f-{job_type}-{execution_id}-{job_id}",
        artifact_files=(receipt_path, *files),
    )


def command_finalize(args: argparse.Namespace) -> None:
    receipt = load_json(args.receipt)
    provenance = load_json(args.provenance)
    receipt["status"] = args.status
    receipt["execution_provenance"] = provenance
    reject_secrets(receipt)
    atomic_json(args.receipt, receipt)
    extra = [path.resolve(strict=True) for path in args.artifact_file]
    wandb = _publish_for_receipt(
        receipt_path=args.receipt.resolve(strict=True),
        receipt=receipt,
        provenance=provenance,
        job_type=args.job_type,
        files=extra,
    )
    receipt["wandb"] = wandb
    atomic_json(args.receipt, receipt)
    atomic_json(args.receipt.parent / "wandb_receipt.json", wandb)
    (args.receipt.parent / "SUCCESS").write_text(f"{args.status}\n", encoding="utf-8")


def command_skip(args: argparse.Namespace) -> None:
    provenance = load_json(args.provenance)
    receipt: dict[str, Any] = {
        "schema_version": args.schema,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": args.status,
        "stage": args.stage,
        "arm": args.arm,
        "rotation": args.rotation,
        "seed": args.seed,
        "optimizer_steps": 0,
        "reason": args.reason,
        "execution_provenance": provenance,
    }
    reject_secrets(receipt)
    atomic_json(args.receipt, receipt)
    wandb = _publish_for_receipt(
        receipt_path=args.receipt.resolve(strict=True),
        receipt=receipt,
        provenance=provenance,
        job_type=args.job_type,
        files=(),
    )
    receipt["wandb"] = wandb
    atomic_json(args.receipt, receipt)
    atomic_json(args.receipt.parent / "wandb_receipt.json", wandb)
    (args.receipt.parent / "SUCCESS").write_text(f"{args.status}\n", encoding="utf-8")


def command_allowed(args: argparse.Namespace) -> None:
    gate = load_json(args.gate)
    allowlist = gate.get(args.field)
    if not isinstance(allowlist, list):
        raise ValueError(f"gate lacks list field {args.field}")
    if args.arm not in allowlist:
        raise SystemExit(1)


def command_list(args: argparse.Namespace) -> None:
    gate = load_json(args.gate)
    allowlist = gate.get(args.field)
    if not isinstance(allowlist, list):
        raise ValueError(f"gate lacks list field {args.field}")
    for arm in allowlist:
        print(str(arm))


def command_verify(args: argparse.Namespace) -> None:
    receipt = load_json(args.receipt)
    provenance = load_json(args.provenance)
    if receipt.get("status") not in SUCCESS_STATUSES:
        raise ValueError("receipt status is not successful")
    if receipt.get("execution_provenance") != provenance:
        raise ValueError("receipt did not embed the exact current-job execution provenance")
    validate_wandb(receipt)
    (args.receipt.parent / "SUCCESS").write_text(f"{receipt['status']}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit-provenance")
    emit.add_argument("--graph", type=Path, required=True)
    emit.add_argument("--label", required=True)
    emit.add_argument("--test-receipt", type=Path, required=True)
    emit.add_argument("--parent", action="append", default=[])
    emit.add_argument("--source", action="append", default=[])
    emit.add_argument("--output", type=Path, required=True)
    emit.set_defaults(func=command_emit)

    finalize = subparsers.add_parser("finalize-receipt")
    finalize.add_argument("--receipt", type=Path, required=True)
    finalize.add_argument("--provenance", type=Path, required=True)
    finalize.add_argument("--job-type", required=True)
    finalize.add_argument("--status", default="success")
    finalize.add_argument("--artifact-file", type=Path, action="append", default=[])
    finalize.set_defaults(func=command_finalize)

    skip = subparsers.add_parser("write-skip")
    skip.add_argument("--receipt", type=Path, required=True)
    skip.add_argument("--provenance", type=Path, required=True)
    skip.add_argument("--schema", required=True)
    skip.add_argument("--status", required=True)
    skip.add_argument("--stage", required=True)
    skip.add_argument("--arm")
    skip.add_argument("--rotation")
    skip.add_argument("--seed", type=int)
    skip.add_argument("--reason", required=True)
    skip.add_argument("--job-type", required=True)
    skip.set_defaults(func=command_skip)

    allowed = subparsers.add_parser("arm-allowed")
    allowed.add_argument("--gate", type=Path, required=True)
    allowed.add_argument("--field", required=True)
    allowed.add_argument("--arm", required=True)
    allowed.set_defaults(func=command_allowed)

    listing = subparsers.add_parser("list-allowlist")
    listing.add_argument("--gate", type=Path, required=True)
    listing.add_argument("--field", required=True)
    listing.set_defaults(func=command_list)
    verify = subparsers.add_parser("verify-receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--provenance", type=Path, required=True)
    verify.set_defaults(func=command_verify)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
