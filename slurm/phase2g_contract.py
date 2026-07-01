#!/usr/bin/env python3
"""Shared contracts for the formal Phase 2g-A Slurm graph.

This module deliberately contains orchestration and receipt validation only.
Scientific training, evaluation, and selection live in the formal runner
entrypoints under :mod:`scripts`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

GRAPH_SCHEMA: Final = "jepa4d-phase2g-dependency-graph-v1"
PREFLIGHT_SCHEMA: Final = "jepa4d-phase2g-preflight-v1"
PROVENANCE_SCHEMA: Final = "jepa4d-phase2g-execution-provenance-v1"
TEST_SCHEMA: Final = "jepa4d-phase2g-test-receipt-v1"
WANDB_SCHEMA: Final = "jepa4d-phase2g-wandb-artifact-receipt-v1"

ACCOUNT: Final = "edgeai_tao-ptm_image-foundation-model-clip"
PARTITIONS: Final = "polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3"
ALLOWED_PARTITIONS: Final = frozenset(PARTITIONS.split(","))
MAX_SECONDS: Final = 4 * 60 * 60

ARMS: Final = ("M0", "M1", "M2", "M3")
ROTATIONS: Final = ("R0", "R1", "R2", "R3")
LEARNING_RATES: Final = ("0.0005", "0.001", "0.002")
SEEDS: Final = (0, 1, 2)
FAMILIES: Final = ("kv1", "kv2", "realsense", "xtion")
ROTATION_FAMILIES: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "R0": {"train": ("kv1", "xtion"), "validation": ("realsense",), "heldout": ("kv2",)},
    "R1": {"train": ("xtion", "realsense"), "validation": ("kv2",), "heldout": ("kv1",)},
    "R2": {"train": ("realsense", "kv2"), "validation": ("kv1",), "heldout": ("xtion",)},
    "R3": {"train": ("kv2", "kv1"), "validation": ("xtion",), "heldout": ("realsense",)},
}

LOGICAL_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+(?:_[0-9]+)?")
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_CREDENTIAL_PATTERN: Final = re.compile(
    r"wandb_v1_[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{20,}|(?:api[_-]?key|password|secret|token)\s*[:=]",
    re.IGNORECASE,
)
_DIODE_PATH_PATTERN: Final = re.compile(r"(?i)(?:^|[/\\])diode(?:[/\\]|$)|(?:^|[/\\])val\.tar\.gz(?:$|[/\\])")

SUBMISSION_POLICY: Final = {
    "all_jobs_submitted_held": True,
    "dependency_type": "afterok",
    "only_root_without_dependency": "T",
    "release_after_atomic_graph_write": True,
    "logical_job_count": 152,
    "scheduler_submission_count": 11,
    "max_parallel_tasks": 8,
    "array_task_throttle": 8,
    "external_final_authorized": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value: Any, location: str = "value") -> None:
    if value is None or isinstance(value, bool | str | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {location}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")
        return
    raise TypeError(f"unsupported value at {location}: {type(value).__name__}")


def reject_credentials(value: Any, location: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = {token for token in re.split(r"[_.-]+", str(key).casefold()) if token}
            if tokens & {"apikey", "authorization", "credential", "password", "secret", "token"}:
                raise ValueError(f"credential-like field at {location}.{key}")
            reject_credentials(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            reject_credentials(child, f"{location}[{index}]")
    elif isinstance(value, str) and _CREDENTIAL_PATTERN.search(value):
        raise ValueError(f"credential-like text at {location}")


def reject_diode_paths(value: Any, location: str = "value") -> None:
    """Reject DIODE archive/root paths while allowing policy labels mentioning DIODE."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if "diode" in normalized and any(token in normalized for token in ("archive", "path", "root", "target")):
                raise ValueError(f"DIODE path-bearing field at {location}.{key}")
            reject_diode_paths(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            reject_diode_paths(child, f"{location}[{index}]")
    elif isinstance(value, str) and (_DIODE_PATH_PATTERN.search(value) or value.startswith(("file://", "s3://"))):
        raise ValueError(f"external-final path at {location}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    finite_tree(value, str(path))
    reject_credentials(value, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"receipt must be a regular file: {path}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    finite_tree(value, str(path))
    reject_credentials(value, str(path))
    return value


def file_identity(path: Path, *, root: Path | None = None, schema: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"identity requires a regular non-symlink file: {path}")
    value: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if root is not None:
        value["relative_path"] = resolved.relative_to(root.resolve(strict=True)).as_posix()
    if schema is not None:
        value["schema"] = schema
    return value


def expected_labels() -> set[str]:
    labels = {"T", "O", "C", "Q", "HG", "S", "G", "Z"}
    labels.update(
        f"H-{arm}-{rotation}-L{lr_index}"
        for arm in ARMS
        for rotation in ROTATIONS
        for lr_index in range(len(LEARNING_RATES))
    )
    labels.update(f"F-{arm}-{rotation}-S{seed}" for arm in ARMS for rotation in ROTATIONS for seed in SEEDS)
    labels.update(f"V-{arm}-{rotation}-S{seed}" for arm in ARMS for rotation in ROTATIONS for seed in SEEDS)
    return labels


def parent_map() -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {"T": [], "O": ["T"], "C": ["T"], "Q": ["O", "C"]}
    tuning = sorted(label for label in expected_labels() if label.startswith("H-"))
    formal = sorted(label for label in expected_labels() if label.startswith("F-"))
    evaluation = sorted(label for label in expected_labels() if label.startswith("V-"))
    parents.update({label: ["Q"] for label in tuning})
    parents["HG"] = tuning
    parents.update({label: ["HG"] for label in formal})
    formal_by_cell = {label.removeprefix("F-"): label for label in formal}
    parents.update({label: [formal_by_cell[label.removeprefix("V-")]] for label in evaluation})
    parents["S"] = evaluation
    parents["G"] = ["S"]
    parents["Z"] = ["G"]
    return parents


def array_coordinates(stage: str, task_id: int) -> dict[str, Any]:
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < 48:
        raise ValueError("Phase 2g array task must be in [0,47]")
    arm = ARMS[task_id // 12]
    within_arm = task_id % 12
    rotation = ROTATIONS[within_arm // 3]
    slot = within_arm % 3
    if stage == "tuning":
        return {
            "arm": arm,
            "rotation": rotation,
            "learning_rate_index": slot,
            "learning_rate": LEARNING_RATES[slot],
            "label": f"H-{arm}-{rotation}-L{slot}",
        }
    if stage in {"formal", "evaluation"}:
        prefix = "F" if stage == "formal" else "V"
        return {"arm": arm, "rotation": rotation, "seed": slot, "label": f"{prefix}-{arm}-{rotation}-S{slot}"}
    raise ValueError(f"unknown Phase 2g array stage: {stage}")


def validate_jobs(jobs: Mapping[str, Mapping[str, Any]]) -> None:
    expected = expected_labels()
    if set(jobs) != expected or len(jobs) != 152:
        raise ValueError(
            f"Phase 2g graph labels differ: missing={sorted(expected - set(jobs))}, extra={sorted(set(jobs) - expected)}"
        )
    expected_parents = parent_map()
    for label, job in jobs.items():
        if list(job.get("parents", [])) != expected_parents[label]:
            raise ValueError(f"parent mapping drifted for {label}")
        if LOGICAL_JOB_ID_PATTERN.fullmatch(str(job.get("job_id", ""))) is None:
            raise ValueError(f"invalid logical scheduler ID for {label}")
        resources = job.get("resources")
        if not isinstance(resources, Mapping) or resources.get("gpu_requested") is not True:
            raise ValueError(f"every approved-partition task requires one GPU: {label}")
        directives = resources.get("directives")
        if not isinstance(directives, Mapping) or directives.get("gres") != "gpu:1":
            raise ValueError(f"task does not request exactly one GPU: {label}")
    roots = [label for label, values in expected_parents.items() if not values]
    if roots != ["T"]:
        raise ValueError(f"only T may be a graph root: {roots}")


def load_graph(path: Path) -> tuple[dict[str, Any], str]:
    graph = load_json(path)
    if graph.get("schema_version") != GRAPH_SCHEMA or graph.get("git_clean") is not True:
        raise ValueError("unexpected or dirty Phase 2g graph")
    if graph.get("git_pushed") is not True or graph.get("submission_policy") != SUBMISSION_POLICY:
        raise ValueError("Phase 2g graph governance policy drifted")
    jobs = graph.get("jobs")
    if not isinstance(jobs, Mapping):
        raise ValueError("Phase 2g graph lacks jobs")
    validate_jobs(jobs)
    reject_diode_paths(graph, "graph")
    return graph, sha256_file(path.resolve(strict=True))


def scheduler_completed_many(job_ids: Sequence[str]) -> bool:
    requested = tuple(dict.fromkeys(job_ids))
    if not requested:
        return True
    if any(LOGICAL_JOB_ID_PATTERN.fullmatch(value) is None for value in requested):
        return False
    result = subprocess.run(
        (
            "sacct",
            "-n",
            "-X",
            "-j",
            ",".join(requested),
            "--format=JobIDRaw%64,State,ExitCode",
            "--parsable2",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    completed = {
        row[0]
        for line in result.stdout.splitlines()
        if (row := line.strip().split("|"))
        and len(row) >= 3
        and row[0] in requested
        and row[1].split("+")[0] == "COMPLETED"
        and row[2] == "0:0"
    }
    return result.returncode == 0 and completed == set(requested)


def validate_wandb(receipt: Mapping[str, Any]) -> None:
    value = receipt.get("wandb")
    if not isinstance(value, Mapping):
        raise ValueError("receipt lacks embedded W&B identity")
    if (
        value.get("schema_version") != WANDB_SCHEMA
        or value.get("mode") != "online"
        or value.get("status") != "success"
    ):
        raise ValueError("W&B receipt is not a successful online upload")
    required = (
        "entity",
        "project",
        "group",
        "job_type",
        "run_name",
        "run_id",
        "run_url",
        "artifact_name",
        "artifact_id",
        "artifact_version",
        "artifact_digest",
    )
    if any(not isinstance(value.get(key), str) or not str(value[key]).strip() for key in required):
        raise ValueError("W&B receipt identity is incomplete")
    execution_id = (
        receipt.get("execution_provenance", {}).get("execution_id")
        if isinstance(receipt.get("execution_provenance"), Mapping)
        else None
    )
    if execution_id is not None and value.get("group") != f"phase2g-quality-{execution_id}":
        raise ValueError("W&B group differs from the formal Phase 2g execution")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("W&B receipt lacks its immutable artifact file manifest")
    names: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("W&B artifact file identity must be an object")
        name = item.get("name")
        if name is None and isinstance(item.get("path"), str):
            name = Path(str(item["path"])).name
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or name in names
            or isinstance(item.get("bytes"), bool)
            or not isinstance(item.get("bytes"), int)
            or int(item["bytes"]) < 0
            or SHA256_PATTERN.fullmatch(str(item.get("sha256", ""))) is None
        ):
            raise ValueError("W&B artifact file manifest contains an invalid identity")
        if isinstance(item.get("path"), str):
            local_path = Path(str(item["path"]))
            if (
                not local_path.is_file()
                or local_path.is_symlink()
                or local_path.stat().st_size != item["bytes"]
                or sha256_file(local_path) != item["sha256"]
            ):
                raise ValueError("local W&B artifact source differs from its embedded file identity")
        names.add(name)


def _current_logical_id(registered: str) -> tuple[str, str]:
    actual = os.environ.get("SLURM_JOB_ID")
    array_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if bool(array_id) != bool(task_id):
        raise ValueError("Slurm array identity is incomplete")
    logical = f"{array_id}_{task_id}" if array_id and task_id else (actual or registered)
    scheduler_id = actual or registered.partition("_")[0]
    if LOGICAL_JOB_ID_PATTERN.fullmatch(logical) is None or not scheduler_id.isdigit():
        raise ValueError("invalid current Slurm identity")
    return logical, scheduler_id


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
    logical_id, scheduler_id = _current_logical_id(str(jobs[label]["job_id"]))
    if logical_id != str(jobs[label]["job_id"]):
        raise ValueError(f"current Slurm task {logical_id} differs from graph job {jobs[label]['job_id']}")
    expected_parents = list(jobs[label]["parents"])
    if list(parents) != expected_parents:
        raise ValueError(f"parent set/order mismatch for {label}")
    root = Path(str(graph["repository_root"])).resolve(strict=True)
    commit = subprocess.check_output(("git", "-C", str(root), "rev-parse", "HEAD"), text=True).strip()
    status = subprocess.check_output(
        ("git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"), text=True
    ).strip()
    upstream = subprocess.check_output(("git", "-C", str(root), "rev-parse", "@{u}"), text=True).strip()
    if commit != graph["git_commit"] or upstream != commit or status:
        raise ValueError("execution requires the graph's exact clean pushed commit")
    test = load_json(test_receipt_path)
    if (
        test.get("schema_version") != TEST_SCHEMA
        or test.get("status") != "pass"
        or test.get("dependency_graph_sha256") != graph_hash
        or test.get("git_commit") != commit
    ):
        raise ValueError("test receipt is not bound to this graph/commit")
    validate_wandb(test)
    if expected_parents and not scheduler_completed_many([str(jobs[parent]["job_id"]) for parent in expected_parents]):
        raise ValueError("one or more direct parents are not scheduler COMPLETED 0:0")
    parent_rows = []
    for parent in expected_parents:
        expected_path = Path(str(jobs[parent]["expected_receipt"])).resolve()
        if parents[parent].resolve() != expected_path or not Path(str(jobs[parent]["expected_success"])).is_file():
            raise ValueError(f"parent receipt/SUCCESS mismatch for {parent}")
        parent_receipt = load_json(parents[parent])
        if parent_receipt.get("status") not in {"pass", "success"}:
            raise ValueError(f"parent {parent} did not succeed")
        validate_wandb(parent_receipt)
        parent_rows.append(
            {
                "label": parent,
                "job_id": jobs[parent]["job_id"],
                "receipt": file_identity(parents[parent]),
                "wandb_run_id": parent_receipt["wandb"]["run_id"],
            }
        )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": graph["execution_id"],
        "job_label": label,
        "git_commit": commit,
        "git_branch": graph["git_branch"],
        "git_clean": True,
        "git_pushed": True,
        "preregistration_sha256": graph["preregistration"]["sha256"],
        "preflight_sha256": graph["preflight"]["sha256"],
        "test_receipt_sha256": sha256_file(test_receipt_path.resolve(strict=True)),
        "dependency_graph_sha256": graph_hash,
        "slurm": {
            "job_id": logical_id,
            "allocation_job_id": scheduler_id,
            "job_name": os.environ.get("SLURM_JOB_NAME", jobs[label]["job_name"]),
            "account": graph["account"],
            "partition_fallback": graph["partition_fallback"],
            "actual_partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
            "resources": jobs[label]["resources"],
        },
        "parents": parent_rows,
        "sources": {name: file_identity(path) for name, path in sorted(sources.items())},
        "data_access_decision": {
            "preflight": dict(graph["preflight"]),
            "registry": dict(graph["registry"]),
            "readiness": dict(graph["readiness"]),
            "data_access_authorized": True,
            "sun_dataset_id": "sun-rgbd.geometry-development",
            "external_final_authorized": False,
        },
        "external_final_authorized": False,
    }
    finite_tree(provenance)
    reject_credentials(provenance)
    reject_diode_paths(provenance)
    return provenance


def verify_receipt(receipt_path: Path, provenance_path: Path) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    provenance = load_json(provenance_path)
    if receipt.get("status") not in {"pass", "success"}:
        raise ValueError("formal Phase 2g stage did not succeed")
    if receipt.get("execution_provenance") != provenance:
        raise ValueError("formal Phase 2g receipt provenance mismatch")
    if not (receipt_path.parent / "SUCCESS").is_file():
        raise ValueError("formal Phase 2g stage lacks terminal SUCCESS")
    validate_wandb(receipt)
    reject_diode_paths(receipt)
    return receipt


def _parse_pairs(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"invalid/duplicate {label}: {value}")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit-provenance")
    emit.add_argument("--graph", type=Path, required=True)
    emit.add_argument("--label", required=True)
    emit.add_argument("--test-receipt", type=Path, required=True)
    emit.add_argument("--parent", action="append", default=[])
    emit.add_argument("--source", action="append", default=[])
    emit.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "emit-provenance":
        value = build_provenance(
            args.graph,
            args.label,
            args.test_receipt,
            _parse_pairs(args.parent, "parent"),
            _parse_pairs(args.source, "source"),
        )
        atomic_json(args.output, value)
    else:
        verify_receipt(args.receipt, args.provenance)


if __name__ == "__main__":
    main()
