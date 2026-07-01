"""Governed provenance, output, and mandatory online-W&B helpers for Phase 2g."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2f_metrics import file_identity
from jepa4d.training.phase2g_protocol import (
    WANDB_ENTITY,
    WANDB_GROUP_PREFIX,
    WANDB_PROJECT,
    WANDB_RECEIPT_SCHEMA,
)


def require_safe_finite_tree(value: Any, *, location: str = "value") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {location}")
        return
    if isinstance(value, str):
        if value.startswith("wandb_v1_"):
            raise ValueError(f"credential-shaped value at {location}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in ("api_key", "password", "secret", "credential", "netrc")):
                raise ValueError(f"credential-like field at {location}.{key}")
            require_safe_finite_tree(child, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            require_safe_finite_tree(child, location=f"{location}[{index}]")
        return
    raise TypeError(f"unsupported metadata type at {location}: {type(value).__name__}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    require_safe_finite_tree(value)
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    require_safe_finite_tree(value, location=str(path))
    return value


def load_execution_provenance(path: Path) -> dict[str, Any]:
    value = load_json(path)
    provenance = value.get("execution_provenance", value)
    if not isinstance(provenance, dict):
        raise TypeError("execution provenance must be a JSON object")
    required = (
        "schema_version",
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "preflight_sha256",
        "test_receipt_sha256",
        "dependency_graph_sha256",
        "slurm",
        "data_access_decision",
    )
    if any(not provenance.get(name) for name in required):
        raise ValueError("Phase 2g execution provenance is incomplete")
    if not isinstance(provenance["slurm"], Mapping) or not provenance["slurm"].get("job_id"):
        raise ValueError("Phase 2g execution provenance lacks Slurm identity")
    if provenance.get("schema_version") != "jepa4d-phase2g-execution-provenance-v1":
        raise ValueError("unexpected Phase 2g execution provenance schema")
    authorization = provenance.get("data_access_decision")
    if not isinstance(authorization, Mapping):
        raise ValueError("Phase 2g execution provenance lacks authorization binding")
    identities = ("preflight", "registry", "readiness")
    if (
        authorization.get("data_access_authorized") is not True
        or authorization.get("sun_dataset_id") != "sun-rgbd.geometry-development"
        or authorization.get("external_final_authorized") is not False
        or any(
            not isinstance(authorization.get(name), Mapping) or len(str(authorization[name].get("sha256", ""))) != 64
            for name in identities
        )
        or authorization["preflight"].get("sha256") != provenance.get("preflight_sha256")
    ):
        raise ValueError("Phase 2g registry/readiness/preflight authorization binding is invalid")
    if provenance.get("git_clean") is not True or provenance.get("git_pushed") is not True:
        raise ValueError("Phase 2g execution provenance is not bound to a clean pushed commit")
    if provenance.get("external_final_authorized") is not False:
        raise ValueError("Phase 2g-A provenance must keep external final unauthorized")
    require_safe_finite_tree(provenance, location="execution_provenance")
    return provenance


def execution_identity(provenance: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = (
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "test_receipt_sha256",
        "dependency_graph_sha256",
    )
    values = tuple(provenance.get(key) for key in keys)
    if any(value is None or value == "" for value in values):
        raise ValueError("execution identity is incomplete")
    return values


def assert_same_execution(receipts: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> None:
    expected = execution_identity(provenance)
    for receipt in receipts:
        embedded = receipt.get("execution_provenance")
        if not isinstance(embedded, Mapping) or execution_identity(embedded) != expected:
            raise ValueError("Phase 2g receipts do not share one execution identity")


def prepare_output(path: Path) -> Path:
    output = path.resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def hardened_wandb_settings(wandb_module: Any) -> Any:
    """Disable implicit host, Git, CLI, code, console, and system-stat capture."""

    settings = wandb_module.Settings(
        console="off",
        disable_git=True,
        save_code=False,
        x_disable_meta=True,
        x_disable_stats=True,
        x_disable_machine_info=True,
    )
    expected = {
        "console": "off",
        "disable_git": True,
        "save_code": False,
        "x_disable_meta": True,
        "x_disable_stats": True,
        "x_disable_machine_info": True,
    }
    if any(getattr(settings, name, None) != value for name, value in expected.items()):
        raise RuntimeError("formal Phase 2g could not enforce hardened W&B metadata settings")
    return settings


def start_wandb_run(
    *,
    provenance: Mapping[str, Any],
    job_type: str,
    semantic_name: str,
    config: Mapping[str, Any],
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
) -> Any:
    """Start one mandatory semantic online run."""

    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("formal Phase 2g requires WANDB_MODE=online")
    require_safe_finite_tree(config, location="wandb.config")
    import wandb

    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    group = f"{WANDB_GROUP_PREFIX}{execution_id}"
    name = f"{execution_id}-{semantic_name}-{job_id}"
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        job_type=job_type,
        name=name,
        mode="online",
        reinit=True,
        settings=hardened_wandb_settings(wandb),
        config={
            **dict(config),
            "execution_id": execution_id,
            "git_commit": provenance["git_commit"],
            "preregistration_sha256": provenance["preregistration_sha256"],
            "preflight_sha256": provenance["preflight_sha256"],
            "registry_sha256": provenance["data_access_decision"]["registry"]["sha256"],
            "readiness_sha256": provenance["data_access_decision"]["readiness"]["sha256"],
            "sun_data_access_authorized": True,
            "external_final_authorized": False,
            "dependency_graph_sha256": provenance["dependency_graph_sha256"],
        },
        tags=["phase-2g", "formal", "sun-rgbd", job_type],
    )
    if run is None or bool(run.offline):
        raise RuntimeError("formal Phase 2g W&B run did not initialize online")
    return run


def _numeric_telemetry_value(value: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    return None if match is None else float(match.group(0))


def snapshot_and_log_gpu_telemetry(run: Any) -> Path | None:
    """Freeze the allocated-node 15-second monitor and mirror numeric rows to W&B.

    ``nvidia-smi --loop`` may still be appending to the source file.  Reading it
    once and retaining only complete CSV rows produces a stable artifact without
    racing the backend uploader against the live monitor.
    """

    source_value = os.environ.get("JEPA4D_GPU_TELEMETRY_PATH")
    if not source_value:
        return None
    source = Path(source_value).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Phase 2g GPU telemetry source must be a regular file")
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) < 2:
        raise RuntimeError("Phase 2g GPU telemetry has no completed sample")
    field_count = len(next(csv.reader([lines[0]])))
    complete = [lines[0]]
    for line in lines[1:]:
        if len(next(csv.reader([line]))) == field_count:
            complete.append(line)
    if len(complete) < 2:
        raise RuntimeError("Phase 2g GPU telemetry has no complete data row")
    snapshot = source.with_name("gpu-telemetry-wandb.csv")
    temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
    temporary.write_text("\n".join(complete) + "\n", encoding="utf-8")
    temporary.replace(snapshot)

    aliases = {
        "temperature.gpu": "gpu/temperature_c",
        "utilization.gpu": "gpu/utilization_percent",
        "utilization.memory": "gpu/memory_utilization_percent",
        "memory.used": "gpu/memory_used_mib",
        "memory.total": "gpu/memory_total_mib",
        "power.draw": "gpu/power_w",
        "clocks.sm": "gpu/sm_clock_mhz",
    }
    rows = list(csv.DictReader(StringIO(snapshot.read_text(encoding="utf-8"))))
    normalized_rows = [{str(name).split(" [", 1)[0].strip(): value for name, value in row.items()} for row in rows]
    gpu_identities = {(str(row.get("index", "")).strip(), str(row.get("uuid", "")).strip()) for row in normalized_rows}
    if (
        len(gpu_identities) != 1
        or any(not value for value in next(iter(gpu_identities), ("", "")))
        or not next(iter(gpu_identities))[0].isdigit()
    ):
        raise RuntimeError("Phase 2g GPU telemetry contains mixed or invalid GPU identities")
    for sample, row in enumerate(normalized_rows):
        logged: dict[str, Any] = {
            "gpu/telemetry_sample": sample,
            "gpu/telemetry_timestamp": str(row.get("timestamp", "")).strip(),
        }
        for normalized, raw_value in row.items():
            metric = aliases.get(normalized)
            numeric = None if raw_value is None else _numeric_telemetry_value(raw_value)
            if metric is not None and numeric is not None:
                logged[metric] = numeric
        run.log(logged)
    run.summary["gpu_telemetry_samples"] = len(rows)
    run.summary["gpu_telemetry_interval_seconds"] = 15
    return snapshot


def finish_wandb_run(
    run: Any,
    *,
    artifact_name: str,
    job_type: str,
    files: Sequence[Path],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Upload immutable files, finish online, and return complete backend identity."""

    require_safe_finite_tree(summary, location="wandb.summary")
    telemetry = snapshot_and_log_gpu_telemetry(run)
    resolved = [path.resolve(strict=True) for path in files]
    if telemetry is not None:
        resolved.append(telemetry.resolve(strict=True))
    if not resolved or any(not path.is_file() for path in resolved):
        raise ValueError("Phase 2g W&B artifacts require regular files")
    import wandb

    for key, value in summary.items():
        run.summary[key] = value
    run.summary["status"] = "success"
    artifact = wandb.Artifact(artifact_name, type=f"phase2g-{job_type}")
    for path in resolved:
        artifact.add_file(str(path), name=path.name)
    logged = run.log_artifact(artifact).wait()
    required = (run.id, run.url, logged.id, logged.version, logged.digest)
    if any(value is None or str(value).strip() == "" for value in required):
        raise RuntimeError("W&B returned an incomplete run/artifact identity")
    receipt = {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "status": "success",
        "mode": "online",
        "entity": str(run.entity),
        "project": str(run.project),
        "group": str(run.group),
        "job_type": job_type,
        "run_name": str(run.name),
        "run_id": str(run.id),
        "run_url": str(run.url),
        "artifact_name": artifact_name,
        "artifact_id": str(logged.id),
        "artifact_version": str(logged.version),
        "artifact_digest": str(logged.digest),
        "files": [file_identity(path) for path in resolved],
    }
    run.finish(exit_code=0)
    require_safe_finite_tree(receipt, location="wandb_receipt")
    return receipt


def complete_output(
    output: Path,
    *,
    receipt_name: str,
    receipt: dict[str, Any],
    wandb_receipt: Mapping[str, Any],
) -> None:
    """Persist terminal identities and write SUCCESS last."""

    if wandb_receipt.get("schema_version") != WANDB_RECEIPT_SCHEMA or wandb_receipt.get("status") != "success":
        raise ValueError("cannot complete Phase 2g output without successful W&B identity")
    receipt["wandb"] = dict(wandb_receipt)
    atomic_json(output / receipt_name, receipt)
    atomic_json(output / "wandb_receipt.json", dict(wandb_receipt))
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")
