"""Mandatory-online W&B publishing for governed, aggregate-only validation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa4d.validation._content import sha256_value

SAFE_WANDB_RECEIPT_SCHEMA = "jepa4d-safe-wandb-receipt-v1"
SAFE_WANDB_FINAL_RECEIPT_SCHEMA = "jepa4d-safe-wandb-final-receipt-v1"

_SAFE_ROLES = frozenset(
    {
        "aggregate-receipt",
        "dashboard-html",
        "dashboard-json",
        "dashboard-receipt",
        "execution-receipt",
        "governance-receipt",
        "postflight-receipt",
    }
)
_SAFE_SUFFIXES = frozenset({".html", ".json"})
_FORBIDDEN_JSON_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "cluster_id",
        "coordinates",
        "data",
        "depth",
        "depths",
        "depth_map",
        "frame_id",
        "frame_ids",
        "image",
        "images",
        "input_path",
        "mask",
        "masks",
        "output_path",
        "path",
        "paths",
        "per_unit",
        "point_cloud",
        "points",
        "pointcloud",
        "prediction",
        "predictions",
        "raw_data",
        "raw_target",
        "raw_targets",
        "rgb",
        "rgbs",
        "sample_id",
        "sample_ids",
        "target",
        "targets",
        "tensor",
        "tensors",
        "unit_id",
        "values",
    }
)
_CREDENTIAL_KEY_TOKENS = frozenset(
    {"api", "apikey", "authorization", "cookie", "credential", "key", "netrc", "password", "secret", "token"}
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"wandb_v1_[A-Za-z0-9_-]+"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|credential|password|secret|token)\b\s*[:=]\s*\S+"),
)
_LOCAL_PATH_PATTERN = re.compile(
    r"(?i)(?:\bfile://|\bs3://|(?<![A-Za-z0-9])(?:/lustre|/home|/root|/tmp|/var/tmp)/\S+)"
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ROLE_JSON_SCHEMAS: Mapping[str, tuple[str, frozenset[str]]] = {
    "dashboard-json": (
        "jepa4d-validation-dashboard-v1",
        frozenset(
            {
                "schema_version",
                "report_id",
                "title",
                "stage",
                "evidence_level",
                "datasets",
                "gate",
                "completeness",
                "claim_boundary",
                "metrics",
                "resource_policy",
                "visualizations",
                "timestamp",
                "wandb_url",
                "governance",
            }
        ),
    ),
    "dashboard-receipt": (
        "jepa4d-validation-dashboard-bundle-v1",
        frozenset({"schema_version", "generation_id", "report_id", "files"}),
    ),
    "governance-receipt": (
        "jepa4d-geometry-official-mini-access-v1",
        frozenset({"schema_version", "dataset_id", "split_id", "operation", "execution_id", "git_commit", "decision"}),
    ),
    "aggregate-receipt": (
        "jepa4d-geometry-official-mini-metric-gate-v1",
        frozenset(
            {
                "schema_version",
                "dataset_id",
                "split_id",
                "operation",
                "execution_id",
                "git_commit",
                "registry_sha256",
                "ledger_sha256",
                "access_decision_sha256",
                "id_manifest_sha256",
                "archive_sha256",
                "model_identity_sha256",
                "depth_alignment_protocol",
                "depth_validity_protocol",
                "aggregation_protocol",
                "evaluated_test_frames",
                "quality_metrics",
                "resource_metrics",
                "gate_conditions",
                "gate_outcome",
            }
        ),
    ),
    "execution-receipt": (
        "jepa4d-geometry-official-mini-execution-v1",
        frozenset(
            {
                "schema_version",
                "status",
                "execution_id",
                "git_commit",
                "dataset_id",
                "split_id",
                "operation",
                "registry_sha256",
                "ledger_sha256",
                "access_decision_sha256",
                "metric_gate_sha256",
                "validation_status",
                "validation_status_sha256",
                "dashboard_generation_id",
                "wandb_receipt_sha256",
                "wandb_run_id",
                "wandb_entity",
                "wandb_project",
                "wandb_group",
                "wandb_job_type",
                "wandb_run_name",
                "wandb_artifact_name",
                "wandb_artifact_id",
                "wandb_artifact_digest",
                "wandb_config_sha256",
                "wandb_summary_sha256",
            }
        ),
    ),
    "postflight-receipt": (
        "jepa4d-geometry-official-mini-postflight-v1",
        frozenset(
            {
                "schema_version",
                "status",
                "execution_id",
                "git_commit",
                "dataset_id",
                "split_id",
                "operation",
                "registry_sha256",
                "ledger_sha256",
                "access_decision_sha256",
                "metric_gate_sha256",
                "dashboard_generation_id",
                "wandb_receipt_sha256",
                "execution_receipt_sha256",
                "scheduler",
            }
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class SafeArtifactFile:
    """One explicitly classified aggregate/dashboard/receipt file."""

    path: Path
    role: str

    def __post_init__(self) -> None:
        if self.role not in _SAFE_ROLES:
            raise ValueError(f"unsafe governed W&B artifact role: {self.role!r}")


@dataclass(frozen=True, slots=True)
class _StableFileIdentity:
    device: int
    inode: int
    bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def _key_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[_.-]+", value.casefold()) if token}


def _validate_text(value: str, location: str) -> None:
    if any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS):
        raise ValueError(f"credential-like content is forbidden in governed W&B payloads: {location}")
    if _LOCAL_PATH_PATTERN.search(value):
        raise ValueError(f"local/restricted path content is forbidden in governed W&B payloads: {location}")


def _validate_role_json_schema(payload: Any, role: str, filename: str) -> None:
    expected = _ROLE_JSON_SCHEMAS.get(role)
    if expected is None:
        raise ValueError(f"JSON is not allowed for governed W&B artifact role {role!r}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"governed W&B JSON must be an object: {filename}")
    schema, allowed_keys = expected
    if payload.get("schema_version") != schema or not set(payload).issubset(allowed_keys):
        raise ValueError(f"governed W&B artifact does not match its role schema: {filename}")
    if role == "dashboard-json":
        metrics = payload.get("metrics")
        if not isinstance(metrics, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "value", "unit", "domain", "split"}
            or isinstance(item.get("value"), bool)
            or not isinstance(item.get("value"), int | float)
            for item in metrics
        ):
            raise ValueError("dashboard metrics must be aggregate scalar records")
    if role == "aggregate-receipt":
        for name in ("quality_metrics", "resource_metrics", "gate_conditions"):
            values = payload.get(name)
            if not isinstance(values, Mapping) or any(
                isinstance(value, (Mapping, list, tuple)) for value in values.values()
            ):
                raise ValueError(f"aggregate receipt {name} must contain scalar values")


def _validate_tree(value: Any, location: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_JSON_KEYS or _key_tokens(key) & _CREDENTIAL_KEY_TOKENS:
                raise ValueError(f"unsafe field is forbidden in governed W&B payloads: {location}.{key}")
            _validate_tree(child, f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 1024:
            raise ValueError(f"oversized sequence is forbidden in governed W&B payloads: {location}")
        for index, child in enumerate(value):
            _validate_tree(child, f"{location}[{index}]")
        return
    if isinstance(value, str):
        if len(value) > 16_384:
            raise ValueError(f"oversized string is forbidden in governed W&B payloads: {location}")
        _validate_text(value, location)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value is forbidden in governed W&B payloads: {location}")
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"unsupported governed W&B payload value at {location}: {type(value).__name__}")


def _load_and_validate_file(path: Path, role: str) -> _StableFileIdentity:
    if path.suffix.casefold() not in _SAFE_SUFFIXES:
        raise ValueError(f"governed W&B artifacts must be JSON or self-contained HTML: {path.name}")
    before = path.stat()
    if before.st_size < 1 or before.st_size > 8 * 1024 * 1024:
        raise ValueError(f"governed W&B artifact size is outside the aggregate-only envelope: {path.name}")
    document_bytes = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        any(getattr(before, field) != getattr(after, field) for field in fields)
        or len(document_bytes) != after.st_size
    ):
        raise ValueError(f"governed W&B artifact changed while it was inspected: {path.name}")
    try:
        document = document_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"governed W&B artifact is not UTF-8 text: {path.name}") from error
    _validate_text(document, path.name)
    if path.suffix.casefold() == ".json":
        try:
            json_payload = json.loads(document)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid governed W&B JSON artifact: {path.name}") from error
        _validate_tree(json_payload, path.name)
        _validate_role_json_schema(json_payload, role, path.name)
    elif role != "dashboard-html":
        raise ValueError("HTML uploads are allowed only for the governed dashboard")
    lowered = document.casefold()
    if path.suffix.casefold() == ".html" and any(marker in lowered for marker in ("<img", " src=", "src=")):
        raise ValueError("governed dashboard HTML must not embed prediction or external media")
    return _StableFileIdentity(
        device=after.st_dev,
        inode=after.st_ino,
        bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        sha256=hashlib.sha256(document_bytes).hexdigest(),
    )


def validate_safe_artifact_files(
    files: Sequence[SafeArtifactFile],
    *,
    artifact_root: str | Path,
) -> tuple[SafeArtifactFile, ...]:
    """Resolve and inspect the exact upload allowlist before W&B is imported."""

    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact_root must be a directory")
    values = tuple(files)
    if not values:
        raise ValueError("governed W&B publishing requires at least one safe artifact")
    resolved: list[SafeArtifactFile] = []
    names: set[str] = set()
    roles: set[str] = set()
    for item in values:
        if not isinstance(item, SafeArtifactFile):
            raise TypeError("files must contain SafeArtifactFile values")
        path = item.path.resolve(strict=True)
        if not path.is_file() or (path != root and root not in path.parents):
            raise ValueError(f"governed W&B artifact escapes its output root: {path.name}")
        if not _SAFE_NAME.fullmatch(path.name) or path.name in names:
            raise ValueError(f"unsafe or duplicate governed W&B artifact name: {path.name!r}")
        if item.role in roles:
            raise ValueError(f"duplicate governed W&B artifact role: {item.role!r}")
        _load_and_validate_file(path, item.role)
        names.add(path.name)
        roles.add(item.role)
        resolved.append(SafeArtifactFile(path=path, role=item.role))
    return tuple(resolved)


def publish_safe_online_run(
    *,
    entity: str | None,
    project: str,
    group: str,
    job_type: str,
    run_name: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifact_name: str,
    artifact_root: str | Path,
    files: Sequence[SafeArtifactFile],
) -> dict[str, Any]:
    """Upload only inspected aggregate artifacts and return backend-confirmed identities."""

    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("governed validation requires WANDB_MODE=online")
    for name, value in (
        ("project", project),
        ("group", group),
        ("job_type", job_type),
        ("run_name", run_name),
        ("artifact_name", artifact_name),
    ):
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError(f"{name} must be a non-empty path-safe identifier")
    if entity is not None and not _SAFE_NAME.fullmatch(entity):
        raise ValueError("entity must be a path-safe identifier")
    _validate_tree(config, "config")
    published_summary = {**summary, "validation/postflight/status": "pending"}
    _validate_tree(published_summary, "summary")
    upload_files = validate_safe_artifact_files(files, artifact_root=artifact_root)
    inspected = {item.path: _load_and_validate_file(item.path, item.role) for item in upload_files}

    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        job_type=job_type,
        name=run_name,
        config=dict(config),
        mode="online",
        reinit=True,
    )
    if run is None or bool(getattr(run, "offline", True)):
        raise RuntimeError("governed W&B run did not initialize online")
    try:
        artifact = wandb.Artifact(artifact_name, type="governed-validation")
        for item in upload_files:
            artifact.add_file(str(item.path), name=item.path.name)
        for key, value in published_summary.items():
            run.summary[key] = value
        logged = run.log_artifact(artifact)
        logged.wait()
        for item in upload_files:
            if _load_and_validate_file(item.path, item.role) != inspected[item.path]:
                raise RuntimeError(f"governed W&B artifact changed during upload: {item.path.name}")
        backend_values = {
            "entity": getattr(run, "entity", None),
            "project": getattr(run, "project", None),
            "run_id": getattr(run, "id", None),
            "run_url": getattr(run, "url", None),
            "artifact_id": getattr(logged, "id", None),
            "artifact_version": getattr(logged, "version", None),
            "artifact_digest": getattr(logged, "digest", None),
        }
        if any(
            not isinstance(value, str) or not value.strip() or value.casefold() in {"none", "null"}
            for value in backend_values.values()
        ):
            raise RuntimeError("W&B did not return complete online run/artifact identities")
        if not str(backend_values["run_url"]).startswith(("https://", "http://")):
            raise RuntimeError("W&B returned an invalid online run URL")
        receipt = {
            "schema_version": SAFE_WANDB_RECEIPT_SCHEMA,
            "status": "uploaded-preliminary",
            "terminal_status": "pending-postflight",
            "mode": "online",
            "entity": backend_values["entity"],
            "project": backend_values["project"],
            "group": group,
            "job_type": job_type,
            "run_name": run_name,
            "run_id": backend_values["run_id"],
            "run_url": backend_values["run_url"],
            "artifact_name": artifact_name,
            "artifact_id": backend_values["artifact_id"],
            "artifact_version": backend_values["artifact_version"],
            "artifact_digest": backend_values["artifact_digest"],
            "config_sha256": sha256_value(config),
            "summary_sha256": sha256_value(published_summary),
            "files": [
                {
                    "name": item.path.name,
                    "role": item.role,
                    "bytes": inspected[item.path].bytes,
                    "sha256": inspected[item.path].sha256,
                }
                for item in upload_files
            ],
        }
        _validate_tree(receipt, "receipt")
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)
    return receipt


def validate_safe_wandb_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate a persisted safe-publisher receipt without contacting W&B."""

    if (
        receipt.get("schema_version") != SAFE_WANDB_RECEIPT_SCHEMA
        or receipt.get("status") != "uploaded-preliminary"
        or receipt.get("terminal_status") != "pending-postflight"
        or receipt.get("mode") != "online"
    ):
        raise ValueError("W&B receipt is not a successful governed online upload")
    for name in ("run_id", "run_url", "artifact_id", "artifact_version", "artifact_digest"):
        if (
            not isinstance(receipt.get(name), str)
            or not str(receipt[name]).strip()
            or str(receipt[name]).casefold() in {"none", "null"}
        ):
            raise ValueError(f"W&B receipt lacks {name}")
    if not str(receipt["run_url"]).startswith(("https://", "http://")):
        raise ValueError("W&B receipt has an invalid run_url")
    for name in ("config_sha256", "summary_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(name, ""))):
            raise ValueError(f"W&B receipt lacks a valid {name}")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("W&B receipt has no artifact file identities")
    names: set[str] = set()
    roles: set[str] = set()
    for item in files:
        if (
            not isinstance(item, Mapping)
            or item.get("role") not in _SAFE_ROLES
            or not isinstance(item.get("name"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 1
            or "path" in item
        ):
            raise ValueError("W&B receipt contains an invalid safe artifact identity")
        name = str(item["name"])
        role = str(item["role"])
        if name in names or role in roles:
            raise ValueError("W&B receipt contains duplicate artifact names or roles")
        names.add(name)
        roles.add(role)
    _validate_tree(receipt, "receipt")


def finalize_safe_online_run(
    *,
    preliminary_receipt: Mapping[str, Any],
    artifact_root: str | Path,
    files: Sequence[SafeArtifactFile],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Resume the preliminary run and publish terminal execution/postflight evidence."""

    validate_safe_wandb_receipt(preliminary_receipt)
    terminal_summary = {**summary, "validation/postflight/status": "pass"}
    _validate_tree(terminal_summary, "terminal_summary")
    upload_files = validate_safe_artifact_files(files, artifact_root=artifact_root)
    inspected = {item.path: _load_and_validate_file(item.path, item.role) for item in upload_files}
    artifact_name = f"terminal-{preliminary_receipt['run_id']}"
    if not _SAFE_NAME.fullmatch(artifact_name):
        raise ValueError("terminal W&B artifact name is not path-safe")

    import wandb

    run = wandb.init(
        entity=preliminary_receipt["entity"],
        project=preliminary_receipt["project"],
        group=preliminary_receipt["group"],
        job_type=preliminary_receipt["job_type"],
        name=preliminary_receipt["run_name"],
        id=preliminary_receipt["run_id"],
        resume="must",
        mode="online",
        reinit=True,
    )
    if run is None or bool(getattr(run, "offline", True)) or getattr(run, "id", None) != preliminary_receipt["run_id"]:
        raise RuntimeError("governed W&B terminal finalizer did not resume the exact online run")
    try:
        artifact = wandb.Artifact(artifact_name, type="governed-validation-terminal")
        for item in upload_files:
            artifact.add_file(str(item.path), name=item.path.name)
        for key, value in terminal_summary.items():
            run.summary[key] = value
        logged = run.log_artifact(artifact)
        logged.wait()
        for item in upload_files:
            if _load_and_validate_file(item.path, item.role) != inspected[item.path]:
                raise RuntimeError(f"governed W&B terminal artifact changed during upload: {item.path.name}")
        backend_values = {
            "run_url": getattr(run, "url", None),
            "artifact_id": getattr(logged, "id", None),
            "artifact_version": getattr(logged, "version", None),
            "artifact_digest": getattr(logged, "digest", None),
        }
        if any(
            not isinstance(value, str) or not value.strip() or value.casefold() in {"none", "null"}
            for value in backend_values.values()
        ):
            raise RuntimeError("W&B terminal finalizer returned incomplete backend identities")
        if not str(backend_values["run_url"]).startswith(("https://", "http://")):
            raise RuntimeError("W&B terminal finalizer returned an invalid run URL")
        receipt = {
            "schema_version": SAFE_WANDB_FINAL_RECEIPT_SCHEMA,
            "status": "finalized",
            "terminal_status": "postflight-pass",
            "preliminary_receipt_sha256": sha256_value(preliminary_receipt),
            "entity": preliminary_receipt["entity"],
            "project": preliminary_receipt["project"],
            "group": preliminary_receipt["group"],
            "job_type": preliminary_receipt["job_type"],
            "run_name": preliminary_receipt["run_name"],
            "run_id": preliminary_receipt["run_id"],
            "run_url": backend_values["run_url"],
            "artifact_name": artifact_name,
            "artifact_id": backend_values["artifact_id"],
            "artifact_version": backend_values["artifact_version"],
            "artifact_digest": backend_values["artifact_digest"],
            "summary_sha256": sha256_value(terminal_summary),
            "files": [
                {
                    "name": item.path.name,
                    "role": item.role,
                    "bytes": inspected[item.path].bytes,
                    "sha256": inspected[item.path].sha256,
                }
                for item in upload_files
            ],
        }
        _validate_tree(receipt, "terminal_receipt")
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)
    return receipt


def validate_safe_wandb_final_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the backend-confirmed terminal W&B receipt."""

    if (
        receipt.get("schema_version") != SAFE_WANDB_FINAL_RECEIPT_SCHEMA
        or receipt.get("status") != "finalized"
        or receipt.get("terminal_status") != "postflight-pass"
        or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("preliminary_receipt_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("summary_sha256", "")))
    ):
        raise ValueError("W&B terminal receipt is not a finalized postflight pass")
    for name in ("run_id", "run_url", "artifact_id", "artifact_version", "artifact_digest"):
        value = receipt.get(name)
        if not isinstance(value, str) or not value.strip() or value.casefold() in {"none", "null"}:
            raise ValueError(f"W&B terminal receipt lacks {name}")
    if not str(receipt["run_url"]).startswith(("https://", "http://")):
        raise ValueError("W&B terminal receipt has an invalid run URL")
    files = receipt.get("files")
    if not isinstance(files, list) or {item.get("role") for item in files if isinstance(item, Mapping)} != {
        "execution-receipt",
        "postflight-receipt",
    }:
        raise ValueError("W&B terminal receipt lacks execution/postflight artifacts")
    _validate_tree(receipt, "terminal_receipt")
