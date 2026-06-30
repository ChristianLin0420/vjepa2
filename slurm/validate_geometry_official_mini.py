#!/usr/bin/env python3
"""Strict postflight for the governed TUM RGB-D official-mini regression."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jepa4d.validation._content import (
    ContentAddress,
    sha256_file,
    sha256_value,
    verify_content_addressed_json,
    write_content_addressed_json,
)
from jepa4d.validation.geometry_official_mini import (
    ACCESS_RECEIPT_SCHEMA,
    AGGREGATION_PROTOCOL,
    DATASET_ID,
    DEPTH_ALIGNMENT_PROTOCOL,
    DEPTH_VALIDITY_PROTOCOL,
    EXECUTION_RECEIPT_SCHEMA,
    EXPECTED_QUALITY_METRICS,
    EXPECTED_RESOURCE_METRICS,
    EXPECTED_TEST_FRAMES,
    METRIC_GATE_SCHEMA,
    METRIC_UNITS,
    OPERATION,
    PROHIBITED_CLAIMS,
    SPLIT_ID,
    SUPPORTED_CLAIMS,
    validate_official_mini_quality_metrics,
)
from jepa4d.validation.ledger import ConsumedTestLedger
from jepa4d.validation.registry import DatasetRegistry
from jepa4d.validation.wandb import (
    SafeArtifactFile,
    finalize_safe_online_run,
    validate_safe_wandb_final_receipt,
    validate_safe_wandb_receipt,
)
from jepa4d.visualization.validation_dashboard import (
    HTML_FILENAME,
    JSON_FILENAME,
    RECEIPT_FILENAME,
    verify_immutable_validation_dashboard,
    wandb_summary_from_serializable,
)

POSTFLIGHT_SCHEMA = "jepa4d-geometry-official-mini-postflight-v1"
TERMINAL_SCHEMA = "jepa4d-geometry-official-mini-terminal-v1"
APPROVED_ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
APPROVED_PARTITIONS = frozenset(
    {"polar4", "polar3", "polar", "batch_block1", "grizzly", "batch_block2", "batch_block3"}
)
MAX_TIME_SECONDS = 4 * 60 * 60

_JOB_ID = re.compile(r"^[0-9]+$")
_JOB_NAME = re.compile(r"^j4d-gmini-[A-Za-z0-9_.-]+$")
_FORBIDDEN_TEXT = ("diode", "sunrgbd", "sun-rgbd", "sun rgb-d", "wandb_v1_", "hf_")
_FORBIDDEN_LOCAL_PATH = re.compile(r"(?i)(?:file://|s3://|(?:/lustre|/home|/root|/tmp|/var/tmp)/\S+)")


@dataclass(frozen=True, slots=True)
class SchedulerIdentity:
    job_id: str
    job_name: str
    account: str
    partition: str
    time_limit: str
    nodes: int
    tasks: int
    gpus: int


SchedulerLookup = Callable[[str], SchedulerIdentity]
GitLookup = Callable[[Path], tuple[str, bool]]
WandbFinalizer = Callable[..., dict[str, Any]]


def _time_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError(f"unsupported Slurm time limit: {value!r}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid Slurm time limit: {value!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def scheduler_identity(job_id: str) -> SchedulerIdentity:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("postflight requires a numeric Slurm job ID")
    result = subprocess.run(
        ("scontrol", "show", "job", "-o", job_id),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("unable to resolve the allocated Slurm job identity")
    fields = dict(
        token.split("=", 1) for token in result.stdout.strip().split() if "=" in token and token.split("=", 1)[0]
    )
    allocated_tres = str(fields.get("AllocTRES", ""))
    gpu_match = re.search(r"(?:^|,)gres/gpu(?:[:/][^=,]+)?=(\d+)(?:,|$)", allocated_tres)
    try:
        return SchedulerIdentity(
            job_id=str(fields["JobId"]),
            job_name=str(fields["JobName"]),
            account=str(fields["Account"]),
            partition=str(fields["Partition"]),
            time_limit=str(fields["TimeLimit"]),
            nodes=int(fields["NumNodes"]),
            tasks=int(fields["NumTasks"]),
            gpus=0 if gpu_match is None else int(gpu_match.group(1)),
        )
    except KeyError as error:
        raise ValueError(f"Slurm identity lacks {error.args[0]}") from error


def git_identity(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(("git", "-C", str(repo_root), "rev-parse", "HEAD"), text=True).strip()
    status = subprocess.check_output(
        ("git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"), text=True
    )
    return commit, not status.strip()


def _single_content_address(directory: Path, prefix: str) -> tuple[Path, dict[str, Any], str]:
    matches = sorted(directory.glob(f"{prefix}-*.json")) if directory.is_dir() else []
    if len(matches) != 1 or matches[0].is_symlink():
        raise ValueError(f"expected exactly one immutable {prefix} receipt")
    value = verify_content_addressed_json(matches[0], prefix=prefix)
    return matches[0], value, matches[0].stem.removeprefix(f"{prefix}-")


def _finite_tree(value: Any, location: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")


def _validate_scheduler(identity: SchedulerIdentity, expected_job_id: str) -> None:
    if identity.job_id != expected_job_id or not _JOB_ID.fullmatch(identity.job_id):
        raise ValueError("Slurm job identity mismatch")
    if not _JOB_NAME.fullmatch(identity.job_name):
        raise ValueError("Slurm job name is not a unique governed-mini name")
    if identity.account != APPROVED_ACCOUNT:
        raise ValueError("Slurm job used an unapproved account")
    if identity.partition not in APPROVED_PARTITIONS:
        raise ValueError("Slurm job used an unapproved partition")
    if _time_seconds(identity.time_limit) > MAX_TIME_SECONDS:
        raise ValueError("Slurm job exceeded the four-hour maximum")
    if (identity.nodes, identity.tasks, identity.gpus) != (1, 1, 1):
        raise ValueError("Slurm job must use exactly one node, one task, and one GPU")


def _expected_file_set(
    *,
    access_path: Path,
    metric_path: Path,
    dashboard: Path,
    wandb_path: Path,
    execution_path: Path,
    postflight_paths: Sequence[Path],
) -> set[Path]:
    return {
        access_path,
        metric_path,
        dashboard / JSON_FILENAME,
        dashboard / HTML_FILENAME,
        dashboard / RECEIPT_FILENAME,
        wandb_path,
        execution_path,
        *postflight_paths,
    }


def _scan_safe_artifacts(output: Path, expected: set[Path]) -> None:
    all_entries = set(output.rglob("*"))
    if any(path.is_symlink() for path in all_entries):
        raise ValueError("official-mini output may not contain symbolic links")
    actual = {path for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        extra = sorted(path.name for path in actual - expected)
        missing = sorted(path.name for path in expected - actual)
        raise ValueError(f"official-mini artifact allowlist mismatch: extra={extra}, missing={missing}")
    for path in actual:
        if path.is_symlink() or path.suffix.casefold() not in {".json", ".html"}:
            raise ValueError(f"unsafe official-mini artifact type: {path.name}")
        document = path.read_text(encoding="utf-8")
        lowered = document.casefold()
        if any(marker in lowered for marker in _FORBIDDEN_TEXT):
            raise ValueError(f"forbidden dataset/credential content in official-mini artifact: {path.name}")
        if _FORBIDDEN_LOCAL_PATH.search(document):
            raise ValueError(f"local data path in official-mini artifact: {path.name}")
    expected_directories: set[Path] = set()
    for path in expected:
        parent = path.parent
        while parent != output:
            expected_directories.add(parent)
            parent = parent.parent
    actual_directories = {path for path in all_entries if path.is_dir()}
    if actual_directories != expected_directories:
        extra = sorted(path.name for path in actual_directories - expected_directories)
        missing = sorted(path.name for path in expected_directories - actual_directories)
        raise ValueError(f"official-mini directory allowlist mismatch: extra={extra}, missing={missing}")


def _validate_wandb_files(receipt: Mapping[str, Any], expected: Mapping[str, tuple[Path, str]]) -> None:
    validate_safe_wandb_receipt(receipt)
    files = receipt["files"]
    assert isinstance(files, list)
    observed = {str(item["name"]): item for item in files}
    if set(observed) != set(expected):
        raise ValueError("W&B artifact file set differs from the governed allowlist")
    for name, (path, role) in expected.items():
        item = observed[name]
        if (
            item.get("role") != role
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"W&B artifact identity mismatch for {name}")


def _validate_report_metrics(
    values: object,
    quality: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> None:
    if not isinstance(values, list) or len(values) != len(quality) + len(resources):
        raise ValueError("dashboard metric table is incomplete")
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("dashboard metric rows must be objects")
        key = (str(value.get("domain")), str(value.get("name")))
        if key in observed:
            raise ValueError("dashboard metric table contains duplicate rows")
        observed[key] = value
    expected = {
        **{("quality", name): metric for name, metric in quality.items()},
        **{("resource", name): metric for name, metric in resources.items()},
    }
    if set(observed) != set(expected):
        raise ValueError("dashboard metric table differs from the metric-gate schema")
    for (domain, name), expected_value in expected.items():
        row = observed[(domain, name)]
        if row.get("value") != expected_value or row.get("unit") != METRIC_UNITS[name] or row.get("split") != SPLIT_ID:
            raise ValueError(f"dashboard metric row differs from the metric gate: {domain}/{name}")


def _validate_report_gate(value: object, expected_conditions: set[str]) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("name") != "scientific-promotion-not-applicable"
        or value.get("outcome") != "not-applicable"
    ):
        raise ValueError("dashboard must mark scientific promotion as not applicable")
    conditions = value.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(expected_conditions):
        raise ValueError("dashboard gate condition table is incomplete")
    observed: dict[str, Mapping[str, Any]] = {}
    for condition in conditions:
        if not isinstance(condition, Mapping) or not isinstance(condition.get("name"), str):
            raise ValueError("dashboard gate condition rows must be named objects")
        name = str(condition["name"])
        if name in observed:
            raise ValueError("dashboard gate condition table contains duplicates")
        observed[name] = condition
    if set(observed) != expected_conditions or any(
        condition.get("domain") != "integrity" or condition.get("passed") is not True
        for condition in observed.values()
    ):
        raise ValueError("dashboard gate conditions differ from the passing metric gate")


def _validate_geometry_official_mini_locked(
    *,
    output: str | Path,
    repo_root: str | Path,
    job_id: str,
    scheduler_lookup: SchedulerLookup = scheduler_identity,
    git_lookup: GitLookup = git_identity,
    wandb_finalizer: WandbFinalizer = finalize_safe_online_run,
) -> ContentAddress:
    """Validate every governed artifact and atomically write an immutable postflight receipt."""

    root = Path(output).resolve(strict=True)
    repository = Path(repo_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("official-mini output must be a real directory")
    scheduler = scheduler_lookup(job_id)
    _validate_scheduler(scheduler, job_id)
    commit, clean = git_lookup(repository)
    if not clean or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("postflight requires the clean committed source revision")

    registry = DatasetRegistry.load(repository / "configs/validation/dataset_registry.yaml")
    ledger = ConsumedTestLedger.load(repository / "configs/validation/consumed_test_ledger.yaml")
    ledger.validate_against(registry)
    entry, split = registry.split(DATASET_ID, SPLIT_ID)
    registered_archives = {
        value.sha256 for value in entry.hashes if value.kind == "archive" and value.sha256 is not None
    }
    access_path, access, access_sha = _single_content_address(root / "governance", "access-decision")
    metric_path, metric, metric_sha = _single_content_address(root / "metrics", "metric-gate")
    wandb_path, wandb, wandb_sha = _single_content_address(root / "wandb", "wandb-receipt")
    execution_path, execution, execution_sha = _single_content_address(root / "execution", "execution-receipt")
    dashboard_directories = sorted((root / "dashboard").glob("validation-dashboard-*"))
    if len(dashboard_directories) != 1:
        raise ValueError("expected exactly one immutable validation dashboard")
    dashboard = dashboard_directories[0]
    dashboard_receipt = verify_immutable_validation_dashboard(dashboard)
    report = json.loads((dashboard / JSON_FILENAME).read_text(encoding="utf-8"))

    decision = access.get("decision")
    if (
        access.get("schema_version") != ACCESS_RECEIPT_SCHEMA
        or access.get("dataset_id") != DATASET_ID
        or access.get("split_id") != SPLIT_ID
        or access.get("operation") != OPERATION.value
        or access.get("git_commit") != commit
        or not isinstance(decision, Mapping)
        or decision.get("authorized") is not True
        or decision.get("grants_data_access") is not True
        or decision.get("operation") != OPERATION.value
        or decision.get("registry_sha256") != registry.sha256
        or decision.get("ledger_sha256") != ledger.sha256
        or decision.get("ledger_state") != "consumed"
        or decision.get("future_use") != "regression"
    ):
        raise ValueError("access-decision receipt does not prove governed consumed regression authorization")

    quality = metric.get("quality_metrics")
    resources = metric.get("resource_metrics")
    gate_conditions = metric.get("gate_conditions")
    if not isinstance(quality, Mapping):
        raise ValueError("metric/gate receipt lacks its aggregate quality metrics")
    validate_official_mini_quality_metrics(quality)
    expected_conditions = {
        "all_aggregate_metrics_finite",
        "complete_registered_test_frames",
        "finite_geometry_outputs",
    }
    if (
        metric.get("schema_version") != METRIC_GATE_SCHEMA
        or metric.get("dataset_id") != DATASET_ID
        or metric.get("split_id") != SPLIT_ID
        or metric.get("operation") != OPERATION.value
        or metric.get("git_commit") != commit
        or metric.get("registry_sha256") != registry.sha256
        or metric.get("ledger_sha256") != ledger.sha256
        or metric.get("access_decision_sha256") != access_sha
        or metric.get("id_manifest_sha256") != split.id_manifest_sha256
        or metric.get("archive_sha256") not in registered_archives
        or not re.fullmatch(r"[0-9a-f]{64}", str(metric.get("model_identity_sha256", "")))
        or metric.get("depth_alignment_protocol") != DEPTH_ALIGNMENT_PROTOCOL
        or metric.get("depth_validity_protocol") != DEPTH_VALIDITY_PROTOCOL
        or metric.get("aggregation_protocol") != AGGREGATION_PROTOCOL
        or metric.get("evaluated_test_frames") != EXPECTED_TEST_FRAMES
        or set(quality) != EXPECTED_QUALITY_METRICS
        or not isinstance(resources, Mapping)
        or set(resources) != EXPECTED_RESOURCE_METRICS
        or quality.get("finite_fraction") != 1.0
        or any(
            isinstance(value, bool) or not isinstance(value, int | float) or float(value) < 0.0
            for value in resources.values()
        )
        or not isinstance(gate_conditions, Mapping)
        or set(gate_conditions) != expected_conditions
        or any(value is not True for value in gate_conditions.values())
        or metric.get("gate_outcome") != "pass"
    ):
        raise ValueError("metric/gate receipt is incomplete or did not pass")
    _finite_tree(metric, "metric-gate")

    governance = report.get("governance")
    datasets = report.get("datasets")
    report_gate = report.get("gate")
    completeness = report.get("completeness")
    claim_boundary = report.get("claim_boundary")
    validation_status = execution.get("validation_status")
    if not isinstance(validation_status, Mapping):
        raise ValueError("execution receipt lacks its validation status")
    validation_status_sha256 = sha256_value(validation_status)
    if (
        dashboard_receipt.get("generation_id") != execution.get("dashboard_generation_id")
        or report.get("report_id") != execution.get("execution_id")
        or report.get("stage") != "geometry"
        or report.get("evidence_level") != "official-smoke"
        or report.get("resource_policy") != "diagnostic-only"
        or not isinstance(completeness, Mapping)
        or completeness.get("expected_cells") != 1
        or completeness.get("succeeded_cells") != 1
        or completeness.get("failed_cells") != 0
        or completeness.get("missing_cells") != 0
        or completeness.get("status") != "complete"
        or not isinstance(claim_boundary, Mapping)
        or claim_boundary.get("supported") != list(SUPPORTED_CLAIMS)
        or claim_boundary.get("prohibited") != list(PROHIBITED_CLAIMS)
        or not isinstance(governance, Mapping)
        or governance.get("registry_sha256") != registry.sha256
        or governance.get("base_ledger_sha256") != ledger.sha256
        or governance.get("metric_gate_receipt_sha256") != metric_sha
        or governance.get("validation_status_sha256") != validation_status_sha256
        or execution.get("validation_status_sha256") != validation_status_sha256
        or validation_status.get("experiment_id") != execution.get("execution_id")
        or validation_status.get("stage") != "geometry"
        or validation_status.get("protocol") != "frozen"
        or validation_status.get("execution") != "complete"
        or validation_status.get("scientific") != "not_applicable"
        or validation_status.get("evidence_level") != "official-smoke"
        or validation_status.get("expected_cells") != 1
        or validation_status.get("successful_cells") != 1
        or validation_status.get("metric_gate_receipt_sha256") != metric_sha
        or not isinstance(datasets, list)
        or len(datasets) != 1
        or not isinstance(datasets[0], Mapping)
        or datasets[0].get("dataset_id") != DATASET_ID
        or datasets[0].get("split") != SPLIT_ID
        or datasets[0].get("target_access") != "consumed"
        or datasets[0].get("id_manifest_sha256") != split.id_manifest_sha256
    ):
        raise ValueError("governed dashboard does not bind the authorized passing regression")
    _validate_report_gate(report_gate, expected_conditions)
    _validate_report_metrics(report.get("metrics"), quality, resources)
    _finite_tree(report, "dashboard-report")

    _validate_wandb_files(
        wandb,
        {
            access_path.name: (access_path, "governance-receipt"),
            metric_path.name: (metric_path, "aggregate-receipt"),
            (dashboard / JSON_FILENAME).name: (dashboard / JSON_FILENAME, "dashboard-json"),
            (dashboard / HTML_FILENAME).name: (dashboard / HTML_FILENAME, "dashboard-html"),
            (dashboard / RECEIPT_FILENAME).name: (dashboard / RECEIPT_FILENAME, "dashboard-receipt"),
        },
    )
    expected_wandb_config = {
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "operation": OPERATION.value,
        "execution_id": execution.get("execution_id"),
        "git_commit": commit,
        "registry_sha256": registry.sha256,
        "ledger_sha256": ledger.sha256,
        "id_manifest_sha256": metric.get("id_manifest_sha256"),
        "model_identity_sha256": metric.get("model_identity_sha256"),
        "depth_alignment_protocol": DEPTH_ALIGNMENT_PROTOCOL,
        "depth_validity_protocol": DEPTH_VALIDITY_PROTOCOL,
        "aggregation_protocol": AGGREGATION_PROTOCOL,
    }
    expected_preliminary_summary = {
        **wandb_summary_from_serializable(report),
        "validation/postflight/status": "pending",
    }
    if (
        execution.get("schema_version") != EXECUTION_RECEIPT_SCHEMA
        or execution.get("status") != "pass"
        or execution.get("git_commit") != commit
        or execution.get("dataset_id") != DATASET_ID
        or execution.get("split_id") != SPLIT_ID
        or execution.get("operation") != OPERATION.value
        or execution.get("registry_sha256") != registry.sha256
        or execution.get("ledger_sha256") != ledger.sha256
        or execution.get("access_decision_sha256") != access_sha
        or execution.get("metric_gate_sha256") != metric_sha
        or access.get("execution_id") != execution.get("execution_id")
        or metric.get("execution_id") != execution.get("execution_id")
        or execution.get("wandb_receipt_sha256") != wandb_sha
        or execution.get("wandb_run_id") != wandb.get("run_id")
        or execution.get("wandb_entity") != wandb.get("entity")
        or execution.get("wandb_project") != wandb.get("project")
        or execution.get("wandb_group") != execution.get("execution_id")
        or execution.get("wandb_group") != wandb.get("group")
        or execution.get("wandb_job_type") != "geometry-official-mini"
        or execution.get("wandb_job_type") != wandb.get("job_type")
        or execution.get("wandb_run_name") != wandb.get("run_name")
        or execution.get("wandb_artifact_name") != f"geometry-official-mini-{execution.get('execution_id')}"
        or execution.get("wandb_artifact_name") != wandb.get("artifact_name")
        or execution.get("wandb_artifact_id") != wandb.get("artifact_id")
        or execution.get("wandb_artifact_digest") != wandb.get("artifact_digest")
        or execution.get("wandb_config_sha256") != wandb.get("config_sha256")
        or execution.get("wandb_summary_sha256") != wandb.get("summary_sha256")
        or wandb.get("config_sha256") != sha256_value(expected_wandb_config)
        or wandb.get("summary_sha256") != sha256_value(expected_preliminary_summary)
    ):
        raise ValueError("execution receipt does not bind all governed terminal artifacts")

    existing_postflight = (
        sorted((root / "postflight").glob("postflight-*.json")) if (root / "postflight").is_dir() else []
    )
    if len(existing_postflight) > 1:
        raise ValueError("multiple immutable postflight receipts found")
    existing_final_wandb = (
        sorted((root / "wandb-final").glob("wandb-final-*.json")) if (root / "wandb-final").is_dir() else []
    )
    existing_terminal = sorted((root / "terminal").glob("terminal-*.json")) if (root / "terminal").is_dir() else []
    if len(existing_final_wandb) > 1 or len(existing_terminal) > 1 or (existing_terminal and not existing_final_wandb):
        raise ValueError("terminal W&B/output receipts are incomplete or duplicated")

    preliminary_expected = _expected_file_set(
        access_path=access_path,
        metric_path=metric_path,
        dashboard=dashboard,
        wandb_path=wandb_path,
        execution_path=execution_path,
        postflight_paths=existing_postflight,
    )
    preliminary_expected.update(existing_final_wandb)
    preliminary_expected.update(existing_terminal)
    # Fail before publishing any postflight or terminal W&B state when the
    # output already contains an unbound file, directory, symlink, or path.
    _scan_safe_artifacts(root, preliminary_expected)

    postflight_payload = {
        "schema_version": POSTFLIGHT_SCHEMA,
        "status": "pass",
        "execution_id": execution["execution_id"],
        "git_commit": commit,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "operation": OPERATION,
        "registry_sha256": registry.sha256,
        "ledger_sha256": ledger.sha256,
        "access_decision_sha256": access_sha,
        "metric_gate_sha256": metric_sha,
        "dashboard_generation_id": dashboard_receipt["generation_id"],
        "wandb_receipt_sha256": wandb_sha,
        "execution_receipt_sha256": execution_sha,
        "scheduler": {
            "job_id": scheduler.job_id,
            "job_name": scheduler.job_name,
            "account": scheduler.account,
            "partition": scheduler.partition,
            "time_limit": scheduler.time_limit,
            "nodes": scheduler.nodes,
            "tasks": scheduler.tasks,
            "gpus": scheduler.gpus,
        },
    }
    postflight_receipt = write_content_addressed_json(postflight_payload, root / "postflight", prefix="postflight")
    if existing_postflight and postflight_receipt.path != existing_postflight[0]:
        raise ValueError("postflight rerun would create a different immutable receipt")

    terminal_summary = {
        "validation/execution_receipt_sha256": execution_sha,
        "validation/postflight_receipt_sha256": postflight_receipt.sha256,
    }
    if existing_final_wandb:
        final_wandb_path, final_wandb, final_wandb_sha = _single_content_address(root / "wandb-final", "wandb-final")
        validate_safe_wandb_final_receipt(final_wandb)
    else:
        final_wandb = wandb_finalizer(
            preliminary_receipt=wandb,
            artifact_root=root,
            files=(
                SafeArtifactFile(execution_path, "execution-receipt"),
                SafeArtifactFile(postflight_receipt.path, "postflight-receipt"),
            ),
            summary=terminal_summary,
        )
        validate_safe_wandb_final_receipt(final_wandb)

    expected_terminal_files = {
        execution_path.name: (execution_path, "execution-receipt"),
        postflight_receipt.path.name: (postflight_receipt.path, "postflight-receipt"),
    }
    final_files = final_wandb.get("files")
    if not isinstance(final_files, list):
        raise ValueError("terminal W&B receipt lacks its artifact file identities")
    observed_terminal_files = {str(value.get("name")): value for value in final_files if isinstance(value, Mapping)}
    if set(observed_terminal_files) != set(expected_terminal_files):
        raise ValueError("terminal W&B artifact set differs from execution/postflight receipts")
    for name, (path, role) in expected_terminal_files.items():
        item = observed_terminal_files[name]
        if (
            item.get("role") != role
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"terminal W&B artifact identity mismatch for {name}")
    if (
        final_wandb.get("preliminary_receipt_sha256") != wandb_sha
        or final_wandb.get("run_id") != wandb.get("run_id")
        or final_wandb.get("summary_sha256")
        != sha256_value({**terminal_summary, "validation/postflight/status": "pass"})
    ):
        raise ValueError("finalized online W&B receipt does not bind the exact postflight evidence")
    if not existing_final_wandb:
        final_wandb_address = write_content_addressed_json(
            final_wandb,
            root / "wandb-final",
            prefix="wandb-final",
        )
        final_wandb_path = final_wandb_address.path
        final_wandb_sha = final_wandb_address.sha256

    if existing_terminal:
        terminal_path, terminal, terminal_sha = _single_content_address(root / "terminal", "terminal")
    else:
        terminal_payload = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "pass",
            "execution_id": execution["execution_id"],
            "git_commit": commit,
            "postflight_receipt_sha256": postflight_receipt.sha256,
            "preliminary_wandb_receipt_sha256": wandb_sha,
            "final_wandb_receipt_sha256": final_wandb_sha,
            "wandb_run_id": final_wandb["run_id"],
            "wandb_terminal_artifact_id": final_wandb["artifact_id"],
            "wandb_terminal_artifact_digest": final_wandb["artifact_digest"],
        }
        terminal_address = write_content_addressed_json(terminal_payload, root / "terminal", prefix="terminal")
        terminal_path = terminal_address.path
        terminal = terminal_payload
        terminal_sha = terminal_address.sha256

    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("status") != "pass"
        or terminal.get("execution_id") != execution.get("execution_id")
        or terminal.get("git_commit") != commit
        or terminal.get("postflight_receipt_sha256") != postflight_receipt.sha256
        or terminal.get("preliminary_wandb_receipt_sha256") != wandb_sha
        or terminal.get("final_wandb_receipt_sha256") != final_wandb_sha
        or terminal.get("wandb_run_id") != final_wandb.get("run_id")
        or terminal.get("wandb_terminal_artifact_id") != final_wandb.get("artifact_id")
        or terminal.get("wandb_terminal_artifact_digest") != final_wandb.get("artifact_digest")
    ):
        raise ValueError("terminal receipt does not bind postflight and finalized online W&B evidence")
    final_expected = _expected_file_set(
        access_path=access_path,
        metric_path=metric_path,
        dashboard=dashboard,
        wandb_path=wandb_path,
        execution_path=execution_path,
        postflight_paths=(postflight_receipt.path,),
    )
    final_expected.update({final_wandb_path, terminal_path})
    _scan_safe_artifacts(root, final_expected)
    return ContentAddress(terminal_path, terminal_sha, terminal_path.stat().st_size)


def validate_geometry_official_mini(
    *,
    output: str | Path,
    repo_root: str | Path,
    job_id: str,
    scheduler_lookup: SchedulerLookup = scheduler_identity,
    git_lookup: GitLookup = git_identity,
    wandb_finalizer: WandbFinalizer = finalize_safe_online_run,
) -> ContentAddress:
    """Serialize postflight publication and verify the exact final artifact set."""

    root = Path(output).resolve(strict=True)
    lock_path = root.parent / f".{root.name}.postflight.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _validate_geometry_official_mini_locked(
            output=root,
            repo_root=repo_root,
            job_id=job_id,
            scheduler_lookup=scheduler_lookup,
            git_lookup=git_lookup,
            wandb_finalizer=wandb_finalizer,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--allocation-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.allocation_only:
        identity = scheduler_identity(args.job_id)
        _validate_scheduler(identity, args.job_id)
        print(json.dumps({"status": "pass", "scheduler": asdict(identity)}, sort_keys=True))
        return
    if args.output is None or args.repo_root is None:
        raise SystemExit("--output and --repo-root are required unless --allocation-only is used")
    receipt = validate_geometry_official_mini(
        output=args.output,
        repo_root=args.repo_root,
        job_id=args.job_id,
    )
    print(json.dumps({"status": "pass", "terminal_sha256": receipt.sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
