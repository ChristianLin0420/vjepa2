#!/usr/bin/env python3
"""Governed postflight for the synthetic Phase 2g training smoke."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from jepa4d.models.phase2f_scale_geometry import (
    DEFAULT_PHASE2F_ARMS,
    Phase2fGeometryConfig,
    Phase2fScaleGeometryProbe,
)
from jepa4d.training.phase2f_losses import Phase2fLossConfig
from jepa4d.training.phase2f_training import PHASE2F_CHECKPOINT_SCHEMA, phase2f_arm_configs
from jepa4d.validation._content import (
    ContentAddress,
    json_value,
    sha256_file,
    sha256_value,
    verify_content_addressed_json,
    write_content_addressed_json,
)

SMOKE_SCHEMA = "jepa4d-phase2g-training-instrumentation-smoke-v2"
STEP_SCHEMA = "jepa4d-phase2g-training-instrumentation-step-v2"
PRELIMINARY_WANDB_SCHEMA = "jepa4d-phase2g-training-instrumentation-wandb-v2"
POSTFLIGHT_SCHEMA = "jepa4d-phase2g-training-instrumentation-postflight-v1"
FINAL_WANDB_SCHEMA = "jepa4d-phase2g-training-instrumentation-wandb-final-v1"
TERMINAL_SCHEMA = "jepa4d-phase2g-training-instrumentation-terminal-v1"
CLAIM_BOUNDARY = (
    "Synthetic integration smoke only: no dataset, cache, held-out target, or scientific-quality evidence was used or "
    "produced. Metrics validate training and logging instrumentation, not model quality or Phase 2g promotion."
)
ARMS = tuple(DEFAULT_PHASE2F_ARMS)
STEPS_PER_ARM = 3
TOTAL_STEPS = len(ARMS) * STEPS_PER_ARM

APPROVED_ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
APPROVED_PARTITIONS = frozenset(
    {"polar4", "polar3", "polar", "batch_block1", "grizzly", "batch_block2", "batch_block3"}
)
MAX_TIME_SECONDS = 30 * 60
EXPECTED_CPUS = 8
EXPECTED_MEMORY_MIB = 32 * 1024
GOVERNED_SEED = 260630
GOVERNED_INPUT_DIM = 768
GOVERNED_SPATIAL_SIZE = 24
GOVERNED_SOURCE_GROUPS = 2
GOVERNED_GRADIENT_CLIP = 5.0
GOVERNED_LEARNING_RATE = 1e-3
GOVERNED_WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM_EPSILON = 1e-6

_JOB_ID = re.compile(r"^[0-9]+$")
_JOB_NAME = re.compile(r"^j4d-p2g-smoke-[A-Za-z0-9_.-]+$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL = re.compile(r"wandb_v1_[A-Za-z0-9_-]+|(?:^|[._-])hf_[A-Za-z0-9]{16,}", re.IGNORECASE)
_LOCAL_PATH = re.compile(r"(?i)(?:\bfile://|\bs3://|(?:/lustre|/home|/root|/tmp|/var/tmp)/\S+)")

_STEP_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "arm_step",
        "global_step",
        "objectives",
        "diagnostics",
        "gradients",
        "optimizer",
        "resources",
        "parameter_counts",
        "gradient_firewall_passed",
    }
)
_OBJECTIVE_NAMES = frozenset(
    {
        "total",
        "shape_objective",
        "scale_objective",
        "field_objective",
        "monolithic_nll",
        "monolithic_scale_invariant",
        "monolithic_gradient",
        "monolithic_distillation",
        "shape_nll",
        "shape_l1",
        "shape_gradient",
        "scale_nll",
        "scale_l1",
        "paired_scale_consistency",
        "scale_field_objective",
        "scale_field_fit",
        "scale_field_tv",
    }
)
_DIAGNOSTIC_NAMES = frozenset(
    {"joint_nll_diagnostic_only", "optimal_log_scale_mean", "scale_field_zero_mean_error", "scale_field_max_abs"}
)
_EXPECTED_OBJECTIVES = {
    "M0": frozenset(
        {
            "total",
            "shape_objective",
            "scale_objective",
            "field_objective",
            "monolithic_nll",
            "monolithic_scale_invariant",
            "monolithic_gradient",
            "monolithic_distillation",
        }
    ),
    **{
        arm: frozenset(
            {
                "total",
                "shape_objective",
                "scale_objective",
                "scale_field_objective",
                "shape_nll",
                "shape_l1",
                "shape_gradient",
                "scale_nll",
                "scale_l1",
                "paired_scale_consistency",
                "scale_field_fit",
                "scale_field_tv",
            }
        )
        for arm in ("M1", "M2", "M3")
    },
}
_EXPECTED_DIAGNOSTICS = {
    "M0": frozenset(),
    **{arm: _DIAGNOSTIC_NAMES for arm in ("M1", "M2", "M3")},
}
_GRADIENT_NAMES = frozenset(
    {
        "norm_shape",
        "norm_scale",
        "norm_field",
        "norm_total_before_clip",
        "firewall_max_forbidden_norm",
        "firewall_shape_to_shape",
        "firewall_shape_to_scale",
        "firewall_shape_to_field",
        "firewall_scale_to_shape",
        "firewall_scale_to_scale",
        "firewall_scale_to_field",
        "firewall_field_to_shape",
        "firewall_field_to_scale",
        "firewall_field_to_field",
    }
)
_FORBIDDEN_GRADIENTS = (
    "firewall_shape_to_scale",
    "firewall_shape_to_field",
    "firewall_scale_to_shape",
    "firewall_scale_to_field",
    "firewall_field_to_shape",
    "firewall_field_to_scale",
)
_OPTIMIZER_KEYS = frozenset(
    {
        "learning_rate",
        "gradient_clip_threshold",
        "pre_clip_gradient_norm",
        "post_clip_gradient_norm",
        "applied_clip_coefficient",
        "was_clipped",
        "parameter_update_norm",
    }
)
_MEMORY_KEYS = frozenset({"allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"})
_NVIDIA_KEYS = frozenset(
    {
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_temperature_c",
        "gpu_power_w",
        "gpu_sm_clock_mhz",
        "gpu_memory_clock_mhz",
    }
)


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
    cpus: int
    allocated_cpus: int
    cpus_per_task: int
    memory_mib: int
    array_job_id: str | None
    array_task_id: str | None


@dataclass(frozen=True, slots=True)
class _StableIdentity:
    device: int
    inode: int
    bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


SchedulerLookup = Callable[[str], SchedulerIdentity]
GitLookup = Callable[[Path], tuple[str, bool]]
WandbPreliminaryVerifier = Callable[..., dict[str, Any]]
WandbFinalizer = Callable[..., dict[str, Any]]


def _time_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError(f"unsupported Slurm time limit: {value!r}")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid Slurm time limit: {value!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _memory_mib(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unsupported Slurm memory quantity: {value!r}")
    try:
        quantity = Decimal(match.group(1))
    except InvalidOperation as error:
        raise ValueError(f"invalid Slurm memory quantity: {value!r}") from error
    unit = match.group(2).upper()
    factors = {
        "": Decimal(1),
        "K": Decimal(1) / 1024,
        "M": Decimal(1),
        "G": Decimal(1024),
        "T": Decimal(1024**2),
        "P": Decimal(1024**3),
    }
    mib = quantity * factors[unit]
    if mib != mib.to_integral_value():
        raise ValueError(f"Slurm memory quantity is not an integral number of MiB: {value!r}")
    return int(mib)


def scheduler_identity(job_id: str) -> SchedulerIdentity:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("postflight requires a numeric Slurm job ID")
    result = subprocess.run(
        ("scontrol", "show", "job", "-o", job_id),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("unable to resolve the active Slurm allocation")
    fields = dict(token.split("=", 1) for token in result.stdout.split() if "=" in token)
    tres = dict(item.split("=", 1) for item in str(fields.get("AllocTRES", "")).split(",") if "=" in item)
    gpu_values = [int(value) for name, value in tres.items() if re.fullmatch(r"gres/gpu(?::[^=,]+)?", name)]
    try:
        return SchedulerIdentity(
            job_id=str(fields["JobId"]),
            job_name=str(fields["JobName"]),
            account=str(fields["Account"]),
            partition=str(fields["Partition"]),
            time_limit=str(fields["TimeLimit"]),
            nodes=int(fields["NumNodes"]),
            tasks=int(fields["NumTasks"]),
            gpus=gpu_values[0] if len(gpu_values) == 1 else -1,
            cpus=int(fields["NumCPUs"]),
            allocated_cpus=int(tres["cpu"]),
            cpus_per_task=int(fields["CPUs/Task"]),
            memory_mib=_memory_mib(tres["mem"]),
            array_job_id=None if fields.get("ArrayJobId") in {None, "", "N/A"} else str(fields["ArrayJobId"]),
            array_task_id=None if fields.get("ArrayTaskId") in {None, "", "N/A"} else str(fields["ArrayTaskId"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Slurm identity is incomplete or malformed: {error}") from error


def _validate_scheduler(identity: SchedulerIdentity, expected_job_id: str) -> None:
    if identity.job_id != expected_job_id or not _JOB_ID.fullmatch(identity.job_id):
        raise ValueError("Slurm job identity mismatch")
    if not _JOB_NAME.fullmatch(identity.job_name):
        raise ValueError("Slurm job name is not a unique Phase 2g smoke name")
    if identity.account != APPROVED_ACCOUNT:
        raise ValueError("Slurm job used an unapproved account")
    if identity.partition not in APPROVED_PARTITIONS:
        raise ValueError("Slurm job used an unapproved partition")
    if not 0 < _time_seconds(identity.time_limit) <= MAX_TIME_SECONDS:
        raise ValueError("Slurm job exceeded the 30-minute maximum")
    if (identity.nodes, identity.tasks, identity.gpus) != (1, 1, 1):
        raise ValueError("Phase 2g smoke requires exactly one node, one task, and one GPU")
    if (identity.cpus, identity.allocated_cpus, identity.cpus_per_task) != (
        EXPECTED_CPUS,
        EXPECTED_CPUS,
        EXPECTED_CPUS,
    ):
        raise ValueError("Phase 2g smoke requires exactly eight allocated CPUs for its one task")
    if identity.memory_mib != EXPECTED_MEMORY_MIB:
        raise ValueError("Phase 2g smoke requires exactly 32 GiB of allocated memory")
    if identity.array_job_id is not None or identity.array_task_id is not None:
        raise ValueError("Phase 2g smoke must not run as an array task")


def git_identity(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(("git", "-C", str(repo_root), "rev-parse", "HEAD"), text=True).strip()
    status = subprocess.check_output(
        ("git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"), text=True
    )
    return commit, not status.strip()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSONL artifact {path.name}: {error}") from error
    for index, line in enumerate(lines):
        if not line:
            raise ValueError(f"blank JSONL row at line {index + 1}")
        try:
            value = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSONL row at line {index + 1}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at line {index + 1}")
        rows.append(value)
    return rows


def _safe_finite_tree(value: Any, location: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            key_tokens = {token for token in re.split(r"[_.-]+", name.casefold()) if token}
            if key_tokens & {
                "api",
                "apikey",
                "authorization",
                "cookie",
                "credential",
                "key",
                "netrc",
                "password",
                "secret",
                "token",
            }:
                raise ValueError(f"credential-like field at {location}.{name}")
            _safe_finite_tree(child, f"{location}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _safe_finite_tree(child, f"{location}[{index}]")
        return
    if isinstance(value, str):
        if _CREDENTIAL.search(value):
            raise ValueError(f"credential-shaped value at {location}")
        if _LOCAL_PATH.search(value):
            raise ValueError(f"local path value at {location}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if value is not None and not isinstance(value, (bool, int, float)):
        raise TypeError(f"unsupported value at {location}: {type(value).__name__}")


def _online_run_url_matches(value: object, *, entity: object, project: object, run_id: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or not isinstance(entity, str)
        or not entity
        or not isinstance(project, str)
        or not project
        or not isinstance(run_id, str)
        or not run_id
    ):
        return False
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    return parts[-4:] == [entity, project, "runs", run_id]


def _backend_artifact_name_matches(name: object, *, collection: str, version: object) -> bool:
    return isinstance(name, str) and isinstance(version, str) and name in {collection, f"{collection}:{version}"}


def _stable_identity(path: Path) -> _StableIdentity:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    final = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields) or any(
        getattr(after, field) != getattr(final, field) for field in fields
    ):
        raise RuntimeError(f"artifact changed while it was inspected: {path.name}")
    return _StableIdentity(
        device=after.st_dev,
        inode=after.st_ino,
        bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        sha256=digest.hexdigest(),
    )


def _identity_matches(value: object, path: Path, *, published_name: str | None = None) -> bool:
    if not isinstance(value, Mapping):
        return False
    identity = _stable_identity(path)
    return (
        value.get("name") == (path.name if published_name is None else published_name)
        and value.get("bytes") == identity.bytes
        and value.get("sha256") == identity.sha256
    )


def _state_files(root: Path, directory: str, prefix: str) -> list[Path]:
    state_dir = root / directory
    return sorted(state_dir.glob(f"{prefix}-*.json")) if state_dir.is_dir() else []


def _scan_exact_artifacts(root: Path, expected: set[Path]) -> None:
    entries = set(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Phase 2g output may not contain symbolic links")
    if any(not path.is_file() and not path.is_dir() for path in entries):
        raise ValueError("Phase 2g output may contain only regular files and directories")
    actual_files = {path for path in entries if path.is_file()}
    if actual_files != expected:
        extra = sorted(str(path.relative_to(root)) for path in actual_files - expected)
        missing = sorted(str(path.relative_to(root)) for path in expected - actual_files)
        raise ValueError(f"Phase 2g artifact allowlist mismatch: extra={extra}, missing={missing}")
    expected_directories: set[Path] = set()
    for path in expected:
        parent = path.parent
        while parent != root:
            expected_directories.add(parent)
            parent = parent.parent
    actual_directories = {path for path in entries if path.is_dir()}
    if actual_directories != expected_directories:
        extra = sorted(str(path.relative_to(root)) for path in actual_directories - expected_directories)
        missing = sorted(str(path.relative_to(root)) for path in expected_directories - actual_directories)
        raise ValueError(f"Phase 2g directory allowlist mismatch: extra={extra}, missing={missing}")
    for path in actual_files:
        if path.stat().st_size < 1:
            raise ValueError(f"Phase 2g artifact is empty: {path.name}")
        if path.suffix in {".json", ".jsonl"} or path.name == "SUCCESS":
            document = path.read_text(encoding="utf-8")
            if _CREDENTIAL.search(document) or _LOCAL_PATH.search(document):
                raise ValueError(f"unsafe credential or local path content in artifact: {path.name}")
        elif path.suffix != ".pt":
            raise ValueError(f"unsafe Phase 2g artifact type: {path.name}")


def _single_content_address(root: Path, directory: str, prefix: str) -> tuple[Path, dict[str, Any], str]:
    matches = _state_files(root, directory, prefix)
    if len(matches) != 1 or matches[0].is_symlink():
        raise ValueError(f"expected exactly one immutable {prefix} receipt")
    value = verify_content_addressed_json(matches[0], prefix=prefix)
    return matches[0], value, matches[0].stem.removeprefix(f"{prefix}-")


def _require_numeric_mapping(value: object, expected: frozenset[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{location} has an unexpected schema")
    if any(
        isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item))
        for item in value.values()
    ):
        raise ValueError(f"{location} must contain finite numeric scalars")
    return value


def _validate_runtime_identity(runtime: object, *, repo_root: Path, commit: str, job_id: str) -> None:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "git_commit",
        "scheduler_job_id",
        "python_version",
        "torch_version",
        "torch_cuda_build",
        "cudnn_version",
        "hardware",
        "code",
    }:
        raise ValueError("training receipt lacks runtime identity")
    if runtime.get("git_commit") != commit or runtime.get("scheduler_job_id") != job_id:
        raise ValueError("training runtime does not bind the clean Git commit and Slurm job")
    if (
        any(
            not isinstance(runtime.get(name), str) or not str(runtime[name]).strip()
            for name in ("python_version", "torch_version", "torch_cuda_build")
        )
        or isinstance(runtime.get("cudnn_version"), bool)
        or not isinstance(runtime.get("cudnn_version"), int)
        or int(runtime["cudnn_version"]) <= 0
    ):
        raise ValueError("training runtime software identity is incomplete")
    hardware = runtime.get("hardware")
    if (
        not isinstance(hardware, Mapping)
        or set(hardware)
        != {
            "device_type",
            "device_name",
            "device_uuid_sha256",
            "compute_capability",
            "total_memory_bytes",
        }
        or hardware.get("device_type") != "cuda"
    ):
        raise ValueError("governed Phase 2g smoke requires a CUDA runtime identity")
    if (
        not isinstance(hardware.get("device_name"), str)
        or not str(hardware["device_name"]).strip()
        or not _SHA256.fullmatch(str(hardware.get("device_uuid_sha256", "")))
        or not isinstance(hardware.get("compute_capability"), str)
        or re.fullmatch(r"[0-9]+\.[0-9]+", str(hardware["compute_capability"])) is None
        or isinstance(hardware.get("total_memory_bytes"), bool)
        or not isinstance(hardware.get("total_memory_bytes"), int)
        or int(hardware["total_memory_bytes"]) <= 0
    ):
        raise ValueError("CUDA hardware identity is incomplete or unsafe")
    code = runtime.get("code")
    expected_paths = {
        "runner": repo_root / "scripts" / "run_phase2g_training_smoke.py",
        "model_module": repo_root / "jepa4d" / "models" / "phase2f_scale_geometry.py",
        "training_module": repo_root / "jepa4d" / "training" / "phase2f_training.py",
    }
    if not isinstance(code, Mapping) or set(code) != set(expected_paths):
        raise ValueError("runtime code identity set is incomplete")
    for name, path in expected_paths.items():
        if (
            not isinstance(code[name], Mapping)
            or set(code[name]) != {"name", "bytes", "sha256"}
            or not _identity_matches(code[name], path)
        ):
            raise ValueError(f"runtime code identity mismatch: {name}")


def _validate_config(
    config: object,
    *,
    execution_id: str,
    runtime: object,
    expected_run_name: str,
    expected_project: str,
    expected_entity: str | None,
) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("training receipt lacks its configuration")
    required = {
        "schema_version",
        "execution_id",
        "evidence_level",
        "claim_boundary",
        "arms",
        "max_steps_per_arm",
        "seed",
        "device_type",
        "input_dim",
        "spatial_size",
        "source_groups",
        "views_per_group",
        "gradient_clip",
        "nll_convention",
        "arm_configs",
        "loss_config",
        "optimizer",
        "determinism",
        "resource_policy",
        "requested_wandb_identity",
        "synthetic_inputs_only",
        "dataset_or_cache_access",
        "runtime_identity",
    }
    if set(config) != required:
        raise ValueError("Phase 2g configuration schema is incomplete or extended")
    if (
        config.get("schema_version") != SMOKE_SCHEMA
        or config.get("execution_id") != execution_id
        or config.get("evidence_level") != "integration-smoke"
        or config.get("claim_boundary") != CLAIM_BOUNDARY
        or config.get("arms") != list(ARMS)
        or config.get("max_steps_per_arm") != STEPS_PER_ARM
        or config.get("device_type") != "cuda"
        or config.get("views_per_group") != 2
        or config.get("nll_convention")
        != "Gaussian NLL omits its additive constant; optimized NLL terms may be negative"
        or config.get("resource_policy") != "diagnostic-only"
        or config.get("synthetic_inputs_only") is not True
        or config.get("dataset_or_cache_access") is not False
        or config.get("runtime_identity") != runtime
    ):
        raise ValueError("Phase 2g configuration violates the governed smoke contract")
    for name in ("seed", "input_dim", "spatial_size", "source_groups"):
        if isinstance(config.get(name), bool) or not isinstance(config.get(name), int) or int(config[name]) <= 0:
            raise ValueError(f"configuration {name} must be a positive integer")
    clip = config.get("gradient_clip")
    if (
        isinstance(clip, bool)
        or not isinstance(clip, int | float)
        or not math.isfinite(float(clip))
        or float(clip) <= 0
    ):
        raise ValueError("configuration gradient_clip must be finite and positive")
    governed_scalars = {
        "seed": GOVERNED_SEED,
        "input_dim": GOVERNED_INPUT_DIM,
        "spatial_size": GOVERNED_SPATIAL_SIZE,
        "source_groups": GOVERNED_SOURCE_GROUPS,
        "gradient_clip": GOVERNED_GRADIENT_CLIP,
    }
    if any(config.get(name) != value for name, value in governed_scalars.items()):
        raise ValueError("configuration differs from the frozen governed numeric smoke contract")
    arm_configs = config.get("arm_configs")
    if not isinstance(arm_configs, Mapping) or set(arm_configs) != set(ARMS):
        raise ValueError("configuration arm matrix is incomplete")
    expected_arm_configs = {
        arm: json_value(asdict(value)) for arm, value in phase2f_arm_configs(int(config["input_dim"])).items()
    }
    if arm_configs != expected_arm_configs:
        raise ValueError("configuration arm matrix differs from the registered Phase 2g contract")
    for arm in ARMS:
        if not isinstance(arm_configs[arm], Mapping) or arm_configs[arm].get("arm") != arm:
            raise ValueError(f"configuration arm identity mismatch: {arm}")
    optimizer = config.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer)
        != {
            "name",
            "learning_rate",
            "weight_decay",
            "betas",
            "eps",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        }
        or optimizer.get("name") != "torch.optim.AdamW"
        or optimizer.get("betas") != [0.9, 0.999]
        or optimizer.get("eps") != 1e-8
        or any(
            optimizer.get(name) is not False
            for name in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused")
        )
    ):
        raise ValueError("configuration does not bind the exact AdamW optimizer")
    for name in ("learning_rate", "weight_decay"):
        value = optimizer.get(name) if isinstance(optimizer, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"optimizer {name} must be finite and non-negative")
    if (
        optimizer.get("learning_rate") != GOVERNED_LEARNING_RATE
        or optimizer.get("weight_decay") != GOVERNED_WEIGHT_DECAY
    ):
        raise ValueError("optimizer differs from the frozen governed numeric smoke contract")
    determinism = config.get("determinism")
    if (
        not isinstance(determinism, Mapping)
        or set(determinism)
        != {
            "synthetic_cpu_generator_seeded_per_arm_step",
            "torch_manual_seeded_per_arm",
            "cuda_manual_seeded_per_arm",
            "deterministic_algorithms_enabled",
            "deterministic_algorithms_warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "bitwise_reproducibility_claimed",
        }
        or determinism.get("bitwise_reproducibility_claimed") is not False
    ):
        raise ValueError("configuration must not claim unsupported bitwise reproducibility")
    if (
        determinism.get("synthetic_cpu_generator_seeded_per_arm_step") is not True
        or determinism.get("torch_manual_seeded_per_arm") is not True
        or determinism.get("cuda_manual_seeded_per_arm") is not True
    ):
        raise ValueError("configuration lacks the required deterministic seeding evidence")
    if config.get("loss_config") != json_value(asdict(Phase2fLossConfig())):
        raise ValueError("configuration differs from the registered loss contract")
    requested_wandb = config.get("requested_wandb_identity")
    if not isinstance(requested_wandb, Mapping) or set(requested_wandb) != {
        "entity",
        "project",
        "group",
        "run_name",
        "job_type",
    }:
        raise ValueError("configuration lacks the requested W&B identity")
    if requested_wandb != {
        "entity": expected_entity,
        "project": expected_project,
        "group": f"phase2g-smoke-{execution_id}",
        "run_name": expected_run_name,
        "job_type": "phase2g-instrumentation-smoke",
    }:
        raise ValueError("configured W&B identity differs from the submitted identity")
    return config


def _validate_checkpoint(
    path: Path,
    arm: str,
    identity: object,
    expected_config: object,
) -> dict[str, int]:
    if not isinstance(identity, Mapping) or set(identity) != {
        "name",
        "bytes",
        "sha256",
        "schema_version",
        "exact_reload",
        "parameter_counts",
    }:
        raise ValueError(f"checkpoint receipt schema mismatch: {arm}")
    if (
        not _identity_matches(identity, path)
        or identity.get("schema_version") != PHASE2F_CHECKPOINT_SCHEMA
        or identity.get("exact_reload") is not True
    ):
        raise ValueError(f"checkpoint identity mismatch: {arm}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"checkpoint cannot be loaded safely: {arm}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "config", "state_dict", "parameter_counts"}:
        raise ValueError(f"checkpoint payload schema mismatch: {arm}")
    if (
        payload.get("schema_version") != PHASE2F_CHECKPOINT_SCHEMA
        or json_value(payload.get("config")) != expected_config
    ):
        raise ValueError(f"checkpoint configuration mismatch: {arm}")
    config_values = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_values, dict) or not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"checkpoint is missing config or state tensors: {arm}")
    if any(
        not isinstance(tensor, torch.Tensor)
        or ((tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()))
        for tensor in state_dict.values()
    ):
        raise ValueError(f"checkpoint contains invalid or non-finite tensors: {arm}")
    try:
        model = Phase2fScaleGeometryProbe(Phase2fGeometryConfig(**config_values))
        expected_state = model.state_dict()
        if set(state_dict) != set(expected_state):
            raise ValueError("checkpoint state tensor names differ from the reconstructed model")
        for name, tensor in state_dict.items():
            expected = expected_state[name]
            if (
                tensor.dtype != expected.dtype
                or tensor.layout != expected.layout
                or tensor.device.type != "cpu"
                or tensor.shape != expected.shape
            ):
                raise ValueError(f"checkpoint tensor contract mismatch: {name}")
        model.load_state_dict(state_dict, strict=True)
        loaded_state = model.state_dict()
        if any(not torch.equal(loaded_state[name], tensor) for name, tensor in state_dict.items()):
            raise ValueError("checkpoint reload changed an exact state tensor")
    except Exception as error:
        raise ValueError(f"checkpoint does not strictly reconstruct its Phase 2g arm: {arm}") from error
    counts = model.parameter_counts()
    if payload.get("parameter_counts") != counts or identity.get("parameter_counts") != counts:
        raise ValueError(f"checkpoint parameter receipt mismatch: {arm}")
    if not _identity_matches(identity, path):
        raise RuntimeError(f"checkpoint changed while it was validated: {arm}")
    return counts


def _validate_step_rows(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    parameter_counts: Mapping[str, Mapping[str, int]],
) -> None:
    if len(rows) != TOTAL_STEPS:
        raise ValueError(f"steps.jsonl must contain exactly {TOTAL_STEPS} optimizer steps")
    expected_sequence = [(arm, arm_step) for arm in ARMS for arm_step in range(STEPS_PER_ARM)]
    learning_rate = float(config["optimizer"]["learning_rate"])
    clip_threshold = float(config["gradient_clip"])
    source_groups = int(config["source_groups"])
    views = int(config["views_per_group"])
    for global_step, (row, (arm, arm_step)) in enumerate(zip(rows, expected_sequence, strict=True)):
        location = f"steps[{global_step}]"
        if set(row) != _STEP_KEYS or row.get("schema_version") != STEP_SCHEMA:
            raise ValueError(f"step row schema mismatch at {global_step}")
        if (
            type(row.get("arm_step")) is not int
            or type(row.get("global_step")) is not int
            or (row.get("arm"), row.get("arm_step"), row.get("global_step")) != (arm, arm_step, global_step)
        ):
            raise ValueError(f"step order or index mismatch at {global_step}")
        if row.get("gradient_firewall_passed") is not True:
            raise ValueError(f"gradient firewall did not pass at {global_step}")

        objectives = row.get("objectives")
        diagnostics = row.get("diagnostics")
        if not isinstance(objectives, Mapping) or not isinstance(diagnostics, Mapping):
            raise ValueError(f"objective/diagnostic schema mismatch at {global_step}")
        if set(objectives) != _EXPECTED_OBJECTIVES[arm] or set(diagnostics) != _EXPECTED_DIAGNOSTICS[arm]:
            raise ValueError(f"objective/diagnostic metric set mismatch at {global_step}")
        field_name = "field_objective" if arm == "M0" else "scale_field_objective"
        other_field = "scale_field_objective" if arm == "M0" else "field_objective"
        required_objectives = {"total", "shape_objective", "scale_objective", field_name}
        if not required_objectives.issubset(objectives) or other_field in objectives:
            raise ValueError(f"optimized objective ownership mismatch at {global_step}")
        for group in (objectives, diagnostics):
            if any(
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value))
                for value in group.values()
            ):
                raise ValueError(f"non-finite objective/diagnostic metric at {global_step}")
        expected_total = sum(float(objectives[name]) for name in ("shape_objective", "scale_objective", field_name))
        if not math.isclose(float(objectives["total"]), expected_total, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"optimized objective decomposition mismatch at {global_step}")
        loss = config["loss_config"]
        if arm == "M0":
            if any(float(objectives[name]) != 0.0 for name in ("scale_objective", "field_objective")):
                raise ValueError(f"inactive M0 objective is nonzero at {global_step}")
            expected_shape = (
                float(objectives["monolithic_nll"])
                + 0.25 * float(objectives["monolithic_scale_invariant"])
                + 0.1 * float(objectives["monolithic_gradient"])
                + 0.25 * float(objectives["monolithic_distillation"])
            )
            if float(objectives["monolithic_distillation"]) != 0.0 or not math.isclose(
                float(objectives["shape_objective"]), expected_shape, rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError(f"M0 component objective mismatch at {global_step}")
        else:
            expected_shape = (
                float(loss["centered_shape_weight"]) * float(objectives["shape_l1"])
                + float(loss["shape_gradient_weight"]) * float(objectives["shape_gradient"])
                + float(loss["shape_nll_weight"]) * float(objectives["shape_nll"])
            )
            expected_scale = (
                float(loss["global_scale_weight"]) * float(objectives["scale_l1"])
                + float(loss["scale_nll_weight"]) * float(objectives["scale_nll"])
                + float(loss["paired_scale_consistency_weight"]) * float(objectives["paired_scale_consistency"])
            )
            expected_field = float(loss["scale_field_fit_weight"]) * float(objectives["scale_field_fit"]) + float(
                loss["scale_field_tv_weight"]
            ) * float(objectives["scale_field_tv"])
            if not all(
                (
                    math.isclose(float(objectives["shape_objective"]), expected_shape, rel_tol=1e-5, abs_tol=1e-6),
                    math.isclose(float(objectives["scale_objective"]), expected_scale, rel_tol=1e-5, abs_tol=1e-6),
                    math.isclose(
                        float(objectives["scale_field_objective"]), expected_field, rel_tol=1e-5, abs_tol=1e-6
                    ),
                )
            ):
                raise ValueError(f"factorized component objective mismatch at {global_step}")
            if arm in {"M1", "M2"} and any(
                float(value) != 0.0
                for value in (
                    objectives["scale_field_objective"],
                    objectives["scale_field_fit"],
                    objectives["scale_field_tv"],
                    diagnostics["scale_field_zero_mean_error"],
                    diagnostics["scale_field_max_abs"],
                )
            ):
                raise ValueError(f"inactive scale-field metric is nonzero at {global_step}")

        gradients = _require_numeric_mapping(row.get("gradients"), _GRADIENT_NAMES, f"{location}.gradients")
        if any(float(value) < 0 for value in gradients.values()):
            raise ValueError(f"gradient norm is negative at {global_step}")
        forbidden = [float(gradients[name]) for name in _FORBIDDEN_GRADIENTS]
        if any(value != 0.0 for value in forbidden) or float(gradients["firewall_max_forbidden_norm"]) != max(
            forbidden
        ):
            raise ValueError(f"gradient firewall evidence mismatch at {global_step}")
        owned = {
            "shape": float(gradients["norm_shape"]),
            "scale": float(gradients["norm_scale"]),
            "field": float(gradients["norm_field"]),
        }
        expected_owned = {
            "shape": True,
            "scale": arm in {"M1", "M2", "M3"},
            "field": arm == "M3",
        }
        for name, active in expected_owned.items():
            if (owned[name] > 0) is not active:
                raise ValueError(f"owned gradient evidence mismatch for {arm}/{name} at {global_step}")
            diagonal = float(gradients[f"firewall_{name}_to_{name}"])
            if (diagonal > 0) is not active or not math.isclose(
                diagonal,
                owned[name],
                rel_tol=2e-5,
                abs_tol=1e-7,
            ):
                raise ValueError(f"owned firewall gradient mismatch for {arm}/{name} at {global_step}")
        recomputed_pre_clip = math.sqrt(sum(value * value for value in owned.values()))
        if not math.isclose(
            float(gradients["norm_total_before_clip"]), recomputed_pre_clip, rel_tol=2e-5, abs_tol=1e-7
        ):
            raise ValueError(f"owned gradients do not reconstruct the total norm at {global_step}")

        optimizer = _require_numeric_mapping(row.get("optimizer"), _OPTIMIZER_KEYS, f"{location}.optimizer")
        if optimizer["learning_rate"] != learning_rate or optimizer["gradient_clip_threshold"] != clip_threshold:
            raise ValueError(f"optimizer configuration mismatch at {global_step}")
        pre_clip = float(optimizer["pre_clip_gradient_norm"])
        post_clip = float(optimizer["post_clip_gradient_norm"])
        coefficient = float(optimizer["applied_clip_coefficient"])
        was_clipped = optimizer["was_clipped"]
        if was_clipped not in {0, 1} or isinstance(was_clipped, bool):
            raise ValueError(f"optimizer clipping flag is invalid at {global_step}")
        if pre_clip < 0 or post_clip < 0 or post_clip > clip_threshold + max(1e-6, clip_threshold * 1e-5):
            raise ValueError(f"post-clip norm exceeds the governed threshold at {global_step}")
        if not math.isclose(pre_clip, float(gradients["norm_total_before_clip"]), rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"optimizer pre-clip norm differs from gradient evidence at {global_step}")
        expected_coefficient = min(1.0, clip_threshold / (pre_clip + CLIP_GRAD_NORM_EPSILON))
        expected_post_clip = pre_clip * expected_coefficient
        expected_clipped = int(expected_coefficient < 1.0)
        if (
            not math.isclose(coefficient, expected_coefficient, rel_tol=2e-6, abs_tol=1e-8)
            or not math.isclose(post_clip, expected_post_clip, rel_tol=2e-5, abs_tol=1e-7)
            or was_clipped != expected_clipped
        ):
            raise ValueError(f"actual clipping fields violate torch clip_grad_norm_ semantics at {global_step}")
        if float(optimizer["parameter_update_norm"]) <= 0:
            raise ValueError(f"optimizer did not update parameters at {global_step}")

        if row.get("parameter_counts") != parameter_counts[arm]:
            raise ValueError(f"step parameter counts differ from checkpoint for {arm}")
        resources = row.get("resources")
        if not isinstance(resources, Mapping) or set(resources) != {"policy", "timing", "memory", "nvidia_smi"}:
            raise ValueError(f"resource diagnostic schema mismatch at {global_step}")
        if resources.get("policy") != "diagnostic-only":
            raise ValueError(f"resource evidence is not diagnostic-only at {global_step}")
        timing = _require_numeric_mapping(
            resources.get("timing"),
            frozenset({"step_seconds", "samples_per_second", "source_groups_per_second"}),
            f"{location}.resources.timing",
        )
        seconds = float(timing["step_seconds"])
        if (
            seconds <= 0
            or not math.isclose(float(timing["samples_per_second"]), source_groups * views / seconds, rel_tol=1e-9)
            or not math.isclose(float(timing["source_groups_per_second"]), source_groups / seconds, rel_tol=1e-9)
        ):
            raise ValueError(f"resource timing/throughput evidence mismatch at {global_step}")
        memory = _require_numeric_mapping(resources.get("memory"), _MEMORY_KEYS, f"{location}.resources.memory")
        if (
            any(float(value) < 0 for value in memory.values())
            or float(memory["allocated_bytes"]) > float(memory["reserved_bytes"])
            or float(memory["allocated_bytes"]) > float(memory["peak_allocated_bytes"])
            or float(memory["reserved_bytes"]) > float(memory["peak_reserved_bytes"])
            or float(memory["peak_allocated_bytes"]) > float(memory["peak_reserved_bytes"])
        ):
            raise ValueError(f"GPU memory evidence is inconsistent at {global_step}")
        nvidia = _require_numeric_mapping(
            resources.get("nvidia_smi"), _NVIDIA_KEYS, f"{location}.resources.nvidia_smi"
        )
        if (
            not 0 <= float(nvidia["gpu_utilization_percent"]) <= 100
            or not 0 <= float(nvidia["gpu_memory_used_mib"]) <= float(nvidia["gpu_memory_total_mib"])
            or float(nvidia["gpu_memory_total_mib"]) <= 0
            or float(nvidia["gpu_temperature_c"]) <= 0
            or any(float(nvidia[name]) < 0 for name in ("gpu_power_w", "gpu_sm_clock_mhz", "gpu_memory_clock_mhz"))
        ):
            raise ValueError(f"nvidia-smi evidence is outside valid bounds at {global_step}")


def _validate_preliminary_wandb(
    value: object,
    *,
    config_sha256: str,
    execution_id: str,
    expected_run_name: str,
    expected_project: str,
    expected_entity: str | None,
    paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "mode",
        "status",
        "terminal_status",
        "entity",
        "project",
        "group",
        "run_name",
        "job_type",
        "run_id",
        "run_url",
        "artifact_name",
        "artifact_id",
        "artifact_version",
        "artifact_digest",
        "config_sha256",
        "files_sha256",
        "files",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("preliminary W&B receipt schema is incomplete or extended")
    if (
        value.get("schema_version") != PRELIMINARY_WANDB_SCHEMA
        or value.get("mode") != "online"
        or value.get("status") != "uploaded-preliminary"
        or value.get("terminal_status") != "pending-postflight"
        or value.get("config_sha256") != config_sha256
    ):
        raise ValueError("preliminary W&B receipt is not an online pending-postflight upload")
    for name in ("entity", "project", "group", "run_name", "job_type", "run_id"):
        if not isinstance(value.get(name), str) or not _SAFE_IDENTIFIER.fullmatch(str(value[name])):
            raise ValueError(f"preliminary W&B receipt lacks safe {name}")
    if value.get("group") != f"phase2g-smoke-{execution_id}":
        raise ValueError("preliminary W&B group does not bind the Phase 2g execution")
    if (
        value.get("run_name") != expected_run_name
        or value.get("project") != expected_project
        or value.get("job_type") != "phase2g-instrumentation-smoke"
    ):
        raise ValueError("preliminary W&B run/project/job type differs from the submitted identity")
    if expected_entity is not None and value.get("entity") != expected_entity:
        raise ValueError("preliminary W&B entity differs from the submitted identity")
    if not _online_run_url_matches(
        value.get("run_url"),
        entity=value.get("entity"),
        project=value.get("project"),
        run_id=value.get("run_id"),
    ):
        raise ValueError("preliminary W&B receipt has an invalid online URL")
    expected_artifact = f"phase2g-instrumentation-smoke-{execution_id}-{value['run_id']}"
    if value.get("artifact_name") != expected_artifact:
        raise ValueError("preliminary W&B artifact name does not bind the execution/run")
    for name in ("artifact_id", "artifact_version", "artifact_digest"):
        if not isinstance(value.get(name), str) or not str(value[name]).strip():
            raise ValueError(f"preliminary W&B receipt lacks {name}")
    files = value.get("files")
    expected_names = ["steps.jsonl", *(f"checkpoints/{arm}.pt" for arm in ARMS)]
    if (
        not isinstance(files, list)
        or len(files) != len(expected_names)
        or any(not isinstance(item, Mapping) for item in files)
        or [item.get("name") for item in files if isinstance(item, Mapping)] != expected_names
    ):
        raise ValueError("preliminary W&B file manifest is incomplete or out of order")
    for item, name in zip(files, expected_names, strict=True):
        if not isinstance(item, Mapping) or set(item) != {"name", "bytes", "sha256"}:
            raise ValueError(f"preliminary W&B file schema mismatch: {name}")
        if not _identity_matches(item, paths[name], published_name=name):
            raise ValueError(f"preliminary W&B file identity mismatch: {name}")
    if value.get("files_sha256") != sha256_value({"files": files}):
        raise ValueError("preliminary W&B file-manifest digest mismatch")
    _safe_finite_tree(value, "wandb_preliminary")
    return value


def _validate_training_artifacts(
    *,
    root: Path,
    repo_root: Path,
    job_id: str,
    commit: str,
    expected_run_name: str,
    expected_project: str,
    expected_entity: str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    training_path = root / "training_receipt.json"
    wandb_path = root / "wandb_receipt.json"
    steps_path = root / "steps.jsonl"
    preliminary_identities = {path: _stable_identity(path) for path in (training_path, wandb_path, steps_path)}
    training = _load_json(training_path)
    wandb = _load_json(wandb_path)
    rows = _load_jsonl(steps_path)
    _safe_finite_tree(training, "training_receipt")
    _safe_finite_tree(wandb, "wandb_receipt")
    _safe_finite_tree(rows, "steps")
    execution_id = training.get("execution_id")
    if set(training) != {
        "schema_version",
        "status",
        "terminal_status",
        "postflight_required",
        "execution_id",
        "created_utc",
        "evidence_level",
        "claim_boundary",
        "synthetic_inputs_only",
        "dataset_or_cache_access",
        "resource_policy",
        "config",
        "config_sha256",
        "runtime_identity",
        "total_optimizer_steps",
        "expected_optimizer_steps",
        "steps",
        "checkpoints",
        "arms",
        "elapsed_seconds",
        "finite",
        "wandb",
    }:
        raise ValueError("training receipt schema is incomplete or extended")
    if not isinstance(execution_id, str) or not _SAFE_IDENTIFIER.fullmatch(execution_id) or len(execution_id) > 95:
        raise ValueError("training receipt has an unsafe execution ID")
    if (
        training.get("schema_version") != SMOKE_SCHEMA
        or training.get("status") != "pending-postflight"
        or training.get("terminal_status") != "pending-postflight"
        or training.get("postflight_required") is not True
        or training.get("evidence_level") != "integration-smoke"
        or training.get("claim_boundary") != CLAIM_BOUNDARY
        or training.get("synthetic_inputs_only") is not True
        or training.get("dataset_or_cache_access") is not False
        or training.get("resource_policy") != "diagnostic-only"
        or training.get("finite") is not True
        or training.get("total_optimizer_steps") != TOTAL_STEPS
        or training.get("expected_optimizer_steps") != TOTAL_STEPS
    ):
        raise ValueError("training receipt is not a complete pending-postflight v2 smoke")
    runtime = training.get("runtime_identity")
    _validate_runtime_identity(runtime, repo_root=repo_root, commit=commit, job_id=job_id)
    config = _validate_config(
        training.get("config"),
        execution_id=execution_id,
        runtime=runtime,
        expected_run_name=expected_run_name,
        expected_project=expected_project,
        expected_entity=expected_entity,
    )
    config_sha = sha256_value(config)
    if training.get("config_sha256") != config_sha:
        raise ValueError("training configuration digest mismatch")
    steps_identity = training.get("steps")
    if (
        not isinstance(steps_identity, Mapping)
        or set(steps_identity) != {"name", "bytes", "sha256"}
        or not _identity_matches(steps_identity, steps_path)
    ):
        raise ValueError("steps.jsonl identity differs from the training receipt")
    if (
        not isinstance(training.get("created_utc"), str)
        or isinstance(training.get("elapsed_seconds"), bool)
        or not isinstance(training.get("elapsed_seconds"), int | float)
        or not math.isfinite(float(training["elapsed_seconds"]))
        or float(training["elapsed_seconds"]) <= 0
    ):
        raise ValueError("training receipt lacks valid creation/elapsed-time metadata")

    checkpoints = training.get("checkpoints")
    arm_summaries = training.get("arms")
    arm_configs = config["arm_configs"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(ARMS):
        raise ValueError("training receipt checkpoint set is incomplete")
    if not isinstance(arm_summaries, Mapping) or set(arm_summaries) != set(ARMS):
        raise ValueError("training receipt arm summary set is incomplete")
    parameter_counts: dict[str, Mapping[str, int]] = {}
    for arm in ARMS:
        counts = _validate_checkpoint(root / "checkpoints" / f"{arm}.pt", arm, checkpoints[arm], arm_configs[arm])
        parameter_counts[arm] = counts
    _validate_step_rows(rows, config=config, parameter_counts=parameter_counts)
    for arm_index, arm in enumerate(ARMS):
        summary = arm_summaries[arm]
        arm_rows = rows[arm_index * STEPS_PER_ARM : (arm_index + 1) * STEPS_PER_ARM]
        if not isinstance(summary, Mapping) or set(summary) != {
            "optimizer_steps",
            "final_total_objective",
            "final_parameter_update_norm",
            "maximum_forbidden_gradient_norm",
            "exact_reload",
        }:
            raise ValueError(f"arm summary schema mismatch: {arm}")
        if (
            summary.get("optimizer_steps") != STEPS_PER_ARM
            or summary.get("exact_reload") is not True
            or not math.isclose(
                float(summary.get("final_total_objective", math.nan)),
                float(arm_rows[-1]["objectives"]["total"]),
                rel_tol=1e-12,
            )
            or not math.isclose(
                float(summary.get("final_parameter_update_norm", math.nan)),
                float(arm_rows[-1]["optimizer"]["parameter_update_norm"]),
                rel_tol=1e-12,
            )
            or float(summary.get("maximum_forbidden_gradient_norm", math.nan))
            != max(float(row["gradients"]["firewall_max_forbidden_norm"]) for row in arm_rows)
        ):
            raise ValueError(f"arm summary differs from optimizer-step evidence: {arm}")

    paths = {
        "steps.jsonl": steps_path,
        **{f"checkpoints/{arm}.pt": root / "checkpoints" / f"{arm}.pt" for arm in ARMS},
    }
    preliminary = _validate_preliminary_wandb(
        wandb,
        config_sha256=config_sha,
        execution_id=execution_id,
        expected_run_name=expected_run_name,
        expected_project=expected_project,
        expected_entity=expected_entity,
        paths=paths,
    )
    if training.get("wandb") != preliminary:
        raise ValueError("training receipt does not embed the exact preliminary W&B receipt")
    if any(_stable_identity(path) != identity for path, identity in preliminary_identities.items()):
        raise RuntimeError("Phase 2g receipt or step evidence changed during validation")
    return training, wandb, rows


def _inspect_terminal_upload(path: Path, *, root: Path) -> _StableIdentity:
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("terminal W&B evidence escapes the artifact root")
    if path.suffix == ".json":
        _safe_finite_tree(_load_json(path), path.name)
    elif path.suffix == ".jsonl":
        _safe_finite_tree(_load_jsonl(path), path.name)
    elif path.suffix != ".pt":
        raise ValueError(f"unsupported terminal W&B evidence type: {path.name}")
    return _stable_identity(path)


def _terminal_uploads(
    *,
    root: Path,
    training_path: Path,
    postflight_path: Path,
) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("training_receipt.json", training_path, "training-receipt"),
        ("wandb_receipt.json", root / "wandb_receipt.json", "preliminary-wandb-receipt"),
        (postflight_path.name, postflight_path, "postflight-receipt"),
        ("steps.jsonl", root / "steps.jsonl", "optimizer-step-log"),
        *((f"checkpoints/{arm}.pt", root / "checkpoints" / f"{arm}.pt", "checkpoint") for arm in ARMS),
    )


def finalize_phase2g_online_run(
    *,
    preliminary_receipt: Mapping[str, Any],
    artifact_root: Path,
    training_path: Path,
    postflight_path: Path,
    summary: Mapping[str, Any],
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Resume the exact preliminary run and publish validated terminal evidence."""

    root = artifact_root.resolve(strict=True)
    uploads = _terminal_uploads(root=root, training_path=training_path, postflight_path=postflight_path)
    upload_identities = {name: _inspect_terminal_upload(path, root=root) for name, path, _ in uploads}
    _safe_finite_tree(summary, "terminal_summary")
    module = wandb_module
    if module is None:
        if os.environ.get("WANDB_MODE") != "online":
            raise RuntimeError("Phase 2g terminal publication requires WANDB_MODE=online")
        import wandb

        module = wandb

    run = module.init(
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
    if run is None or bool(getattr(run, "offline", True)):
        raise RuntimeError("Phase 2g terminal finalizer did not resume online")
    expected_run = {
        "entity": preliminary_receipt["entity"],
        "project": preliminary_receipt["project"],
        "group": preliminary_receipt["group"],
        "run_name": preliminary_receipt["run_name"],
        "job_type": preliminary_receipt["job_type"],
        "run_id": preliminary_receipt["run_id"],
    }
    actual_run = {
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "group": getattr(run, "group", None),
        "run_name": getattr(run, "name", None),
        "job_type": getattr(run, "job_type", None),
        "run_id": getattr(run, "id", None),
    }
    if actual_run != expected_run:
        run.finish(exit_code=1)
        raise RuntimeError("Phase 2g terminal finalizer resumed a different W&B identity")
    artifact_name = f"phase2g-terminal-{preliminary_receipt['run_id']}"
    if not _SAFE_IDENTIFIER.fullmatch(artifact_name):
        run.finish(exit_code=1)
        raise ValueError("terminal W&B artifact name is unsafe")
    try:
        artifact = module.Artifact(artifact_name, type="phase2g-governed-terminal")
        for name, path, _ in uploads:
            artifact.add_file(str(path), name=name)
        logged = run.log_artifact(artifact)
        logged.wait()
        if any(_inspect_terminal_upload(path, root=root) != upload_identities[name] for name, path, _ in uploads):
            raise RuntimeError("Phase 2g terminal evidence changed during W&B upload")
        backend = {
            "run_url": getattr(run, "url", None),
            "artifact_name": getattr(logged, "name", None),
            "artifact_id": getattr(logged, "id", None),
            "artifact_version": getattr(logged, "version", None),
            "artifact_digest": getattr(logged, "digest", None),
        }
        if any(not isinstance(value, str) or not value.strip() for value in backend.values()):
            raise RuntimeError("W&B terminal finalizer returned incomplete backend identities")
        if backend["run_url"] != preliminary_receipt["run_url"] or not _online_run_url_matches(
            backend["run_url"],
            entity=expected_run["entity"],
            project=expected_run["project"],
            run_id=expected_run["run_id"],
        ):
            raise RuntimeError("W&B terminal finalizer returned a mismatched run URL")
        if not _backend_artifact_name_matches(
            backend["artifact_name"],
            collection=artifact_name,
            version=backend["artifact_version"],
        ):
            raise RuntimeError("W&B terminal finalizer returned a mismatched artifact name")
        receipt = {
            "schema_version": FINAL_WANDB_SCHEMA,
            "status": "finalized",
            "terminal_status": "postflight-pass",
            "mode": "online",
            "preliminary_receipt_sha256": sha256_value(preliminary_receipt),
            **expected_run,
            "run_url": backend["run_url"],
            "artifact_name": artifact_name,
            "artifact_id": backend["artifact_id"],
            "artifact_version": backend["artifact_version"],
            "artifact_digest": backend["artifact_digest"],
            "summary_sha256": sha256_value(summary),
            "files": [
                {
                    "name": name,
                    "role": role,
                    "bytes": upload_identities[name].bytes,
                    "sha256": upload_identities[name].sha256,
                }
                for name, _, role in uploads
            ],
        }
        _safe_finite_tree(receipt, "wandb_final")
        run.summary.update(dict(summary))
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)
    return receipt


def _validate_final_wandb(
    value: Mapping[str, Any],
    *,
    preliminary: Mapping[str, Any],
    training_path: Path,
    postflight_path: Path,
    summary: Mapping[str, Any],
) -> None:
    if set(value) != {
        "schema_version",
        "status",
        "terminal_status",
        "mode",
        "preliminary_receipt_sha256",
        "entity",
        "project",
        "group",
        "run_name",
        "job_type",
        "run_id",
        "run_url",
        "artifact_name",
        "artifact_id",
        "artifact_version",
        "artifact_digest",
        "summary_sha256",
        "files",
    }:
        raise ValueError("final W&B receipt schema is incomplete or extended")
    if (
        value.get("schema_version") != FINAL_WANDB_SCHEMA
        or value.get("status") != "finalized"
        or value.get("terminal_status") != "postflight-pass"
        or value.get("mode") != "online"
        or value.get("preliminary_receipt_sha256") != sha256_value(preliminary)
        or value.get("summary_sha256") != sha256_value(summary)
    ):
        raise ValueError("final W&B receipt does not bind the postflight pass")
    for name in ("entity", "project", "group", "run_name", "job_type", "run_id"):
        if value.get(name) != preliminary.get(name):
            raise ValueError(f"final W&B receipt changed the preliminary {name}")
    if value.get("run_url") != preliminary.get("run_url"):
        raise ValueError("final W&B receipt changed the preliminary run_url")
    root = training_path.parent
    expected = {
        name: (path, role)
        for name, path, role in _terminal_uploads(
            root=root,
            training_path=training_path,
            postflight_path=postflight_path,
        )
    }
    files = value.get("files")
    if (
        not isinstance(files, list)
        or len(files) != len(expected)
        or any(not isinstance(item, Mapping) for item in files)
        or {str(item.get("name")): item for item in files if isinstance(item, Mapping)}.keys() != expected.keys()
    ):
        raise ValueError("final W&B receipt has an invalid terminal file set")
    observed = {str(item["name"]): item for item in files if isinstance(item, Mapping)}
    for name, (path, role) in expected.items():
        if (
            set(observed[name]) != {"name", "role", "bytes", "sha256"}
            or observed[name].get("role") != role
            or not _identity_matches(observed[name], path, published_name=name)
        ):
            raise ValueError(f"final W&B artifact identity mismatch: {name}")
    for name in ("run_url", "artifact_id", "artifact_version", "artifact_digest"):
        if not isinstance(value.get(name), str) or not str(value[name]).strip():
            raise ValueError(f"final W&B receipt lacks {name}")
    if value.get("artifact_name") != f"phase2g-terminal-{preliminary['run_id']}":
        raise ValueError("final W&B artifact name does not bind the resumed run")
    if not _online_run_url_matches(
        value.get("run_url"),
        entity=value.get("entity"),
        project=value.get("project"),
        run_id=value.get("run_id"),
    ):
        raise ValueError("final W&B receipt has an invalid online URL")
    _safe_finite_tree(value, "wandb_final")


def _write_success(root: Path, terminal_sha256: str) -> None:
    target = root / "SUCCESS"
    expected = f"terminal_sha256={terminal_sha256}\n".encode()
    if target.exists():
        if target.is_symlink() or target.read_bytes() != expected:
            raise ValueError("SUCCESS marker does not bind the terminal receipt")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{root.name}.success-", dir=root.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != expected:
                raise ValueError("conflicting SUCCESS marker appeared during postflight") from None
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _base_files(root: Path) -> set[Path]:
    return {
        root / "training_receipt.json",
        root / "wandb_receipt.json",
        root / "steps.jsonl",
        *(root / "checkpoints" / f"{arm}.pt" for arm in ARMS),
    }


def _assert_postflight_inputs_unchanged(root: Path, postflight: Mapping[str, Any]) -> None:
    checkpoint_hashes = {arm: sha256_file(root / "checkpoints" / f"{arm}.pt") for arm in ARMS}
    if (
        postflight.get("training_receipt_sha256") != sha256_file(root / "training_receipt.json")
        or postflight.get("preliminary_wandb_receipt_sha256") != sha256_file(root / "wandb_receipt.json")
        or postflight.get("steps_sha256") != sha256_file(root / "steps.jsonl")
        or postflight.get("checkpoint_sha256") != checkpoint_hashes
    ):
        raise RuntimeError("validated Phase 2g evidence changed during terminal publication")


def _validate_phase2g_locked(
    *,
    output: Path,
    repo_root: Path,
    job_id: str,
    expected_run_name: str,
    expected_wandb_project: str,
    expected_wandb_entity: str | None,
    scheduler_lookup: SchedulerLookup,
    git_lookup: GitLookup,
    wandb_finalizer: WandbFinalizer,
) -> ContentAddress:
    root = output
    scheduler = scheduler_lookup(job_id)
    _validate_scheduler(scheduler, job_id)
    commit, clean = git_lookup(repo_root)
    if not clean or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("postflight requires a clean committed Git worktree")

    postflight_files = _state_files(root, "postflight", "postflight")
    final_files = _state_files(root, "wandb-final", "wandb-final")
    terminal_files = _state_files(root, "terminal", "terminal")
    if len(postflight_files) > 1 or len(final_files) > 1 or len(terminal_files) > 1:
        raise ValueError("multiple immutable Phase 2g state receipts found")
    if terminal_files and not final_files:
        raise ValueError("terminal receipt exists without finalized W&B evidence")
    success_path = root / "SUCCESS"
    preliminary_expected = _base_files(root) | set(postflight_files) | set(final_files) | set(terminal_files)
    if success_path.exists() or success_path.is_symlink():
        preliminary_expected.add(success_path)
    _scan_exact_artifacts(root, preliminary_expected)
    if success_path.exists() and not terminal_files:
        raise ValueError("SUCCESS exists before terminal validation")

    training, preliminary_wandb, rows = _validate_training_artifacts(
        root=root,
        repo_root=repo_root,
        job_id=job_id,
        commit=commit,
        expected_run_name=expected_run_name,
        expected_project=expected_wandb_project,
        expected_entity=expected_wandb_entity,
    )
    training_path = root / "training_receipt.json"
    wandb_path = root / "wandb_receipt.json"
    steps_path = root / "steps.jsonl"
    checkpoint_hashes = {arm: sha256_file(root / "checkpoints" / f"{arm}.pt") for arm in ARMS}
    postflight_payload = {
        "schema_version": POSTFLIGHT_SCHEMA,
        "status": "pass",
        "execution_id": training["execution_id"],
        "git_commit": commit,
        "training_receipt_sha256": sha256_file(training_path),
        "preliminary_wandb_receipt_sha256": sha256_file(wandb_path),
        "config_sha256": training["config_sha256"],
        "steps_sha256": sha256_file(steps_path),
        "checkpoint_sha256": checkpoint_hashes,
        "total_optimizer_steps": len(rows),
        "arms": list(ARMS),
        "gradient_firewall_passed": True,
        "synthetic_inputs_only": True,
        "dataset_or_cache_access": False,
        "scheduler": asdict(scheduler),
    }
    if postflight_files:
        postflight_path, existing_postflight, postflight_sha = _single_content_address(
            root, "postflight", "postflight"
        )
        if existing_postflight != postflight_payload:
            raise ValueError("existing postflight receipt differs from recomputed evidence")
        postflight = ContentAddress(postflight_path, postflight_sha, postflight_path.stat().st_size)
    else:
        postflight = write_content_addressed_json(postflight_payload, root / "postflight", prefix="postflight")

    terminal_summary = {
        "status": "success",
        "terminal_status": "postflight-pass",
        "validation/postflight/status": "pass",
        "validation/training_receipt_sha256": sha256_file(training_path),
        "validation/postflight_receipt_sha256": postflight.sha256,
        "validation/total_optimizer_steps": TOTAL_STEPS,
        "validation/gradient_firewall_passed": True,
        "validation/synthetic_inputs_only": True,
    }
    if final_files:
        final_path, final_wandb, final_sha = _single_content_address(root, "wandb-final", "wandb-final")
        _validate_final_wandb(
            final_wandb,
            preliminary=preliminary_wandb,
            training_path=training_path,
            postflight_path=postflight.path,
            summary=terminal_summary,
        )
    else:
        final_wandb = wandb_finalizer(
            preliminary_receipt=preliminary_wandb,
            artifact_root=root,
            training_path=training_path,
            postflight_path=postflight.path,
            summary=terminal_summary,
        )
        _validate_final_wandb(
            final_wandb,
            preliminary=preliminary_wandb,
            training_path=training_path,
            postflight_path=postflight.path,
            summary=terminal_summary,
        )
        final_address = write_content_addressed_json(final_wandb, root / "wandb-final", prefix="wandb-final")
        final_path, final_sha = final_address.path, final_address.sha256

    _assert_postflight_inputs_unchanged(root, postflight_payload)
    final_scheduler = scheduler_lookup(job_id)
    _validate_scheduler(final_scheduler, job_id)
    if final_scheduler != scheduler:
        raise RuntimeError("Slurm allocation identity changed during terminal publication")
    final_commit, final_clean = git_lookup(repo_root)
    if final_commit != commit or not final_clean:
        raise RuntimeError("Git worktree changed during terminal publication")
    _validate_runtime_identity(training["runtime_identity"], repo_root=repo_root, commit=commit, job_id=job_id)

    if terminal_files:
        terminal_path, terminal, terminal_sha = _single_content_address(root, "terminal", "terminal")
    else:
        terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "pass",
            "execution_id": training["execution_id"],
            "git_commit": commit,
            "training_receipt_sha256": sha256_file(training_path),
            "preliminary_wandb_receipt_sha256": sha256_file(wandb_path),
            "postflight_receipt_sha256": postflight.sha256,
            "final_wandb_receipt_sha256": final_sha,
            "wandb_run_id": final_wandb["run_id"],
            "wandb_terminal_artifact_id": final_wandb["artifact_id"],
            "wandb_terminal_artifact_digest": final_wandb["artifact_digest"],
        }
        terminal_address = write_content_addressed_json(terminal, root / "terminal", prefix="terminal")
        terminal_path, terminal_sha = terminal_address.path, terminal_address.sha256
    if set(terminal) != {
        "schema_version",
        "status",
        "execution_id",
        "git_commit",
        "training_receipt_sha256",
        "preliminary_wandb_receipt_sha256",
        "postflight_receipt_sha256",
        "final_wandb_receipt_sha256",
        "wandb_run_id",
        "wandb_terminal_artifact_id",
        "wandb_terminal_artifact_digest",
    } or (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("status") != "pass"
        or terminal.get("execution_id") != training["execution_id"]
        or terminal.get("git_commit") != commit
        or terminal.get("training_receipt_sha256") != sha256_file(training_path)
        or terminal.get("preliminary_wandb_receipt_sha256") != sha256_file(wandb_path)
        or terminal.get("postflight_receipt_sha256") != postflight.sha256
        or terminal.get("final_wandb_receipt_sha256") != final_sha
        or terminal.get("wandb_run_id") != final_wandb.get("run_id")
        or terminal.get("wandb_terminal_artifact_id") != final_wandb.get("artifact_id")
        or terminal.get("wandb_terminal_artifact_digest") != final_wandb.get("artifact_digest")
    ):
        raise ValueError("terminal receipt does not bind the complete Phase 2g postflight")

    without_success = _base_files(root) | {postflight.path, final_path, terminal_path}
    _assert_postflight_inputs_unchanged(root, postflight_payload)
    _scan_exact_artifacts(root, without_success | ({success_path} if success_path.exists() else set()))
    _write_success(root, terminal_sha)
    _scan_exact_artifacts(root, without_success | {success_path})
    return ContentAddress(terminal_path, terminal_sha, terminal_path.stat().st_size)


def validate_phase2g_training_smoke(
    *,
    output: str | Path,
    repo_root: str | Path,
    job_id: str,
    expected_run_name: str,
    expected_wandb_project: str,
    expected_wandb_entity: str | None = None,
    scheduler_lookup: SchedulerLookup = scheduler_identity,
    git_lookup: GitLookup = git_identity,
    wandb_finalizer: WandbFinalizer = finalize_phase2g_online_run,
) -> ContentAddress:
    """Serialize postflight publication and verify the final exact artifact set."""

    source = Path(output)
    if source.is_symlink():
        raise ValueError("Phase 2g output root may not be a symbolic link")
    root = source.resolve(strict=True)
    repository = Path(repo_root).resolve(strict=True)
    if not root.is_dir() or not repository.is_dir():
        raise ValueError("output and repo_root must be directories")
    for name, value in (("expected_run_name", expected_run_name), ("expected_wandb_project", expected_wandb_project)):
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError(f"{name} must be a path-safe identifier")
    if expected_wandb_entity is not None and not _SAFE_IDENTIFIER.fullmatch(expected_wandb_entity):
        raise ValueError("expected_wandb_entity must be a path-safe identifier")
    lock_path = root.parent / f".{root.name}.postflight.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _validate_phase2g_locked(
            output=root,
            repo_root=repository,
            job_id=job_id,
            expected_run_name=expected_run_name,
            expected_wandb_project=expected_wandb_project,
            expected_wandb_entity=expected_wandb_entity,
            scheduler_lookup=scheduler_lookup,
            git_lookup=git_lookup,
            wandb_finalizer=wandb_finalizer,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--allocation-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.allocation_only:
        identity = scheduler_identity(args.job_id)
        _validate_scheduler(identity, args.job_id)
        print(json.dumps({"status": "pass", "scheduler": asdict(identity)}, sort_keys=True))
        return
    if args.output is None or args.repo_root is None or args.run_name is None or args.wandb_project is None:
        raise SystemExit(
            "--output, --repo-root, --run-name, and --wandb-project are required unless --allocation-only is used"
        )
    receipt = validate_phase2g_training_smoke(
        output=args.output,
        repo_root=args.repo_root,
        job_id=args.job_id,
        expected_run_name=args.run_name,
        expected_wandb_project=args.wandb_project,
        expected_wandb_entity=args.wandb_entity,
    )
    print(json.dumps({"status": "pass", "terminal_sha256": receipt.sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
