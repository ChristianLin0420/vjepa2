#!/usr/bin/env python3
"""Run a bounded synthetic Phase 2g training-and-observability smoke.

This runner is intentionally incapable of reading a dataset or feature cache. It
exercises the M0--M3 optimizer, gradient-firewall, checkpoint, and online W&B
boundaries with deterministic generated tensors. Its outputs are integration
evidence only and must never be interpreted as architecture-quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from jepa4d.models.phase2f_scale_geometry import DEFAULT_PHASE2F_ARMS, Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_losses import Phase2fLossConfig
from jepa4d.training.phase2f_training import (
    assert_strict_phase2f_reload,
    load_phase2f_checkpoint,
    phase2f_arm_configs,
    save_phase2f_checkpoint,
    train_phase2f_step,
)

ARMS = DEFAULT_PHASE2F_ARMS
SMOKE_SCHEMA = "jepa4d-phase2g-training-instrumentation-smoke-v2"
STEP_SCHEMA = "jepa4d-phase2g-training-instrumentation-step-v2"
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2g-training-instrumentation-wandb-v2"
WANDB_JOB_TYPE = "phase2g-instrumentation-smoke"
CLIP_GRAD_NORM_EPSILON = 1e-6
CLAIM_BOUNDARY = (
    "Synthetic integration smoke only: no dataset, cache, held-out target, or scientific-quality evidence was used or "
    "produced. Metrics validate training and logging instrumentation, not model quality or Phase 2g promotion."
)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CREDENTIAL_SHAPE = re.compile(r"wandb_v1_[A-Za-z0-9_-]+|(?:^|[._-])hf_[A-Za-z0-9]{16,}", re.IGNORECASE)
_OBJECTIVE_METRICS = frozenset(
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
_DIAGNOSTIC_METRICS = frozenset(
    {
        "joint_nll_diagnostic_only",
        "optimal_log_scale_mean",
        "scale_field_zero_mean_error",
        "scale_field_max_abs",
    }
)
_ADAMW_BETAS = (0.9, 0.999)
_ADAMW_EPS = 1e-8
_NVIDIA_SMI_FIELDS = (
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
)


@dataclass(frozen=True, slots=True)
class SmokeSettings:
    """Configuration for the synthetic-only instrumentation boundary."""

    output: Path
    execution_id: str
    git_commit: str = "0" * 40
    scheduler_job_id: str = "unit-test"
    max_steps: int = 3
    seed: int = 260630
    device: str = "cpu"
    input_dim: int = 768
    spatial_size: int = 24
    source_groups: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    wandb_project: str = "jepa4d-worldmodel"
    wandb_entity: str | None = None
    wandb_group: str = "phase2g-training-instrumentation-smoke"
    wandb_run_name: str = "phase2g-training-instrumentation-smoke"

    def __post_init__(self) -> None:
        _validate_identifier("execution_id", self.execution_id)
        if len(self.execution_id) > 95:
            raise ValueError("execution_id must contain at most 95 characters")
        if not _GIT_COMMIT.fullmatch(self.git_commit):
            raise ValueError("git_commit must be a lowercase 40-character Git object ID")
        _validate_identifier("scheduler_job_id", self.scheduler_job_id)
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or not 1 <= self.max_steps <= 10:
            raise ValueError("max_steps must be an integer in [1, 10]")
        for name in ("seed", "input_dim", "spatial_size", "source_groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_dim < 2:
            raise ValueError("input_dim must be at least 2 for the synthetic target construction")
        if self.spatial_size < 2:
            raise ValueError("spatial_size must be at least 2 for the spatial-gradient losses")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        for name in ("wandb_project", "wandb_group", "wandb_run_name"):
            _validate_identifier(name, getattr(self, name))
        if self.wandb_entity is not None:
            _validate_identifier("wandb_entity", self.wandb_entity)


def _validate_identifier(name: str, value: str) -> None:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty path-safe identifier")
    if _CREDENTIAL_SHAPE.search(value) or any(word in value.lower() for word in ("api-key", "api_key", "secret")):
        raise ValueError(f"{name} resembles credential material")


def _bounded_max_steps(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max steps must be an integer") from error
    if not 1 <= parsed <= 10:
        raise argparse.ArgumentTypeError("max steps must be in [1, 10]")
    return parsed


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if not isinstance(decoded, dict):
        raise TypeError("expected a JSON object")
    return decoded


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_identity_and_state(
    path: Path,
    *,
    published_name: str | None = None,
) -> tuple[dict[str, Any], tuple[int, int, int, int, int]]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"smoke artifact must be a regular non-symlink file: {path.name}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    final = resolved.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields) or any(
        getattr(after, field) != getattr(final, field) for field in fields
    ):
        raise RuntimeError(f"smoke artifact changed while hashing: {path.name}")
    name = resolved.name if published_name is None else published_name
    if not name or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"unsafe published smoke artifact name: {name!r}")
    identity = {"name": name, "bytes": after.st_size, "sha256": digest.hexdigest()}
    state = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    return identity, state


def _file_identity(path: Path, *, published_name: str | None = None) -> dict[str, Any]:
    identity, _ = _file_identity_and_state(path, published_name=published_name)
    return identity


def _runtime_identity(settings: SmokeSettings, device: torch.device) -> dict[str, Any]:
    model_module = Path(sys.modules[Phase2fScaleGeometryProbe.__module__].__file__ or "").resolve(strict=True)
    training_module = Path(sys.modules[train_phase2f_step.__module__].__file__ or "").resolve(strict=True)
    hardware: dict[str, Any] = {"device_type": device.type}
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        raw_uuid = getattr(properties, "uuid", None)
        hardware.update(
            {
                "device_name": properties.name,
                "device_uuid_sha256": None if raw_uuid is None else _sha256_text(str(raw_uuid)),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return {
        "git_commit": settings.git_commit,
        "scheduler_job_id": settings.scheduler_job_id,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "hardware": hardware,
        "code": {
            "runner": _file_identity(Path(__file__)),
            "model_module": _file_identity(model_module),
            "training_module": _file_identity(training_module),
        },
    }


def _determinism_settings(device: torch.device) -> dict[str, Any]:
    return {
        "synthetic_cpu_generator_seeded_per_arm_step": True,
        "torch_manual_seeded_per_arm": True,
        "cuda_manual_seeded_per_arm": device.type == "cuda",
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "bitwise_reproducibility_claimed": False,
    }


def _optimizer_settings(settings: SmokeSettings) -> dict[str, Any]:
    return {
        "name": "torch.optim.AdamW",
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "betas": list(_ADAMW_BETAS),
        "eps": _ADAMW_EPS,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }


def _require_safe_finite_tree(value: Any, location: str = "root") -> None:
    """Reject tensors, non-finite numbers, and credential-shaped serialized values."""

    if isinstance(value, torch.Tensor):
        raise TypeError(f"raw tensors are forbidden in persisted smoke metadata: {location}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if any(token in key_text.lower() for token in ("api_key", "password", "credential", "private_key")):
                raise ValueError(f"credential-like metadata key is forbidden: {location}.{key_text}")
            _require_safe_finite_tree(child, f"{location}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_safe_finite_tree(child, f"{location}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")
    if isinstance(value, str) and _CREDENTIAL_SHAPE.search(value):
        raise ValueError(f"credential-shaped text is forbidden at {location}")


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    _require_safe_finite_tree(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    _require_safe_finite_tree(value)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_batch(
    settings: SmokeSettings,
    *,
    arm_index: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic paired views without touching any external data path."""

    generator = torch.Generator(device="cpu").manual_seed(settings.seed + arm_index * 10_003 + step * 101)
    groups = settings.source_groups
    size = settings.spatial_size
    base = torch.randn(groups, settings.input_dim, size, size, generator=generator)
    view_delta = torch.linspace(-0.025, 0.025, groups).view(groups, 1, 1, 1)
    features = torch.stack((base, base * 0.985 + view_delta), dim=1).flatten(0, 1)

    coordinate = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    source_scale = torch.linspace(0.7, 1.3, groups).view(groups, 1, 1)
    log_depth = 0.16 * base[:, 0] + 0.08 * base[:, 1] + 0.12 * xx + 0.07 * yy + source_scale.log()
    base_target = log_depth.clamp(-1.5, 1.5).exp()
    target_depth = torch.stack((base_target, base_target * 1.01), dim=1).flatten(0, 1)
    valid = torch.ones_like(target_depth, dtype=torch.bool)

    cameras: list[torch.Tensor] = []
    for group in range(groups):
        for view in range(2):
            focal = 300.0 + 17.0 * group + 5.0 * view
            cameras.append(
                torch.tensor(
                    [[focal, 0.0, 190.0 + view], [0.0, focal + 9.0, 188.0 - view], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )
            )
    intrinsics = torch.stack(cameras)
    return features, target_depth, valid, intrinsics


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _parameter_update_norm(model: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for name, parameter in model.named_parameters():
        delta = parameter.detach().cpu().double() - before[name].detach().cpu().double()
        squared += delta.square().sum()
    return float(squared.sqrt())


def _parameter_gradient_norm(model: torch.nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().cpu().double().square().sum()
    value = float(squared.sqrt())
    if not math.isfinite(value):
        raise RuntimeError("training smoke produced a non-finite post-clip gradient norm")
    return value


def _split_step_metrics(metrics: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    scalar_metrics = {name: float(value) for name, value in metrics.items() if not name.startswith("gradient_")}
    unknown = set(scalar_metrics) - _OBJECTIVE_METRICS - _DIAGNOSTIC_METRICS
    if unknown:
        raise RuntimeError(f"unclassified Phase 2g smoke metrics: {sorted(unknown)}")
    objectives = {name: value for name, value in scalar_metrics.items() if name in _OBJECTIVE_METRICS}
    diagnostics = {name: value for name, value in scalar_metrics.items() if name in _DIAGNOSTIC_METRICS}
    if "total" not in objectives:
        raise RuntimeError("Phase 2g smoke step lacks its total optimized objective")
    field_name = "field_objective" if "field_objective" in objectives else "scale_field_objective"
    expected_total = (
        objectives.get("shape_objective", 0.0)
        + objectives.get("scale_objective", 0.0)
        + objectives.get(field_name, 0.0)
    )
    if not math.isclose(objectives["total"], expected_total, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("Phase 2g optimized objective does not match its shape/scale/field decomposition")
    return objectives, diagnostics


def _memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0.0,
            "reserved_bytes": 0.0,
            "peak_allocated_bytes": 0.0,
            "peak_reserved_bytes": 0.0,
        }
    return {
        "allocated_bytes": float(torch.cuda.memory_allocated(device)),
        "reserved_bytes": float(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
    }


def nvidia_smi_telemetry(device: torch.device) -> dict[str, float]:
    """Read one finite scalar sample for required CUDA hardware diagnostics."""

    if device.type != "cuda":
        return {}
    index = 0 if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    selector = None if uuid is None else str(uuid)
    if selector and not selector.startswith(("GPU-", "MIG-")):
        selector = f"GPU-{selector}"
    if not selector:
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if index >= len(visible):
            raise RuntimeError("unable to map the logical CUDA device to an allocated nvidia-smi identity")
        selector = visible[index]
    query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("nvidia-smi telemetry query failed for the allocated CUDA device") from None
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi telemetry query failed for the allocated CUDA device")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"nvidia-smi returned {len(rows)} telemetry rows for the allocated CUDA device")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != len(_NVIDIA_SMI_FIELDS):
        raise RuntimeError("nvidia-smi returned an unexpected telemetry schema")
    try:
        telemetry = {name: float(value) for name, value in zip(_NVIDIA_SMI_FIELDS, fields, strict=True)}
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned a non-numeric telemetry value") from error
    _require_safe_finite_tree(telemetry, "nvidia_smi")
    return telemetry


def _wandb_payload(row: Mapping[str, Any]) -> dict[str, float | int | str]:
    arm = str(row["arm"])
    prefix = f"arms/{arm}"
    payload: dict[str, float | int | str] = {
        "global_step": int(row["global_step"]),
        "arm_step": int(row["arm_step"]),
        "arm": arm,
        f"{prefix}/optimizer/learning_rate": float(row["optimizer"]["learning_rate"]),
        f"{prefix}/optimizer/gradient_clip_threshold": float(row["optimizer"]["gradient_clip_threshold"]),
        f"{prefix}/optimizer/pre_clip_gradient_norm": float(row["optimizer"]["pre_clip_gradient_norm"]),
        f"{prefix}/optimizer/post_clip_gradient_norm": float(row["optimizer"]["post_clip_gradient_norm"]),
        f"{prefix}/optimizer/applied_clip_coefficient": float(row["optimizer"]["applied_clip_coefficient"]),
        f"{prefix}/optimizer/was_clipped": int(row["optimizer"]["was_clipped"]),
        f"{prefix}/optimizer/parameter_update_norm": float(row["optimizer"]["parameter_update_norm"]),
        f"{prefix}/resource_diagnostic/timing/step_seconds": float(row["resources"]["timing"]["step_seconds"]),
        f"{prefix}/resource_diagnostic/throughput/samples_per_second": float(
            row["resources"]["timing"]["samples_per_second"]
        ),
        f"{prefix}/resource_diagnostic/throughput/source_groups_per_second": float(
            row["resources"]["timing"]["source_groups_per_second"]
        ),
    }
    payload.update({f"{prefix}/objective/{name}": float(value) for name, value in row["objectives"].items()})
    payload.update({f"{prefix}/diagnostic/{name}": float(value) for name, value in row["diagnostics"].items()})
    payload.update({f"{prefix}/gradients/{name}": float(value) for name, value in row["gradients"].items()})
    payload.update(
        {
            f"{prefix}/resource_diagnostic/memory/{name}": float(value)
            for name, value in row["resources"]["memory"].items()
        }
    )
    payload.update(
        {
            f"{prefix}/resource_diagnostic/nvidia_smi/{name}": float(value)
            for name, value in row["resources"]["nvidia_smi"].items()
        }
    )
    payload.update(
        {
            f"{prefix}/architecture/parameter_count/{name}": int(value)
            for name, value in row["parameter_counts"].items()
        }
    )
    return payload


def _required_backend_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.casefold() in {"none", "null"}:
        raise RuntimeError(f"W&B did not return a valid {label}")
    _require_safe_finite_tree(value, f"wandb_backend.{label}")
    return value


def _backend_artifact_name_matches(name: str, *, collection: str, version: str) -> bool:
    return name in {collection, f"{collection}:{version}"}


def _validated_run_identity(run: Any, settings: SmokeSettings) -> dict[str, str]:
    identity = {
        "entity": _required_backend_string(getattr(run, "entity", None), "entity"),
        "project": _required_backend_string(getattr(run, "project", None), "project"),
        "group": _required_backend_string(getattr(run, "group", None), "group"),
        "run_name": _required_backend_string(getattr(run, "name", None), "run_name"),
        "job_type": _required_backend_string(getattr(run, "job_type", None), "job_type"),
        "run_id": _required_backend_string(getattr(run, "id", None), "run_id"),
        "run_url": _required_backend_string(getattr(run, "url", None), "run_url"),
    }
    if settings.wandb_entity is not None and identity["entity"] != settings.wandb_entity:
        raise RuntimeError("W&B backend entity differs from the requested entity")
    if (identity["project"], identity["group"], identity["run_name"], identity["job_type"]) != (
        settings.wandb_project,
        settings.wandb_group,
        settings.wandb_run_name,
        WANDB_JOB_TYPE,
    ):
        raise RuntimeError("W&B backend project/group/run name/job type differs from the requested identity")
    parsed_url = urlsplit(identity["run_url"])
    if parsed_url.scheme not in {"https", "http"} or not parsed_url.netloc:
        raise RuntimeError("W&B backend returned an invalid online run URL")
    path_parts = [part for part in parsed_url.path.split("/") if part]
    expected_suffix = [identity["entity"], identity["project"], "runs", identity["run_id"]]
    if path_parts[-4:] != expected_suffix:
        raise RuntimeError("W&B backend run URL does not bind the returned run identity")
    return identity


def _wandb_identity(
    run: Any,
    artifact: Any,
    *,
    artifact_name: str,
    settings: SmokeSettings,
    files: Sequence[Mapping[str, Any]],
    config_sha256: str,
) -> dict[str, Any]:
    run_identity = _validated_run_identity(run, settings)
    artifact_version = _required_backend_string(getattr(artifact, "version", None), "artifact_version")
    backend_artifact_name = _required_backend_string(getattr(artifact, "name", None), "artifact_name")
    if not _backend_artifact_name_matches(
        backend_artifact_name,
        collection=artifact_name,
        version=artifact_version,
    ):
        raise RuntimeError("W&B backend artifact name differs from the requested artifact identity")
    artifact_id = _required_backend_string(getattr(artifact, "id", None), "artifact_id")
    artifact_digest = _required_backend_string(getattr(artifact, "digest", None), "artifact_digest")
    file_values = [dict(value) for value in files]
    files_sha256 = _canonical_sha256({"files": file_values})
    values = {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "mode": "online",
        "status": "uploaded-preliminary",
        "terminal_status": "pending-postflight",
        **run_identity,
        "artifact_name": artifact_name,
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "artifact_digest": artifact_digest,
        "config_sha256": config_sha256,
        "files_sha256": files_sha256,
        "files": file_values,
    }
    _require_safe_finite_tree(values, "wandb_receipt")
    return values


def run_training_smoke(
    settings: SmokeSettings,
    *,
    wandb_module: Any,
    telemetry_reader: Callable[[torch.device], dict[str, float]] = nvidia_smi_telemetry,
) -> dict[str, Any]:
    """Execute all four arms and return a sanitized integration receipt."""

    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")
    output = settings.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    steps_path = output / "steps.jsonl"
    steps_path.touch()
    runtime_identity = _runtime_identity(settings, device)
    configs = phase2f_arm_configs(settings.input_dim)
    loss_config = Phase2fLossConfig()
    optimizer_settings = _optimizer_settings(settings)

    config = {
        "schema_version": SMOKE_SCHEMA,
        "execution_id": settings.execution_id,
        "evidence_level": "integration-smoke",
        "claim_boundary": CLAIM_BOUNDARY,
        "arms": list(ARMS),
        "max_steps_per_arm": settings.max_steps,
        "seed": settings.seed,
        "device_type": device.type,
        "input_dim": settings.input_dim,
        "spatial_size": settings.spatial_size,
        "source_groups": settings.source_groups,
        "views_per_group": 2,
        "gradient_clip": settings.gradient_clip,
        "arm_configs": {arm: asdict(configs[arm]) for arm in ARMS},
        "loss_config": asdict(loss_config),
        "nll_convention": "Gaussian NLL omits its additive constant; optimized NLL terms may be negative",
        "optimizer": optimizer_settings,
        "determinism": _determinism_settings(device),
        "resource_policy": "diagnostic-only",
        "requested_wandb_identity": {
            "entity": settings.wandb_entity,
            "project": settings.wandb_project,
            "group": settings.wandb_group,
            "run_name": settings.wandb_run_name,
            "job_type": WANDB_JOB_TYPE,
        },
        "synthetic_inputs_only": True,
        "dataset_or_cache_access": False,
        "runtime_identity": runtime_identity,
    }
    config = _json_safe_mapping(config)
    _require_safe_finite_tree(config, "config")
    run = None
    finished = False
    try:
        run = wandb_module.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            group=settings.wandb_group,
            name=settings.wandb_run_name,
            job_type=WANDB_JOB_TYPE,
            mode="online",
            reinit="finish_previous",
            config=config,
            tags=["phase-2g", "integration-smoke", "synthetic-only", "training-observability"],
        )
        if run is None or bool(run.offline):
            raise RuntimeError("the instrumentation smoke requires an online W&B run")
        initial_run_identity = _validated_run_identity(run, settings)
        if hasattr(run, "define_metric"):
            run.define_metric("global_step")
            run.define_metric("arms/*", step_metric="global_step")

        checkpoint_identities: dict[str, dict[str, Any]] = {}
        arm_summaries: dict[str, dict[str, Any]] = {}
        global_step = 0
        total_started = time.perf_counter()
        for arm_index, arm in enumerate(ARMS):
            torch.manual_seed(settings.seed + arm_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(settings.seed + arm_index)
                torch.cuda.reset_peak_memory_stats(device)
            model = Phase2fScaleGeometryProbe(configs[arm]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=settings.learning_rate,
                weight_decay=settings.weight_decay,
                betas=_ADAMW_BETAS,
                eps=_ADAMW_EPS,
                amsgrad=False,
                maximize=False,
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
            )
            final_features: torch.Tensor | None = None
            final_intrinsics: torch.Tensor | None = None
            final_row: dict[str, Any] | None = None
            maximum_forbidden_gradient_norm = 0.0
            for arm_step in range(settings.max_steps):
                features_cpu, targets_cpu, valid_cpu, intrinsics_cpu = _synthetic_batch(
                    settings,
                    arm_index=arm_index,
                    step=arm_step,
                )
                features = features_cpu.to(device)
                targets = targets_cpu.to(device)
                valid = valid_cpu.to(device)
                intrinsics = intrinsics_cpu.to(device) if configs[arm].consumes_intrinsics else None
                before = _parameter_snapshot(model)
                _synchronize(device)
                started = time.perf_counter()
                result = train_phase2f_step(
                    model,
                    optimizer,
                    features,
                    targets,
                    intrinsics=intrinsics,
                    intrinsics_image_size=(384, 384) if intrinsics is not None else None,
                    valid_mask=valid,
                    loss_config=loss_config,
                    group_count=settings.source_groups,
                    views=2,
                    maximum_gradient_norm=settings.gradient_clip,
                    verify_firewall=True,
                    firewall_tolerance=0.0,
                )
                _synchronize(device)
                step_seconds = time.perf_counter() - started
                if step_seconds <= 0 or result.firewall is None or not result.firewall.passed:
                    raise RuntimeError(f"arm {arm} did not produce a valid timed firewall-audited step")
                maximum_forbidden_gradient_norm = max(
                    maximum_forbidden_gradient_norm,
                    result.firewall.maximum_forbidden_norm,
                )
                update_norm = _parameter_update_norm(model, before)
                objectives, diagnostics = _split_step_metrics(result.metrics)
                gradients = {
                    name.removeprefix("gradient_"): float(value)
                    for name, value in result.metrics.items()
                    if name.startswith("gradient_")
                }
                pre_clip_norm = float(result.metrics["gradient_norm_total_before_clip"])
                post_clip_norm = _parameter_gradient_norm(model)
                if pre_clip_norm < 0 or not math.isfinite(pre_clip_norm):
                    raise RuntimeError("training smoke produced an invalid pre-clip gradient norm")
                if post_clip_norm > settings.gradient_clip + max(1e-6, settings.gradient_clip * 1e-5):
                    raise RuntimeError("training smoke post-clip gradient norm exceeds its configured threshold")
                applied_clip_coefficient = min(
                    1.0,
                    settings.gradient_clip / (pre_clip_norm + CLIP_GRAD_NORM_EPSILON),
                )
                expected_post_clip_norm = pre_clip_norm * applied_clip_coefficient
                if not math.isclose(post_clip_norm, expected_post_clip_norm, rel_tol=2e-5, abs_tol=1e-7):
                    raise RuntimeError("training smoke gradients violate torch clip_grad_norm_ semantics")
                was_clipped = int(applied_clip_coefficient < 1.0)
                memory = _memory_metrics(device)
                telemetry = telemetry_reader(device)
                if device.type == "cuda" and set(telemetry) != set(_NVIDIA_SMI_FIELDS):
                    raise RuntimeError("CUDA smoke telemetry must contain the complete nvidia-smi scalar schema")
                row = {
                    "schema_version": STEP_SCHEMA,
                    "arm": arm,
                    "arm_step": arm_step,
                    "global_step": global_step,
                    "objectives": objectives,
                    "diagnostics": diagnostics,
                    "gradients": gradients,
                    "optimizer": {
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_clip_threshold": settings.gradient_clip,
                        "pre_clip_gradient_norm": pre_clip_norm,
                        "post_clip_gradient_norm": post_clip_norm,
                        "applied_clip_coefficient": applied_clip_coefficient,
                        "was_clipped": was_clipped,
                        "parameter_update_norm": update_norm,
                    },
                    "resources": {
                        "policy": "diagnostic-only",
                        "timing": {
                            "step_seconds": step_seconds,
                            "samples_per_second": len(features) / step_seconds,
                            "source_groups_per_second": settings.source_groups / step_seconds,
                        },
                        "memory": memory,
                        "nvidia_smi": telemetry,
                    },
                    "parameter_counts": result.parameter_counts,
                    "gradient_firewall_passed": True,
                }
                _append_jsonl(steps_path, row)
                run.log(_wandb_payload(row), step=global_step)
                final_features = features
                final_intrinsics = intrinsics
                final_row = row
                global_step += 1

            if final_features is None or final_row is None:
                raise RuntimeError(f"arm {arm} completed no optimizer steps")
            checkpoint = save_phase2f_checkpoint(model, checkpoint_dir / f"{arm}.pt")
            reloaded, _ = load_phase2f_checkpoint(checkpoint, device=device)
            assert_strict_phase2f_reload(
                model,
                reloaded,
                final_features,
                intrinsics=final_intrinsics,
                intrinsics_image_size=(384, 384) if final_intrinsics is not None else None,
            )
            checkpoint_identities[arm] = {
                **_file_identity(checkpoint),
                "schema_version": "jepa4d-phase2f-checkpoint-v1",
                "exact_reload": True,
                "parameter_counts": model.parameter_counts(),
            }
            arm_summaries[arm] = {
                "optimizer_steps": settings.max_steps,
                "final_total_objective": float(final_row["objectives"]["total"]),
                "final_parameter_update_norm": float(final_row["optimizer"]["parameter_update_norm"]),
                "maximum_forbidden_gradient_norm": maximum_forbidden_gradient_norm,
                "exact_reload": True,
            }

        steps_identity = _file_identity(steps_path)
        upload_paths = (
            ("steps.jsonl", steps_path),
            *((f"checkpoints/{arm}.pt", checkpoint_dir / f"{arm}.pt") for arm in ARMS),
        )
        uploaded_file_snapshots = [
            _file_identity_and_state(path, published_name=published_name) for published_name, path in upload_paths
        ]
        uploaded_file_identities = [identity for identity, _ in uploaded_file_snapshots]
        uploaded_files_sha256 = _canonical_sha256({"files": uploaded_file_identities})
        config_sha256 = _canonical_sha256(config)
        artifact_name = f"phase2g-instrumentation-smoke-{settings.execution_id}-{initial_run_identity['run_id']}"
        artifact = wandb_module.Artifact(
            artifact_name,
            type="training-instrumentation-smoke",
            metadata={
                "schema_version": SMOKE_SCHEMA,
                "evidence_level": "integration-smoke",
                "synthetic_inputs_only": True,
                "terminal_status": "pending-postflight",
                "config_sha256": config_sha256,
                "files_sha256": uploaded_files_sha256,
            },
        )
        for published_name, path in upload_paths:
            artifact.add_file(str(path), name=published_name)
        logged_artifact = run.log_artifact(artifact)
        logged_artifact.wait()
        after_upload_snapshots = [
            _file_identity_and_state(path, published_name=published_name) for published_name, path in upload_paths
        ]
        if after_upload_snapshots != uploaded_file_snapshots:
            raise RuntimeError("a Phase 2g smoke artifact changed during W&B upload")
        wandb_receipt = _wandb_identity(
            run,
            logged_artifact,
            artifact_name=artifact_name,
            settings=settings,
            files=uploaded_file_identities,
            config_sha256=config_sha256,
        )

        receipt = {
            "schema_version": SMOKE_SCHEMA,
            "status": "pending-postflight",
            "terminal_status": "pending-postflight",
            "postflight_required": True,
            "execution_id": settings.execution_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "evidence_level": "integration-smoke",
            "claim_boundary": CLAIM_BOUNDARY,
            "synthetic_inputs_only": True,
            "dataset_or_cache_access": False,
            "resource_policy": "diagnostic-only",
            "config": config,
            "config_sha256": config_sha256,
            "runtime_identity": runtime_identity,
            "total_optimizer_steps": global_step,
            "expected_optimizer_steps": len(ARMS) * settings.max_steps,
            "steps": steps_identity,
            "checkpoints": checkpoint_identities,
            "arms": arm_summaries,
            "elapsed_seconds": time.perf_counter() - total_started,
            "finite": True,
            "wandb": wandb_receipt,
        }
        run.summary.update(
            {
                "status": "pending-postflight",
                "validation/postflight/status": "pending",
                "evidence_level": "integration-smoke",
                "total_optimizer_steps": global_step,
                "synthetic_inputs_only": True,
                "config_sha256": config_sha256,
                "artifact_files_sha256": uploaded_files_sha256,
            }
        )
        run.finish(exit_code=0)
        finished = True
        _write_json(output / "wandb_receipt.json", wandb_receipt)
        _write_json(output / "training_receipt.json", receipt)
        return receipt
    except Exception:
        if run is not None and not finished:
            with suppress(Exception):
                run.summary.update({"status": "failed", "evidence_level": "integration-smoke"})
            with suppress(Exception):
                run.log({"terminal/failure": 1})
            with suppress(Exception):
                run.finish(exit_code=1)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--max-steps", type=_bounded_max_steps, default=3, help="optimizer steps per arm (1..10)")
    parser.add_argument("--seed", type=int, default=260630)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def _require_cli_runtime(device: torch.device) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("the Phase 2g instrumentation smoke may execute only inside a Slurm allocation")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the Phase 2g instrumentation smoke CLI requires an allocated CUDA device")
    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("the Phase 2g instrumentation smoke requires WANDB_MODE=online")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device)
    _require_cli_runtime(device)
    import wandb

    slurm_id = os.environ["SLURM_JOB_ID"]
    run_name = args.run_name or f"{args.execution_id}-training-smoke-{slurm_id}"
    wandb_group = args.wandb_group or f"phase2g-smoke-{args.execution_id}"
    settings = SmokeSettings(
        output=args.output,
        execution_id=args.execution_id,
        git_commit=os.environ.get("JEPA4D_GIT_COMMIT", ""),
        scheduler_job_id=slurm_id,
        max_steps=args.max_steps,
        seed=args.seed,
        device=str(device),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=wandb_group,
        wandb_run_name=run_name,
    )
    receipt = run_training_smoke(settings, wandb_module=wandb)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "evidence_level": receipt["evidence_level"],
                "total_optimizer_steps": receipt["total_optimizer_steps"],
                "output": str(settings.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
