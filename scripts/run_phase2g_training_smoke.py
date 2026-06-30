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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from jepa4d.models.phase2f_scale_geometry import DEFAULT_PHASE2F_ARMS, Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_training import (
    assert_strict_phase2f_reload,
    load_phase2f_checkpoint,
    phase2f_arm_configs,
    save_phase2f_checkpoint,
    train_phase2f_step,
)

ARMS = DEFAULT_PHASE2F_ARMS
SMOKE_SCHEMA = "jepa4d-phase2g-training-instrumentation-smoke-v1"
STEP_SCHEMA = "jepa4d-phase2g-training-instrumentation-step-v1"
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2g-training-instrumentation-wandb-v1"
CLAIM_BOUNDARY = (
    "Synthetic integration smoke only: no dataset, cache, held-out target, or scientific-quality evidence was used or "
    "produced. Metrics validate training and logging instrumentation, not model quality or Phase 2g promotion."
)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CREDENTIAL_SHAPE = re.compile(r"wandb_v1_[A-Za-z0-9_-]+|(?:^|[._-])hf_[A-Za-z0-9]{16,}", re.IGNORECASE)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"name": resolved.name, "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _runtime_identity(settings: SmokeSettings, device: torch.device) -> dict[str, Any]:
    model_module = Path(sys.modules[Phase2fScaleGeometryProbe.__module__].__file__ or "").resolve(strict=True)
    training_module = Path(sys.modules[train_phase2f_step.__module__].__file__ or "").resolve(strict=True)
    hardware: dict[str, Any] = {"device_type": device.type}
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        hardware.update(
            {
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "unavailable")),
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
    if not selector:
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if index >= len(visible):
            raise RuntimeError("unable to map the logical CUDA device to an allocated nvidia-smi identity")
        selector = visible[index]
    query = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={selector}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
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
        f"{prefix}/optimizer/unclipped_gradient_norm": float(row["optimizer"]["unclipped_gradient_norm"]),
        f"{prefix}/optimizer/clipped_gradient_norm": float(row["optimizer"]["clipped_gradient_norm"]),
        f"{prefix}/optimizer/clip_coefficient": float(row["optimizer"]["clip_coefficient"]),
        f"{prefix}/optimizer/was_clipped": int(row["optimizer"]["was_clipped"]),
        f"{prefix}/optimizer/parameter_update_norm": float(row["optimizer"]["parameter_update_norm"]),
        f"{prefix}/timing/step_seconds": float(row["timing"]["step_seconds"]),
        f"{prefix}/throughput/samples_per_second": float(row["timing"]["samples_per_second"]),
        f"{prefix}/throughput/source_groups_per_second": float(row["timing"]["source_groups_per_second"]),
    }
    payload.update({f"{prefix}/loss/{name}": float(value) for name, value in row["loss"].items()})
    payload.update({f"{prefix}/gradients/{name}": float(value) for name, value in row["gradients"].items()})
    payload.update({f"{prefix}/memory/{name}": float(value) for name, value in row["memory"].items()})
    payload.update({f"{prefix}/nvidia_smi/{name}": float(value) for name, value in row["nvidia_smi"].items()})
    return payload


def _wandb_identity(run: Any, artifact: Any, *, artifact_name: str) -> dict[str, Any]:
    values = {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "mode": "online",
        "entity": str(run.entity),
        "project": str(run.project),
        "group": str(run.group),
        "run_name": str(run.name),
        "run_id": str(run.id),
        "run_url": str(run.url),
        "artifact_name": artifact_name,
        "artifact_id": str(artifact.id),
        "artifact_version": str(artifact.version),
        "artifact_digest": str(artifact.digest),
        "status": "success",
    }
    if any(not values[name] for name in values if name not in {"entity"}):
        raise RuntimeError("W&B did not return complete online run/artifact identities")
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
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "gradient_clip": settings.gradient_clip,
        "synthetic_inputs_only": True,
        "dataset_or_cache_access": False,
        "runtime_identity": runtime_identity,
    }
    _require_safe_finite_tree(config, "config")
    run = None
    finished = False
    try:
        run = wandb_module.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            group=settings.wandb_group,
            name=settings.wandb_run_name,
            job_type="phase2g-instrumentation-smoke",
            mode="online",
            reinit=True,
            config=config,
            tags=["phase-2g", "integration-smoke", "synthetic-only", "training-observability"],
        )
        if run is None or bool(run.offline):
            raise RuntimeError("the instrumentation smoke requires an online W&B run")
        if hasattr(run, "define_metric"):
            run.define_metric("global_step")
            run.define_metric("arms/*", step_metric="global_step")

        configs = phase2f_arm_configs(settings.input_dim)
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
                loss = {
                    name: float(value) for name, value in result.metrics.items() if not name.startswith("gradient_")
                }
                gradients = {
                    name.removeprefix("gradient_"): float(value)
                    for name, value in result.metrics.items()
                    if name.startswith("gradient_")
                }
                unclipped = float(result.metrics["gradient_norm_total_before_clip"])
                clip_coefficient = min(1.0, settings.gradient_clip / max(unclipped, 1e-12))
                memory = _memory_metrics(device)
                telemetry = telemetry_reader(device)
                if device.type == "cuda" and set(telemetry) != set(_NVIDIA_SMI_FIELDS):
                    raise RuntimeError("CUDA smoke telemetry must contain the complete nvidia-smi scalar schema")
                row = {
                    "schema_version": STEP_SCHEMA,
                    "arm": arm,
                    "arm_step": arm_step,
                    "global_step": global_step,
                    "loss": loss,
                    "gradients": gradients,
                    "optimizer": {
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_clip_threshold": settings.gradient_clip,
                        "unclipped_gradient_norm": unclipped,
                        "clipped_gradient_norm": min(unclipped, settings.gradient_clip),
                        "clip_coefficient": clip_coefficient,
                        "was_clipped": int(unclipped > settings.gradient_clip),
                        "parameter_update_norm": update_norm,
                    },
                    "timing": {
                        "step_seconds": step_seconds,
                        "samples_per_second": len(features) / step_seconds,
                        "source_groups_per_second": settings.source_groups / step_seconds,
                    },
                    "memory": memory,
                    "nvidia_smi": telemetry,
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
                "final_total_loss": float(final_row["loss"]["total"]),
                "final_parameter_update_norm": float(final_row["optimizer"]["parameter_update_norm"]),
                "maximum_forbidden_gradient_norm": maximum_forbidden_gradient_norm,
                "exact_reload": True,
            }

        steps_identity = _file_identity(steps_path)
        artifact_name = f"phase2g-instrumentation-smoke-{settings.execution_id}-{run.id}"
        artifact = wandb_module.Artifact(
            artifact_name,
            type="training-instrumentation-smoke",
            metadata={
                "schema_version": SMOKE_SCHEMA,
                "evidence_level": "integration-smoke",
                "synthetic_inputs_only": True,
                "config_sha256": _canonical_sha256(config),
            },
        )
        artifact.add_file(str(steps_path), name="steps.jsonl")
        for arm in ARMS:
            artifact.add_file(str(checkpoint_dir / f"{arm}.pt"), name=f"checkpoints/{arm}.pt")
        logged_artifact = run.log_artifact(artifact)
        logged_artifact.wait()
        wandb_receipt = _wandb_identity(run, logged_artifact, artifact_name=artifact_name)
        _write_json(output / "wandb_receipt.json", wandb_receipt)

        receipt = {
            "schema_version": SMOKE_SCHEMA,
            "status": "success",
            "execution_id": settings.execution_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "evidence_level": "integration-smoke",
            "claim_boundary": CLAIM_BOUNDARY,
            "synthetic_inputs_only": True,
            "dataset_or_cache_access": False,
            "config": config,
            "config_sha256": _canonical_sha256(config),
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
        _write_json(output / "training_receipt.json", receipt)
        run.summary.update(
            {
                "status": "success",
                "evidence_level": "integration-smoke",
                "total_optimizer_steps": global_step,
                "synthetic_inputs_only": True,
            }
        )
        run.finish(exit_code=0)
        finished = True
        (output / "SUCCESS").write_text("success\n", encoding="utf-8")
        return receipt
    except Exception:
        if run is not None and not finished:
            try:
                run.summary.update({"status": "failed", "evidence_level": "integration-smoke"})
                run.log({"terminal/failure": 1})
                run.finish(exit_code=1)
            except Exception:
                pass
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
