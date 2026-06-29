"""Train Phase-2e factorized geometry probes from an immutable feature cache."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from PIL import Image

from jepa4d.models.factorized_geometry import (
    FactorizedGeometryConfig,
    FactorizedGeometryOutput,
    FactorizedShapeScaleGeometryProbe,
)
from jepa4d.models.geometry_student import geometry_probe_loss

CACHE_SCHEMA = "jepa4d-phase2e-feature-cache-v1"
SHARD_SCHEMA = "jepa4d-phase2e-training-shard-v1"
ARTIFACT_MANIFEST_SCHEMA = "jepa4d-phase2e-artifact-manifest-v1"
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2e-wandb-artifact-receipt-v1"
FEATURE_SHAPE = (768, 24, 24)
DEFAULT_VARIANTS = (
    "monolithic_final",
    "factorized_bias",
    "factorized_vjepa",
    "factorized_rgb",
    "factorized_vjepa_rgb",
    "factorized_vjepa_k",
    "factorized_full",
    "factorized_full_teacher",
)
LOSS_WEIGHTS = {
    "geometry_probe": 1.0,
    "global_log_scale": 1.0,
    "centered_gt_shape": 0.25,
    "centered_teacher": 0.25,
    "paired_scale_consistency": 0.1,
}

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(slots=True)
class CachedSplit:
    name: str
    features: torch.Tensor
    rgb: torch.Tensor
    intrinsics_384: torch.Tensor
    targets: torch.Tensor
    teacher_centered_shape: torch.Tensor | None
    sample_ids: list[str]
    sensor_ids: list[str]
    paired: bool

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    @property
    def views(self) -> int:
        return 2 if self.paired else 1


@dataclass(slots=True)
class FeatureCache:
    path: Path
    sha256: str
    train: CachedSplit
    validation: CachedSplit


@dataclass(frozen=True, slots=True)
class VariantSpec:
    name: str
    config: FactorizedGeometryConfig
    use_teacher: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _require_string_list(value: Any, length: int, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} strings")
    result = [str(item) for item in value]
    if any(not item for item in result):
        raise ValueError(f"{label} contains an empty value")
    return result


def _valid_depth(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target) & (target > 0.1) & (target < 10.0)


def _validate_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{label} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value.detach().cpu().contiguous()


def _validate_split(name: str, value: Any) -> CachedSplit:
    if not isinstance(value, dict):
        raise TypeError(f"cache split {name} must be an object")
    required = {"features", "rgb", "intrinsics_384", "targets", "sample_ids", "sensor_ids"}
    allowed = required | {"teacher_centered_shape"}
    if set(value) - allowed or not required <= set(value):
        raise ValueError(
            f"cache split {name} fields differ: missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - allowed)}"
        )
    features = _validate_tensor(value["features"], f"{name}.features")
    paired = name == "train" and features.ndim == 5
    if paired:
        if features.shape[1] != 2 or tuple(features.shape[2:]) != FEATURE_SHAPE:
            raise ValueError(f"paired train features must be [N,2,{FEATURE_SHAPE[0]},24,24]")
        prefix = tuple(features.shape[:2])
    else:
        if features.ndim != 4 or tuple(features.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"{name} features must be [N,{FEATURE_SHAPE[0]},24,24]")
        prefix = (features.shape[0],)
    if name == "validation" and features.ndim != 4:
        raise ValueError("validation must be unpaired [N,...] data")
    count = int(features.shape[0])
    if count <= 0:
        raise ValueError(f"cache split {name} is empty")

    rgb = _validate_tensor(value["rgb"], f"{name}.rgb")
    intrinsics = _validate_tensor(value["intrinsics_384"], f"{name}.intrinsics_384")
    targets = _validate_tensor(value["targets"], f"{name}.targets")
    expected_rgb_ndim = 5 if paired else 4
    if rgb.ndim != expected_rgb_ndim or tuple(rgb.shape[: len(prefix)]) != prefix or rgb.shape[len(prefix)] != 3:
        raise ValueError(f"{name}.rgb does not match the feature leading dimensions or RGB channels")
    if rgb.shape[-2] <= 0 or rgb.shape[-1] <= 0:
        raise ValueError(f"{name}.rgb has an empty spatial dimension")
    expected_intrinsics_shape = (*prefix, 3, 3)
    if tuple(intrinsics.shape) != expected_intrinsics_shape:
        raise ValueError(f"{name}.intrinsics_384 must have shape {expected_intrinsics_shape}")
    if not bool((intrinsics[..., 0, 0] > 0).all()) or not bool((intrinsics[..., 1, 1] > 0).all()):
        raise ValueError(f"{name}.intrinsics_384 has a non-positive focal length")
    expected_last_row = intrinsics.new_tensor((0.0, 0.0, 1.0)).expand_as(intrinsics[..., 2, :])
    if not torch.allclose(intrinsics[..., 2, :], expected_last_row, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name}.intrinsics_384 has an invalid pinhole last row")
    expected_target_shape = (*prefix, 24, 24)
    if tuple(targets.shape) != expected_target_shape:
        raise ValueError(f"{name}.targets must have shape {expected_target_shape}")
    flat_targets = targets.reshape(-1, 24, 24)
    if any(int(_valid_depth(target).sum()) < 100 for target in flat_targets):
        raise ValueError(f"{name}.targets contains a sample with fewer than 100 valid pixels")

    teacher_value = value.get("teacher_centered_shape")
    teacher: torch.Tensor | None = None
    if teacher_value is not None:
        teacher = _validate_tensor(teacher_value, f"{name}.teacher_centered_shape")
        if tuple(teacher.shape) != expected_target_shape:
            raise ValueError(f"{name}.teacher_centered_shape must have shape {expected_target_shape}")
    sample_ids = _require_string_list(value["sample_ids"], count, f"{name}.sample_ids")
    sensor_ids = _require_string_list(value["sensor_ids"], count, f"{name}.sensor_ids")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{name}.sample_ids must be unique")
    return CachedSplit(name, features, rgb, intrinsics, targets, teacher, sample_ids, sensor_ids, paired)


def load_feature_cache(path: Path) -> FeatureCache:
    """Load only the formal train/validation schema and fail closed on any test split."""

    resolved = path.resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"feature cache schema must be {CACHE_SCHEMA}")
    if "test" in payload:
        raise ValueError("Phase2e feature cache must not expose a test split")
    if set(payload) - {"schema_version", "splits"}:
        raise ValueError(
            f"feature cache has unexpected root fields: {sorted(set(payload) - {'schema_version', 'splits'})}"
        )
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        found = sorted(splits) if isinstance(splits, dict) else type(splits).__name__
        raise ValueError(f"feature cache must contain exactly train/validation and no test split; found {found}")
    train = _validate_split("train", splits["train"])
    validation = _validate_split("validation", splits["validation"])
    overlap = set(train.sample_ids) & set(validation.sample_ids)
    if overlap:
        raise ValueError(f"train/validation sample IDs overlap: {sorted(overlap)[:3]}")
    return FeatureCache(resolved, _sha256(resolved), train, validation)


def variant_spec(name: str, hidden_dim: int = 64) -> VariantSpec:
    common: dict[str, Any] = {"input_dim": 768, "hidden_dim": hidden_dim}
    definitions: dict[str, tuple[str, str, tuple[str, ...], bool]] = {
        "monolithic_final": ("monolithic", "none", (), False),
        "factorized_bias": ("factorized", "none", (), False),
        "factorized_vjepa": ("factorized", "none", ("vjepa",), False),
        "factorized_rgb": ("factorized", "none", ("rgb",), False),
        "factorized_vjepa_rgb": ("factorized", "none", ("vjepa", "rgb"), False),
        "factorized_vjepa_k": (
            "factorized",
            "known_rays",
            ("vjepa", "intrinsics", "ray_summary"),
            False,
        ),
        "factorized_full": (
            "factorized",
            "known_rays",
            ("vjepa", "rgb", "intrinsics", "ray_summary"),
            False,
        ),
        "factorized_full_teacher": (
            "factorized",
            "known_rays",
            ("vjepa", "rgb", "intrinsics", "ray_summary"),
            True,
        ),
    }
    if name not in definitions:
        raise ValueError(f"unknown Phase2e variant {name}; choose from {', '.join(DEFAULT_VARIANTS)}")
    mode, camera_mode, scale_inputs, use_teacher = definitions[name]
    config = FactorizedGeometryConfig(
        **common,
        mode=mode,  # type: ignore[arg-type]
        camera_mode=camera_mode,  # type: ignore[arg-type]
        scale_inputs=scale_inputs,  # type: ignore[arg-type]
    )
    return VariantSpec(name, config, use_teacher)


def _batch(split: CachedSplit, indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    group_count = len(indices)
    views = split.views

    def selected(tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.index_select(0, indices)
        return value.reshape(group_count * views, *value.shape[2:]) if split.paired else value

    sample_ids = [f"{split.sample_ids[index]}:view{view}" for index in indices.tolist() for view in range(views)]
    sensor_ids = [split.sensor_ids[index] for index in indices.tolist() for _ in range(views)]
    return {
        "features": selected(split.features).to(device),
        "rgb": selected(split.rgb).to(device),
        "intrinsics_384": selected(split.intrinsics_384).to(device),
        "targets": selected(split.targets).to(device),
        "teacher_centered_shape": (
            None if split.teacher_centered_shape is None else selected(split.teacher_centered_shape).to(device)
        ),
        "sample_ids": sample_ids,
        "sensor_ids": sensor_ids,
        "group_count": group_count,
        "views": views,
    }


def _forward(
    model: FactorizedShapeScaleGeometryProbe,
    spec: VariantSpec,
    batch: dict[str, Any],
) -> FactorizedGeometryOutput:
    kwargs: dict[str, Any] = {}
    if "rgb" in spec.config.scale_inputs:
        kwargs["rgb"] = batch["rgb"]
    if spec.config.camera_mode == "known_rays" or {
        "intrinsics",
        "ray_summary",
    } & set(spec.config.scale_inputs):
        kwargs["intrinsics"] = batch["intrinsics_384"]
        kwargs["intrinsics_image_size"] = (384, 384)
    return model(batch["features"], **kwargs)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("masked mean received an empty sample")
    sums = torch.where(mask, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    return sums / counts


def phase2e_loss(
    output: FactorizedGeometryOutput,
    targets: torch.Tensor,
    *,
    teacher_centered_shape: torch.Tensor | None,
    use_teacher: bool,
    group_count: int,
    views: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine the registered geometry loss with explicit scale/shape terms."""

    valid = _valid_depth(targets)
    base, base_parts = geometry_probe_loss(output.log_depth, output.log_variance, targets, valid)
    if output.centered_shape is None:
        if output.global_log_scale is not None or use_teacher:
            raise ValueError("monolithic output cannot use factorized scale/teacher supervision")
        zero = base.detach().new_zeros(())
        return base, {
            "geometry_probe": base.detach(),
            **{f"base_{key}": value for key, value in base_parts.items()},
            "global_log_scale": zero,
            "centered_gt_shape": zero,
            "centered_teacher": zero,
            "paired_scale_consistency": zero,
        }
    target_log = targets.clamp_min(1e-4).log()
    target_scale = _masked_mean(target_log, valid)
    target_shape = target_log - target_scale[:, None, None]
    assert output.global_log_scale is not None
    predicted_scale = output.global_log_scale.flatten()
    predicted_shape = output.centered_shape
    predicted_shape = predicted_shape - _masked_mean(predicted_shape, valid)[:, None, None]
    global_scale = F.smooth_l1_loss(predicted_scale, target_scale)
    centered_shape = F.smooth_l1_loss(predicted_shape[valid], target_shape[valid])
    teacher_loss = output.log_depth.new_zeros(())
    if use_teacher:
        if teacher_centered_shape is None:
            raise ValueError("factorized_full_teacher requires teacher_centered_shape in the train cache")
        if output.centered_shape is None:
            raise ValueError("teacher-centered distillation requires factorized output")
        teacher = teacher_centered_shape - _masked_mean(teacher_centered_shape, valid)[:, None, None]
        teacher_loss = F.smooth_l1_loss(predicted_shape[valid], teacher[valid])
    paired_consistency = output.log_depth.new_zeros(())
    if views == 2:
        scales = predicted_scale.reshape(group_count, views)
        paired_consistency = F.smooth_l1_loss(scales[:, 0], scales[:, 1])
    total = (
        LOSS_WEIGHTS["geometry_probe"] * base
        + LOSS_WEIGHTS["global_log_scale"] * global_scale
        + LOSS_WEIGHTS["centered_gt_shape"] * centered_shape
        + LOSS_WEIGHTS["centered_teacher"] * teacher_loss
        + LOSS_WEIGHTS["paired_scale_consistency"] * paired_consistency
    )
    return total, {
        "geometry_probe": base.detach(),
        **{f"base_{key}": value for key, value in base_parts.items()},
        "global_log_scale": global_scale.detach(),
        "centered_gt_shape": centered_shape.detach(),
        "centered_teacher": teacher_loss.detach(),
        "paired_scale_consistency": paired_consistency.detach(),
    }


def _sample_metrics(log_depth: torch.Tensor, log_variance: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    valid = _valid_depth(target)
    prediction = log_depth.exp()
    pred = prediction[valid]
    truth = target[valid]
    error = pred - truth
    ratio = torch.maximum(pred / truth, truth / pred.clamp_min(1e-8))
    alignment_scale = truth.median() / pred.median().clamp_min(1e-8)
    aligned = pred * alignment_scale
    residual_log = log_depth[valid] - truth.log()
    variance = log_variance[valid].exp().clamp_min(1e-8)
    return {
        "metric_abs_rel": float((error.abs() / truth).mean()),
        "metric_rmse_m": float(error.square().mean().sqrt()),
        "metric_delta_1": float((ratio < 1.25).float().mean()),
        "aligned_abs_rel": float(((aligned - truth).abs() / truth).mean()),
        "abs_log_scale_error": abs(float(alignment_scale.clamp_min(1e-12).log())),
        "log_depth_nll": float((0.5 * (variance.log() + residual_log.square() / variance)).mean()),
    }


def _validation_metrics(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    targets: torch.Tensor,
    sensor_ids: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, float]]]:
    rows = [
        _sample_metrics(depth, variance, target)
        for depth, variance, target in zip(log_depth, log_variance, targets, strict=True)
    ]
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    per_sensor: dict[str, dict[str, float]] = {}
    for sensor in sorted(set(sensor_ids)):
        selected = [row for row, row_sensor in zip(rows, sensor_ids, strict=True) if row_sensor == sensor]
        per_sensor[sensor] = {key: float(np.mean([row[key] for row in selected])) for key in rows[0]}
    return metrics, per_sensor, rows


def _predict_validation(
    model: FactorizedShapeScaleGeometryProbe,
    spec: VariantSpec,
    split: CachedSplit,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    log_depths = []
    log_variances = []
    global_scales = []
    with torch.inference_mode():
        for offset in range(0, split.size, batch_size):
            indices = torch.arange(offset, min(offset + batch_size, split.size))
            batch = _batch(split, indices, device)
            output = _forward(model, spec, batch)
            log_depths.append(output.log_depth.cpu())
            log_variances.append(output.log_variance.cpu())
            if output.global_log_scale is None:
                valid = _valid_depth(batch["targets"])
                scale = _masked_mean(output.log_depth, valid)
            else:
                scale = output.global_log_scale.flatten()
            global_scales.append(scale.cpu())
    log_depth = torch.cat(log_depths)
    log_variance = torch.cat(log_variances)
    global_scale = torch.cat(global_scales)
    metrics, per_sensor, per_sample = _validation_metrics(
        log_depth,
        log_variance,
        split.targets,
        split.sensor_ids,
    )
    return {
        "log_depth": log_depth,
        "log_variance": log_variance,
        "global_log_scale": global_scale,
        "targets": split.targets,
        "metrics": metrics,
        "per_sensor": per_sensor,
        "per_sample": per_sample,
        "sample_ids": split.sample_ids,
        "sensor_ids": split.sensor_ids,
    }


def _same_prediction(left: FactorizedGeometryOutput, right: FactorizedGeometryOutput) -> bool:
    fields = ("log_depth", "log_variance", "centered_shape", "global_log_scale")
    for field in fields:
        first = getattr(left, field)
        second = getattr(right, field)
        if first is None or second is None:
            if first is not None or second is not None:
                return False
        elif not torch.equal(first, second):
            return False
    return True


def _train_one(
    cache: FeatureCache,
    spec: VariantSpec,
    seed: int,
    output: Path,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    run_config_sha256: str,
    run: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FactorizedShapeScaleGeometryProbe(spec.config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_abs_rel = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    history_path = output / "histories" / f"{spec.name}-seed{seed}.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(cache.train.size, generator=generator)
        sums: dict[str, float] = {}
        examples = 0
        gradient_norms = []
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            batch = _batch(cache.train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, spec, batch)
            loss, parts = phase2e_loss(
                prediction,
                batch["targets"],
                teacher_centered_shape=batch["teacher_centered_shape"],
                use_teacher=spec.use_teacher,
                group_count=batch["group_count"],
                views=batch["views"],
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            weight = int(batch["targets"].shape[0])
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach()) * weight
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value) * weight
            examples += weight
            gradient_norms.append(float(gradient_norm))
        validation = _predict_validation(model, spec, cache.validation, device, batch_size)
        validation_abs_rel = float(validation["metrics"]["metric_abs_rel"])
        is_best = validation_abs_rel < best_abs_rel
        if is_best:
            best_abs_rel = validation_abs_rel
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        row: dict[str, Any] = {
            "variant": spec.name,
            "seed": seed,
            "epoch": epoch,
            **{key: value / examples for key, value in sums.items()},
            "gradient_norm": float(np.mean(gradient_norms)),
            **{f"validation_{key}": float(value) for key, value in validation["metrics"].items()},
            "best_validation_metric_abs_rel": best_abs_rel,
            "is_best": is_best,
        }
        history.append(row)
        with history_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if run is not None:
            prefix = f"training/{spec.name}/seed_{seed}"
            if epoch == 0:
                run.define_metric(f"{prefix}/epoch")
                run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")
            run.log({f"{prefix}/{key}": value for key, value in row.items() if key not in {"variant", "seed"}})
    if best_state is None:
        raise RuntimeError(f"training produced no checkpoint for {spec.name} seed {seed}")
    model.load_state_dict(best_state, strict=True)
    checkpoint = output / "checkpoints" / f"{spec.name}-seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "jepa4d-phase2e-checkpoint-v1",
            "variant": spec.name,
            "seed": seed,
            "config": asdict(spec.config),
            "use_teacher": spec.use_teacher,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_metric_abs_rel": best_abs_rel,
            "feature_cache_sha256": cache.sha256,
            "run_config_sha256": run_config_sha256,
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = FactorizedShapeScaleGeometryProbe(FactorizedGeometryConfig(**payload["config"])).to(device)
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    reloaded.eval()
    reload_batch = _batch(cache.validation, torch.arange(min(2, cache.validation.size)), device)
    with torch.inference_mode():
        original_prediction = _forward(model, spec, reload_batch)
        reloaded_prediction = _forward(reloaded, spec, reload_batch)
    if not _same_prediction(original_prediction, reloaded_prediction):
        raise RuntimeError(f"strict checkpoint reload changed predictions for {spec.name} seed {seed}")
    final_validation = _predict_validation(reloaded, spec, cache.validation, device, batch_size)
    prediction_path = output / "validation_predictions" / f"{spec.name}-seed{seed}.pt"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "jepa4d-phase2e-validation-predictions-v1",
            "variant": spec.name,
            "seed": seed,
            "sample_ids": final_validation["sample_ids"],
            "sensor_ids": final_validation["sensor_ids"],
            "prediction_m": final_validation["log_depth"].exp(),
            "target_m": final_validation["targets"],
            "log_variance": final_validation["log_variance"],
            "global_log_scale": final_validation["global_log_scale"],
        },
        prediction_path,
    )
    validation_record = {
        "schema_version": "jepa4d-phase2e-validation-v1",
        "variant": spec.name,
        "seed": seed,
        "metrics": final_validation["metrics"],
        "per_sensor": final_validation["per_sensor"],
        "per_sample": [
            {
                "sample_id": sample_id,
                "sensor_id": sensor_id,
                **metrics,
            }
            for sample_id, sensor_id, metrics in zip(
                final_validation["sample_ids"],
                final_validation["sensor_ids"],
                final_validation["per_sample"],
                strict=True,
            )
        ],
    }
    validation_path = output / "validation_metrics" / f"{spec.name}-seed{seed}.json"
    _write_json(validation_path, validation_record)
    result = {
        "variant": spec.name,
        "seed": seed,
        "config": asdict(spec.config),
        "use_teacher": spec.use_teacher,
        "trainable_parameters": reloaded.trainable_parameter_count,
        "best_epoch": best_epoch,
        "best_validation_metric_abs_rel": best_abs_rel,
        "validation_metrics": final_validation["metrics"],
        "validation_per_sensor": final_validation["per_sensor"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_reload": "strict-prediction-equality-pass",
        "run_config_sha256": run_config_sha256,
        "history": str(history_path),
        "history_sha256": _sha256(history_path),
        "validation_predictions": str(prediction_path),
        "validation_predictions_sha256": _sha256(prediction_path),
        "validation_metrics_path": str(validation_path),
        "validation_metrics_sha256": _sha256(validation_path),
        "training_seconds": time.perf_counter() - started,
    }
    return result, history, prediction_path


def _depth_png(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().numpy()
    finite = np.isfinite(array)
    if not finite.any():
        array = np.zeros_like(array)
    else:
        low, high = np.percentile(array[finite], (2, 98))
        if high <= low:
            high = low + 1.0
        array = np.clip((array - low) / (high - low), 0, 1)
    rgb = np.stack((array, 1.0 - np.abs(2.0 * array - 1.0), 1.0 - array), axis=-1)
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").resize((192, 192), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _sparkline(values: list[float], width: int = 240, height: int = 64) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-12)
    points = []
    for index, value in enumerate(values):
        x = index * width / max(len(values) - 1, 1)
        y = height - (value - low) * height / span
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="validation curve">'
        f'<polyline fill="none" stroke="#55d6be" stroke-width="2" points="{" ".join(points)}"/></svg>'
    )


def snapshot_gpu_telemetry(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Persist and summarize the Slurm-side nvidia-smi monitor, when available."""
    log_root = os.getenv("JEPA4D_JOB_LOG_DIR")
    source = None if not log_root else Path(log_root) / "gpu-telemetry.csv"
    if source is None or not source.is_file() or source.stat().st_size <= 0:
        return {"available": False, "reason": "JEPA4D GPU monitor file is unavailable"}, []
    destination = output / "gpu_telemetry.csv"
    shutil.copyfile(source, destination)
    with destination.open(newline="") as stream:
        rows = [{str(key).strip(): str(value).strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        return {"available": False, "reason": "GPU monitor contains no samples"}, []

    def numeric(value: str) -> float | None:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        return None if match is None else float(match.group())

    fields = {
        "utilization_gpu": ("utilization.gpu [%]", "utilization_gpu_pct"),
        "utilization_memory": ("utilization.memory [%]", "utilization_memory_pct"),
        "memory_used_mib": ("memory.used [MiB]", "memory_used_mib"),
        "temperature_c": ("temperature.gpu", "temperature_c"),
        "power_w": ("power.draw [W]", "power_w"),
    }
    statistics: dict[str, dict[str, float]] = {}
    for label, (field, _) in fields.items():
        values = [parsed for row in rows if (parsed := numeric(row.get(field, ""))) is not None]
        if values:
            statistics[label] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "p95": float(np.percentile(values, 95)),
            }
    numeric_rows: list[dict[str, Any]] = []
    for row in rows:
        numeric_row: dict[str, Any] = {
            "timestamp": row.get("timestamp", ""),
            "gpu_index": numeric(row.get("index", "")),
            "gpu_uuid": row.get("uuid", ""),
            "gpu_name": row.get("name", ""),
            "pstate": row.get("pstate", ""),
        }
        for _, (source_field, output_field) in fields.items():
            numeric_row[output_field] = numeric(row.get(source_field, ""))
        numeric_rows.append(numeric_row)
    summary = {
        "available": True,
        "samples": len(rows),
        "source": str(source.resolve()),
        "snapshot": str(destination.resolve()),
        "snapshot_sha256": _sha256(destination),
        "statistics": statistics,
    }
    _write_json(output / "gpu_telemetry_summary.json", summary)
    return summary, numeric_rows


def build_self_contained_report(
    output: Path,
    results: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
    telemetry: dict[str, Any] | None = None,
) -> Path:
    rows = []
    panels = []
    for result in results:
        key = f"{result['variant']}-seed{result['seed']}"
        metrics = result["validation_metrics"]
        curve = [float(row["validation_metric_abs_rel"]) for row in histories[key]]
        rows.append(
            "<tr>"
            f"<td>{html.escape(result['variant'])}</td><td>{result['seed']}</td>"
            f"<td>{metrics['metric_abs_rel']:.6f}</td><td>{metrics['aligned_abs_rel']:.6f}</td>"
            f"<td>{metrics['abs_log_scale_error']:.6f}</td><td>{metrics['log_depth_nll']:.6f}</td>"
            f"<td>{result['trainable_parameters']}</td><td>{_sparkline(curve)}</td>"
            "</tr>"
        )
        prediction = torch.load(result["validation_predictions"], map_location="cpu", weights_only=True)
        predicted = prediction["prediction_m"][0]
        target = prediction["target_m"][0]
        relative_error = (predicted - target).abs() / target.clamp_min(1e-6)
        images = []
        for title, tensor in (("prediction", predicted), ("target", target), ("relative error", relative_error)):
            images.append(
                f'<figure><img alt="{html.escape(key)} {title}" src="data:image/png;base64,{_depth_png(tensor)}">'
                f"<figcaption>{html.escape(title)}</figcaption></figure>"
            )
        panels.append(f'<section><h3>{html.escape(key)}</h3><div class="images">{"".join(images)}</div></section>')
    telemetry_cards = ""
    if telemetry and telemetry.get("available"):
        stats = telemetry.get("statistics", {})
        values = [f"{telemetry['samples']} monitor samples"]
        if "utilization_gpu" in stats:
            values.append(
                f"GPU utilization mean/p95 {stats['utilization_gpu']['mean']:.1f}%/{stats['utilization_gpu']['p95']:.1f}%"
            )
        if "memory_used_mib" in stats:
            values.append(f"peak memory {stats['memory_used_mib']['max'] / 1024:.2f} GiB")
        telemetry_cards = f"<section><h2>Runtime telemetry</h2><p>{' · '.join(values)}</p></section>"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase2e factorized training shard</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0e1525;color:#e8eefb;margin:2rem}}
table{{border-collapse:collapse;width:100%;background:#17223a}}th,td{{padding:.55rem;border:1px solid #34425f}}
th{{background:#22304d}}svg{{width:240px;height:64px}}section{{margin:1.5rem 0;padding:1rem;background:#17223a}}
.images{{display:flex;gap:1rem;flex-wrap:wrap}}figure{{margin:0}}img{{width:192px;height:192px;image-rendering:pixelated}}
code{{color:#8be9fd}}
</style></head><body>
<h1>Phase2e factorized geometry training shard</h1>
<p>Validation-only checkpoint selection. Cache policy: <code>train + validation only</code>.</p>
<table><thead><tr><th>Variant</th><th>Seed</th><th>Raw AbsRel</th><th>Aligned AbsRel</th>
<th>Abs log-scale</th><th>Log-depth NLL</th><th>Parameters</th><th>Validation curve</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
{telemetry_cards}
<h2>Fixed first-validation-sample diagnostics</h2>{"".join(panels)}
</body></html>"""
    path = output / "phase2e_report.html"
    path.write_text(document, encoding="utf-8")
    return path


def _prepare_output(path: Path) -> Path:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory must be new and empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def validate_shard_artifacts(output_path: Path, shard: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete, immutable evidence bundle before publication."""
    output = output_path.resolve()
    if shard.get("schema_version") != SHARD_SCHEMA or shard.get("status") != "success":
        raise ValueError("only a successful Phase2e shard can be published")
    if shard.get("selection_split") != "validation":
        raise ValueError("Phase2e artifact checkpoint selection must be validation-only")
    results = shard.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Phase2e artifact requires at least one result")

    files: list[dict[str, Any]] = []

    def add_file(role: str, path_value: str | Path, expected_sha256: str | None = None) -> Path:
        path = Path(path_value).resolve()
        try:
            relative = path.relative_to(output)
        except ValueError as error:
            raise ValueError(f"artifact path escapes the shard output: {path}") from error
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"artifact file is absent or empty: {relative}")
        digest = _sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"artifact checksum mismatch for {relative}")
        files.append(
            {
                "role": role,
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        return path

    shard_path = add_file("shard", output / "phase2e_shard.json")
    on_disk_shard = json.loads(shard_path.read_text())
    if _canonical_sha256(on_disk_shard) != _canonical_sha256(shard):
        raise ValueError("phase2e_shard.json does not match the in-memory shard")
    add_file(
        "resolved_config",
        output / "resolved_config.json",
        str(shard["resolved_config_file_sha256"]),
    )
    report = add_file("html_report", str(shard["report"]), str(shard["report_sha256"]))
    report_text = report.read_text()
    if "<script" in report_text.lower() or "data:image/png;base64," not in report_text or "<svg" not in report_text:
        raise ValueError("Phase2e HTML report is not self-contained")
    telemetry = shard.get("gpu_telemetry")
    if isinstance(telemetry, dict) and telemetry.get("available"):
        add_file("gpu_telemetry", str(telemetry["snapshot"]), str(telemetry["snapshot_sha256"]))
        add_file("gpu_telemetry_summary", output / "gpu_telemetry_summary.json")

    result_keys = (
        ("checkpoint", "checkpoint_sha256"),
        ("history", "history_sha256"),
        ("validation_predictions", "validation_predictions_sha256"),
        ("validation_metrics_path", "validation_metrics_sha256"),
    )
    identities: set[tuple[str, int]] = set()
    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each Phase2e result must be an object")
        identity = (str(result["variant"]), int(result["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate Phase2e result: {identity}")
        identities.add(identity)
        for path_key, hash_key in result_keys:
            add_file(f"{path_key}:{identity[0]}:seed{identity[1]}", str(result[path_key]), str(result[hash_key]))

    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "shard_schema_version": SHARD_SCHEMA,
        "config_sha256": shard["config_sha256"],
        "selection_split": "validation",
        "files": files,
    }


def upload_wandb_artifact(
    run: Any,
    output_path: Path,
    shard: dict[str, Any],
    *,
    wandb_module: Any | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Upload a validated shard bundle, wait for commit, and persist its receipt."""
    if wandb_module is None:
        import wandb

        wandb_module = wandb

    output = output_path.resolve()
    manifest = validate_shard_artifacts(output, shard)
    manifest_path = output / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    artifact = wandb_module.Artifact(
        name=f"{run.id}-phase2e-factorized-shard",
        type="phase2e-training-shard",
        metadata={
            "schema_version": SHARD_SCHEMA,
            "config_sha256": shard["config_sha256"],
            "feature_cache_sha256": shard["feature_cache_sha256"],
            "selection_split": "validation",
            "result_rows": len(shard["results"]),
        },
    )
    artifact.add_dir(str(output), name="phase2e")
    uploaded = run.log_artifact(artifact).wait(timeout=timeout_seconds)
    run_id = str(run.id)
    artifact_id = str(uploaded.id)
    if not run_id or run_id == "None" or not artifact_id or artifact_id == "None":
        raise RuntimeError("W&B did not return a durable run/artifact ID after upload")
    receipt = {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "status": "uploaded",
        "mode": "online",
        "run_id": run_id,
        "run_url": str(run.url),
        "run_path": str(run.path),
        "artifact_id": artifact_id,
        "artifact_name": str(uploaded.name),
        "artifact_qualified_name": str(uploaded.qualified_name),
        "artifact_version": str(uploaded.version),
        "artifact_digest": str(uploaded.digest),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "phase2e_shard_sha256": _sha256(output / "phase2e_shard.json"),
    }
    _write_json(output / "wandb_receipt.json", receipt)
    return receipt


def run_training_shard(
    cache_path: Path,
    output_path: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    epochs: int = 60,
    batch_size: int = 8,
    hidden_dim: int = 64,
    device_name: str = "cuda:0",
    wandb_enabled: bool = True,
    wandb_project: str = "jepa4d-worldmodel",
    wandb_entity: str | None = None,
    run_name: str = "phase2e-factorized-shard",
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or hidden_dim <= 0:
        raise ValueError("epochs, batch_size, and hidden_dim must be positive")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty unique tuple")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique tuple")
    specs = [variant_spec(name, hidden_dim) for name in variants]
    cache = load_feature_cache(cache_path)
    if any(spec.use_teacher for spec in specs) and cache.train.teacher_centered_shape is None:
        raise ValueError("factorized_full_teacher was requested but train.teacher_centered_shape is absent")
    output = _prepare_output(output_path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    resolved = {
        "schema_version": SHARD_SCHEMA,
        "feature_cache": {"path": str(cache.path), "sha256": cache.sha256, "schema_version": CACHE_SCHEMA},
        "data_splits": {
            "train": {"groups": cache.train.size, "views": cache.train.views, "paired": cache.train.paired},
            "validation": {
                "groups": cache.validation.size,
                "views": cache.validation.views,
                "paired": cache.validation.paired,
            },
        },
        "variants": list(variants),
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0.0001, "gradient_clip": 5.0},
        "loss_weights": LOSS_WEIGHTS,
        "checkpoint_selection": "minimum validation raw metric_abs_rel only",
        "device": str(device),
        "wandb": {"enabled": wandb_enabled, "mode": "online", "project": wandb_project, "entity": wandb_entity},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
    }
    config_sha256 = _canonical_sha256(resolved)
    resolved["config_sha256"] = config_sha256
    resolved_path = output / "resolved_config.json"
    _write_json(resolved_path, resolved)
    resolved_file_sha256 = _sha256(resolved_path)
    run = None
    if wandb_enabled:
        import wandb

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            job_type="phase2e-factorized-training-shard",
            mode="online",
            config=resolved,
            tags=["phase-2e", "factorized-depth", "feature-cache", "validation-only-selection"],
        )
        if run.offline:
            raise RuntimeError("Phase2e training requires online W&B")
    results: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    try:
        for spec in specs:
            for seed in seeds:
                result, history, _ = _train_one(
                    cache,
                    spec,
                    seed,
                    output,
                    device,
                    epochs=epochs,
                    batch_size=batch_size,
                    run_config_sha256=config_sha256,
                    run=run,
                )
                results.append(result)
                histories[f"{spec.name}-seed{seed}"] = history
        telemetry, telemetry_rows = snapshot_gpu_telemetry(output)
        report = build_self_contained_report(output, results, histories, telemetry)
        shard = {
            "schema_version": SHARD_SCHEMA,
            "status": "success",
            "config_sha256": config_sha256,
            "resolved_config_file_sha256": resolved_file_sha256,
            "feature_cache_sha256": cache.sha256,
            "selection_split": "validation",
            "results": results,
            "report": str(report),
            "report_sha256": _sha256(report),
            "gpu_telemetry": telemetry,
            "wandb_url": None if run is None else run.url,
        }
        _write_json(output / "phase2e_shard.json", shard)
        artifact_manifest = validate_shard_artifacts(output, shard)
        _write_json(output / "artifact_manifest.json", artifact_manifest)
        if run is not None:
            import wandb

            table = wandb.Table(
                columns=["variant", "seed", "raw_absrel", "aligned_absrel", "abs_log_scale", "nll", "parameters"]
            )
            for result in results:
                metrics = result["validation_metrics"]
                table.add_data(
                    result["variant"],
                    result["seed"],
                    metrics["metric_abs_rel"],
                    metrics["aligned_abs_rel"],
                    metrics["abs_log_scale_error"],
                    metrics["log_depth_nll"],
                    result["trainable_parameters"],
                )
            logged = {"validation/results": table, "validation/report": wandb.Html(str(report), inject=False)}
            if telemetry_rows:
                telemetry_table = wandb.Table(columns=list(telemetry_rows[0]))
                for row in telemetry_rows:
                    telemetry_table.add_data(*row.values())
                logged["runtime/gpu_telemetry"] = telemetry_table
            run.log(logged)
            receipt = upload_wandb_artifact(run, output, shard, wandb_module=wandb)
            run.summary.update(
                {
                    "status": "success",
                    "result_rows": len(results),
                    "config_sha256": config_sha256,
                    "feature_cache_sha256": cache.sha256,
                    "artifact_id": receipt["artifact_id"],
                    "artifact_qualified_name": receipt["artifact_qualified_name"],
                    "artifact_digest": receipt["artifact_digest"],
                    "gpu_telemetry_samples": telemetry.get("samples", 0),
                }
            )
            run.finish(exit_code=0)
        return shard
    except Exception as error:
        failure = {"schema_version": SHARD_SCHEMA, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        _write_json(output / "run_failure.json", failure)
        if run is not None:
            run.summary.update(failure)
            run.finish(exit_code=1)
        raise


def _parse_csv(value: str, label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise typer.BadParameter(f"{label} must be a non-empty comma-separated list without duplicates")
    return values


@app.command()
def main(
    cache: Annotated[Path, typer.Option("--cache", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    variants: Annotated[str, typer.Option("--variants")] = ",".join(DEFAULT_VARIANTS),
    seeds: Annotated[str, typer.Option("--seeds")] = "0,1,2",
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 60,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim", min=1)] = 64,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_entity: Annotated[str | None, typer.Option("--wandb-entity")] = None,
    run_name: Annotated[str, typer.Option("--run-name")] = "phase2e-factorized-shard",
) -> None:
    variant_values = _parse_csv(variants, "variants")
    seed_strings = _parse_csv(seeds, "seeds")
    try:
        seed_values = tuple(int(value) for value in seed_strings)
    except ValueError as error:
        raise typer.BadParameter("seeds must be comma-separated integers") from error
    if len(seed_values) != 3:
        raise typer.BadParameter("formal Phase2e CLI requires exactly three seeds")
    result = run_training_shard(
        cache,
        output,
        variant_values,
        seed_values,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        device_name=device,
        wandb_enabled=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
    )
    typer.echo(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    app()
