"""Shared tuning/formal training implementation for Phase 2g-A."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from jepa4d.evaluation.phase2f_metrics import cuda_hardware_identity, file_identity, fit_variance_multiplier
from jepa4d.evaluation.phase2g_data import load_rotation_training_data, write_torch_atomic
from jepa4d.evaluation.phase2g_metrics import evaluate_phase2g_predictions, opaque_frame_id
from jepa4d.evaluation.phase2g_visualization import write_training_visualizations
from jepa4d.models.phase2f_scale_geometry import Phase2fArm, Phase2fGeometryOutput, Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_training import (
    assert_strict_phase2f_reload,
    load_phase2f_checkpoint,
    phase2f_arm_configs,
    save_phase2f_checkpoint,
    train_phase2f_step,
)
from jepa4d.training.phase2g_protocol import (
    ARMS,
    BATCH_SOURCE_GROUPS,
    EXPECTED_PARAMETERS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_STEPS,
    FORMAL_TRAINING_RECEIPT_SCHEMA,
    GRADIENT_CLIP,
    IMAGE_SIZE,
    NORMALIZATION_SCHEMA,
    ROTATIONS,
    SAMPLES_PER_FAMILY,
    TUNING_EPOCHS,
    TUNING_RECEIPT_SCHEMA,
    TUNING_SEED,
    TUNING_STEPS,
    WEIGHT_DECAY,
)
from jepa4d.training.phase2g_runtime import atomic_json, require_safe_finite_tree


def _tensor_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fit_rotation_normalization(feature_rows: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Fit channel population statistics from train-family views only."""

    if not feature_rows:
        raise ValueError("normalization requires train-family feature tensors")
    if any(value.ndim != 5 or value.shape[1:] != (2, 768, 24, 24) for value in feature_rows):
        raise ValueError("normalization rows must have shape [N,2,768,24,24]")
    total_values = 0
    channel_sum = torch.zeros(768, dtype=torch.float64)
    channel_square_sum = torch.zeros(768, dtype=torch.float64)
    for source in feature_rows:
        for offset in range(0, len(source), 16):
            chunk = source[offset : offset + 16].double()
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("normalization features must be finite")
            channel_sum += chunk.sum(dim=(0, 1, 3, 4))
            channel_square_sum += chunk.square().sum(dim=(0, 1, 3, 4))
            total_values += int(chunk.shape[0] * chunk.shape[1] * chunk.shape[3] * chunk.shape[4])
    mean_vector = channel_sum / total_values
    variance_vector = (channel_square_sum / total_values - mean_vector.square()).clamp_min(0)
    mean = mean_vector.float().reshape(1, 768, 1, 1)
    std = variance_vector.sqrt().clamp_min(1e-6).float().reshape(1, 768, 1, 1)
    tensors = {"mean": mean, "std": std}
    rows = sum(len(value) for value in feature_rows)
    return {
        "schema_version": NORMALIZATION_SCHEMA,
        "fit_rows": rows,
        "views_per_row": 2,
        "channels": 768,
        "spatial_size": [24, 24],
        "method": "per-channel-population-mean-std-two-training-families-both-views",
        "tensor_sha256": _tensor_sha256(tensors),
        **tensors,
    }


def validate_normalization(value: Mapping[str, Any], *, expected_rows: int = 2 * SAMPLES_PER_FAMILY) -> None:
    if value.get("schema_version") != NORMALIZATION_SCHEMA or value.get("fit_rows") != expected_rows:
        raise ValueError("unexpected Phase 2g normalization schema/fit rows")
    mean, std = value.get("mean"), value.get("std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError("normalization tensors are missing")
    if mean.shape != (1, 768, 1, 1) or std.shape != mean.shape or not bool(torch.isfinite(mean).all()):
        raise ValueError("normalization tensor shape/values changed")
    if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("normalization standard deviation must be finite and positive")
    if value.get("tensor_sha256") != _tensor_sha256({"mean": mean, "std": std}):
        raise ValueError("normalization tensor SHA-256 mismatch")


def load_normalization(path: Path, *, expected_rows: int = 2 * SAMPLES_PER_FAMILY) -> dict[str, Any]:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("normalization must be a mapping")
    validate_normalization(value, expected_rows=expected_rows)
    return value


def _normalize(features: torch.Tensor, normalization: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    mean = normalization["mean"].to(device=device, dtype=torch.float32)
    std = normalization["std"].to(device=device, dtype=torch.float32)
    result = (features.to(device=device, dtype=torch.float32) - mean) / std
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("normalized Phase 2g features are non-finite")
    return result


def _uses_camera(arm: str) -> bool:
    return arm in {"M2", "M3"}


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _parameter_delta_norm(model: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().cpu().double() - before[name].double()
        squared += float(delta.square().sum())
    return math.sqrt(squared)


def _state_changed(model: torch.nn.Module, initial: Mapping[str, torch.Tensor]) -> bool:
    return any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters())


def _concatenate_role(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not bundles:
        raise ValueError("Phase 2g role has no family bundles")
    return {
        "sample_ids": [sample_id for bundle in bundles for sample_id in bundle["input"]["samples"]["sample_ids"]],
        "family_ids": [
            bundle["input"]["samples"]["family"]
            for bundle in bundles
            for _ in bundle["input"]["samples"]["sample_ids"]
        ],
        "features": torch.cat([bundle["feature"]["ordinary_features"] for bundle in bundles]),
        "targets": torch.cat([bundle["target"]["ordinary_targets"]["depth_24"] for bundle in bundles]),
        "valid": torch.cat([bundle["target"]["ordinary_targets"]["valid_24"] for bundle in bundles]),
        "intrinsics": torch.cat([bundle["input"]["ordinary_inputs"]["intrinsics_384"] for bundle in bundles]),
    }


def _predict(
    model: Phase2fScaleGeometryProbe,
    features: torch.Tensor,
    intrinsics: torch.Tensor,
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, torch.Tensor | None]:
    values: dict[str, list[torch.Tensor]] = {
        "log_depth": [],
        "log_variance": [],
        "centered_shape": [],
        "global_log_scale": [],
        "scale_field": [],
    }
    presence: dict[str, bool | None] = {name: None for name in values}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = _normalize(features[offset : offset + batch_size], normalization, device)
            camera = (
                intrinsics[offset : offset + batch_size].to(device).float() if _uses_camera(model.config.arm) else None
            )
            output: Phase2fGeometryOutput = model(
                batch,
                intrinsics=camera,
                intrinsics_image_size=IMAGE_SIZE if camera is not None else None,
            )
            for name in values:
                item = getattr(output, name)
                exists = item is not None
                if presence[name] is None:
                    presence[name] = exists
                elif presence[name] != exists:
                    raise RuntimeError(f"model output presence changed for {name}")
                if item is not None:
                    values[name].append(item.detach().cpu())
    return {name: torch.cat(rows) if rows else None for name, rows in values.items()}


def _evaluate_role(
    model: Phase2fScaleGeometryProbe,
    role: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
    variance_multiplier: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    features = role["features"][:, 0]
    intrinsics = role["intrinsics"][:, 0]
    targets = role["targets"][:, 0].float()
    valid = role["valid"][:, 0]
    predictions = _predict(model, features, intrinsics, normalization, device=device)
    metrics = evaluate_phase2g_predictions(
        predictions["log_depth"],  # type: ignore[arg-type]
        predictions["log_variance"],  # type: ignore[arg-type]
        targets,
        valid_mask=valid,
        variance_multiplier=variance_multiplier,
        frame_ids=[opaque_frame_id(value) for value in role["sample_ids"]],
        family_ids=role["family_ids"],
    )
    predictions["target_depth"] = targets
    predictions["valid_mask"] = valid
    return metrics, predictions


def _train_scale_median(role: Mapping[str, Any]) -> float:
    targets = role["targets"][:, 0].double()
    valid = role["valid"][:, 0]
    frame_medians = [float(targets[index][valid[index]].log().median()) for index in range(len(targets))]
    return float(np.median(frame_medians))


def _allowed_gradient_checks(arm: str, metrics: Mapping[str, float]) -> dict[str, bool]:
    checks = {"shape": float(metrics["gradient_norm_shape"]) > 0.0}
    if arm in {"M1", "M2", "M3"}:
        checks["scale"] = float(metrics["gradient_norm_scale"]) > 0.0
    if arm == "M3":
        checks["field"] = float(metrics["gradient_norm_field"]) > 0.0
    return checks


def run_training_cell(
    *,
    stage: str,
    arm: str,
    rotation: str,
    seed: int,
    learning_rate: float,
    cache_root: Path,
    output: Path,
    provenance: Mapping[str, Any],
    device: torch.device,
    wandb_run: Any,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Run one exact tuning or formal cell; held-out targets are unavailable."""

    if stage not in {"tuning", "formal"} or arm not in ARMS or rotation not in ROTATIONS:
        raise ValueError("invalid Phase 2g training cell identity")
    if stage == "tuning" and seed != TUNING_SEED:
        raise ValueError("tuning seed differs from the frozen protocol")
    if stage == "formal" and seed not in FORMAL_SEEDS:
        raise ValueError("formal seed differs from the frozen protocol")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase 2g training requires an allocated CUDA device")
    bundles, descriptor = load_rotation_training_data(cache_root, rotation)
    train = _concatenate_role(bundles["train"])
    validation = _concatenate_role(bundles["validation"])
    if len(train["sample_ids"]) != 2 * SAMPLES_PER_FAMILY or len(validation["sample_ids"]) != SAMPLES_PER_FAMILY:
        raise ValueError("formal Phase 2g rotation view has incorrect train/validation counts")
    if descriptor["heldout_family"] in set(train["family_ids"] + validation["family_ids"]):
        raise RuntimeError("held-out family leaked into a training role")

    normalization = fit_rotation_normalization([bundle["feature"]["ordinary_features"] for bundle in bundles["train"]])
    validate_normalization(normalization)
    normalization_path = write_torch_atomic(output / "feature_normalization.pt", normalization)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    configs = phase2f_arm_configs(768)
    arm_key = cast(Phase2fArm, arm)
    model = Phase2fScaleGeometryProbe(configs[arm_key]).to(device)
    if model.trainable_parameter_count != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError("Phase 2g architecture parameter identity changed")
    initial = _parameter_snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    epochs = TUNING_EPOCHS if stage == "tuning" else FORMAL_EPOCHS
    expected_steps = TUNING_STEPS if stage == "tuning" else FORMAL_STEPS
    checkpoint = output / "checkpoint.pt"
    history: list[dict[str, Any]] = []
    step_objectives: list[float] = []
    best_key = (math.inf, math.inf, epochs + 1)
    best_epoch = -1
    best_model: Phase2fScaleGeometryProbe | None = None
    optimizer_steps = 0
    maximum_forbidden = 0.0
    allowed_seen: defaultdict[str, bool] = defaultdict(bool)
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + epoch)
        order = torch.randperm(len(train["sample_ids"]), generator=generator)
        sums: defaultdict[str, float] = defaultdict(float)
        update_norm_sum = 0.0
        batches = 0
        for offset in range(0, len(order), BATCH_SOURCE_GROUPS):
            selected = order[offset : offset + BATCH_SOURCE_GROUPS]
            before = _parameter_snapshot(model)
            features = _normalize(train["features"].index_select(0, selected).flatten(0, 1), normalization, device)
            targets = train["targets"].index_select(0, selected).flatten(0, 1).to(device).float()
            valid = train["valid"].index_select(0, selected).flatten(0, 1).to(device)
            targets = torch.where(valid, targets, torch.ones_like(targets))
            camera = (
                train["intrinsics"].index_select(0, selected).flatten(0, 1).to(device).float()
                if _uses_camera(arm)
                else None
            )
            step = train_phase2f_step(
                model,
                optimizer,
                features,
                targets,
                intrinsics=camera,
                intrinsics_image_size=IMAGE_SIZE if camera is not None else None,
                valid_mask=valid,
                group_count=len(selected),
                views=2,
                maximum_gradient_norm=GRADIENT_CLIP,
                firewall_tolerance=0.0,
            )
            for name, value in step.metrics.items():
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite training metric: {name}")
                sums[name] += value
            objective = float(step.metrics["total"])
            step_objectives.append(objective)
            forbidden = step.firewall.maximum_forbidden_norm if step.firewall is not None else math.inf
            if forbidden != 0.0:
                raise RuntimeError("Phase 2g zero-tolerance gradient firewall failed")
            maximum_forbidden = max(maximum_forbidden, forbidden)
            for name, passed in _allowed_gradient_checks(arm, step.metrics).items():
                allowed_seen[name] = allowed_seen[name] or passed
            update_norm_sum += _parameter_delta_norm(model, before)
            optimizer_steps += 1
            batches += 1
        validation_metrics, _ = _evaluate_role(
            model,
            validation,
            normalization,
            device=device,
            variance_multiplier=1.0,
        )
        macro = validation_metrics["equal_family_macro"]
        key = (float(macro["raw_abs_rel"]), float(macro["absolute_log_scale_error"]), epoch)
        selected_best = key < best_key
        if selected_best:
            best_key, best_epoch = key, epoch
            save_phase2f_checkpoint(model, checkpoint)
            best_model = copy.deepcopy(model)
        elapsed = time.perf_counter() - epoch_started
        row: dict[str, Any] = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "train_total": sums["total"] / batches,
            "train_epoch_seconds": elapsed,
            "throughput_source_groups_per_second": len(order) / elapsed,
            "learning_rate": learning_rate,
            "parameter_update_norm": update_norm_sum / batches,
            "validation_raw_abs_rel": macro["raw_abs_rel"],
            "validation_aligned_abs_rel": macro["aligned_abs_rel"],
            "validation_absolute_log_scale_error": macro["absolute_log_scale_error"],
            "validation_signed_log_scale_error": macro["signed_log_scale_error"],
            "validation_nll": macro["nll"],
            "validation_ause": macro["ause"],
            "validation_raw_rmse": macro["raw_rmse"],
            "validation_delta1": macro["delta1"],
            "validation_reliability_error": macro["reliability_error"],
            "validation_coverage_50": macro["coverage_50"],
            "validation_coverage_80": macro["coverage_80"],
            "validation_coverage_90": macro["coverage_90"],
            "validation_coverage_95": macro["coverage_95"],
            "checkpoint_selected": selected_best,
            "checkpoint_rank_key": list(key),
            "gradient_firewall_max_forbidden_norm": maximum_forbidden,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        for name, value in sorted(sums.items()):
            row[f"train_{name}"] = value / batches
        require_safe_finite_tree(row, location=f"epoch[{epoch}]")
        history.append(row)
        wandb_run.log(row, step=epoch)
    if optimizer_steps != expected_steps or best_epoch < 0 or best_model is None or not checkpoint.is_file():
        raise RuntimeError("Phase 2g training did not complete its frozen steps/checkpoint selection")
    decile = max(1, math.ceil(len(step_objectives) * 0.1))
    first_decile = float(np.mean(step_objectives[:decile]))
    final_decile = float(np.mean(step_objectives[-decile:]))
    health = {
        "all_finite": True,
        "maximum_forbidden_gradient_norm": maximum_forbidden,
        "allowed_gradient_seen": dict(allowed_seen),
        "all_expected_allowed_gradients_seen": all(allowed_seen.values()),
        "model_changed_from_initialization": _state_changed(model, initial),
        "first_objective_decile_mean": first_decile,
        "final_objective_decile_mean": final_decile,
        "objective_decreased": final_decile < first_decile,
    }
    if not all(
        (
            health["maximum_forbidden_gradient_norm"] == 0.0,
            health["all_expected_allowed_gradients_seen"],
            health["model_changed_from_initialization"],
            health["objective_decreased"],
        )
    ):
        raise RuntimeError(f"Phase 2g training health gate failed: {health}")

    reloaded, _ = load_phase2f_checkpoint(checkpoint, device=device)
    fixed_features = _normalize(validation["features"][:2, 0], normalization, device)
    fixed_k = validation["intrinsics"][:2, 0].to(device).float() if _uses_camera(arm) else None
    assert_strict_phase2f_reload(
        best_model,
        reloaded,
        fixed_features,
        intrinsics=fixed_k,
        intrinsics_image_size=IMAGE_SIZE if fixed_k is not None else None,
    )
    uncalibrated, tensors = _evaluate_role(
        reloaded,
        validation,
        normalization,
        device=device,
        variance_multiplier=1.0,
    )
    calibration = fit_variance_multiplier(
        tensors["log_depth"],  # type: ignore[arg-type]
        tensors["log_variance"],  # type: ignore[arg-type]
        tensors["target_depth"],  # type: ignore[arg-type]
        tensors["valid_mask"],  # type: ignore[arg-type]
    )
    calibrated, _ = _evaluate_role(
        reloaded,
        validation,
        normalization,
        device=device,
        variance_multiplier=float(calibration["multiplier"]),
    )
    history_jsonl = output / "epochs.jsonl"
    history_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in history), encoding="utf-8"
    )
    history_csv = output / "epochs.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({name for row in history for name in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    atomic_json(
        output / "validation_metrics.json",
        {"uncalibrated": uncalibrated, "calibrated": calibrated},
    )
    calibration_path = atomic_json(output / "variance_calibration.json", calibration)
    training_figure, resource_figure, training_report = write_training_visualizations(output, history)
    normalization_identity = file_identity(normalization_path, schema=NORMALIZATION_SCHEMA)
    normalization_identity.update(
        {
            "tensor_sha256": normalization["tensor_sha256"],
            "fit_families": list(ROTATIONS[rotation]["train"]),
            "heldout_family_absent": descriptor["heldout_family"],
        }
    )
    receipt_schema = TUNING_RECEIPT_SCHEMA if stage == "tuning" else FORMAL_TRAINING_RECEIPT_SCHEMA
    receipt = {
        "schema_version": receipt_schema,
        "status": "success",
        "stage": stage,
        "arm": arm,
        "rotation": rotation,
        "seed": seed,
        "learning_rate": learning_rate,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": {
            "epochs": epochs,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "batch_source_groups": BATCH_SOURCE_GROUPS,
            "views_per_source_group": 2,
            "gradient_clip": GRADIENT_CLIP,
            "scheduler": None,
            "early_stopping": False,
            "model": asdict(configs[arm_key]),
        },
        "rotation_view": {
            "schema_version": descriptor["schema_version"],
            "view_sha256": descriptor["view_sha256"],
            "membership_sha256": descriptor["membership_sha256"],
            "train_families": descriptor["train_families"],
            "validation_family": descriptor["validation_family"],
            "heldout_family": descriptor["heldout_family"],
            "heldout_target_exposed": False,
        },
        "checkpoint": file_identity(checkpoint, schema="jepa4d-phase2f-checkpoint-v1"),
        "feature_normalization": normalization_identity,
        "validation_variance_calibration": {**calibration, **file_identity(calibration_path)},
        "train_median_log_scale": _train_scale_median(train),
        "best_epoch": best_epoch,
        "checkpoint_selection_key": list(best_key),
        "validation_metrics": calibrated,
        "epoch_diagnostics": [
            {
                "epoch": row["epoch"],
                "train_total": row["train_total"],
                "allowed_gradient_norm_shape": row.get("train_gradient_norm_shape", 0.0),
                "allowed_gradient_norm_scale": row.get("train_gradient_norm_scale", 0.0),
                "allowed_gradient_norm_field": row.get("train_gradient_norm_field", 0.0),
                "forbidden_gradient_norm": row["gradient_firewall_max_forbidden_norm"],
                "epoch_seconds": row["train_epoch_seconds"],
                "throughput_source_groups_per_second": row["throughput_source_groups_per_second"],
                "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": row["peak_cuda_reserved_bytes"],
            }
            for row in history
        ],
        "health": health,
        "parameter_counts": reloaded.parameter_counts(),
        "optimizer_steps": optimizer_steps,
        "exact_reload": True,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            **cuda_hardware_identity(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "resource_metrics_are_descriptive_only": True,
        "external_final_authorized": False,
        "execution_provenance": dict(provenance),
    }
    require_safe_finite_tree(receipt, location="training_receipt")
    atomic_json(output / "training_receipt.json", receipt)
    return receipt, (
        history_jsonl,
        history_csv,
        training_figure,
        resource_figure,
        training_report,
    )
