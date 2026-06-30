#!/usr/bin/env python3
"""Run one preregistered Phase 2f pilot or formal SUN RGB-D training job."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from jepa4d.evaluation.phase2f_data_cache import (
    ROTATIONS,
    SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA,
    SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA,
    SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA,
    reject_external_target_references,
    rotation_indices,
    validate_sun_development_feature_cache,
    validate_sun_development_input_cache,
    validate_sun_development_target_cache,
)
from jepa4d.evaluation.phase2f_metrics import (
    atomic_json,
    cuda_hardware_identity,
    evaluate_depth_predictions,
    file_identity,
    fit_variance_multiplier,
    require_finite_tree,
    self_contained_html,
)
from jepa4d.models.phase2f_scale_geometry import Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_training import (
    assert_strict_phase2f_reload,
    load_phase2f_checkpoint,
    phase2f_arm_configs,
    save_phase2f_checkpoint,
    train_phase2f_step,
)

ARMS = ("M0", "M1", "M2", "M3")
SEEDS = (0, 1, 2)
TRAINING_RECEIPT_SCHEMA = "jepa4d-phase2f-training-run-v1"
PILOT_GATE_SCHEMA = "jepa4d-phase2f-pilot-qualification-v1"
NORMALIZATION_SCHEMA = "jepa4d-phase2f-rotation-feature-normalization-v1"
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4
BATCH_GROUPS = 8
GRADIENT_CLIP = 5.0
PILOT_EPOCHS = 10
FORMAL_EPOCHS = 60
IMAGE_SIZE = (384, 384)
EXPECTED_PARAMETERS = {"M0": 86_402, "M1": 92_820, "M2": 92_916, "M3": 93_685}


def _load_torch(path: Path) -> dict[str, Any]:
    reject_external_target_references(str(path))
    resolved = path.resolve(strict=True)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"cache must contain a mapping: {path}")
    return value


def load_development_caches(
    input_path: Path,
    feature_path: Path,
    target_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the physically separated SUN caches and prove exact row/hash binding."""

    input_cache = _load_torch(input_path)
    feature_cache = _load_torch(feature_path)
    target_cache = _load_torch(target_path)
    validate_sun_development_input_cache(input_cache)
    validate_sun_development_feature_cache(feature_cache)
    validate_sun_development_target_cache(target_cache)
    if input_cache.get("schema_version") != SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA:
        raise ValueError("unexpected Phase 2f input cache schema")
    if feature_cache.get("schema_version") != SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA:
        raise ValueError("unexpected Phase 2f feature cache schema")
    if target_cache.get("schema_version") != SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA:
        raise ValueError("unexpected Phase 2f target cache schema")
    input_identity = file_identity(input_path, schema=SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA)
    if feature_cache["input_cache_sha256"] != input_identity["sha256"]:
        raise ValueError("feature cache is not hash-bound to the supplied input cache")
    if target_cache["input_cache_sha256"] != input_identity["sha256"]:
        raise ValueError("target cache is not hash-bound to the supplied input cache")
    samples = input_cache["samples"]
    if feature_cache["samples"] != samples or target_cache["samples"] != samples:
        raise ValueError("Phase 2f cache row identities differ")
    if not (
        feature_cache["sample_manifest_sha256"]
        == target_cache["sample_manifest_sha256"]
        == input_cache["sample_manifest_sha256"]
    ):
        raise ValueError("Phase 2f sample-manifest identities differ")
    identities = {
        "input": input_identity,
        "feature": file_identity(feature_path, schema=SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA),
        "target": file_identity(target_path, schema=SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA),
    }
    return input_cache, feature_cache, target_cache, identities


def _tensor_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fit_rotation_normalization(raw_features: torch.Tensor, train_indices: torch.Tensor) -> dict[str, Any]:
    """Fit channel statistics from exactly two train families and both views."""

    train = raw_features.index_select(0, train_indices).float()
    if train.shape != (256, 2, 768, 24, 24) or not bool(torch.isfinite(train).all()):
        raise ValueError("rotation normalization requires exactly 256x2 finite training feature grids")
    mean = train.mean(dim=(0, 1, 3, 4), keepdim=False).reshape(1, 768, 1, 1)
    variance = (train - mean[:, None]).square().mean(dim=(0, 1, 3, 4)).reshape(1, 768, 1, 1)
    std = variance.sqrt().clamp_min(1e-6)
    tensors = {"mean": mean.cpu(), "std": std.cpu()}
    return {
        "schema_version": NORMALIZATION_SCHEMA,
        "fit_rows": 256,
        "views_per_row": 2,
        "channels": 768,
        "spatial_size": [24, 24],
        "method": "per-channel-population-mean-std-train-families-both-views",
        "tensor_sha256": _tensor_sha256(tensors),
        **tensors,
    }


def _normalize(features: torch.Tensor, normalization: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    mean = normalization["mean"].to(device=device, dtype=torch.float32)
    std = normalization["std"].to(device=device, dtype=torch.float32)
    values = (features.to(device=device, dtype=torch.float32) - mean) / std
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("normalized Phase 2f features are non-finite")
    return values


def _uses_camera(arm: str) -> bool:
    return arm in {"M2", "M3"}


def _predict(
    model: Phase2fScaleGeometryProbe,
    features: torch.Tensor,
    intrinsics: torch.Tensor,
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            feature_batch = _normalize(features[offset : offset + batch_size], normalization, device)
            camera = (
                intrinsics[offset : offset + batch_size].to(device).float() if _uses_camera(model.config.arm) else None
            )
            output = model(
                feature_batch,
                intrinsics=camera,
                intrinsics_image_size=IMAGE_SIZE if camera is not None else None,
            )
            means.append(output.log_depth.detach().cpu())
            variances.append(output.log_variance.detach().cpu())
    return torch.cat(means), torch.cat(variances)


def _split_metrics(
    model: Phase2fScaleGeometryProbe,
    indices: torch.Tensor,
    input_cache: Mapping[str, Any],
    feature_cache: Mapping[str, Any],
    target_cache: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
    variance_multiplier: float = 1.0,
) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    features = feature_cache["ordinary_features"].index_select(0, indices)[:, 0]
    intrinsics = input_cache["ordinary_inputs"]["intrinsics_384"].index_select(0, indices)[:, 0]
    target = target_cache["ordinary_targets"]["depth_24"].index_select(0, indices)[:, 0]
    valid = target_cache["ordinary_targets"]["valid_24"].index_select(0, indices)[:, 0]
    means, variances = _predict(model, features, intrinsics, normalization, device=device)
    sample_ids = [input_cache["samples"]["sample_ids"][int(index)] for index in indices]
    family_ids = [input_cache["samples"]["family_ids"][int(index)] for index in indices]
    metrics = evaluate_depth_predictions(
        means,
        variances,
        target,
        valid_mask=valid,
        variance_multiplier=variance_multiplier,
        frame_ids=sample_ids,
        group_ids=family_ids,
    )
    return metrics, (means, variances, target, valid)


def evaluate_camera_controls(
    model: Phase2fScaleGeometryProbe,
    indices: torch.Tensor,
    input_cache: Mapping[str, Any],
    feature_cache: Mapping[str, Any],
    target_cache: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate P1-P7 with identical features/targets and four K conditions."""

    if not _uses_camera(model.config.arm):
        return {}
    features = feature_cache["paired_features"].index_select(0, indices)[:, 1:].flatten(0, 1)
    targets = target_cache["paired_targets"]["depth_24"].index_select(0, indices)[:, 1:].flatten(0, 1)
    valid = target_cache["paired_targets"]["valid_24"].index_select(0, indices)[:, 1:].flatten(0, 1)
    ids = [
        f"{input_cache['samples']['sample_ids'][int(index)]}::P{profile}"
        for index in indices
        for profile in range(1, 8)
    ]
    groups = [input_cache["samples"]["family_ids"][int(index)] for index in indices for _ in range(7)]
    predictions: dict[str, torch.Tensor] = {}
    raw_abs_rel: dict[str, float] = {}
    for control in ("updated", "stale", "wrong", "permuted"):
        camera = input_cache["paired_inputs"][f"{control}_k"].index_select(0, indices)[:, 1:].flatten(0, 1)
        means, variances = _predict(model, features, camera, normalization, device=device)
        predictions[control] = means.exp()
        metrics = evaluate_depth_predictions(
            means,
            variances,
            targets,
            valid_mask=valid,
            frame_ids=ids,
            group_ids=groups,
        )
        raw_abs_rel[control] = float(metrics["group_macro"]["raw_abs_rel"])
    deltas = {
        name: float((predictions["updated"] - predictions[name]).abs().mean())
        for name in ("stale", "wrong", "permuted")
    }
    permutation = input_cache["paired_inputs"]["profile_permutation"].cpu()
    return {
        "profiles": [f"P{index}" for index in range(1, 8)],
        "identity_profile_excluded": True,
        "raw_abs_rel": raw_abs_rel,
        "output_delta_m": deltas,
        "minimum_output_delta_m": min(deltas.values()),
        "permutation_bijective": torch.equal(permutation.sort().values, torch.arange(8)),
        "permutation_change_fraction": float((permutation != torch.arange(8)).float().mean()),
        "equal_family_macro": True,
    }


def _write_normalization(path: Path, normalization: Mapping[str, Any]) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(normalization), temporary)
    temporary.replace(path)
    return path


def _epoch_plot(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    image = Image.new("RGB", (1040, 540), "#f7f8fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Phase 2f training: loss and validation raw AbsRel", fill="#17202e")
    if history:
        loss = np.asarray([float(row["train_total"]) for row in history])
        metric = np.asarray([float(row["validation_raw_abs_rel"]) for row in history])
        for values, color, top, bottom, label in (
            (loss, "#2f6fed", 80, 260, "train total"),
            (metric, "#238636", 310, 490, "validation raw AbsRel"),
        ):
            draw.text((30, top), label, fill="#17202e")
            minimum, maximum = float(values.min()), float(values.max())
            points = []
            for index, value in enumerate(values):
                x = 120 + int(index * 850 / max(1, len(values) - 1))
                fraction = 0.5 if maximum <= minimum else (float(value) - minimum) / (maximum - minimum)
                y = bottom - int(fraction * (bottom - top - 24))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
            draw.text((880, top), f"min {minimum:.5f}", fill=color)
    image.save(path)


def _validate_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    provenance = value.get("execution_provenance", value) if isinstance(value, dict) else None
    if not isinstance(provenance, dict):
        raise TypeError("execution provenance must be a JSON object")
    required = {
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "test_receipt_sha256",
        "dependency_graph_sha256",
        "slurm",
    }
    if not required <= set(provenance) or any(not provenance[name] for name in required):
        raise ValueError("execution provenance is incomplete")

    def reject_secret(item: Any, location: str = "provenance") -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if any(token in str(key).lower() for token in ("token", "secret", "password", "api_key")):
                    raise ValueError(f"credential-like provenance field is forbidden: {location}.{key}")
                reject_secret(child, f"{location}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                reject_secret(child, f"{location}[{index}]")

    reject_secret(provenance)
    require_finite_tree(provenance, "execution_provenance")
    return provenance


def _qualification_allowed(path: Path, arm: str, *, schema: str, allowlist_field: str) -> bool:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("schema_version") != schema or value.get("status") != "pass":
        raise ValueError(f"training requires a passing {schema} receipt")
    allowlist = value.get(allowlist_field)
    if not isinstance(allowlist, list) or "M0" not in allowlist or any(item not in ARMS for item in allowlist):
        raise ValueError("pilot gate formal allowlist is invalid")
    return arm in allowlist


def _finish_wandb(
    run: Any,
    *,
    artifact_name: str,
    job_type: str,
    files: Sequence[Path],
) -> dict[str, Any]:
    import wandb

    run.summary["status"] = "success"
    artifact = wandb.Artifact(artifact_name, type=f"phase2f-{job_type}")
    for path in files:
        artifact.add_file(str(path.resolve(strict=True)), name=path.name)
    logged = run.log_artifact(artifact)
    logged.wait()
    if not all((run.id, run.url, logged.id, logged.version, logged.digest)):
        raise RuntimeError("W&B did not return complete online run/artifact identities")
    receipt = {
        "schema_version": "jepa4d-phase2f-wandb-artifact-receipt-v1",
        "mode": "online",
        "entity": str(run.entity),
        "project": str(run.project),
        "group": str(run.group),
        "job_type": job_type,
        "run_name": str(run.name),
        "run_id": str(run.id),
        "run_url": str(run.url),
        "artifact_name": artifact_name,
        "artifact_version": str(logged.version),
        "artifact_id": str(logged.id),
        "artifact_digest": str(logged.digest),
        "status": "success",
    }
    run.finish(exit_code=0)
    return receipt


def _write_skip(
    output: Path,
    *,
    arm: str,
    rotation: str,
    seed: int,
    stage: str,
    reason: str,
    provenance: Mapping[str, Any],
    run: Any,
) -> None:
    result = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": "skipped_not_qualified",
        "stage": stage,
        "arm": arm,
        "rotation": rotation,
        "seed": seed,
        "optimizer_steps": 0,
        "reason": reason,
        "hardware": cuda_hardware_identity(torch.device("cuda")),
        "execution_provenance": dict(provenance),
    }
    receipt = atomic_json(output / "training_receipt.json", result)
    report = output / "report.html"
    report.write_text(
        self_contained_html(
            f"Phase 2f {stage} skip",
            {"arm": arm, "rotation": rotation, "seed": seed, "optimizer_steps": 0},
            claim_boundary="Predeclared formal job skipped after failed qualification; no optimizer step was run.",
        ),
        encoding="utf-8",
    )
    run.log({"terminal/status": "skipped_not_qualified", "terminal/optimizer_steps": 0})
    wandb_receipt = _finish_wandb(
        run,
        artifact_name=f"phase2f-{stage}-{provenance['execution_id']}-{arm}-{rotation}-s{seed}",
        job_type=stage,
        files=(report,),
    )
    result["wandb"] = wandb_receipt
    atomic_json(receipt, result)
    atomic_json(output / "wandb_receipt.json", wandb_receipt)
    (output / "SUCCESS").write_text("skipped_not_qualified\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--rotation", choices=tuple(ROTATIONS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--latency-gate", type=Path)
    parser.add_argument("--pilot-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2f experiment CLIs may run only inside a Slurm job")
    if args.stage == "pilot" and (args.rotation != "R0" or args.seed != 0):
        raise ValueError("the frozen pilot is R0/seed 0 only")
    if args.stage == "pilot" and args.latency_gate is None:
        raise ValueError("pilot jobs require --latency-gate")
    if args.stage == "formal" and args.pilot_gate is None:
        raise ValueError("formal jobs require --pilot-gate")
    if args.stage == "pilot" and args.pilot_gate is not None:
        raise ValueError("pilot jobs cannot consume a pilot gate")
    if args.stage == "formal" and args.latency_gate is not None:
        raise ValueError("formal jobs consume the pilot gate, not the latency gate directly")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 2f experiment training must run on a Slurm CUDA allocation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = _validate_provenance(args.provenance)
    execution_id = str(provenance["execution_id"])
    slurm_id = str(provenance["slurm"].get("job_id", os.environ.get("SLURM_JOB_ID", "unknown")))
    job_type = "pilot" if args.stage == "pilot" else "formal"
    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"phase2f-{execution_id}",
        job_type=job_type,
        name=f"{execution_id}-{job_type}-{args.arm}-{args.rotation}-s{args.seed}-{slurm_id}",
        mode="online",
        reinit=True,
        config={
            "arm": args.arm,
            "rotation": args.rotation,
            "seed": args.seed,
            "stage": args.stage,
            "epochs": PILOT_EPOCHS if args.stage == "pilot" else FORMAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_source_groups": BATCH_GROUPS,
            "git_commit": provenance["git_commit"],
            "test_receipt_sha256": provenance["test_receipt_sha256"],
            "dependency_graph_sha256": provenance["dependency_graph_sha256"],
        },
    )
    if run is None or run.offline:
        raise RuntimeError("Phase 2f training requires online W&B")
    qualified = True
    skip_reason = ""
    if args.stage == "pilot":
        assert args.latency_gate is not None
        qualified = _qualification_allowed(
            args.latency_gate,
            args.arm,
            schema="jepa4d-phase2f-latency-qualification-v1",
            allowlist_field="qualified_arms",
        )
        skip_reason = "not_in_frozen_latency_allowlist"
    elif args.stage == "formal":
        assert args.pilot_gate is not None
        qualified = _qualification_allowed(
            args.pilot_gate,
            args.arm,
            schema=PILOT_GATE_SCHEMA,
            allowlist_field="formal_allowlist",
        )
        skip_reason = "not_in_frozen_pilot_allowlist"
    if not qualified:
        _write_skip(
            output,
            arm=args.arm,
            rotation=args.rotation,
            seed=args.seed,
            stage=args.stage,
            reason=skip_reason,
            provenance=provenance,
            run=run,
        )
        print(json.dumps({"status": "skipped_not_qualified", "stage": args.stage, "arm": args.arm}, sort_keys=True))
        return

    input_cache, feature_cache, target_cache, cache_identities = load_development_caches(
        args.input_cache, args.feature_cache, args.target_cache
    )
    split_indices = rotation_indices(feature_cache, args.rotation)
    normalization = fit_rotation_normalization(feature_cache["ordinary_features"], split_indices["train"])
    normalization_path = _write_normalization(output / "feature_normalization.pt", normalization)
    normalization_identity = file_identity(normalization_path, schema=NORMALIZATION_SCHEMA)
    normalization_identity["tensor_sha256"] = normalization["tensor_sha256"]
    normalization_identity["fit_families"] = list(ROTATIONS[args.rotation]["train"])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    configs = phase2f_arm_configs(768)
    model = Phase2fScaleGeometryProbe(configs[args.arm]).to(device)
    if model.trainable_parameter_count != EXPECTED_PARAMETERS[args.arm]:
        raise RuntimeError(
            f"{args.arm} parameter count {model.trainable_parameter_count} != frozen {EXPECTED_PARAMETERS[args.arm]}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = PILOT_EPOCHS if args.stage == "pilot" else FORMAL_EPOCHS
    raw_features = feature_cache["ordinary_features"]
    raw_targets = target_cache["ordinary_targets"]["depth_24"]
    raw_valid = target_cache["ordinary_targets"]["valid_24"]
    raw_intrinsics = input_cache["ordinary_inputs"]["intrinsics_384"]
    best_key = (math.inf, math.inf, epochs + 1)
    best_epoch = -1
    checkpoint = output / "checkpoint.pt"
    history: list[dict[str, Any]] = []
    maximum_forbidden = 0.0
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + epoch)
        order = split_indices["train"][torch.randperm(len(split_indices["train"]), generator=generator)]
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for offset in range(0, len(order), BATCH_GROUPS):
            selected = order[offset : offset + BATCH_GROUPS]
            group_count = len(selected)
            features = _normalize(raw_features.index_select(0, selected).flatten(0, 1), normalization, device)
            targets = raw_targets.index_select(0, selected).flatten(0, 1).to(device).float()
            valid = raw_valid.index_select(0, selected).flatten(0, 1).to(device)
            targets = torch.where(valid, targets, torch.ones_like(targets))
            intrinsics = (
                raw_intrinsics.index_select(0, selected).flatten(0, 1).to(device).float()
                if _uses_camera(args.arm)
                else None
            )
            step = train_phase2f_step(
                model,
                optimizer,
                features,
                targets,
                intrinsics=intrinsics,
                intrinsics_image_size=IMAGE_SIZE if intrinsics is not None else None,
                valid_mask=valid,
                group_count=group_count,
                views=2,
                maximum_gradient_norm=GRADIENT_CLIP,
            )
            for name, value in step.metrics.items():
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite training metric {name}")
                sums[name] += value
            forbidden = 0.0 if step.firewall is None else step.firewall.maximum_forbidden_norm
            maximum_forbidden = max(maximum_forbidden, forbidden)
            if forbidden != 0.0:
                raise RuntimeError("Phase 2f strict gradient firewall was violated")
            optimizer_steps += 1
            batches += 1
        validation, _ = _split_metrics(
            model,
            split_indices["validation"],
            input_cache,
            feature_cache,
            target_cache,
            normalization,
            device=device,
        )
        macro = validation["group_macro"]
        key = (float(macro["raw_abs_rel"]), float(macro["absolute_log_scale_error"]), epoch)
        selected_best = key < best_key
        if selected_best:
            best_key = key
            best_epoch = epoch
            save_phase2f_checkpoint(model, checkpoint)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_total": sums["total"] / batches,
            "train_epoch_seconds": time.perf_counter() - epoch_start,
            "learning_rate": LEARNING_RATE,
            "validation_raw_abs_rel": macro["raw_abs_rel"],
            "validation_aligned_abs_rel": macro["aligned_abs_rel"],
            "validation_absolute_log_scale_error": macro["absolute_log_scale_error"],
            "validation_nll": macro["nll"],
            "validation_ause": macro["ause"],
            "checkpoint_selected": selected_best,
            "gradient_firewall_max_forbidden_norm": maximum_forbidden,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        for name, value in sorted(sums.items()):
            row[f"train_{name}"] = value / batches
        require_finite_tree(row, f"epoch[{epoch}]")
        history.append(row)
        run.log(row, step=epoch)
    if best_epoch < 0 or not checkpoint.exists():
        raise RuntimeError("training did not produce a selected checkpoint")

    selected_model, selected_payload = load_phase2f_checkpoint(checkpoint, device=device)
    reloaded, reloaded_payload = load_phase2f_checkpoint(checkpoint, device=device)
    if set(selected_payload["state_dict"]) != set(reloaded_payload["state_dict"]) or any(
        not torch.equal(selected_payload["state_dict"][name], reloaded_payload["state_dict"][name])
        for name in selected_payload["state_dict"]
    ):
        raise RuntimeError("two independent reloads produced different CPU state dictionaries")
    validation_indices = split_indices["validation"][:2]
    fixed_features = _normalize(raw_features.index_select(0, validation_indices)[:, 0], normalization, device)
    fixed_k = (
        raw_intrinsics.index_select(0, validation_indices)[:, 0].to(device).float() if _uses_camera(args.arm) else None
    )
    assert_strict_phase2f_reload(
        selected_model,
        reloaded,
        fixed_features,
        intrinsics=fixed_k,
        intrinsics_image_size=IMAGE_SIZE if fixed_k is not None else None,
    )
    validation_uncalibrated, validation_tensors = _split_metrics(
        reloaded,
        split_indices["validation"],
        input_cache,
        feature_cache,
        target_cache,
        normalization,
        device=device,
    )
    calibration = fit_variance_multiplier(
        validation_tensors[0],
        validation_tensors[1],
        validation_tensors[2],
        validation_tensors[3],
    )
    validation_calibrated, _ = _split_metrics(
        reloaded,
        split_indices["validation"],
        input_cache,
        feature_cache,
        target_cache,
        normalization,
        device=device,
        variance_multiplier=float(calibration["multiplier"]),
    )
    development_test, _ = _split_metrics(
        reloaded,
        split_indices["development_test"],
        input_cache,
        feature_cache,
        target_cache,
        normalization,
        device=device,
        variance_multiplier=float(calibration["multiplier"]),
    )
    camera_controls = evaluate_camera_controls(
        reloaded,
        split_indices["development_test"],
        input_cache,
        feature_cache,
        target_cache,
        normalization,
        device=device,
    )

    history_jsonl = output / "epochs.jsonl"
    history_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in history),
        encoding="utf-8",
    )
    with (output / "epochs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({name for row in history for name in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        output / "curves.npz",
        epoch=np.arange(epochs),
        train_total=np.asarray([row["train_total"] for row in history]),
        validation_raw_abs_rel=np.asarray([row["validation_raw_abs_rel"] for row in history]),
        validation_scale_error=np.asarray([row["validation_absolute_log_scale_error"] for row in history]),
    )
    figure = output / "curves.png"
    _epoch_plot(figure, history)
    metrics_path = atomic_json(
        output / "metrics.json",
        {
            "validation_uncalibrated": validation_uncalibrated,
            "validation": validation_calibrated,
            "development_test": development_test,
            "camera_controls": camera_controls,
        },
    )
    calibration_path = atomic_json(output / "variance_calibration.json", calibration)
    report = output / "report.html"
    report.write_text(
        self_contained_html(
            f"Phase 2f {args.stage}: {args.arm} {args.rotation} seed {args.seed}",
            {
                "best_epoch": best_epoch,
                "dev raw AbsRel": development_test["group_macro"]["raw_abs_rel"],
                "dev scale error": development_test["group_macro"]["absolute_log_scale_error"],
                "forbidden gradient max": maximum_forbidden,
                "parameters": model.trainable_parameter_count,
            },
            images=(("Training and validation curves", figure),),
            claim_boundary="SUN RGB-D development evidence only; no DIODE archive or target was accessed.",
        ),
        encoding="utf-8",
    )
    checkpoint_identity = file_identity(checkpoint, schema="jepa4d-phase2f-checkpoint-v1")
    artifact_name = f"phase2f-{args.stage}-{execution_id}-{args.arm.lower()}-{args.rotation.lower()}-s{args.seed}"
    run.log(
        {
            "terminal/status": "success",
            "terminal/best_epoch": best_epoch,
            "terminal/development_raw_abs_rel": development_test["group_macro"]["raw_abs_rel"],
            "terminal/maximum_forbidden_gradient_norm": maximum_forbidden,
        },
        step=epochs,
    )
    wandb_receipt = _finish_wandb(
        run,
        artifact_name=artifact_name,
        job_type=job_type,
        files=(
            checkpoint,
            normalization_path,
            history_jsonl,
            output / "epochs.csv",
            metrics_path,
            calibration_path,
            figure,
            report,
        ),
    )
    receipt = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": "success",
        "stage": args.stage,
        "arm": args.arm,
        "rotation": args.rotation,
        "seed": args.seed,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": {
            "epochs": epochs,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_source_groups": BATCH_GROUPS,
            "gradient_clip": GRADIENT_CLIP,
            "scheduler": None,
            "early_stopping": False,
            "model": asdict(configs[args.arm]),
        },
        "cache_inputs": cache_identities,
        "checkpoint": checkpoint_identity,
        "feature_normalization": normalization_identity,
        "validation_variance_calibration": {**calibration, **file_identity(calibration_path)},
        "best_epoch": best_epoch,
        "checkpoint_selection_key": list(best_key),
        "metrics": {"validation": validation_calibrated, "development_test": development_test},
        "camera_controls": camera_controls,
        "parameter_counts": reloaded.parameter_counts(),
        "optimizer_steps": optimizer_steps,
        "maximum_forbidden_gradient_norm": maximum_forbidden,
        "finite": True,
        "exact_reload": True,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            **cuda_hardware_identity(device),
            "peak_allocation_bytes": torch.cuda.max_memory_allocated(device),
        },
        "wandb": wandb_receipt,
        "execution_provenance": provenance,
    }
    require_finite_tree(receipt, "training_receipt")
    atomic_json(output / "wandb_receipt.json", wandb_receipt)
    atomic_json(output / "training_receipt.json", receipt)
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "success",
                "arm": args.arm,
                "rotation": args.rotation,
                "seed": args.seed,
                "best_epoch": best_epoch,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
