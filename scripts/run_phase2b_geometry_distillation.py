"""Train and compare Phase 2b RGB/V-JEPA geometry probes on an immutable split."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from PIL import Image

from jepa4d.benchmarks.geometry.tum_rgbd import depth_metrics, load_tum_indices, validate_archive
from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.comparison import ComparisonRecord, VariantResult
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.geometry_student import DenseGeometryProbe, geometry_probe_loss, rgb_grid_features
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor

app = typer.Typer(add_completion=False)


def _write_json(path: Path, payload: Any) -> None:
    """Atomically persist a JSON artifact so interrupted jobs never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    return (result.stdout or result.stderr).strip()


def _environment_snapshot(device: str) -> dict[str, Any]:
    git_diff = _command(["git", "diff", "--binary", "HEAD"])
    gpu: dict[str, Any] = {}
    if torch.cuda.is_available():
        index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_gb": properties.total_memory / 2**30,
            "uuid": getattr(properties, "uuid", None),
        }
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,pstate",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "git_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
    }


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _log_event(output: Path, stage: str, **values: Any) -> None:
    row = {"timestamp_utc": datetime.now(UTC).isoformat(), "stage": stage, **values}
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    typer.echo(json.dumps(row, sort_keys=True, allow_nan=False))


def _checkpoint_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    patterns = ("*.safetensors", "*.pt", "*.pth", "*.json", "*.py")
    for name, root in paths.items():
        if not root.exists():
            raise FileNotFoundError(f"required asset is missing: {root}")
        files = [root] if root.is_file() else sorted({path for pattern in patterns for path in root.rglob(pattern)})
        if not files:
            raise FileNotFoundError(f"required asset contains no recognized files: {root}")
        manifest[name] = {
            "path": str(root.resolve()),
            "files": [
                {
                    "path": str(path.relative_to(root) if root.is_dir() else path.name),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            ],
        }
    return manifest


def _dataset_fingerprint(dataset_root: Path, splits: dict[str, list[Any]], archive: Path) -> dict[str, Any]:
    root = dataset_root.resolve()
    required = [root / name for name in ("rgb.txt", "depth.txt", "groundtruth.txt")]
    selected = sorted(
        {item.rgb_path.resolve() for values in splits.values() for item in values}
        | {item.depth_path.resolve() for values in splits.values() for item in values}
    )
    for path in required + selected:
        if not path.is_file():
            raise FileNotFoundError(f"dataset file is missing: {path}")
        if not path.is_relative_to(root):
            raise ValueError(f"dataset sample escapes root {root}: {path}")
    extracted = {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in required + selected
    }
    wanted_members = {f"{root.name}/{relative}": relative for relative in extracted}
    archive_members: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle:
            relative = wanted_members.get(member.name.removeprefix("./"))
            if relative is None:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"unable to read required archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            archive_members[relative] = {"bytes": size, "sha256": digest.hexdigest()}
    missing = sorted(set(extracted) - set(archive_members))
    mismatched = sorted(relative for relative in archive_members if archive_members[relative] != extracted[relative])
    if missing or mismatched:
        raise ValueError(f"dataset root does not match verified archive: missing={missing}, mismatched={mismatched}")
    return {
        "root": str(root),
        "archive": {"path": str(archive.resolve()), "bytes": archive.stat().st_size, "sha256": _sha256(archive)},
        "index_files": {path.name: extracted[str(path.relative_to(root))] for path in required},
        "selected_files": [
            {"path": str(path.relative_to(root)), **extracted[str(path.relative_to(root))]} for path in selected
        ],
        "selected_file_count": len(selected),
        "splits": {
            name: [
                {
                    "sample_id": item.sample_id,
                    "timestamp": item.timestamp,
                    "rgb": str(item.rgb_path.resolve().relative_to(root)),
                    "depth": str(item.depth_path.resolve().relative_to(root)),
                }
                for item in values
            ]
            for name, values in splits.items()
        },
        "archive_extraction_verified": True,
    }


def _center_crop_square(values: torch.Tensor) -> torch.Tensor:
    height, width = values.shape[-2:]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return values[..., top : top + size, left : left + size]


def _images(samples: list[Any], size: int = 384) -> torch.Tensor:
    values = [
        torch.from_numpy(np.asarray(Image.open(item.rgb_path).convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)
        for item in samples
    ]
    cropped = torch.stack([_center_crop_square(value) for value in values]).float() / 255.0
    return F.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)


def _targets(samples: list[Any], size: tuple[int, int]) -> torch.Tensor:
    values = []
    for item in samples:
        raw = np.asarray(Image.open(item.depth_path), dtype=np.uint16).copy()
        depth = torch.from_numpy(raw.astype(np.float32) / 5000.0)
        values.append(_center_crop_square(depth))
    return F.interpolate(torch.stack(values).unsqueeze(1), size=size, mode="nearest")[:, 0]


def _single_image_batch(samples: list[Any], *, size: int = 384) -> Any:
    images = _images(samples, size=size)
    return collate_rgb_inputs([from_view_sequences([[image]]) for image in images])


def _valid(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target) & (target > 0.1) & (target < 10.0)


def _extract_vjepa(
    extractor: VJEPA21FeatureExtractor, samples: list[Any], chunk_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    layers: dict[int, list[torch.Tensor]] = defaultdict(list)
    final: list[torch.Tensor] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(samples), chunk_size):
            chunk = samples[offset : offset + chunk_size]
            batch = _single_image_batch(chunk)
            bundle = extractor(batch)
            final.append(bundle.dense_tokens[:, 0, 0].detach().cpu())
            for layer, value in bundle.layer_tokens.items():
                layers[layer].append(value[:, 0, 0].detach().cpu())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    def grid(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.reshape(len(samples), 24, 24, -1).permute(0, 3, 1, 2).contiguous().half()

    final_grid = grid(torch.cat(final))
    layer_grids = {f"vjepa_layer_{layer}": grid(torch.cat(layers[layer])) for layer in sorted(layers)}
    return (
        {"vjepa_final": final_grid, **layer_grids},
        {
            "total_seconds": elapsed,
            "per_frame_ms": elapsed * 1000.0 / len(samples),
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        },
    )


def _teacher_depth(
    head: GeometryBeliefHead, samples: list[Any], chunk_size: int
) -> tuple[torch.Tensor, dict[str, float]]:
    predictions = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(samples), chunk_size):
            chunk = samples[offset : offset + chunk_size]
            belief = head(_single_image_batch(chunk, size=518))
            assert belief.depth_mean is not None
            depth = belief.depth_mean[:, 0, 0].float().cpu()
            predictions.append(depth)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return torch.cat(predictions), {
        "total_seconds": elapsed,
        "per_frame_ms": elapsed * 1000.0 / len(samples),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def _normalize(
    train: torch.Tensor, validation: torch.Tensor, test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mean = train.float().mean(dim=(0, 2, 3), keepdim=True)
    std = train.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    return (
        ((train.float() - mean) / std).half(),
        ((validation.float() - mean) / std).half(),
        ((test.float() - mean) / std).half(),
        {"mean": mean.cpu(), "std": std.cpu()},
    )


def _normalize_multilayer(
    train: dict[str, torch.Tensor], validation: dict[str, torch.Tensor], test: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
    keys = sorted(key for key in train if key.startswith("vjepa_layer_"))
    if len(keys) != 4:
        raise ValueError(f"expected four V-JEPA hierarchy layers, found {keys}")
    normalized_train = []
    normalized_validation = []
    normalized_test = []
    statistics: dict[str, dict[str, torch.Tensor]] = {}
    for key in keys:
        values = _normalize(train[key], validation[key], test[key])
        normalized_train.append(values[0])
        normalized_validation.append(values[1])
        normalized_test.append(values[2])
        statistics[key] = values[3]
    # Averaging standardized layers keeps the candidate at the same 768 input
    # channels and therefore exactly parameter-matches the final-layer probe.
    return (
        torch.stack(normalized_train).mean(dim=0),
        torch.stack(normalized_validation).mean(dim=0),
        torch.stack(normalized_test).mean(dim=0),
        statistics,
    )


def _raw_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = _valid(target)
    values = predicted[mask]
    if int(mask.sum()) < 100:
        raise ValueError("fewer than 100 valid target pixels")
    if not torch.isfinite(values).all() or not (values > 0).all():
        invalid = int((~torch.isfinite(values) | (values <= 0)).sum())
        raise ValueError(f"prediction has {invalid} non-finite or non-positive values on valid target pixels")
    prediction, truth = predicted[mask], target[mask]
    error = prediction - truth
    ratio = torch.maximum(prediction / truth, truth / prediction.clamp_min(1e-8))
    return {
        "metric_abs_rel": float((error.abs() / truth).mean()),
        "metric_rmse_m": float(torch.sqrt(error.square().mean())),
        "metric_delta_1": float((ratio < 1.25).float().mean()),
    }


def _median_align(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Align relative predictions to GT only where protocol explicitly permits it."""
    aligned = []
    for prediction, target in zip(predictions, targets, strict=True):
        mask = _valid(target)
        if int(mask.sum()) < 100:
            raise ValueError("fewer than 100 valid pixels for median alignment")
        if not torch.isfinite(prediction[mask]).all() or not (prediction[mask] > 0).all():
            raise ValueError("invalid prediction values on target-defined mask")
        scale = target[mask].median() / prediction[mask].median().clamp_min(1e-8)
        aligned.append(prediction * scale)
    return torch.stack(aligned)


def _fit_metric_scale(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Fit one frozen teacher scale using training pixels only."""
    prediction_values = []
    target_values = []
    for prediction, target in zip(predictions, targets, strict=True):
        mask = _valid(target)
        values = prediction[mask]
        if not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError("teacher scale calibration received invalid predictions")
        prediction_values.append(values)
        target_values.append(target[mask])
    prediction = torch.cat(prediction_values)
    target = torch.cat(target_values)
    return float(target.median() / prediction.median().clamp_min(1e-8))


def _evaluate_depths(
    predictions: torch.Tensor, targets: torch.Tensor, *, include_metric: bool = True
) -> dict[str, float]:
    rows = []
    for predicted, target in zip(predictions, targets, strict=True):
        # Validate on the target-defined mask before calling the aligned helper,
        # which historically excluded invalid predictions from its denominator.
        _raw_metrics(predicted, target)
        aligned, _, _ = depth_metrics(predicted, target)
        row = {f"aligned_{key}": value for key, value in aligned.items()}
        if include_metric:
            row.update(_raw_metrics(predicted, target))
        rows.append(row)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _per_frame_rows(
    variant: str,
    seed: int | None,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    samples: list[Any],
) -> list[dict[str, Any]]:
    rows = []
    for prediction, target, sample in zip(predictions, targets, samples, strict=True):
        raw = _raw_metrics(prediction, target)
        aligned, alignment_scale, _ = depth_metrics(prediction, target)
        valid = _valid(target)
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "frame_id": sample.sample_id,
                "timestamp": sample.timestamp,
                "valid_target_fraction": float(valid.float().mean()),
                "prediction_coverage_on_valid_target": 1.0,
                "prediction_mean_m": float(prediction[valid].mean()),
                "target_mean_m": float(target[valid].mean()),
                "alignment_scale": alignment_scale,
                **raw,
                **{f"aligned_{key}": value for key, value in aligned.items()},
            }
        )
    return rows


def _calibrate_log_variance(
    model: DenseGeometryProbe,
    validation_features: torch.Tensor,
    validation_target: torch.Tensor,
    test_features: torch.Tensor,
    test_target: torch.Tensor,
    device: str,
) -> tuple[float, float, float]:
    model.eval()
    with torch.inference_mode():
        val_log_depth, val_logvar = model(validation_features.to(device))
        test_log_depth, test_logvar = model(test_features.to(device))
    val_truth = validation_target.to(device).clamp_min(1e-4).log()
    test_truth = test_target.to(device).clamp_min(1e-4).log()
    val_mask, test_mask = _valid(validation_target).to(device), _valid(test_target).to(device)
    multiplier = float(
        (((val_log_depth - val_truth).square() / val_logvar.exp().clamp_min(1e-8))[val_mask]).mean().clamp(1e-4, 1e4)
    )
    residual = (test_log_depth - test_truth)[test_mask]
    raw_variance = test_logvar.exp()[test_mask].clamp_min(1e-8)
    calibrated_variance = raw_variance * multiplier
    raw_nll = 0.5 * (raw_variance.log() + residual.square() / raw_variance)
    calibrated_nll = 0.5 * (calibrated_variance.log() + residual.square() / calibrated_variance)
    return multiplier, float(raw_nll.mean()), float(calibrated_nll.mean())


def _head_latency(model: DenseGeometryProbe, features: torch.Tensor, device: str) -> float:
    value = features[:1].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(value)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            model(value)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 10.0


def _train_variant(
    variant: str,
    seed: int,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    test_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_target: torch.Tensor,
    test_target_518: torch.Tensor,
    teacher_target: torch.Tensor,
    output: Path,
    device: str,
    epochs: int,
    run: Any,
    encoder_runtime: dict[str, float],
) -> tuple[VariantResult, list[dict[str, Any]], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    model = DenseGeometryProbe(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_score = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    history_path = output / "histories" / f"{variant}-seed{seed}.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 8
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_features), generator=generator)
        weighted_loss_sum = 0.0
        valid_pixel_count = 0
        component_sums: dict[str, float] = defaultdict(float)
        gradient_norms: list[float] = []
        for offset in range(0, len(order), batch_size):
            index = order[offset : offset + batch_size]
            features = train_features[index].to(device)
            target = train_target[index].to(device)
            teacher = teacher_target[index].to(device)
            batch_valid_pixels = int(_valid(target).sum())
            optimizer.zero_grad(set_to_none=True)
            log_depth, logvar = model(features)
            loss, parts = geometry_probe_loss(log_depth, logvar, target, _valid(target), teacher_depth=teacher)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            weighted_loss_sum += float(loss.detach()) * batch_valid_pixels
            valid_pixel_count += batch_valid_pixels
            gradient_norms.append(float(gradient_norm.detach()))
            for key, value in parts.items():
                component_sums[key] += float(value) * batch_valid_pixels
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_features.to(device))[0].exp().cpu()
        validation_absrel = _evaluate_depths(validation_prediction, validation_target)["metric_abs_rel"]
        if validation_absrel < best_score:
            best_score = validation_absrel
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.cuda.synchronize()
        row: dict[str, Any] = {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "loss": weighted_loss_sum / valid_pixel_count,
            "validation_metric_abs_rel": validation_absrel,
            "gradient_norm": float(np.mean(gradient_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_validation_metric_abs_rel": best_score,
            "is_best": epoch == best_epoch,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
        row.update({key: value / valid_pixel_count for key, value in component_sums.items()})
        history.append(row)
        with history_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if run is not None:
            prefix = f"training/{variant}/seed_{seed}"
            if epoch == 0:
                run.define_metric(f"{prefix}/epoch")
                run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")
            run.log(
                {
                    f"{prefix}/epoch": epoch,
                    f"{prefix}/loss": row["loss"],
                    f"{prefix}/validation_abs_rel": validation_absrel,
                    f"{prefix}/gradient_norm": row["gradient_norm"],
                    f"{prefix}/epoch_seconds": row["epoch_seconds"],
                    f"{prefix}/gpu_allocated_gb": row["gpu_allocated_gb"],
                    f"{prefix}/gpu_reserved_gb": row["gpu_reserved_gb"],
                    f"{prefix}/nll": row["nll"],
                    f"{prefix}/scale_invariant": row["scale_invariant"],
                    f"{prefix}/gradient": row["gradient"],
                    f"{prefix}/distillation": row["distillation"],
                }
            )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            _log_event(
                output,
                "training_epoch",
                variant=variant,
                seed=seed,
                epoch=epoch,
                loss=row["loss"],
                validation_metric_abs_rel=validation_absrel,
                best_validation_metric_abs_rel=best_score,
            )
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    peak_training_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / f"{variant}-seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "seed": seed,
            "input_dim": train_features.shape[1],
            "state_dict": best_state,
            "validation_abs_rel": best_score,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    model.eval()
    with torch.inference_mode():
        validation_log_depth, validation_logvar = model(validation_features.to(device))
        test_log_depth, test_logvar = model(test_features.to(device))
        test_prediction = F.interpolate(
            test_log_depth.exp().unsqueeze(1), size=test_target_518.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0].cpu()
    metrics = _evaluate_depths(test_prediction, test_target_518)
    multiplier, raw_nll, calibrated_nll = _calibrate_log_variance(
        model,
        validation_features,
        validation_target,
        test_features,
        F.interpolate(test_target_518.unsqueeze(1), size=test_features.shape[-2:], mode="nearest")[:, 0],
        device,
    )
    metrics.update(
        {
            "validation_metric_abs_rel": best_score,
            "variance_multiplier": multiplier,
            "raw_log_depth_nll": raw_nll,
            "calibrated_log_depth_nll": calibrated_nll,
        }
    )
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    head_ms = _head_latency(model, test_features, device)
    peak_head_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    trainable_parameters = sum(value.numel() for value in model.parameters())
    encoder_parameters = int(encoder_runtime.get("parameters", 0))
    parameters = trainable_parameters + encoder_parameters
    family = "rgb" if variant == "rgb_probe" else "vjepa"
    role = "non_jepa_baseline" if variant == "rgb_probe" else ("ablation" if variant == "vjepa_final" else "ours")
    notes = [
        "Best checkpoint selected only on validation metric AbsRel.",
        "VGGT training-scale auxiliary loss weight=0.25; test targets are never used for training or selection.",
    ]
    if variant == "rgb_probe":
        notes.append("Non-JEPA representation baseline uses the same VGGT-assisted supervision as the JEPA probes.")
    result = VariantResult(
        variant_id=variant,
        family=family,
        role=role,
        seed=seed,
        metrics=metrics,
        runtime={
            "encoder_ms_per_frame": encoder_runtime["per_frame_ms"],
            "head_ms_per_frame": head_ms,
            "total_ms_per_frame": encoder_runtime["per_frame_ms"] + head_ms,
            "peak_encoder_memory_gb": encoder_runtime["peak_memory_gb"],
            "peak_head_memory_gb": peak_head_memory_gb,
            "peak_training_memory_gb": peak_training_memory_gb,
            "training_seconds": training_seconds,
            "encoder_model_load_seconds": encoder_runtime.get("model_load_seconds", 0.0),
        },
        parameters=parameters,
        trainable_parameters=trainable_parameters,
        encoder_parameters=encoder_parameters,
        checkpoint=str(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
        notes=notes,
    )
    diagnostics = {
        "prediction_m": test_prediction,
        "target_m": test_target_518.cpu(),
        "log_variance_24": test_logvar.cpu(),
        "validation_prediction_24_m": validation_log_depth.exp().cpu(),
        "validation_target_24_m": validation_target.cpu(),
        "validation_log_variance_24": validation_logvar.cpu(),
    }
    return result, history, diagnostics


def _aggregate(results: list[VariantResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[VariantResult]] = defaultdict(list)
    for value in results:
        grouped[value.variant_id].append(value)
    output: dict[str, dict[str, float]] = {}
    for variant, values in grouped.items():
        metrics: dict[str, float] = {}
        for key in values[0].metrics:
            numbers = np.asarray([value.metrics[key] for value in values])
            metrics[f"{key}_mean"] = float(numbers.mean())
            metrics[f"{key}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        metrics["total_ms_per_frame_mean"] = float(np.mean([value.runtime["total_ms_per_frame"] for value in values]))
        metrics["parameters"] = float(values[0].parameters)
        output[variant] = metrics
    return output


def _artifact_manifest(output: Path) -> dict[str, dict[str, Any]]:
    # The manifest cannot hash itself, and the backend artifact receipt only
    # exists after the immutable directory snapshot has uploaded successfully.
    excluded = {
        "artifact_manifest.json",
        "artifact_manifest.json.tmp",
        "wandb_artifact_receipt.json",
        "wandb_artifact_receipt.json.tmp",
    }
    files = [path for path in output.rglob("*") if path.is_file() and path.name not in excluded]
    return {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(files)
    }


def _upload_wandb_artifact(run: Any, output: Path, status: str) -> dict[str, Any]:
    """Upload the final immutable snapshot and persist the backend receipt."""
    import wandb

    suffix = "comparison" if status == "success" else "failed-comparison"
    artifact = wandb.Artifact(f"{run.id}-phase2b-{suffix}", type="geometry-comparison")
    artifact.add_dir(str(output), name="phase2b")
    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait(timeout=900)
    receipt = {
        "schema_version": "jepa4d-phase2b-wandb-artifact-v1",
        "status": status,
        "mode": "online",
        "run_id": run.id,
        "run_url": run.url,
        "run_path": run.path,
        "artifact_name": logged_artifact.name,
        "artifact_version": logged_artifact.version,
        "artifact_digest": logged_artifact.digest,
        "artifact_manifest_sha256": _sha256(output / "artifact_manifest.json"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    required = ("run_id", "run_url", "artifact_name", "artifact_version", "artifact_digest")
    if any(not receipt[key] for key in required):
        raise RuntimeError(f"W&B returned an incomplete formal artifact receipt: {receipt}")
    _write_json(output / "wandb_artifact_receipt.json", receipt)
    return receipt


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _validate_complete_results(results: list[VariantResult], failures: list[dict[str, str]]) -> None:
    expected = {
        "vggt_teacher": {None},
        "rgb_probe": {0, 1, 2},
        "vjepa_final": {0, 1, 2},
        "vjepa_multilayer": {0, 1, 2},
    }
    actual: dict[str, set[int | None]] = defaultdict(set)
    for result in results:
        if result.seed in actual[result.variant_id]:
            raise RuntimeError(f"duplicate result for {result.variant_id} seed {result.seed}")
        actual[result.variant_id].add(result.seed)
        numbers = [*result.metrics.values(), *result.runtime.values()]
        if not all(np.isfinite(float(value)) for value in numbers):
            raise RuntimeError(f"non-finite metrics/runtime for {result.variant_id} seed {result.seed}")
        if result.seed is not None:
            if result.checkpoint is None or result.checkpoint_sha256 is None:
                raise RuntimeError(f"missing checkpoint metadata for {result.variant_id} seed {result.seed}")
            checkpoint = Path(result.checkpoint)
            if not checkpoint.is_file() or _sha256(checkpoint) != result.checkpoint_sha256:
                raise RuntimeError(f"checkpoint hash mismatch for {result.variant_id} seed {result.seed}")
    if dict(actual) != expected:
        raise RuntimeError(f"incomplete result set: expected={expected}, actual={dict(actual)}")
    if failures:
        raise RuntimeError(f"Phase 2b recorded {len(failures)} seed failures")


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")],
    archive: Annotated[Path, typer.Option("--archive")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/jepa4d_phase2b/tum_rgbd_v1"),
    manifest_path: Annotated[Path, typer.Option("--manifest")] = Path(
        "jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml"
    ),
    vjepa_checkpoint: Annotated[Path, typer.Option("--vjepa-checkpoint")] = Path(
        "checkpoints/vjepa2.1-vitb-fpc64-384"
    ),
    vjepa_implementation: Annotated[Path, typer.Option("--vjepa-implementation")] = Path(
        "checkpoints/vjepa21_hf_impl"
    ),
    vggt_checkpoint: Annotated[Path, typer.Option("--vggt-checkpoint")] = Path("checkpoints/VGGT-1B"),
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    epochs: Annotated[int, typer.Option("--epochs")] = 60,
    wandb_enabled: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_entity: Annotated[str | None, typer.Option("--wandb-entity")] = None,
    run_name: Annotated[str, typer.Option("--run-name")] = "phase2b-jepa-geometry-distillation-v1",
) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise typer.BadParameter("Phase 2b training requires CUDA")
    if epochs <= 0:
        raise typer.BadParameter("epochs must be positive")
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _configure_determinism()
    _log_event(output, "initializing", device=device, epochs=epochs, wandb=wandb_enabled)

    manifest = validate_archive(archive, manifest_path)
    split_names = ("train", "validation", "test")
    split_indices = {name: [int(value) for value in manifest[f"{name}_indices"]] for name in split_names}
    for name, indices in split_indices.items():
        if len(indices) != len(set(indices)) or indices != sorted(indices):
            raise ValueError(f"{name} indices must be unique and chronological")
    if any(
        set(split_indices[first]) & set(split_indices[second])
        for first in split_names
        for second in split_names
        if first < second
    ):
        raise ValueError("train, validation, and test indices must be disjoint")
    split_hash = hashlib.sha256(
        json.dumps(
            {key: manifest[key] for key in ("train_indices", "validation_indices", "test_indices")}, sort_keys=True
        ).encode()
    ).hexdigest()
    splits = {name: load_tum_indices(dataset_root, split_indices[name]) for name in split_names}
    if {name: len(values) for name, values in splits.items()} != {"train": 64, "validation": 16, "test": 8}:
        raise ValueError("formal split must contain exactly 64/16/8 frames")

    _log_event(output, "validating_dataset_extraction")
    dataset_record = _dataset_fingerprint(dataset_root, splits, archive)
    _write_json(output / "dataset_fingerprint.json", dataset_record)
    _log_event(output, "validating_model_assets")
    asset_record = _checkpoint_manifest(
        {
            "vjepa_checkpoint": vjepa_checkpoint,
            "vjepa_implementation": vjepa_implementation,
            "vggt_checkpoint": vggt_checkpoint,
        }
    )
    _write_json(output / "asset_manifest.json", asset_record)

    targets_24 = {name: _targets(samples, (24, 24)) for name, samples in splits.items()}
    targets_518 = {name: _targets(samples, (518, 518)) for name, samples in splits.items()}
    environment = _environment_snapshot(device)
    _write_json(output / "environment.json", environment)
    (output / "pip-freeze.txt").write_text(_command([sys.executable, "-m", "pip", "freeze"]) + "\n")
    resolved_config = {
        "dataset_root": str(dataset_root.resolve()),
        "archive": str(archive.resolve()),
        "manifest": str(manifest_path.resolve()),
        "split_hash": split_hash,
        "output": str(output.resolve()),
        "vjepa_checkpoint": str(vjepa_checkpoint.resolve()),
        "vjepa_implementation": str(vjepa_implementation.resolve()),
        "vggt_checkpoint": str(vggt_checkpoint.resolve()),
        "device": device,
        "epochs": epochs,
        "seeds": [0, 1, 2],
        "encoder_chunk_size": 8,
        "probe_batch_size": 8,
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0.0001, "gradient_clip": 5.0},
        "loss_weights": {"nll": 1.0, "scale_invariant": 0.25, "gradient": 0.1, "teacher": 0.25},
        "preprocessing": "center-crop shortest side to square; RGB bilinear to 384; depth nearest to evaluation grid",
        "teacher_scale_policy": "one global scale fitted on training pixels and frozen before validation/test",
        "multilayer_policy": "train-standardize each of layers 2/5/8/11, then average; parameter-matched to final layer",
        "wandb": {"enabled": wandb_enabled, "project": wandb_project, "entity": wandb_entity, "mode": "online"},
    }
    _write_json(output / "resolved_config.json", resolved_config)

    run = None
    run_finished = False
    artifact_receipt: dict[str, Any] | None = None
    try:
        if wandb_enabled:
            import wandb

            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                job_type="phase2b-training",
                mode="online",
                tags=["phase-2b", "geometry-distillation", "TUM-RGBD", "vjepa", "baselines", "cuda"],
                config={
                    **resolved_config,
                    "manifest_id": manifest["dataset_id"],
                    "manifest_version": manifest["version"],
                },
            )
            if run.offline:
                raise RuntimeError("formal Phase 2b requires online W&B logging")
            _log_event(output, "wandb_online", url=run.url)

        _log_event(output, "vjepa_loading")
        load_started = time.perf_counter()
        extractor = VJEPA21FeatureExtractor(
            checkpoint=vjepa_checkpoint,
            implementation_path=vjepa_implementation,
            backend="hf_compat",
            device=device,
        )
        vjepa_load_seconds = time.perf_counter() - load_started
        vjepa_parameters = (
            sum(value.numel() for value in extractor.model.parameters()) if extractor.model is not None else 0
        )
        vjepa_features: dict[str, dict[str, torch.Tensor]] = {}
        vjepa_runtime: dict[str, dict[str, float]] = {}
        feature_statistics: dict[str, Any] = {}
        for name, samples in splits.items():
            _log_event(output, "vjepa_extracting", split=name, frames=len(samples))
            vjepa_features[name], vjepa_runtime[name] = _extract_vjepa(extractor, samples, chunk_size=8)
            vjepa_runtime[name].update(
                {"model_load_seconds": vjepa_load_seconds, "parameters": float(vjepa_parameters)}
            )
            feature_statistics[name] = {}
            for key, value in vjepa_features[name].items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"non-finite V-JEPA features in {name}/{key}")
                feature_statistics[name][key] = {
                    "shape": list(value.shape),
                    "mean": float(value.float().mean()),
                    "std": float(value.float().std()),
                    "min": float(value.float().min()),
                    "max": float(value.float().max()),
                }
                if run is not None:
                    run.log(
                        {
                            f"features/{name}/{key}/mean": feature_statistics[name][key]["mean"],
                            f"features/{name}/{key}/std": feature_statistics[name][key]["std"],
                            f"features/{name}/{key}/finite_fraction": 1.0,
                        }
                    )
        _write_json(output / "feature_statistics.json", feature_statistics)
        del extractor
        _release_cuda()
        _log_event(output, "vjepa_released")

        _log_event(output, "vggt_loading")
        load_started = time.perf_counter()
        teacher = GeometryBeliefHead(
            backend="vggt", device=device, model_id=str(vggt_checkpoint), precision="bfloat16"
        )
        teacher_load_seconds = time.perf_counter() - load_started
        teacher_parameters = (
            sum(value.numel() for value in teacher.model.parameters()) if teacher.model is not None else 0
        )
        teacher_raw_518: dict[str, torch.Tensor] = {}
        teacher_runtime: dict[str, dict[str, float]] = {}
        for name, samples in splits.items():
            _log_event(output, "vggt_extracting", split=name, frames=len(samples))
            teacher_raw_518[name], teacher_runtime[name] = _teacher_depth(teacher, samples, chunk_size=8)
            teacher_runtime[name].update(
                {"model_load_seconds": teacher_load_seconds, "parameters": float(teacher_parameters)}
            )
        teacher_scale = _fit_metric_scale(teacher_raw_518["train"], targets_518["train"])
        teacher_train_24 = F.interpolate(
            (teacher_raw_518["train"] * teacher_scale).unsqueeze(1),
            size=(24, 24),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        teacher_test_metric = teacher_raw_518["test"] * teacher_scale
        teacher_metrics = _evaluate_depths(teacher_test_metric, targets_518["test"])
        teacher_metrics["training_fitted_metric_scale"] = teacher_scale
        teacher_result = VariantResult(
            variant_id="vggt_teacher",
            family="vggt",
            role="teacher_baseline",
            seed=None,
            metrics=teacher_metrics,
            runtime={
                "encoder_ms_per_frame": teacher_runtime["test"]["per_frame_ms"],
                "head_ms_per_frame": 0.0,
                "total_ms_per_frame": teacher_runtime["test"]["per_frame_ms"],
                "peak_encoder_memory_gb": teacher_runtime["test"]["peak_memory_gb"],
                "peak_head_memory_gb": 0.0,
                "training_seconds": 0.0,
                "encoder_model_load_seconds": teacher_load_seconds,
            },
            parameters=teacher_parameters,
            trainable_parameters=0,
            encoder_parameters=teacher_parameters,
            notes=[
                "Official VGGT-1B BF16 teacher evaluated at native 518px.",
                "One metric scale was fitted only on training pixels and frozen for test; aligned_* remains per-frame scale aligned.",
            ],
        )
        diagnostics_dir = output / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        teacher_diagnostics_path = diagnostics_dir / "vggt_teacher.npz"
        np.savez_compressed(
            teacher_diagnostics_path,
            prediction_m=teacher_test_metric.numpy(),
            target_m=targets_518["test"].numpy(),
            relative_prediction=teacher_raw_518["test"].numpy(),
            fitted_scale=np.asarray(teacher_scale),
        )
        per_frame_metrics = _per_frame_rows(
            "vggt_teacher", None, teacher_test_metric, targets_518["test"], splits["test"]
        )
        if run is not None:
            run.log({f"teacher/{key}": value for key, value in teacher_metrics.items()})
        del teacher
        _release_cuda()
        _log_event(output, "vggt_released", fitted_metric_scale=teacher_scale)

        rgb_features: dict[str, torch.Tensor] = {}
        rgb_runtime_by_split: dict[str, dict[str, float]] = {}
        for name, samples in splits.items():
            started = time.perf_counter()
            rgb_features[name] = rgb_grid_features(_images(samples), 24).half()
            elapsed = time.perf_counter() - started
            rgb_runtime_by_split[name] = {
                "total_seconds": elapsed,
                "per_frame_ms": elapsed * 1000.0 / len(samples),
                "peak_memory_gb": 0.0,
                "model_load_seconds": 0.0,
                "parameters": 0.0,
            }

        variant_features: dict[str, dict[str, torch.Tensor]] = {}
        normalization_hashes: dict[str, str] = {}
        rgb_normalized = _normalize(rgb_features["train"], rgb_features["validation"], rgb_features["test"])
        variant_features["rgb_probe"] = {
            "train": rgb_normalized[0],
            "validation": rgb_normalized[1],
            "test": rgb_normalized[2],
        }
        rgb_normalization_path = output / "rgb_probe-normalization.pt"
        torch.save(rgb_normalized[3], rgb_normalization_path)
        normalization_hashes[str(rgb_normalization_path.relative_to(output))] = _sha256(rgb_normalization_path)

        final_normalized = _normalize(
            vjepa_features["train"]["vjepa_final"],
            vjepa_features["validation"]["vjepa_final"],
            vjepa_features["test"]["vjepa_final"],
        )
        variant_features["vjepa_final"] = {
            "train": final_normalized[0],
            "validation": final_normalized[1],
            "test": final_normalized[2],
        }
        final_normalization_path = output / "vjepa_final-normalization.pt"
        torch.save(final_normalized[3], final_normalization_path)
        normalization_hashes[str(final_normalization_path.relative_to(output))] = _sha256(final_normalization_path)

        multilayer_normalized = _normalize_multilayer(
            vjepa_features["train"], vjepa_features["validation"], vjepa_features["test"]
        )
        variant_features["vjepa_multilayer"] = {
            "train": multilayer_normalized[0],
            "validation": multilayer_normalized[1],
            "test": multilayer_normalized[2],
        }
        multilayer_normalization_path = output / "vjepa_multilayer-normalization.pt"
        torch.save(multilayer_normalized[3], multilayer_normalization_path)
        normalization_hashes[str(multilayer_normalization_path.relative_to(output))] = _sha256(
            multilayer_normalization_path
        )
        del vjepa_features, rgb_features
        gc.collect()

        results = [teacher_result]
        failures: list[dict[str, str]] = []
        histories: list[dict[str, Any]] = []
        diagnostics: dict[str, str] = {"vggt_teacher": str(teacher_diagnostics_path)}
        for variant, features in variant_features.items():
            for seed in (0, 1, 2):
                _log_event(output, "seed_start", variant=variant, seed=seed)
                try:
                    runtime = rgb_runtime_by_split["test"] if variant == "rgb_probe" else vjepa_runtime["test"]
                    result, history, prediction_diagnostics = _train_variant(
                        variant,
                        seed,
                        features["train"],
                        features["validation"],
                        features["test"],
                        targets_24["train"],
                        targets_24["validation"],
                        targets_518["test"],
                        teacher_train_24,
                        output,
                        device,
                        epochs,
                        run,
                        runtime,
                    )
                    diagnostics_path = diagnostics_dir / f"{variant}-seed{seed}.npz"
                    np.savez_compressed(
                        diagnostics_path,
                        **{key: value.detach().cpu().numpy() for key, value in prediction_diagnostics.items()},
                    )
                    diagnostics[f"{variant}-seed{seed}"] = str(diagnostics_path)
                    results.append(result)
                    histories.extend(history)
                    per_frame_metrics.extend(
                        _per_frame_rows(
                            variant,
                            seed,
                            prediction_diagnostics["prediction_m"],
                            prediction_diagnostics["target_m"],
                            splits["test"],
                        )
                    )
                    _log_event(
                        output,
                        "seed_complete",
                        variant=variant,
                        seed=seed,
                        metric_abs_rel=result.metrics["metric_abs_rel"],
                        checkpoint_sha256=result.checkpoint_sha256,
                    )
                except Exception as error:
                    failure = {
                        "variant": variant,
                        "seed": str(seed),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    _write_json(output / "failures" / f"{variant}-seed{seed}.json", failure)
                    _log_event(output, "seed_failed", variant=variant, seed=seed, error=failure["error"])
                finally:
                    _release_cuda()

        aggregates = _aggregate(results)
        artifact_hashes = dict(normalization_hashes)
        artifact_hashes.update(
            {
                str(Path(value.checkpoint).relative_to(output)): str(value.checkpoint_sha256)
                for value in results
                if value.checkpoint is not None and value.checkpoint_sha256 is not None
            }
        )
        record = ComparisonRecord(
            experiment_id=run_name,
            schema_version="jepa4d-phase2b-comparison-v1",
            dataset_manifest=str(manifest_path),
            split_hash=split_hash,
            metric_policy={
                "primary": "metric_abs_rel on target-defined valid pixels of chronological held-out test frames",
                "secondary": "per-frame median-aligned depth, validation-fitted uncertainty NLL, latency, and peak GPU memory",
                "checkpoint_selection": "minimum validation metric_abs_rel",
                "teacher_scale": "one global scale fitted only on training pixels and frozen before test",
                "preprocessing": resolved_config["preprocessing"],
                "seeds": [0, 1, 2],
                "teacher_auxiliary_weight": 0.25,
                "multilayer_fusion": resolved_config["multilayer_policy"],
            },
            variants=results,
            failures=failures,
            aggregates=aggregates,
            wandb_url=None if run is None else run.url,
            environment=environment,
            artifacts=artifact_hashes,
        )
        comparison_path = output / "comparison.json"
        failures_path = output / "failures.json"
        per_frame_path = output / "per_frame_metrics.json"
        _write_json(comparison_path, record.to_serializable())
        _write_json(failures_path, failures)
        _write_json(per_frame_path, per_frame_metrics)
        diagnostics["per_frame_metrics"] = str(per_frame_path)

        from jepa4d.visualization.geometry_student_report import write_phase2b_report

        html_report = write_phase2b_report(
            output=output,
            comparison=record.to_serializable(),
            histories=histories,
            diagnostics=diagnostics,
        )

        completion_error: str | None = None
        try:
            _validate_complete_results(results, failures)
        except Exception as error:
            completion_error = f"{type(error).__name__}: {error}"
        completion = {
            "status": "success" if completion_error is None else "failed",
            "error": completion_error,
            "result_rows": len(results),
            "probe_checkpoints": len(list((output / "checkpoints").glob("*.pt"))),
            "seed_failures": len(failures),
        }
        _write_json(output / "completion_gate.json", completion)
        _log_event(output, "completion_gate", **completion)
        manifest_files = _artifact_manifest(output)
        _write_json(output / "artifact_manifest.json", manifest_files)

        postflight_command = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "slurm" / "validate_phase2b_output.py"),
            "--output",
            str(output),
        ]
        postflight = subprocess.run(postflight_command, check=False, capture_output=True, text=True, timeout=300)
        (output / "postflight-validation.log").write_text((postflight.stdout or "") + (postflight.stderr or ""))
        if postflight.returncode != 0 and completion_error is None:
            completion_error = f"strict postflight validation exited {postflight.returncode}"
            completion["status"] = "failed"
            completion["error"] = completion_error
            _write_json(output / "completion_gate.json", completion)
        _log_event(output, "postflight_validation", status=completion["status"], returncode=postflight.returncode)
        manifest_files = _artifact_manifest(output)
        _write_json(output / "artifact_manifest.json", manifest_files)
        if completion_error is not None:
            terminal_failure = {
                "error": completion_error,
                "traceback": None,
                "wandb_url": None if run is None else run.url,
            }
            _write_json(output / "run_failure.json", terminal_failure)
            _log_event(output, "run_failed", error=completion_error)
            manifest_files = _artifact_manifest(output)
            _write_json(output / "artifact_manifest.json", manifest_files)

        if run is not None:
            import wandb

            results_table = wandb.Table(
                columns=[
                    "variant",
                    "role",
                    "seed",
                    "metric_abs_rel",
                    "aligned_abs_rel",
                    "aligned_rmse_m",
                    "calibrated_nll",
                    "encoder_ms",
                    "head_ms",
                    "total_ms",
                    "encoder_peak_gb",
                    "head_peak_gb",
                    "training_peak_gb",
                    "trainable_parameters",
                    "encoder_parameters",
                    "total_parameters",
                ]
            )
            learned_table = wandb.Table(columns=["variant_seed", "total_ms", "metric_abs_rel", "peak_memory_gb"])
            for variant_result in results:
                results_table.add_data(
                    variant_result.variant_id,
                    variant_result.role,
                    variant_result.seed,
                    variant_result.metrics.get("metric_abs_rel"),
                    variant_result.metrics.get("aligned_abs_rel"),
                    variant_result.metrics.get("aligned_rmse_m"),
                    variant_result.metrics.get("calibrated_log_depth_nll"),
                    variant_result.runtime["encoder_ms_per_frame"],
                    variant_result.runtime["head_ms_per_frame"],
                    variant_result.runtime["total_ms_per_frame"],
                    variant_result.runtime["peak_encoder_memory_gb"],
                    variant_result.runtime["peak_head_memory_gb"],
                    variant_result.runtime.get("peak_training_memory_gb"),
                    variant_result.trainable_parameters,
                    variant_result.encoder_parameters,
                    variant_result.parameters,
                )
                learned_table.add_data(
                    f"{variant_result.variant_id}-seed{variant_result.seed}",
                    variant_result.runtime["total_ms_per_frame"],
                    variant_result.metrics.get("metric_abs_rel"),
                    max(
                        variant_result.runtime["peak_encoder_memory_gb"],
                        variant_result.runtime["peak_head_memory_gb"],
                    ),
                )
            history_columns = [
                "variant",
                "seed",
                "epoch",
                "loss",
                "validation_metric_abs_rel",
                "nll",
                "scale_invariant",
                "gradient",
                "distillation",
                "gradient_norm",
                "learning_rate",
                "best_validation_metric_abs_rel",
                "epoch_seconds",
                "gpu_allocated_gb",
                "gpu_reserved_gb",
            ]
            history_table = wandb.Table(
                columns=history_columns,
                data=[[row.get(column) for column in history_columns] for row in histories],
            )
            frame_columns = [
                "variant",
                "seed",
                "frame_id",
                "timestamp",
                "metric_abs_rel",
                "metric_rmse_m",
                "aligned_abs_rel",
                "aligned_rmse_m",
                "alignment_scale",
                "valid_target_fraction",
            ]
            frame_table = wandb.Table(
                columns=frame_columns,
                data=[[row.get(column) for column in frame_columns] for row in per_frame_metrics],
            )
            diagnostic_media: dict[str, Any] = {}
            for label, path_value in diagnostics.items():
                if label == "per_frame_metrics" or (label != "vggt_teacher" and not label.endswith("seed0")):
                    continue
                with np.load(path_value) as values:
                    prediction = values["prediction_m"][0]
                    target = values["target_m"][0]
                    relative_error = np.abs(prediction - target) / np.maximum(target, 1e-6)
                    validation_prediction = (
                        values["validation_prediction_24_m"] if "validation_prediction_24_m" in values.files else None
                    )
                    validation_target = (
                        values["validation_target_24_m"] if "validation_target_24_m" in values.files else None
                    )
                    validation_logvar = (
                        values["validation_log_variance_24"] if "validation_log_variance_24" in values.files else None
                    )
                diagnostic_media[f"diagnostics/{label}/prediction"] = wandb.Image(
                    prediction, caption=f"{label}: held-out frame 0 prediction (m)"
                )
                diagnostic_media[f"diagnostics/{label}/target"] = wandb.Image(
                    target, caption=f"{label}: held-out frame 0 target (m)"
                )
                diagnostic_media[f"diagnostics/{label}/relative_error"] = wandb.Image(
                    np.clip(relative_error, 0, 1), caption=f"{label}: held-out frame 0 clipped relative error"
                )
                if (
                    validation_prediction is not None
                    and validation_target is not None
                    and validation_logvar is not None
                ):
                    validation_relative_error = np.abs(validation_prediction[0] - validation_target[0]) / np.maximum(
                        validation_target[0], 1e-6
                    )
                    diagnostic_media[f"diagnostics/{label}/validation_prediction"] = wandb.Image(
                        validation_prediction[0], caption=f"{label}: fixed validation frame prediction (m)"
                    )
                    diagnostic_media[f"diagnostics/{label}/validation_relative_error"] = wandb.Image(
                        np.clip(validation_relative_error, 0, 1),
                        caption=f"{label}: fixed validation frame clipped relative error",
                    )
                    diagnostic_media[f"diagnostics/{label}/validation_uncertainty"] = wandb.Image(
                        np.exp(validation_logvar[0]), caption=f"{label}: fixed validation frame predicted variance"
                    )
            run.log(
                {
                    "comparison/results": results_table,
                    "comparison/training_history": history_table,
                    "comparison/per_frame_metrics": frame_table,
                    "comparison/accuracy_latency": wandb.plot.scatter(
                        learned_table, "total_ms", "metric_abs_rel", title="Accuracy–latency trade-off"
                    ),
                    "comparison/failures": len(failures),
                    "comparison/report": wandb.Html(str(html_report), inject=False),
                    **diagnostic_media,
                }
            )
            for variant, values in aggregates.items():
                run.summary.update({f"comparison/{variant}/{key}": value for key, value in values.items()})
            artifact_receipt = _upload_wandb_artifact(run, output, str(completion["status"]))
            run.summary.update(
                {
                    "result": completion["status"],
                    "variants": len(results),
                    "failures": len(failures),
                    "report": str(html_report),
                    "artifact_files": len(manifest_files),
                    "artifact_name": artifact_receipt["artifact_name"],
                    "artifact_version": artifact_receipt["artifact_version"],
                    "artifact_digest": artifact_receipt["artifact_digest"],
                }
            )
        typer.echo(
            json.dumps(
                {
                    "comparison": str(comparison_path),
                    "report": str(html_report),
                    "wandb_url": record.wandb_url,
                    "wandb_artifact": artifact_receipt,
                    "aggregates": aggregates,
                    "completion": completion,
                },
                indent=2,
                allow_nan=False,
            )
        )
        if run is not None:
            run.finish(exit_code=0 if completion_error is None else 1)
            run_finished = True
        if completion_error is not None:
            raise RuntimeError(completion_error)
    except Exception as error:
        run_failure: dict[str, Any] = {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "wandb_url": None if run is None else run.url,
        }
        failure_published = bool(artifact_receipt is not None and artifact_receipt.get("status") == "failed")
        if not failure_published:
            if not (output / "run_failure.json").exists():
                _write_json(output / "run_failure.json", run_failure)
                _log_event(output, "run_failed", error=run_failure["error"])
            # A previous success receipt is not evidence for the terminal
            # process state and must not enter the failure supplement.
            (output / "wandb_artifact_receipt.json").unlink(missing_ok=True)
            manifest_files = _artifact_manifest(output)
            _write_json(output / "artifact_manifest.json", manifest_files)
        if run is not None and not run_finished:
            try:
                run.summary.update({"result": "failed", "failure": run_failure["error"]})
                run.log({"failure/error": run_failure["error"]})
                if not failure_published:
                    artifact_receipt = _upload_wandb_artifact(run, output, "failed")
                    failure_published = True
                    run.summary.update(
                        {
                            "artifact_name": artifact_receipt["artifact_name"],
                            "artifact_version": artifact_receipt["artifact_version"],
                            "artifact_digest": artifact_receipt["artifact_digest"],
                        }
                    )
                run.finish(exit_code=1)
                run_finished = True
            except Exception as upload_error:
                if not failure_published:
                    upload_failure = {
                        "error": f"{type(upload_error).__name__}: {upload_error}",
                        "traceback": traceback.format_exc(),
                    }
                    _write_json(output / "wandb_upload_failure.json", upload_failure)
                    _log_event(output, "wandb_artifact_failed", error=upload_failure["error"])
                    _write_json(output / "artifact_manifest.json", _artifact_manifest(output))
                try:
                    run.finish(exit_code=1)
                    run_finished = True
                except Exception:
                    pass
        raise


if __name__ == "__main__":
    app()
