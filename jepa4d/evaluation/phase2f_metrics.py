"""Strict metric, calibration, hashing, and report helpers for Phase 2f.

This module intentionally has no dataset loader.  Development and external-final
callers use the same functions so metric definitions cannot drift between gates.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch
from PIL import Image

METRICS_SCHEMA = "jepa4d-phase2f-depth-metrics-v1"
CALIBRATION_SCHEMA = "jepa4d-phase2f-variance-calibration-v1"
PRIMARY_METRIC_NAMES = ("raw_abs_rel", "absolute_log_scale_error", "aligned_abs_rel", "nll", "ause")
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2f-wandb-artifact-receipt-v1"


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value under the repository's canonical encoding."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> Path:
    """Write finite UTF-8 JSON atomically."""

    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def file_identity(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    """Return a strict content identity for one existing file."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if schema is not None:
        result["schema"] = schema
    return result


def require_finite_tree(value: Any, label: str = "value") -> None:
    """Reject NaN/Inf recursively before values reach receipts or gates."""

    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}[{index}]")
        return
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def _validate_prediction_inputs(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (log_depth, log_variance, target_depth)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("predictions and targets must be torch tensors")
    if log_depth.ndim != 3 or log_depth.shape != log_variance.shape or log_depth.shape != target_depth.shape:
        raise ValueError("log_depth, log_variance, and target_depth must share shape [N,H,W]")
    if len(log_depth) == 0 or any(not torch.is_floating_point(value) for value in tensors):
        raise ValueError("metric tensors must be non-empty floating-point [N,H,W]")
    if not bool(torch.isfinite(log_depth).all()) or not bool(torch.isfinite(log_variance).all()):
        raise ValueError("predicted means and variances must be finite")
    valid = torch.isfinite(target_depth) & (target_depth > 0) if valid_mask is None else valid_mask
    if valid.shape != target_depth.shape or valid.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean and match target_depth")
    if bool((valid.flatten(1).sum(1) == 0).any()):
        raise ValueError("every evaluated frame must contain a valid target pixel")
    return (
        log_depth.detach().cpu().double(),
        log_variance.detach().cpu().double(),
        target_depth.detach().cpu().double(),
        valid.detach().cpu(),
    )


def fit_variance_multiplier(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Fit the preregistered single positive validation variance multiplier."""

    prediction, variance_log, target, valid = _validate_prediction_inputs(
        log_depth, log_variance, target_depth, valid_mask
    )
    residual = prediction - target.clamp_min(torch.finfo(target.dtype).tiny).log()
    ratio = residual[valid].square() / variance_log[valid].exp()
    unclipped = float(ratio.mean())
    multiplier = min(1e3, max(1e-3, unclipped))
    result = {
        "schema_version": CALIBRATION_SCHEMA,
        "method": "mean_squared_log_residual_over_predicted_variance",
        "unclipped_multiplier": unclipped,
        "multiplier": multiplier,
        "clip_interval": [1e-3, 1e3],
        "pixels": int(valid.sum()),
    }
    require_finite_tree(result, "calibration")
    return result


def _risk_coverage(error: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if error.ndim != 1 or uncertainty.shape != error.shape or len(error) == 0:
        raise ValueError("risk-coverage inputs must be non-empty one-dimensional arrays")
    if not np.isfinite(error).all() or not np.isfinite(uncertainty).all():
        raise ValueError("risk-coverage inputs must be finite")
    count = len(error)
    coverage = np.arange(1, count + 1, dtype=np.float64) / count
    predicted_order = np.argsort(uncertainty, kind="stable")
    oracle_order = np.argsort(error, kind="stable")
    predicted_risk = np.cumsum(error[predicted_order]) / np.arange(1, count + 1)
    oracle_risk = np.cumsum(error[oracle_order]) / np.arange(1, count + 1)
    ause = float(np.trapz(predicted_risk - oracle_risk, coverage))  # noqa: NPY201 - NumPy 1.x cluster env.
    return coverage, predicted_risk, max(0.0, ause)


def evaluate_depth_predictions(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    variance_multiplier: float = 1.0,
    frame_ids: Sequence[str] | None = None,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute raw/aligned/scale, calibrated NLL, coverage, and AUSE metrics.

    All macro results first average pixels within a frame. ``group_macro`` then
    averages frames within each group and groups equally, as required for the
    SUN family and DIODE domain aggregates.
    """

    if not math.isfinite(variance_multiplier) or not 1e-3 <= variance_multiplier <= 1e3:
        raise ValueError("variance_multiplier must be finite and in [1e-3,1e3]")
    prediction, variance_log, target, valid = _validate_prediction_inputs(
        log_depth, log_variance, target_depth, valid_mask
    )
    frame_count = len(prediction)
    ids = list(frame_ids) if frame_ids is not None else [f"frame-{index:06d}" for index in range(frame_count)]
    groups = list(group_ids) if group_ids is not None else ["all"] * frame_count
    if len(ids) != frame_count or len(set(ids)) != frame_count or any(not value for value in ids):
        raise ValueError("frame_ids must be unique non-empty strings matching the batch")
    if len(groups) != frame_count or any(not value for value in groups):
        raise ValueError("group_ids must be non-empty strings matching the batch")

    target_log = target.clamp_min(torch.finfo(target.dtype).tiny).log()
    calibrated_log_variance = variance_log + math.log(variance_multiplier)
    rows: list[dict[str, Any]] = []
    all_error: list[np.ndarray] = []
    all_uncertainty: list[np.ndarray] = []
    coverage_levels = (0.5, 0.8, 0.9, 0.95)
    covered = {level: 0 for level in coverage_levels}
    total_pixels = 0
    for index in range(frame_count):
        mask = valid[index]
        predicted_log = prediction[index][mask]
        expected_log = target_log[index][mask]
        expected_depth = target[index][mask]
        predicted_depth = predicted_log.exp()
        log_residual = predicted_log - expected_log
        log_scale = float(log_residual.median())
        aligned_depth = (predicted_log - log_scale).exp()
        absolute_relative = (predicted_depth - expected_depth).abs() / expected_depth
        aligned_relative = (aligned_depth - expected_depth).abs() / expected_depth
        calibrated_variance = calibrated_log_variance[index][mask].exp()
        nll = 0.5 * (log_residual.square() / calibrated_variance + calibrated_variance.log())
        uncertainty = calibrated_variance.sqrt()
        pixel_error = absolute_relative.numpy()
        pixel_uncertainty = uncertainty.numpy()
        all_error.append(pixel_error)
        all_uncertainty.append(pixel_uncertainty)
        total_pixels += len(pixel_error)
        for level in coverage_levels:
            z_value = NormalDist().inv_cdf((1 + level) / 2)
            covered[level] += int((log_residual.abs() <= z_value * uncertainty).sum())
        _, _, frame_ause = _risk_coverage(pixel_error, pixel_uncertainty)
        rows.append(
            {
                "frame_id": ids[index],
                "group_id": groups[index],
                "valid_pixels": int(mask.sum()),
                "raw_abs_rel": float(absolute_relative.mean()),
                "aligned_abs_rel": float(aligned_relative.mean()),
                "signed_log_scale_error": log_scale,
                "absolute_log_scale_error": abs(log_scale),
                "nll": float(nll.mean()),
                "ause": frame_ause,
            }
        )

    group_values: dict[str, dict[str, float]] = {}
    for group in sorted(set(groups)):
        selected = [row for row in rows if row["group_id"] == group]
        group_values[group] = {
            name: float(np.mean([float(row[name]) for row in selected])) for name in PRIMARY_METRIC_NAMES
        }
    frame_macro = {name: float(np.mean([float(row[name]) for row in rows])) for name in PRIMARY_METRIC_NAMES}
    group_macro = {
        name: float(np.mean([values[name] for values in group_values.values()])) for name in PRIMARY_METRIC_NAMES
    }
    flat_error = np.concatenate(all_error)
    flat_uncertainty = np.concatenate(all_uncertainty)
    coverage, risk, pixel_ause = _risk_coverage(flat_error, flat_uncertainty)
    stride = max(1, len(coverage) // 256)
    reliability = [{"nominal": level, "empirical": covered[level] / total_pixels} for level in coverage_levels]
    result = {
        "schema_version": METRICS_SCHEMA,
        "variance_multiplier": variance_multiplier,
        "frames": frame_count,
        "valid_pixels": total_pixels,
        "per_frame": rows,
        "per_group": group_values,
        "frame_macro": frame_macro,
        "group_macro": group_macro,
        "coverage": reliability,
        "risk_coverage": {
            "coverage": coverage[::stride].tolist(),
            "risk": risk[::stride].tolist(),
            "pixel_ause": pixel_ause,
            "ranking": "uncalibrated_variance_equivalent_under_scalar_multiplier",
        },
    }
    require_finite_tree(result, "metrics")
    return result


def png_data_uri(path: Path) -> str:
    """Embed a local PNG for a self-contained HTML report."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def array_to_png_bytes(values: np.ndarray, *, minimum: float | None = None, maximum: float | None = None) -> bytes:
    """Create a compact deterministic RGB heatmap without a plotting backend."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("heatmap values must be a finite two-dimensional array")
    low = float(array.min()) if minimum is None else minimum
    high = float(array.max()) if maximum is None else maximum
    normalized = np.zeros_like(array) if high <= low else np.clip((array - low) / (high - low), 0, 1)
    red = np.clip(1.5 * normalized, 0, 1)
    blue = np.clip(1.5 * (1 - normalized), 0, 1)
    green = np.clip(1.5 - np.abs(2 * normalized - 1) * 1.5, 0, 1)
    rgb = np.round(np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def self_contained_html(
    title: str,
    summary: Mapping[str, Any],
    *,
    images: Sequence[tuple[str, Path]] = (),
    claim_boundary: str,
) -> str:
    """Return a readable report with inline CSS, JSON, and image bytes only."""

    cards = "".join(
        f"<div class='card'><b>{html.escape(str(key))}</b><span>{html.escape(str(value))}</span></div>"
        for key, value in summary.items()
    )
    figures = "".join(
        f"<figure><img src='{png_data_uri(path)}' alt='{html.escape(label)}'><figcaption>"
        f"{html.escape(label)}</figcaption></figure>"
        for label, path in images
    )
    raw = html.escape(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return f"""<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f6f7fb;color:#18212f;margin:0;padding:28px}}
main{{max-width:1100px;margin:auto}} .boundary{{padding:14px;border-left:5px solid #e18b27;background:#fff4dd}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:18px 0}}
.card{{background:white;border-radius:10px;padding:14px;box-shadow:0 1px 5px #ccd2dc;display:flex;flex-direction:column}}
.card span{{font-size:1.2rem;margin-top:8px}} .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
figure{{background:white;padding:12px;margin:0;border-radius:10px}} img{{width:100%;height:auto}} pre{{background:#17202e;color:#eef3fa;padding:16px;overflow:auto}}
</style><main><h1>{html.escape(title)}</h1><p class='boundary'>{html.escape(claim_boundary)}</p>
<div class='cards'>{cards}</div><div class='figures'>{figures}</div><details><summary>Machine-readable summary</summary><pre>{raw}</pre></details></main>"""


def publish_online_wandb(
    *,
    entity: str,
    project: str,
    group: str,
    job_type: str,
    run_name: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifact_name: str,
    artifact_files: Sequence[Path],
) -> dict[str, Any]:
    """Publish one mandatory online run/artifact and return its strict identity."""

    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("Phase 2f requires WANDB_MODE=online")
    require_finite_tree(config, "wandb.config")
    require_finite_tree(summary, "wandb.summary")
    files = [path.resolve(strict=True) for path in artifact_files]
    if not files or any(not path.is_file() for path in files):
        raise ValueError("a W&B artifact requires at least one regular file")
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
    if run is None or run.offline:
        raise RuntimeError("Phase 2f W&B run did not initialize online")
    artifact = wandb.Artifact(artifact_name, type=f"phase2f-{job_type}")
    for path in files:
        artifact.add_file(str(path), name=path.name)
    for key, value in summary.items():
        run.summary[key] = value
    run.summary["status"] = "success"
    logged = run.log_artifact(artifact)
    logged.wait()
    run_id = str(run.id)
    run_url = str(run.url)
    entity_name = str(run.entity)
    project_name = str(run.project)
    artifact_id = str(logged.id)
    artifact_version = str(logged.version)
    artifact_digest = str(logged.digest)
    run.finish(exit_code=0)
    if not all((run_id, run_url, artifact_id, artifact_version, artifact_digest)):
        raise RuntimeError("W&B did not return complete online run/artifact identities")
    return {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "mode": "online",
        "entity": entity_name,
        "project": project_name,
        "group": group,
        "job_type": job_type,
        "run_name": run_name,
        "run_id": run_id,
        "run_url": run_url,
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "artifact_id": artifact_id,
        "artifact_digest": artifact_digest,
        "files": [file_identity(path) for path in files],
        "status": "success",
    }


def cuda_hardware_identity(device: torch.device) -> dict[str, Any]:
    """Return finite, non-secret CUDA hardware/runtime fields for receipts."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA hardware identity requires an available CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    result: dict[str, Any] = {
        "gpu_index": index,
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_uuid": "unknown",
        "driver_version": "unknown",
        "cuda_runtime": torch.version.cuda or "unknown",
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    query = subprocess.run(
        (
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode == 0 and query.stdout.strip():
        fields = [value.strip() for value in query.stdout.splitlines()[0].split(",")]
        if len(fields) == 2 and all(fields):
            result["gpu_uuid"], result["driver_version"] = fields
    return result
