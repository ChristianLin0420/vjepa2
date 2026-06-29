"""Official TUM RGB-D mini-subset evaluation for Phase 2 geometry."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


@dataclass(frozen=True, slots=True)
class TUMSample:
    sample_id: str
    timestamp: float
    rgb_path: Path
    depth_path: Path
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


def _read_index(path: Path, columns: int) -> list[tuple[float, list[str]]]:
    rows: list[tuple[float, list[str]]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != columns:
            raise ValueError(f"unexpected {path.name} row: {line}")
        rows.append((float(fields[0]), fields[1:]))
    return rows


def _nearest(rows: list[tuple[float, list[str]]], timestamp: float, max_delta: float) -> tuple[float, list[str]]:
    result = min(rows, key=lambda item: abs(item[0] - timestamp))
    if abs(result[0] - timestamp) > max_delta:
        raise ValueError(f"no timestamp association within {max_delta:.3f}s for {timestamp:.6f}")
    return result


def validate_archive(archive: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text())
    expected_bytes = int(manifest["archive"]["bytes"])
    if archive.stat().st_size != expected_bytes:
        raise ValueError(f"archive size mismatch: {archive.stat().st_size} != {expected_bytes}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = str(manifest["archive"]["sha256"])
    if actual != expected:
        raise ValueError(f"archive SHA-256 mismatch: {actual} != {expected}")
    return manifest


def load_tum_subset(root: Path, frame_count: int = 8, max_delta: float = 0.03) -> list[TUMSample]:
    rgb = _read_index(root / "rgb.txt", 2)
    margin = max(1, len(rgb) // 20)
    indices = np.linspace(margin, len(rgb) - margin - 1, frame_count, dtype=int).tolist()
    return load_tum_indices(root, indices, max_delta=max_delta)


def load_tum_indices(root: Path, indices: list[int], max_delta: float = 0.03) -> list[TUMSample]:
    rgb = _read_index(root / "rgb.txt", 2)
    depth = _read_index(root / "depth.txt", 2)
    groundtruth = _read_index(root / "groundtruth.txt", 8)
    samples: list[TUMSample] = []
    for index in indices:
        if index < 0 or index >= len(rgb):
            raise IndexError(f"RGB frame index out of range: {index}")
        timestamp, rgb_values = rgb[int(index)]
        _, depth_values = _nearest(depth, timestamp, max_delta)
        _, pose_values = _nearest(groundtruth, timestamp, max_delta)
        samples.append(
            TUMSample(
                sample_id=f"fr1_xyz_{index:04d}",
                timestamp=timestamp,
                rgb_path=root / rgb_values[0],
                depth_path=root / depth_values[0],
                translation=np.asarray(pose_values[:3], dtype=np.float64),
                quaternion_xyzw=np.asarray(pose_values[3:], dtype=np.float64),
            )
        )
    return samples


def load_depth(path: Path, output_size: tuple[int, int]) -> torch.Tensor:
    values = np.asarray(Image.open(path), dtype=np.uint16).copy()
    depth = torch.from_numpy(values.astype(np.float32) / 5000.0)[None, None]
    return F.interpolate(depth, size=output_size, mode="nearest")[0, 0]


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_mean, target - target_mean
    covariance = centered_target.T @ centered_source / source.shape[0]
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vh
    variance = np.mean(np.sum(centered_source**2, axis=1))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _geodesic_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def pose_metrics(extrinsics: torch.Tensor, samples: list[TUMSample]) -> dict[str, float]:
    camera_from_world = extrinsics.detach().cpu().double().numpy()
    world_from_camera = np.linalg.inv(camera_from_world)
    predicted_positions = world_from_camera[:, :3, 3]
    target_positions = np.stack([sample.translation for sample in samples])
    scale, rotation, translation = umeyama_similarity(predicted_positions, target_positions)
    aligned_positions = (scale * (rotation @ predicted_positions.T)).T + translation
    errors = np.linalg.norm(aligned_positions - target_positions, axis=1)
    predicted_rotations = [rotation @ value[:3, :3] for value in world_from_camera]
    target_rotations = [quaternion_xyzw_to_matrix(sample.quaternion_xyzw) for sample in samples]
    rotation_errors = [
        _geodesic_degrees(predicted_rotations[index], target_rotations[index]) for index in range(len(samples))
    ]
    relative_translation_errors = []
    for index in range(len(samples) - 1):
        predicted_delta = aligned_positions[index + 1] - aligned_positions[index]
        target_delta = target_positions[index + 1] - target_positions[index]
        relative_translation_errors.append(float(np.linalg.norm(predicted_delta - target_delta)))
    return {
        "pose_ate_rmse_m_sim3": float(np.sqrt(np.mean(errors**2))),
        "pose_ate_mean_m_sim3": float(np.mean(errors)),
        "pose_rotation_mean_deg_sim3": float(np.mean(rotation_errors)),
        "pose_relative_translation_mean_m_sim3": float(np.mean(relative_translation_errors)),
        "pose_alignment_scale": scale,
    }


def _depth_values(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    valid = torch.isfinite(predicted) & torch.isfinite(target) & (target > 0.1) & (target < 10.0) & (predicted > 0)
    predicted_valid, target_valid = predicted[valid].float(), target[valid].float()
    if predicted_valid.numel() < 100:
        raise ValueError("fewer than 100 valid depth pixels")
    scale = float(target_valid.median() / predicted_valid.median().clamp_min(1e-8))
    return predicted_valid * scale, target_valid, scale


def depth_metrics(predicted: torch.Tensor, target: torch.Tensor) -> tuple[dict[str, float], float, torch.Tensor]:
    aligned, truth, scale = _depth_values(predicted, target)
    error = aligned - truth
    ratio = torch.maximum(aligned / truth, truth / aligned.clamp_min(1e-8))
    values = {
        "abs_rel": float((error.abs() / truth).mean()),
        "rmse_m": float(torch.sqrt((error**2).mean())),
        "log_rmse": float(torch.sqrt(((aligned.log() - truth.log()) ** 2).mean())),
        "delta_1": float((ratio < 1.25).float().mean()),
        "delta_2": float((ratio < 1.25**2).float().mean()),
        "delta_3": float((ratio < 1.25**3).float().mean()),
    }
    return values, scale, error


def point_metrics(predicted: torch.Tensor, target: torch.Tensor, intrinsics: torch.Tensor) -> dict[str, float]:
    aligned, truth, _ = _depth_values(predicted, target)
    valid = torch.isfinite(predicted) & torch.isfinite(target) & (target > 0.1) & (target < 10.0) & (predicted > 0)
    y, x = torch.where(valid)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    predicted_points = torch.stack(((x - cx) * aligned / fx, (y - cy) * aligned / fy, aligned), dim=-1)
    target_points = torch.stack(((x - cx) * truth / fx, (y - cy) * truth / fy, truth), dim=-1)
    distance = torch.linalg.vector_norm(predicted_points - target_points, dim=-1)
    return {
        "point_error_mean_m_aligned": float(distance.mean()),
        "point_error_median_m_aligned": float(distance.median()),
        "point_fscore_5cm_aligned": float((distance < 0.05).float().mean()),
        "point_fscore_10cm_aligned": float((distance < 0.10).float().mean()),
    }


def calibration_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    logvar: torch.Tensor,
    calibration_indices: list[int],
    test_indices: list[int],
) -> dict[str, float]:
    calibration_ratios: list[torch.Tensor] = []
    test_values: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for index in calibration_indices + test_indices:
        aligned, truth, scale = _depth_values(predicted[index], target[index])
        valid = torch.isfinite(predicted[index]) & torch.isfinite(target[index]) & (target[index] > 0.1) & (
            target[index] < 10.0
        ) & (predicted[index] > 0)
        variance = logvar[index][valid].float().exp() * scale**2
        squared_error = (aligned - truth) ** 2
        if index in calibration_indices:
            calibration_ratios.append(squared_error / variance.clamp_min(1e-8))
        else:
            test_values.append((aligned - truth, variance, torch.exp(-logvar[index][valid].float())))
    variance_scale = float(torch.cat(calibration_ratios).mean().clamp(1e-4, 1e4))
    errors = torch.cat([value[0] for value in test_values])
    raw_variance = torch.cat([value[1] for value in test_values]).clamp_min(1e-8)
    calibrated_variance = raw_variance * variance_scale
    raw_nll = 0.5 * (raw_variance.log() + errors**2 / raw_variance)
    calibrated_nll = 0.5 * (calibrated_variance.log() + errors**2 / calibrated_variance)
    confidence = torch.cat([value[2] for value in test_values])
    relative_error = errors.abs() / torch.cat(
        [_depth_values(predicted[index], target[index])[1] for index in test_indices]
    )
    order = torch.argsort(confidence, descending=True)
    oracle = torch.argsort(relative_error)
    coverages = torch.linspace(0.1, 1.0, 10)
    risk = []
    oracle_risk = []
    for coverage in coverages:
        count = max(1, int(relative_error.numel() * float(coverage)))
        risk.append(float(relative_error[order[:count]].mean()))
        oracle_risk.append(float(relative_error[oracle[:count]].mean()))
    return {
        "uncertainty_variance_scale": variance_scale,
        "uncertainty_raw_nll": float(raw_nll.mean()),
        "uncertainty_calibrated_nll": float(calibrated_nll.mean()),
        "uncertainty_ause": float(np.trapz(np.asarray(risk) - np.asarray(oracle_risk), coverages.numpy())),
        "uncertainty_error_correlation": float(np.corrcoef(confidence.numpy(), relative_error.numpy())[0, 1]),
    }
