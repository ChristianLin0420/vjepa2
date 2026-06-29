"""Portable NPZ and PLY exports for geometry beliefs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jepa4d.data.schemas import RGBInputBatch
from jepa4d.models.geometry_belief import GeometryBelief


def export_geometry_npz(belief: GeometryBelief, path: str | Path) -> Path:
    return belief.save_npz(path)


def export_pointcloud_ply(
    belief: GeometryBelief,
    batch: RGBInputBatch,
    path: str | Path,
    *,
    max_points: int = 100_000,
    max_logvar: float | None = None,
) -> Path:
    """Write an ASCII PLY with RGB colors and optional uncertainty filtering."""
    if belief.pointmap_mean is None:
        raise ValueError("point-map belief is required for PLY export")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    points = belief.pointmap_mean[0].reshape(-1, 3).detach().cpu()
    views, steps, height, width = belief.pointmap_mean.shape[1:5]
    images = batch.images[0].reshape(views * steps, 3, *batch.images.shape[-2:])
    colors = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
    colors = (colors.permute(0, 2, 3, 1).reshape(-1, 3).clamp(0, 1) * 255).byte().cpu()
    valid = torch.isfinite(points).all(dim=-1)
    if max_logvar is not None and belief.pointmap_logvar is not None:
        uncertainty = belief.pointmap_logvar[0].reshape(-1, 3).mean(dim=-1).detach().cpu()
        valid &= uncertainty <= max_logvar
    indices = torch.where(valid)[0]
    if indices.numel() > max_points:
        positions = torch.linspace(0, indices.numel() - 1, max_points).long()
        indices = indices[positions]
    points_np = points[indices].numpy()
    colors_np = colors[indices].numpy()
    with target.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points_np)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.concatenate((points_np, colors_np), axis=1), fmt="%.6f %.6f %.6f %d %d %d")
    return target
