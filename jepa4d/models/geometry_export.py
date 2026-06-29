"""Portable NPZ, PLY, and COLMAP text exports for geometry beliefs."""

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


def _rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to COLMAP's scalar-first quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qvec = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        nxt = (axis + 1) % 3
        last = (axis + 2) % 3
        scale = 2.0 * np.sqrt(max(1.0 + matrix[axis, axis] - matrix[nxt, nxt] - matrix[last, last], 0.0))
        qvec = np.zeros(4, dtype=np.float64)
        qvec[axis + 1] = 0.25 * scale
        qvec[0] = (matrix[last, nxt] - matrix[nxt, last]) / max(scale, 1e-12)
        qvec[nxt + 1] = (matrix[nxt, axis] + matrix[axis, nxt]) / max(scale, 1e-12)
        qvec[last + 1] = (matrix[last, axis] + matrix[axis, last]) / max(scale, 1e-12)
    qvec /= max(np.linalg.norm(qvec), 1e-12)
    return qvec if qvec[0] >= 0 else -qvec


def export_colmap_text(belief: GeometryBelief, batch: RGBInputBatch, directory: str | Path) -> Path:
    """Export cameras and poses in COLMAP text format.

    The belief stores camera-from-world transforms, which is the convention
    expected by COLMAP's ``images.txt``. Dense points remain authoritative in
    the NPZ/PLY outputs, so ``points3D.txt`` is intentionally empty.
    """
    if belief.camera_extrinsics is None or belief.camera_intrinsics is None:
        raise ValueError("camera extrinsics and intrinsics are required for COLMAP export")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    extrinsics = belief.camera_extrinsics[0].reshape(-1, 4, 4).detach().cpu().numpy()
    intrinsics = belief.camera_intrinsics[0].reshape(-1, 3, 3).detach().cpu().numpy()
    height, width = belief.metadata.get("output_resolution", batch.images.shape[-2:])
    source_refs = [ref for view in batch.source_refs[0] for ref in view] if batch.source_refs else []

    camera_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
    ]
    for index, (intrinsic, extrinsic) in enumerate(zip(intrinsics, extrinsics, strict=True), start=1):
        camera_lines.append(
            f"{index} PINHOLE {int(width)} {int(height)} {intrinsic[0, 0]:.12g} {intrinsic[1, 1]:.12g} "
            f"{intrinsic[0, 2]:.12g} {intrinsic[1, 2]:.12g}"
        )
        qvec = _rotation_matrix_to_qvec(extrinsic[:3, :3])
        tvec = extrinsic[:3, 3]
        name = Path(source_refs[index - 1]).name if index <= len(source_refs) else f"frame_{index - 1:06d}.png"
        values = [*qvec.tolist(), *tvec.tolist()]
        image_lines.append(f"{index} {' '.join(f'{value:.12g}' for value in values)} {index} {name}")
        image_lines.append("")

    (target / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (target / "images.txt").write_text("\n".join(image_lines) + "\n")
    (target / "points3D.txt").write_text(
        "# Empty: dense geometry is stored in geometry_belief.npz and pointcloud.ply\n"
    )
    return target
