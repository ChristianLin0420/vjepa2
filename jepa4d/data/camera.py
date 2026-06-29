"""Camera calibration helpers."""

from __future__ import annotations

import torch


def default_pinhole_intrinsics(
    batch: int, views: int, height: int, width: int, focal_ratio: float = 1.2
) -> torch.Tensor:
    """Return a conservative prior, explicitly intended as an uncertain initialization."""
    matrix = torch.eye(3).repeat(batch, views, 1, 1)
    matrix[..., 0, 0] = focal_ratio * width
    matrix[..., 1, 1] = focal_ratio * height
    matrix[..., 0, 2] = (width - 1) / 2
    matrix[..., 1, 2] = (height - 1) / 2
    return matrix
