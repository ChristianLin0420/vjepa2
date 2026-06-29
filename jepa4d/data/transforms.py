"""Deterministic V-JEPA image normalization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_center_crop_normalize(images: torch.Tensor, size: int = 384) -> torch.Tensor:
    """Resize shortest side, center crop, and ImageNet-normalize [...,3,H,W]."""
    if images.shape[-3] != 3:
        raise ValueError("expected RGB channels in the third-to-last dimension")
    leading = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:]).float()
    height, width = flat.shape[-2:]
    scale = size / min(height, width)
    resized = F.interpolate(
        flat, size=(round(height * scale), round(width * scale)), mode="bilinear", align_corners=False
    )
    top = (resized.shape[-2] - size) // 2
    left = (resized.shape[-1] - size) // 2
    cropped = resized[..., top : top + size, left : left + size]
    mean = torch.tensor(IMAGENET_MEAN, dtype=cropped.dtype, device=cropped.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=cropped.dtype, device=cropped.device).view(1, 3, 1, 1)
    return ((cropped - mean) / std).reshape(*leading, 3, size, size)
