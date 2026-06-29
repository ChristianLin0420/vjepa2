"""Pinhole-camera geometry utilities for camera-aware dense prediction.

The resize convention matches ``torch.nn.functional.interpolate`` with
``align_corners=False``: integer pixel indices denote pixel centres and the
resize transform preserves the half-pixel coordinate convention.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

IntrinsicsControl = Literal["correct", "wrong", "shuffled"]


def _batched_intrinsics(intrinsics: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if intrinsics.ndim == 2:
        if intrinsics.shape != (3, 3):
            raise ValueError(f"expected intrinsics [3,3], got {tuple(intrinsics.shape)}")
        values = intrinsics.unsqueeze(0)
        unbatched = True
    elif intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        values = intrinsics
        unbatched = False
    else:
        raise ValueError(f"expected intrinsics [3,3] or [B,3,3], got {tuple(intrinsics.shape)}")
    if not torch.is_floating_point(values):
        raise TypeError("intrinsics must use a floating-point dtype")
    if not torch.isfinite(values).all():
        raise ValueError("intrinsics must be finite")
    if not bool((values[..., 0, 0] > 0).all()) or not bool((values[..., 1, 1] > 0).all()):
        raise ValueError("pinhole focal lengths must be positive")
    expected_last_row = values.new_tensor((0.0, 0.0, 1.0)).expand_as(values[..., 2, :])
    if not torch.allclose(values[..., 2, :], expected_last_row, rtol=1e-5, atol=1e-6):
        raise ValueError("intrinsics must have the pinhole last row [0,0,1]")
    determinant = torch.linalg.det(values.float() if values.dtype in {torch.float16, torch.bfloat16} else values)
    if not bool((determinant.abs() > 1e-12).all()):
        raise ValueError("intrinsics must be invertible")
    return values, unbatched


def _positive_image_size(size: tuple[int, int], label: str) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{label} must be (height, width)")
    height, width = size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError(f"{label} must contain positive integers, got {size}")
    return height, width


def update_intrinsics_for_crop_resize(
    intrinsics: torch.Tensor,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    *,
    crop: tuple[int, int, int, int] | None = None,
    half_pixel_centers: bool = True,
) -> torch.Tensor:
    """Update a pinhole matrix after a crop followed by a resize.

    Args:
        intrinsics: ``[3,3]`` or ``[B,3,3]`` matrices in ``input_size`` pixels.
        input_size: Source ``(height, width)``.
        output_size: Destination ``(height, width)``.
        crop: Optional ``(top, left, height, width)`` in source pixels. The
            full source image is used when omitted.
        half_pixel_centers: Use the half-pixel mapping employed by bilinear
            interpolation with ``align_corners=False``. Disable only when the
            caller's image transform uses the simple corner-origin mapping.

    Returns:
        Intrinsics with the same batched/unbatched shape as the input.
    """

    values, unbatched = _batched_intrinsics(intrinsics)
    input_height, input_width = _positive_image_size(input_size, "input_size")
    output_height, output_width = _positive_image_size(output_size, "output_size")
    if crop is None:
        top, left, crop_height, crop_width = 0, 0, input_height, input_width
    else:
        if len(crop) != 4:
            raise ValueError("crop must be (top, left, height, width)")
        top, left, crop_height, crop_width = crop
        if any(isinstance(value, bool) or not isinstance(value, int) for value in crop):
            raise ValueError("crop values must be integers")
        if top < 0 or left < 0 or crop_height <= 0 or crop_width <= 0:
            raise ValueError(f"crop must be positive and in bounds, got {crop}")
        if top + crop_height > input_height or left + crop_width > input_width:
            raise ValueError(f"crop {crop} exceeds input size {input_size}")

    scale_x = output_width / crop_width
    scale_y = output_height / crop_height
    if half_pixel_centers:
        offset_x = scale_x * (0.5 - left) - 0.5
        offset_y = scale_y * (0.5 - top) - 0.5
    else:
        offset_x = -scale_x * left
        offset_y = -scale_y * top
    transform = values.new_zeros((values.shape[0], 3, 3))
    transform[:, 0, 0] = scale_x
    transform[:, 1, 1] = scale_y
    transform[:, 0, 2] = offset_x
    transform[:, 1, 2] = offset_y
    transform[:, 2, 2] = 1.0
    updated = transform @ values
    return updated[0] if unbatched else updated


def resize_intrinsics(
    intrinsics: torch.Tensor,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    *,
    half_pixel_centers: bool = True,
) -> torch.Tensor:
    """Update intrinsics for a full-frame resize."""

    return update_intrinsics_for_crop_resize(
        intrinsics,
        input_size,
        output_size,
        half_pixel_centers=half_pixel_centers,
    )


def apply_intrinsics_control(
    intrinsics: torch.Tensor,
    control: IntrinsicsControl = "correct",
    *,
    permutation: torch.Tensor | None = None,
    wrong_focal_scale: float = 1.25,
    wrong_principal_shift: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Apply an explicit camera-information control for causal ablations.

    ``wrong`` deterministically perturbs focal length and, optionally, the
    principal point. ``shuffled`` mismatches matrices across batch examples;
    its default permutation is a one-position cyclic roll. A caller may pass
    an explicit permutation to make the mismatch identity part of a protocol.
    """

    values, unbatched = _batched_intrinsics(intrinsics)
    if control == "correct":
        controlled = values.clone()
    elif control == "wrong":
        shift_x, shift_y = wrong_principal_shift
        if not all(math.isfinite(value) for value in (wrong_focal_scale, shift_x, shift_y)):
            raise ValueError("wrong-camera control values must be finite")
        if wrong_focal_scale <= 0:
            raise ValueError("wrong_focal_scale must be positive")
        if wrong_focal_scale == 1.0 and shift_x == 0.0 and shift_y == 0.0:
            raise ValueError("wrong-camera control must actually perturb the intrinsics")
        controlled = values.clone()
        controlled[..., 0, 0] *= wrong_focal_scale
        controlled[..., 0, 1] *= wrong_focal_scale
        controlled[..., 1, 1] *= wrong_focal_scale
        controlled[..., 0, 2] += shift_x
        controlled[..., 1, 2] += shift_y
    elif control == "shuffled":
        if unbatched or values.shape[0] < 2:
            raise ValueError("shuffled-camera control requires batched intrinsics with at least two examples")
        if permutation is None:
            permutation = torch.arange(values.shape[0], device=values.device).roll(1)
        if permutation.ndim != 1 or len(permutation) != values.shape[0]:
            raise ValueError("camera permutation must have one index per batch example")
        if permutation.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("camera permutation must use an integer dtype")
        permutation = permutation.to(device=values.device, dtype=torch.long)
        expected = torch.arange(values.shape[0], device=values.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("camera permutation must contain every batch index exactly once")
        controlled = values.index_select(0, permutation)
    else:
        raise ValueError(f"unknown intrinsics control: {control}")
    return controlled[0] if unbatched else controlled


def normalized_camera_rays(intrinsics: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Return a unit camera ray at every integer pixel centre.

    The output is ``[3,H,W]`` for an unbatched matrix and ``[B,3,H,W]`` for
    batched matrices. The matrices must already be expressed in the coordinate
    system of ``image_size``; use :func:`update_intrinsics_for_crop_resize`
    before this function when images were transformed.
    """

    values, unbatched = _batched_intrinsics(intrinsics)
    height, width = _positive_image_size(image_size, "image_size")
    y, x = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=values.dtype),
        torch.arange(width, device=values.device, dtype=values.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x))).reshape(1, 3, height * width)
    pixels = pixels.expand(values.shape[0], -1, -1)
    solve_values = values.float() if values.dtype in {torch.float16, torch.bfloat16} else values
    rays = torch.linalg.solve(solve_values, pixels.to(dtype=solve_values.dtype))
    epsilon = torch.finfo(rays.dtype).eps
    rays = rays / torch.linalg.vector_norm(rays, dim=1, keepdim=True).clamp_min(epsilon)
    rays = rays.reshape(values.shape[0], 3, height, width)
    return rays[0] if unbatched else rays


def normalized_intrinsics_summary(intrinsics: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Summarize focal length/FoV and principal-point offset in four values."""

    values, unbatched = _batched_intrinsics(intrinsics)
    height, width = _positive_image_size(image_size, "image_size")
    summary = torch.stack(
        (
            (values[..., 0, 0] / width).log(),
            (values[..., 1, 1] / height).log(),
            (values[..., 0, 2] + 0.5) / width - 0.5,
            (values[..., 1, 2] + 0.5) / height - 0.5,
        ),
        dim=-1,
    )
    return summary[0] if unbatched else summary


def camera_ray_summary(rays: torch.Tensor) -> torch.Tensor:
    """Return per-axis mean and standard deviation of a normalized ray map."""

    if rays.ndim == 3:
        if rays.shape[0] != 3:
            raise ValueError(f"expected rays [3,H,W], got {tuple(rays.shape)}")
        values = rays.unsqueeze(0)
        unbatched = True
    elif rays.ndim == 4 and rays.shape[1] == 3:
        values = rays
        unbatched = False
    else:
        raise ValueError(f"expected rays [3,H,W] or [B,3,H,W], got {tuple(rays.shape)}")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError("ray maps must be finite floating-point tensors")
    flattened = values.flatten(2)
    summary = torch.cat((flattened.mean(dim=-1), flattened.std(dim=-1, unbiased=False)), dim=-1)
    return summary[0] if unbatched else summary
