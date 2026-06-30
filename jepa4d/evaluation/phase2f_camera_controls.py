"""Exact preregistered Phase 2f paired camera controls.

All profiles operate on an already-created 384x384 SUN RGB-D center-square
base. The fixed permutation is within the eight profiles of each source frame.
This module is development-only and must never be applied to DIODE targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from jepa4d.data.camera_geometry import update_intrinsics_for_crop_resize

CAMERA_CONTROL_SCHEMA = "jepa4d-phase2f-paired-camera-controls-v1"
BASE_SIZE = (384, 384)
PROFILE_COUNT = 8
PROFILE_IDS = tuple(f"P{index}" for index in range(PROFILE_COUNT))
PROFILE_PERMUTATION = (5, 6, 3, 2, 1, 7, 0, 4)
MIN_DISTINCT_INTRINSICS = 8
MIN_PERMUTATION_CHANGE_FRACTION = 0.95
WRONG_FOCAL_SCALE = 1.25
WRONG_PRINCIPAL_SHIFT = (38.4, -38.4)


def _validate_base_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.ndim == 2:
        values = intrinsics.unsqueeze(0)
    elif intrinsics.ndim == 3:
        values = intrinsics
    else:
        raise ValueError(f"base intrinsics must be [3,3] or [N,3,3], got {tuple(intrinsics.shape)}")
    if tuple(values.shape[-2:]) != (3, 3) or len(values) == 0:
        raise ValueError("base intrinsics must contain at least one 3x3 matrix")
    if not torch.is_floating_point(values) or not bool(torch.isfinite(values).all()):
        raise ValueError("base intrinsics must be finite floating-point matrices")
    if not bool((values[:, 0, 0] > 0).all()) or not bool((values[:, 1, 1] > 0).all()):
        raise ValueError("base intrinsics must have positive focal lengths")
    expected = values.new_tensor((0.0, 0.0, 1.0)).expand(len(values), -1)
    if not torch.allclose(values[:, 2], expected, rtol=1e-6, atol=1e-7):
        raise ValueError("base intrinsics must have pinhole last row [0,0,1]")
    return values


@dataclass(frozen=True)
class CameraProfile:
    profile_id: str
    crop: tuple[int, int, int, int]
    resized_size: tuple[int, int]
    padding_tlbr: tuple[int, int, int, int]

    def validate(self) -> None:
        if self.profile_id not in PROFILE_IDS:
            raise ValueError(f"invalid Phase 2f profile ID: {self.profile_id}")
        top, left, height, width = self.crop
        if min(top, left) < 0 or min(height, width) <= 0 or top + height > 384 or left + width > 384:
            raise ValueError(f"profile {self.profile_id} crop is outside the 384x384 base")
        resized_height, resized_width = self.resized_size
        pad_top, pad_left, pad_bottom, pad_right = self.padding_tlbr
        if min(resized_height, resized_width) <= 0 or min(self.padding_tlbr) < 0:
            raise ValueError(f"profile {self.profile_id} has invalid resize/padding")
        if resized_height + pad_top + pad_bottom != 384 or resized_width + pad_left + pad_right != 384:
            raise ValueError(f"profile {self.profile_id} does not produce 384x384")


def frozen_camera_profiles() -> tuple[CameraProfile, ...]:
    """Return the exact P0-P7 profile order frozen in the preregistration."""

    profiles = (
        CameraProfile("P0", (0, 0, 384, 384), (384, 384), (0, 0, 0, 0)),
        CameraProfile("P1", (29, 29, 326, 326), (384, 384), (0, 0, 0, 0)),
        CameraProfile("P2", (0, 29, 326, 326), (384, 384), (0, 0, 0, 0)),
        CameraProfile("P3", (58, 29, 326, 326), (384, 384), (0, 0, 0, 0)),
        CameraProfile("P4", (29, 0, 326, 326), (384, 384), (0, 0, 0, 0)),
        CameraProfile("P5", (29, 58, 326, 326), (384, 384), (0, 0, 0, 0)),
        CameraProfile("P6", (0, 0, 384, 384), (326, 326), (29, 29, 29, 29)),
        CameraProfile("P7", (0, 0, 384, 384), (326, 384), (29, 0, 29, 0)),
    )
    if tuple(profile.profile_id for profile in profiles) != PROFILE_IDS:
        raise RuntimeError("Phase 2f profile order changed")
    for profile in profiles:
        profile.validate()
    return profiles


def _apply_profile_to_chw(value: torch.Tensor, profile: CameraProfile, *, mode: str) -> torch.Tensor:
    if value.ndim != 3 or tuple(value.shape[-2:]) != BASE_SIZE or not torch.is_floating_point(value):
        raise ValueError("profile inputs must be floating-point [C,384,384]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("profile inputs must be finite")
    profile.validate()
    top, left, height, width = profile.crop
    cropped = value[:, top : top + height, left : left + width].unsqueeze(0)
    if mode == "bilinear":
        resized = F.interpolate(
            cropped,
            size=profile.resized_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
    elif mode == "area":
        resized = F.interpolate(cropped, size=profile.resized_size, mode="area")[0]
    else:
        raise ValueError(f"unsupported profile interpolation mode: {mode}")
    pad_top, pad_left, pad_bottom, pad_right = profile.padding_tlbr
    output = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
    if tuple(output.shape[-2:]) != BASE_SIZE:
        raise RuntimeError(f"profile {profile.profile_id} did not produce 384x384")
    return output


def apply_profile_to_rgb(image: torch.Tensor, profile: CameraProfile) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("RGB profile input must have shape [3,384,384]")
    return _apply_profile_to_chw(image, profile, mode="bilinear")


def apply_profile_to_intrinsics(intrinsics: torch.Tensor, profile: CameraProfile) -> torch.Tensor:
    """Apply the exact P0-P7 crop/resize/pad transform to one base K."""

    if intrinsics.ndim != 2 or tuple(intrinsics.shape) != (3, 3):
        raise ValueError("one camera profile requires one [3,3] base matrix")
    _validate_base_intrinsics(intrinsics)
    profile.validate()
    updated = update_intrinsics_for_crop_resize(
        intrinsics,
        BASE_SIZE,
        profile.resized_size,
        crop=profile.crop,
        half_pixel_centers=True,
    ).clone()
    pad_top, pad_left, _, _ = profile.padding_tlbr
    updated[0, 2] += pad_left
    updated[1, 2] += pad_top
    return updated


def transform_and_reduce_depth(
    depth: torch.Tensor,
    valid: torch.Tensor,
    profile: CameraProfile,
    *,
    output_size: tuple[int, int] = (24, 24),
    min_valid_mass: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one profile and preregistered mask-weighted area reduction."""

    if depth.ndim != 2 or valid.ndim != 2 or depth.shape != valid.shape or tuple(depth.shape) != BASE_SIZE:
        raise ValueError("depth and valid must both have shape [384,384]")
    if not torch.is_floating_point(depth) or valid.dtype != torch.bool:
        raise TypeError("depth must be floating point and valid must be bool")
    if not bool(torch.isfinite(depth[valid]).all()) or not bool((depth[valid] > 0).all()):
        raise ValueError("valid depth values must be finite and positive")
    if len(output_size) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in output_size
    ):
        raise ValueError("output_size must contain two positive integers")
    if not math.isfinite(min_valid_mass) or not 0 < min_valid_mass <= 1:
        raise ValueError("min_valid_mass must be in (0,1]")
    weighted = torch.where(valid, depth, torch.zeros_like(depth)).unsqueeze(0)
    mass = valid.float().unsqueeze(0)
    transformed_weighted = _apply_profile_to_chw(weighted, profile, mode="area")
    transformed_mass = _apply_profile_to_chw(mass, profile, mode="area")
    numerator = F.interpolate(transformed_weighted.unsqueeze(0), size=output_size, mode="area")[0, 0]
    denominator = F.interpolate(transformed_mass.unsqueeze(0), size=output_size, mode="area")[0, 0]
    reduced_valid = denominator >= min_valid_mass
    reduced_depth = torch.zeros_like(numerator)
    reduced_depth[reduced_valid] = numerator[reduced_valid] / denominator[reduced_valid]
    if not bool(torch.isfinite(reduced_depth).all()):
        raise ValueError("mask-weighted target reduction produced non-finite values")
    return reduced_depth, reduced_valid


def _distinct_count(matrices: torch.Tensor, tolerance: float = 1e-6) -> int:
    quantized = torch.round(matrices.detach().cpu().double() / tolerance).to(torch.int64).reshape(len(matrices), -1)
    return int(torch.unique(quantized, dim=0).shape[0])


def matrix_change_fraction(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.ndim < 3 or tuple(first.shape[-2:]) != (3, 3):
        raise ValueError("camera matrix batches must have equal [...,3,3] shape")
    left = first.reshape(-1, 3, 3)
    right = second.reshape(-1, 3, 3)
    unchanged = torch.isclose(left, right, rtol=1e-7, atol=1e-6).flatten(1).all(dim=1)
    return float((~unchanged).double().mean().item())


def validate_profile_permutation(
    updated: torch.Tensor,
    permutation: torch.Tensor,
    *,
    minimum_change_fraction: float = MIN_PERMUTATION_CHANGE_FRACTION,
) -> tuple[torch.Tensor, float, float]:
    """Apply a within-source profile permutation and fail closed below 95%."""

    if updated.ndim != 4 or updated.shape[1:] != (PROFILE_COUNT, 3, 3):
        raise ValueError("updated K must have shape [N,8,3,3]")
    if permutation.ndim != 1 or len(permutation) != PROFILE_COUNT or permutation.dtype != torch.long:
        raise ValueError("profile permutation must be int64 with shape [8]")
    values = permutation.detach().cpu()
    if not torch.equal(values.sort().values, torch.arange(PROFILE_COUNT)):
        raise ValueError("profile permutation must be bijective")
    if not math.isfinite(minimum_change_fraction) or not 0 < minimum_change_fraction <= 1:
        raise ValueError("minimum_change_fraction must be in (0,1]")
    assignment_fraction = float((values != torch.arange(PROFILE_COUNT)).double().mean().item())
    permuted = updated.index_select(1, values.to(updated.device))
    matrix_fraction = matrix_change_fraction(updated, permuted)
    if assignment_fraction + 1e-12 < minimum_change_fraction:
        raise ValueError(
            f"profile permutation changes only {assignment_fraction:.6f} of assignments; "
            f"required >= {minimum_change_fraction:.6f}"
        )
    if matrix_fraction + 1e-12 < minimum_change_fraction:
        raise ValueError(
            f"profile permutation changes only {matrix_fraction:.6f} of matrices; "
            f"required >= {minimum_change_fraction:.6f}"
        )
    return permuted, assignment_fraction, matrix_fraction


@dataclass(frozen=True)
class PairedCameraControls:
    updated: torch.Tensor
    stale: torch.Tensor
    wrong: torch.Tensor
    permuted: torch.Tensor
    permutation: torch.Tensor
    distinct_updated_per_source: tuple[int, ...]
    permutation_assignment_change_fraction: float
    permutation_matrix_change_fraction: float


def build_paired_camera_controls(base_intrinsics: torch.Tensor) -> PairedCameraControls:
    """Build exact ``[source,profile,3,3]`` updated/stale/wrong/permuted K."""

    base = _validate_base_intrinsics(base_intrinsics)
    profiles = frozen_camera_profiles()
    updated = torch.stack(
        [torch.stack([apply_profile_to_intrinsics(matrix, profile) for profile in profiles]) for matrix in base]
    )
    stale = base[:, None].expand(-1, PROFILE_COUNT, -1, -1).clone()
    wrong = updated.clone()
    wrong[..., 0, 0] *= WRONG_FOCAL_SCALE
    wrong[..., 1, 1] *= WRONG_FOCAL_SCALE
    wrong[..., 0, 2] += WRONG_PRINCIPAL_SHIFT[0]
    wrong[..., 1, 2] += WRONG_PRINCIPAL_SHIFT[1]
    permutation = torch.tensor(PROFILE_PERMUTATION, dtype=torch.long)
    permuted, assignment_fraction, matrix_fraction = validate_profile_permutation(
        updated,
        permutation,
        minimum_change_fraction=1.0,
    )
    distinct = tuple(_distinct_count(source) for source in updated)
    if any(count != MIN_DISTINCT_INTRINSICS for count in distinct):
        raise ValueError(f"all sources require exactly eight distinct analytic K matrices; found {distinct}")
    if matrix_change_fraction(updated, wrong) != 1.0:
        raise RuntimeError("wrong-K construction did not change every analytic matrix")
    return PairedCameraControls(
        updated=updated,
        stale=stale,
        wrong=wrong,
        permuted=permuted,
        permutation=permutation,
        distinct_updated_per_source=distinct,
        permutation_assignment_change_fraction=assignment_fraction,
        permutation_matrix_change_fraction=matrix_fraction,
    )


def validate_camera_output_delta(
    updated_output_m: torch.Tensor,
    controlled_output_m: torch.Tensor,
    *,
    minimum_mean_absolute_delta_m: float = 1e-6,
) -> float:
    """Fail closed when a camera-conditioned model is insensitive to a K control."""

    if updated_output_m.shape != controlled_output_m.shape or updated_output_m.numel() == 0:
        raise ValueError("camera outputs must have equal non-empty shapes")
    if not torch.is_floating_point(updated_output_m) or not torch.is_floating_point(controlled_output_m):
        raise TypeError("camera outputs must be floating point")
    if not bool(torch.isfinite(updated_output_m).all()) or not bool(torch.isfinite(controlled_output_m).all()):
        raise ValueError("camera outputs must be finite")
    if not math.isfinite(minimum_mean_absolute_delta_m) or minimum_mean_absolute_delta_m <= 0:
        raise ValueError("minimum output delta must be finite and positive")
    delta = float((updated_output_m - controlled_output_m).abs().double().mean().item())
    if delta <= minimum_mean_absolute_delta_m:
        raise ValueError(
            f"camera-conditioned output delta {delta:.9g} m does not exceed {minimum_mean_absolute_delta_m:.9g} m"
        )
    return delta
