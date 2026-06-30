"""Separated SUN RGB-D development caches for Phase 2f.

RGB/K, frozen V-JEPA features, and SUN development targets are physically
separate payloads. No API in this module accepts an external-final archive or
creates an external-final feature cache.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from jepa4d.benchmarks.geometry.sun_rgbd import SUNRGBDFrame, decode_sunrgbd_depth
from jepa4d.data.camera_geometry import update_intrinsics_for_crop_resize
from jepa4d.evaluation.phase2e_feature_cache import deterministic_crop_views
from jepa4d.evaluation.phase2f_camera_controls import (
    BASE_SIZE,
    CAMERA_CONTROL_SCHEMA,
    PROFILE_COUNT,
    PROFILE_IDS,
    PROFILE_PERMUTATION,
    apply_profile_to_rgb,
    build_paired_camera_controls,
    frozen_camera_profiles,
    transform_and_reduce_depth,
)

SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA = "jepa4d-phase2f-sun-dev-input-cache-v1"
SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA = "jepa4d-phase2f-sun-dev-target-cache-v1"
SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA = "jepa4d-phase2f-sun-dev-feature-cache-v1"
SUN_DEVELOPMENT_RECEIPT_SCHEMA = "jepa4d-phase2f-sun-dev-cache-receipt-v1"
SUN_FAMILIES = ("kv1", "xtion", "realsense", "kv2")
ROTATIONS = {
    "R0": {"train": ("kv1", "xtion"), "validation": "realsense", "development_test": "kv2"},
    "R1": {"train": ("xtion", "realsense"), "validation": "kv2", "development_test": "kv1"},
    "R2": {"train": ("realsense", "kv2"), "validation": "kv1", "development_test": "xtion"},
    "R3": {"train": ("kv2", "kv1"), "validation": "xtion", "development_test": "realsense"},
}
CLAIM_BOUNDARY = "SUN RGB-D development cache only; external-final archive and targets are absent"
_FORBIDDEN_EXTERNAL_TOKENS = ("diode", "external_final", "external-target", "final_target", "val.tar.gz")


@dataclass(frozen=True)
class PreparedSunDevelopment:
    sample_ids: list[str]
    family_ids: list[str]
    images_384: torch.Tensor
    rgb_96: torch.Tensor
    intrinsics_384: torch.Tensor
    ordinary_depth_24: torch.Tensor
    ordinary_valid_24: torch.Tensor
    center_depth_384: torch.Tensor
    center_valid_384: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def reject_external_target_references(value: Any, *, location: str = "payload") -> None:
    """Reject external-final names/paths from every development-cache loader."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_external_target_references(key, location=f"{location}.<key>")
            reject_external_target_references(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_external_target_references(item, location=f"{location}[{index}]")
    elif isinstance(value, (str, Path)):
        text = str(value).lower()
        if any(token in text for token in _FORBIDDEN_EXTERNAL_TOKENS):
            raise ValueError(f"external-final reference is forbidden in SUN development cache at {location}")


def _validate_identities(
    sample_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    expected_per_family: int,
) -> tuple[list[str], list[str]]:
    if isinstance(expected_per_family, bool) or not isinstance(expected_per_family, int) or expected_per_family <= 0:
        raise ValueError("expected_per_family must be a positive integer")
    samples = list(sample_ids)
    families = list(family_ids)
    if not samples or len(samples) != len(families):
        raise ValueError("sample_ids and family_ids must be equal non-empty sequences")
    if any(not isinstance(value, str) or not value or not value.isascii() for value in samples + families):
        raise ValueError("sample/family IDs must be non-empty ASCII strings")
    if len(set(samples)) != len(samples):
        raise ValueError("SUN development sample IDs must be unique")
    counts = Counter(families)
    expected = {family: expected_per_family for family in SUN_FAMILIES}
    if dict(counts) != expected:
        raise ValueError(f"SUN development cache requires family counts {expected}; found {dict(counts)}")
    return samples, families


def _to_uint8_images(value: torch.Tensor, expected_shape: tuple[int, ...], label: str) -> torch.Tensor:
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}, got {tuple(value.shape)}")
    if value.dtype == torch.uint8:
        return value.detach().cpu().contiguous()
    if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be uint8 or finite floating point")
    if float(value.min()) < 0 or float(value.max()) > 1:
        raise ValueError(f"floating-point {label} must be in [0,1]")
    return torch.round(value.detach().cpu() * 255).to(torch.uint8).contiguous()


def _validate_k(value: torch.Tensor, expected_shape: tuple[int, ...], label: str) -> torch.Tensor:
    if (
        tuple(value.shape) != expected_shape
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite floating point with shape {expected_shape}")
    expected_last_row = value.new_tensor((0.0, 0.0, 1.0)).expand(*value.shape[:-2], 3)
    if not torch.allclose(value[..., 2, :], expected_last_row, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{label} must have pinhole last row [0,0,1]")
    if not bool((value[..., 0, 0] > 0).all()) or not bool((value[..., 1, 1] > 0).all()):
        raise ValueError(f"{label} must have positive focal lengths")
    return value.detach().cpu().float().contiguous()


def _resize_rgb(rgb: torch.Tensor, crop: tuple[int, int, int, int], size: tuple[int, int]) -> torch.Tensor:
    top, left, height, width = crop
    value = rgb[:, top : top + height, left : left + width].unsqueeze(0)
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False, antialias=True)[0]


def _masked_area_resize(
    depth: torch.Tensor,
    valid: torch.Tensor,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    top, left, height, width = crop
    cropped_depth = depth[top : top + height, left : left + width]
    cropped_valid = valid[top : top + height, left : left + width]
    numerator = torch.where(cropped_valid, cropped_depth, torch.zeros_like(cropped_depth))[None, None]
    mass = cropped_valid.float()[None, None]
    numerator_out = F.interpolate(numerator, size=size, mode="area")[0, 0]
    mass_out = F.interpolate(mass, size=size, mode="area")[0, 0]
    valid_out = mass_out >= 0.25
    depth_out = torch.zeros_like(numerator_out)
    depth_out[valid_out] = numerator_out[valid_out] / mass_out[valid_out]
    return depth_out, valid_out


def prepare_sun_development_frames(
    frames: Sequence[SUNRGBDFrame],
    *,
    clamp_max_depth_m: float | None,
) -> PreparedSunDevelopment:
    """Decode selected SUN frames into the frozen two-view and center-base tensors."""

    if not frames:
        raise ValueError("selected SUN development frames must not be empty")
    images_rows: list[torch.Tensor] = []
    rgb_rows: list[torch.Tensor] = []
    intrinsics_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    center_depth_rows: list[torch.Tensor] = []
    center_valid_rows: list[torch.Tensor] = []
    sample_ids: list[str] = []
    family_ids: list[str] = []
    for frame in frames:
        if frame.intrinsics is None or frame.image_size_hw is None or frame.depth_size_hw is None:
            raise ValueError(f"SUN frame was not validated before preprocessing: {frame.sample_id}")
        if frame.image_size_hw != frame.depth_size_hw:
            raise ValueError(f"SUN frame RGB/depth sizes differ: {frame.sample_id}")
        with Image.open(frame.image_path) as image:
            rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).float().div_(255)
        depth = torch.from_numpy(decode_sunrgbd_depth(frame.depth_path, clamp_max_depth_m=clamp_max_depth_m)).float()
        source_size = tuple(int(value) for value in rgb.shape[-2:])
        if source_size != frame.image_size_hw or tuple(depth.shape) != source_size:
            raise ValueError(f"SUN decoded shape changed: {frame.sample_id}")
        valid = torch.isfinite(depth) & (depth > 0.1) & (depth < 10.0)
        if int(valid.sum()) < 100:
            raise ValueError(f"SUN frame has fewer than 100 valid depth pixels: {frame.sample_id}")
        views = deterministic_crop_views(source_size, paired=True)
        if tuple(view.name for view in views) != ("center_square", "center_crop_0.85"):
            raise RuntimeError("Phase 2f ordinary SUN view policy changed")
        intrinsics = torch.as_tensor(frame.intrinsics, dtype=torch.float32)
        image_views: list[torch.Tensor] = []
        rgb_views: list[torch.Tensor] = []
        k_views: list[torch.Tensor] = []
        depth_views: list[torch.Tensor] = []
        valid_views: list[torch.Tensor] = []
        for view in views:
            image_views.append(_resize_rgb(rgb, view.crop_box, (384, 384)))
            rgb_views.append(_resize_rgb(rgb, view.crop_box, (96, 96)))
            k_views.append(
                update_intrinsics_for_crop_resize(
                    intrinsics,
                    source_size,
                    (384, 384),
                    crop=view.crop_box,
                    half_pixel_centers=True,
                )
            )
            reduced_depth, reduced_valid = _masked_area_resize(depth, valid, view.crop_box, (24, 24))
            depth_views.append(reduced_depth)
            valid_views.append(reduced_valid)
        center_depth, center_valid = _masked_area_resize(depth, valid, views[0].crop_box, BASE_SIZE)
        images_rows.append(torch.stack(image_views))
        rgb_rows.append(torch.stack(rgb_views))
        intrinsics_rows.append(torch.stack(k_views))
        target_rows.append(torch.stack(depth_views))
        valid_rows.append(torch.stack(valid_views))
        center_depth_rows.append(center_depth)
        center_valid_rows.append(center_valid)
        sample_ids.append(frame.sample_id)
        family_ids.append(frame.sensor)
    return PreparedSunDevelopment(
        sample_ids=sample_ids,
        family_ids=family_ids,
        images_384=torch.stack(images_rows),
        rgb_96=torch.stack(rgb_rows),
        intrinsics_384=torch.stack(intrinsics_rows),
        ordinary_depth_24=torch.stack(target_rows),
        ordinary_valid_24=torch.stack(valid_rows),
        center_depth_384=torch.stack(center_depth_rows),
        center_valid_384=torch.stack(center_valid_rows),
    )


def build_sun_development_input_cache(
    *,
    sample_ids: Sequence[str],
    family_ids: Sequence[str],
    images_384: torch.Tensor,
    rgb_96: torch.Tensor,
    intrinsics_384: torch.Tensor,
    sample_manifest_sha256: str,
    expected_per_family: int = 128,
) -> dict[str, Any]:
    """Build two-view ordinary inputs and exact P0-P7 paired RGB/K controls."""

    samples, families = _validate_identities(sample_ids, family_ids, expected_per_family=expected_per_family)
    count = len(samples)
    ordinary_images = _to_uint8_images(images_384, (count, 2, 3, 384, 384), "images_384")
    ordinary_rgb = _to_uint8_images(rgb_96, (count, 2, 3, 96, 96), "rgb_96")
    ordinary_k = _validate_k(intrinsics_384, (count, 2, 3, 3), "intrinsics_384")
    profiles = frozen_camera_profiles()
    paired_images_uint8 = torch.empty((count, PROFILE_COUNT, 3, 384, 384), dtype=torch.uint8)
    for source_index, image_uint8 in enumerate(ordinary_images[:, 0]):
        image = image_uint8.float().div(255)
        for profile_index, profile in enumerate(profiles):
            transformed = apply_profile_to_rgb(image, profile)
            paired_images_uint8[source_index, profile_index] = torch.round(transformed.clamp(0, 1) * 255).to(
                torch.uint8
            )
    controls = build_paired_camera_controls(ordinary_k[:, 0])
    payload: dict[str, Any] = {
        "schema_version": SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": "sun-rgbd-official",
        "sample_manifest_sha256": _validate_sha256(sample_manifest_sha256, "sample_manifest_sha256"),
        "samples": {"sample_ids": samples, "family_ids": families},
        "ordinary_inputs": {
            "view_ids": ["center_square", "center_crop_0.85"],
            "images_384_uint8": ordinary_images,
            "rgb_96_uint8": ordinary_rgb,
            "intrinsics_384": ordinary_k,
        },
        "paired_inputs": {
            "profile_ids": list(PROFILE_IDS),
            "profile_permutation": controls.permutation,
            "images_384_uint8": paired_images_uint8,
            "updated_k": controls.updated.float().contiguous(),
            "stale_k": controls.stale.float().contiguous(),
            "wrong_k": controls.wrong.float().contiguous(),
            "permuted_k": controls.permuted.float().contiguous(),
        },
        "audit": {
            "camera_control_schema": CAMERA_CONTROL_SCHEMA,
            "sample_count": count,
            "expected_per_family": expected_per_family,
            "family_counts": dict(Counter(families)),
            "profiles_per_sample": PROFILE_COUNT,
            "distinct_updated_intrinsics_per_source_min": min(controls.distinct_updated_per_source),
            "distinct_updated_intrinsics_per_source_max": max(controls.distinct_updated_per_source),
            "permutation_assignment_change_fraction": controls.permutation_assignment_change_fraction,
            "permutation_matrix_change_fraction": controls.permutation_matrix_change_fraction,
            "sealed_archive_paths_present": False,
        },
    }
    validate_sun_development_input_cache(payload)
    return payload


def _identity_fields(payload: Mapping[str, Any]) -> tuple[list[str], list[str], int]:
    samples = payload.get("samples")
    if not isinstance(samples, Mapping) or set(samples) != {"sample_ids", "family_ids"}:
        raise ValueError("cache samples must contain only sample_ids/family_ids")
    sample_ids = samples.get("sample_ids")
    family_ids = samples.get("family_ids")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping) or not isinstance(audit.get("expected_per_family"), int):
        raise ValueError("cache audit is missing expected_per_family")
    if not isinstance(sample_ids, list) or not isinstance(family_ids, list):
        raise ValueError("cache sample/family IDs must be lists")
    samples_valid, families_valid = _validate_identities(
        sample_ids,
        family_ids,
        expected_per_family=audit["expected_per_family"],
    )
    return samples_valid, families_valid, len(samples_valid)


def validate_sun_development_input_cache(payload: Mapping[str, Any]) -> None:
    reject_external_target_references(payload)
    expected_root = {
        "schema_version",
        "claim_boundary",
        "dataset_id",
        "sample_manifest_sha256",
        "samples",
        "ordinary_inputs",
        "paired_inputs",
        "audit",
    }
    if set(payload) != expected_root or payload.get("schema_version") != SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA:
        raise ValueError("unexpected SUN development input-cache root/schema")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY or payload.get("dataset_id") != "sun-rgbd-official":
        raise ValueError("SUN development input-cache identity changed")
    _validate_sha256(payload.get("sample_manifest_sha256"), "sample_manifest_sha256")
    _, families, count = _identity_fields(payload)
    ordinary = payload.get("ordinary_inputs")
    if not isinstance(ordinary, Mapping) or set(ordinary) != {
        "view_ids",
        "images_384_uint8",
        "rgb_96_uint8",
        "intrinsics_384",
    }:
        raise ValueError("ordinary input cache has unexpected fields")
    if ordinary.get("view_ids") != ["center_square", "center_crop_0.85"]:
        raise ValueError("ordinary two-view policy changed")
    _to_uint8_images(ordinary["images_384_uint8"], (count, 2, 3, 384, 384), "images_384_uint8")
    _to_uint8_images(ordinary["rgb_96_uint8"], (count, 2, 3, 96, 96), "rgb_96_uint8")
    ordinary_k = _validate_k(ordinary["intrinsics_384"], (count, 2, 3, 3), "intrinsics_384")
    paired = payload.get("paired_inputs")
    if not isinstance(paired, Mapping) or set(paired) != {
        "profile_ids",
        "profile_permutation",
        "images_384_uint8",
        "updated_k",
        "stale_k",
        "wrong_k",
        "permuted_k",
    }:
        raise ValueError("paired input cache has unexpected fields")
    if paired.get("profile_ids") != list(PROFILE_IDS):
        raise ValueError("paired profile order changed")
    permutation = paired.get("profile_permutation")
    if not isinstance(permutation, torch.Tensor) or not torch.equal(
        permutation.cpu(), torch.tensor(PROFILE_PERMUTATION)
    ):
        raise ValueError("paired profile permutation changed")
    _to_uint8_images(paired["images_384_uint8"], (count, 8, 3, 384, 384), "paired images")
    controls = {
        name: _validate_k(paired[f"{name}_k"], (count, 8, 3, 3), f"{name}_k")
        for name in ("updated", "stale", "wrong", "permuted")
    }
    expected_controls = build_paired_camera_controls(ordinary_k[:, 0])
    for name in controls:
        if not torch.equal(controls[name], getattr(expected_controls, name).float().cpu()):
            raise ValueError(f"cached {name}_k differs from the frozen analytic control")
    audit = payload.get("audit")
    expected_audit_fields = {
        "camera_control_schema",
        "sample_count",
        "expected_per_family",
        "family_counts",
        "profiles_per_sample",
        "distinct_updated_intrinsics_per_source_min",
        "distinct_updated_intrinsics_per_source_max",
        "permutation_assignment_change_fraction",
        "permutation_matrix_change_fraction",
        "sealed_archive_paths_present",
    }
    if not isinstance(audit, Mapping) or set(audit) != expected_audit_fields:
        raise ValueError("input-cache audit has unexpected fields")
    if audit.get("sample_count") != count or audit.get("family_counts") != dict(Counter(families)):
        raise ValueError("input-cache audit count mismatch")
    if (
        audit.get("camera_control_schema") != CAMERA_CONTROL_SCHEMA
        or audit.get("profiles_per_sample") != 8
        or audit.get("distinct_updated_intrinsics_per_source_min") != 8
        or audit.get("distinct_updated_intrinsics_per_source_max") != 8
        or audit.get("permutation_assignment_change_fraction") != 1.0
        or audit.get("permutation_matrix_change_fraction") != 1.0
        or audit.get("sealed_archive_paths_present") is not False
    ):
        raise ValueError("input-cache camera-control audit failed")


def build_sun_development_target_cache(
    input_cache: Mapping[str, Any],
    *,
    ordinary_depth_24: torch.Tensor,
    ordinary_valid_24: torch.Tensor,
    center_depth_384: torch.Tensor,
    center_valid_384: torch.Tensor,
    input_cache_sha256: str,
) -> dict[str, Any]:
    """Build a SUN-only target payload physically separate from RGB/K/features."""

    validate_sun_development_input_cache(input_cache)
    sample_ids, family_ids, count = _identity_fields(input_cache)
    if ordinary_depth_24.shape != (count, 2, 24, 24) or ordinary_valid_24.shape != (count, 2, 24, 24):
        raise ValueError("ordinary SUN targets must have shape [N,2,24,24]")
    if center_depth_384.shape != (count, 384, 384) or center_valid_384.shape != (count, 384, 384):
        raise ValueError("center-square SUN targets must have shape [N,384,384]")
    if not torch.is_floating_point(ordinary_depth_24) or ordinary_valid_24.dtype != torch.bool:
        raise TypeError("ordinary depth must be floating point and validity must be bool")
    if not torch.is_floating_point(center_depth_384) or center_valid_384.dtype != torch.bool:
        raise TypeError("center depth must be floating point and validity must be bool")
    if not bool(torch.isfinite(ordinary_depth_24[ordinary_valid_24]).all()):
        raise ValueError("ordinary valid target values must be finite")
    paired_depth_rows: list[torch.Tensor] = []
    paired_valid_rows: list[torch.Tensor] = []
    profiles = frozen_camera_profiles()
    for depth, valid in zip(center_depth_384, center_valid_384, strict=True):
        transformed = [transform_and_reduce_depth(depth, valid, profile) for profile in profiles]
        paired_depth_rows.append(torch.stack([item[0] for item in transformed]))
        paired_valid_rows.append(torch.stack([item[1] for item in transformed]))
    payload: dict[str, Any] = {
        "schema_version": SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": "sun-rgbd-official",
        "sample_manifest_sha256": input_cache["sample_manifest_sha256"],
        "input_cache_sha256": _validate_sha256(input_cache_sha256, "input_cache_sha256"),
        "samples": {"sample_ids": sample_ids, "family_ids": family_ids},
        "ordinary_targets": {
            "view_ids": ["center_square", "center_crop_0.85"],
            "depth_24": ordinary_depth_24.detach().cpu().float().contiguous(),
            "valid_24": ordinary_valid_24.detach().cpu().contiguous(),
        },
        "paired_targets": {
            "profile_ids": list(PROFILE_IDS),
            "depth_24": torch.stack(paired_depth_rows).float().contiguous(),
            "valid_24": torch.stack(paired_valid_rows).contiguous(),
        },
        "audit": {
            "sample_count": count,
            "expected_per_family": input_cache["audit"]["expected_per_family"],
            "family_counts": dict(Counter(family_ids)),
            "profiles_per_sample": 8,
            "target_reduction": "mask-weighted-area-valid-mass-ge-0.25",
            "sealed_archive_paths_present": False,
        },
    }
    validate_sun_development_target_cache(payload)
    return payload


def validate_sun_development_target_cache(payload: Mapping[str, Any]) -> None:
    reject_external_target_references(payload)
    expected_root = {
        "schema_version",
        "claim_boundary",
        "dataset_id",
        "sample_manifest_sha256",
        "input_cache_sha256",
        "samples",
        "ordinary_targets",
        "paired_targets",
        "audit",
    }
    if set(payload) != expected_root or payload.get("schema_version") != SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA:
        raise ValueError("unexpected SUN development target-cache root/schema")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY or payload.get("dataset_id") != "sun-rgbd-official":
        raise ValueError("SUN development target-cache identity changed")
    _validate_sha256(payload.get("sample_manifest_sha256"), "sample_manifest_sha256")
    _validate_sha256(payload.get("input_cache_sha256"), "input_cache_sha256")
    _, families, count = _identity_fields(payload)
    ordinary = payload.get("ordinary_targets")
    if not isinstance(ordinary, Mapping) or set(ordinary) != {"view_ids", "depth_24", "valid_24"}:
        raise ValueError("ordinary target cache has unexpected fields")
    paired = payload.get("paired_targets")
    if not isinstance(paired, Mapping) or set(paired) != {"profile_ids", "depth_24", "valid_24"}:
        raise ValueError("paired target cache has unexpected fields")
    if ordinary.get("view_ids") != ["center_square", "center_crop_0.85"]:
        raise ValueError("ordinary target view order changed")
    if paired.get("profile_ids") != list(PROFILE_IDS):
        raise ValueError("paired target profile order changed")
    for label, depth, valid, expected_shape in (
        ("ordinary", ordinary.get("depth_24"), ordinary.get("valid_24"), (count, 2, 24, 24)),
        ("paired", paired.get("depth_24"), paired.get("valid_24"), (count, 8, 24, 24)),
    ):
        if not isinstance(depth, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise TypeError(f"{label} target tensors are missing")
        if depth.shape != expected_shape or valid.shape != expected_shape or valid.dtype != torch.bool:
            raise ValueError(f"{label} target tensors have invalid shape/dtype")
        if not torch.is_floating_point(depth) or not bool(torch.isfinite(depth[valid]).all()):
            raise ValueError(f"{label} valid depth values must be finite")
    audit = payload.get("audit")
    expected_audit = {
        "sample_count": count,
        "expected_per_family": audit.get("expected_per_family") if isinstance(audit, Mapping) else None,
        "family_counts": dict(Counter(families)),
        "profiles_per_sample": 8,
        "target_reduction": "mask-weighted-area-valid-mass-ge-0.25",
        "sealed_archive_paths_present": False,
    }
    if audit != expected_audit:
        raise ValueError("target-cache audit failed")


def build_sun_development_feature_cache(
    input_cache: Mapping[str, Any],
    *,
    ordinary_features: torch.Tensor,
    paired_features: torch.Tensor,
    input_cache_sha256: str,
) -> dict[str, Any]:
    """Bind raw final-layer V-JEPA grids to input rows without targets."""

    validate_sun_development_input_cache(input_cache)
    sample_ids, family_ids, count = _identity_fields(input_cache)
    if ordinary_features.shape != (count, 2, 768, 24, 24):
        raise ValueError("ordinary_features must have shape [N,2,768,24,24]")
    if paired_features.shape != (count, 8, 768, 24, 24):
        raise ValueError("paired_features must have shape [N,8,768,24,24]")
    if any(
        not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
        for value in (ordinary_features, paired_features)
    ):
        raise ValueError("Phase 2f feature grids must be finite floating point")
    payload: dict[str, Any] = {
        "schema_version": SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": "sun-rgbd-official",
        "sample_manifest_sha256": input_cache["sample_manifest_sha256"],
        "input_cache_sha256": _validate_sha256(input_cache_sha256, "input_cache_sha256"),
        "samples": {"sample_ids": sample_ids, "family_ids": family_ids},
        "ordinary_features": ordinary_features.detach().cpu().half().contiguous(),
        "paired_features": paired_features.detach().cpu().half().contiguous(),
        "audit": {
            "sample_count": count,
            "expected_per_family": input_cache["audit"]["expected_per_family"],
            "family_counts": dict(Counter(family_ids)),
            "ordinary_views_per_sample": 2,
            "paired_profiles_per_sample": 8,
            "normalization": "not-applied-fit-independently-per-rotation-train-families",
            "targets_present": False,
            "sealed_archive_paths_present": False,
        },
    }
    validate_sun_development_feature_cache(payload)
    return payload


def validate_sun_development_feature_cache(payload: Mapping[str, Any]) -> None:
    reject_external_target_references(payload)
    expected_root = {
        "schema_version",
        "claim_boundary",
        "dataset_id",
        "sample_manifest_sha256",
        "input_cache_sha256",
        "samples",
        "ordinary_features",
        "paired_features",
        "audit",
    }
    if set(payload) != expected_root or payload.get("schema_version") != SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA:
        raise ValueError("unexpected SUN development feature-cache root/schema")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY or payload.get("dataset_id") != "sun-rgbd-official":
        raise ValueError("SUN development feature-cache identity changed")
    _validate_sha256(payload.get("sample_manifest_sha256"), "sample_manifest_sha256")
    _validate_sha256(payload.get("input_cache_sha256"), "input_cache_sha256")
    _, families, count = _identity_fields(payload)
    ordinary = payload.get("ordinary_features")
    paired = payload.get("paired_features")
    if not isinstance(ordinary, torch.Tensor) or ordinary.shape != (count, 2, 768, 24, 24):
        raise ValueError("feature cache ordinary grid shape changed")
    if not isinstance(paired, torch.Tensor) or paired.shape != (count, 8, 768, 24, 24):
        raise ValueError("feature cache paired grid shape changed")
    if any(
        not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()) for value in (ordinary, paired)
    ):
        raise ValueError("feature cache grids must be finite floating point")
    audit = payload.get("audit")
    expected_audit = {
        "sample_count": count,
        "expected_per_family": audit.get("expected_per_family") if isinstance(audit, Mapping) else None,
        "family_counts": dict(Counter(families)),
        "ordinary_views_per_sample": 2,
        "paired_profiles_per_sample": 8,
        "normalization": "not-applied-fit-independently-per-rotation-train-families",
        "targets_present": False,
        "sealed_archive_paths_present": False,
    }
    if audit != expected_audit:
        raise ValueError("feature-cache audit failed")


def rotation_indices(payload: Mapping[str, Any], rotation: str) -> dict[str, torch.Tensor]:
    """Return immutable source indices for one of the four frozen rotations."""

    if payload.get("schema_version") == SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA:
        validate_sun_development_input_cache(payload)
    elif payload.get("schema_version") == SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA:
        validate_sun_development_target_cache(payload)
    elif payload.get("schema_version") == SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA:
        validate_sun_development_feature_cache(payload)
    else:
        raise ValueError("rotation indices require a validated SUN development cache")
    if rotation not in ROTATIONS:
        raise ValueError(f"unknown Phase 2f rotation: {rotation}")
    samples = payload["samples"]
    family_ids = samples["family_ids"]
    policy = ROTATIONS[rotation]

    def indices(families: tuple[str, ...]) -> torch.Tensor:
        return torch.tensor([index for index, family in enumerate(family_ids) if family in families], dtype=torch.long)

    return {
        "train": indices(tuple(policy["train"])),
        "validation": indices((str(policy["validation"]),)),
        "development_test": indices((str(policy["development_test"]),)),
    }


def write_cache(path: Path, payload: Mapping[str, Any]) -> Path:
    schema = payload.get("schema_version")
    validators = {
        SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA: validate_sun_development_input_cache,
        SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA: validate_sun_development_target_cache,
        SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA: validate_sun_development_feature_cache,
    }
    if schema not in validators:
        raise ValueError(f"unsupported Phase 2f cache schema: {schema}")
    validators[schema](payload)
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)
    return output
