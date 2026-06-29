"""Deterministic SUN RGB-D preprocessing and Phase-2e cache contracts.

The functions in this module are model-agnostic.  They prepare camera-aligned
RGB/depth views, apply train-only V-JEPA feature normalization, center a relative
depth teacher, and validate the two deliberately separate cache artifacts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from jepa4d.benchmarks.geometry.sun_rgbd import SUNRGBDFrame, decode_sunrgbd_depth
from jepa4d.data.camera_geometry import update_intrinsics_for_crop_resize

CACHE_SCHEMA = "jepa4d-phase2e-feature-cache-v1"
RECEIPT_SCHEMA = "jepa4d-phase2e-feature-cache-receipt-v1"
VIEW_POLICY = "center-square-plus-deterministic-0.85-center-crop-v1"
FEATURE_SHAPE = (768, 24, 24)
RGB_SHAPE = (3, 96, 96)
TARGET_SHAPE = (24, 24)
FROZEN_DATASET_ID = "sun-rgbd-official"
FROZEN_DATASET_VERSION = "SUNRGBD-v1"
FROZEN_MANIFEST_SHA256 = "174716f4f1bd4a4b709f2a4b1a1cd4dca4fd17ef34cc543c1fc8985b75b44c92"
FROZEN_SPLIT_SHA256 = "d1815109fa0b34dd2270f1da616d4ff65beaa41fcf437b7d75ead557a1ab75c7"
FROZEN_SPLIT_COUNTS = {"train": 384, "validation": 128, "test": 128}
FROZEN_MODEL_IDENTITIES = {
    "vjepa_checkpoint": {
        "files": 4,
        "bytes": 438_859_320,
        "content_manifest_sha256": "8c61f645d6252d619acdd15bca42f210fc27768050cc9995ebaa98cf6d779908",
    },
    "vjepa_implementation": {
        "files": 4,
        "bytes": 50_869,
        "content_manifest_sha256": "2479dbf282e31821dddfea7b8f26b4aee629b762c8fad4023d1f57a7e3f55d8c",
    },
    "vggt_checkpoint": {
        "files": 4,
        "bytes": 5_026_368_503,
        "content_manifest_sha256": "5a388a7fe320cf909c5ee438535a7bc2b2ddfae8aa765dbb60ad8843a766de74",
    },
}

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class CropView:
    name: str
    crop_box: tuple[int, int, int, int]


@dataclass(slots=True)
class PreparedSplit:
    """Uniform internal representation with an explicit view axis."""

    name: SplitName
    images_384: torch.Tensor
    rgb_96: torch.Tensor
    intrinsics_384: torch.Tensor
    targets_24: torch.Tensor
    sample_ids: list[str]
    sensor_ids: list[str]
    group_ids: list[str]
    crop_boxes: torch.Tensor
    source_sizes: torch.Tensor
    view_names: tuple[str, ...]

    @property
    def count(self) -> int:
        return int(self.images_384.shape[0])

    @property
    def views(self) -> int:
        return int(self.images_384.shape[1])

    def metadata_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, sample_id in enumerate(self.sample_ids):
            rows.append(
                {
                    "sample_id": sample_id,
                    "sensor_id": self.sensor_ids[index],
                    "group_id": self.group_ids[index],
                    "views": [
                        {
                            "view_name": name,
                            "crop_box_top_left_height_width": self.crop_boxes[index, view].tolist(),
                            "source_size_height_width": self.source_sizes[index, view].tolist(),
                        }
                        for view, name in enumerate(self.view_names)
                    ],
                }
            )
        return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_crop_views(
    source_size: tuple[int, int], *, paired: bool, crop_scale: float = 0.85
) -> tuple[CropView, ...]:
    """Return the frozen centre-square and optional nested 0.85 crop."""
    height, width = source_size
    if height <= 0 or width <= 0:
        raise ValueError(f"source_size must be positive, got {source_size}")
    if not math.isfinite(crop_scale) or not 0 < crop_scale < 1:
        raise ValueError("crop_scale must be finite and strictly between zero and one")
    square_side = min(height, width)
    square = CropView(
        "center_square",
        ((height - square_side) // 2, (width - square_side) // 2, square_side, square_side),
    )
    if not paired:
        return (square,)
    crop_side = max(1, int(math.floor(square_side * crop_scale)))
    crop = CropView(
        "center_crop_0.85",
        ((height - crop_side) // 2, (width - crop_side) // 2, crop_side, crop_side),
    )
    return square, crop


def _resize_rgb(rgb: torch.Tensor, crop: tuple[int, int, int, int], size: int) -> torch.Tensor:
    top, left, height, width = crop
    value = rgb[:, top : top + height, left : left + width].unsqueeze(0)
    return F.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)[0]


def _resize_depth(depth: torch.Tensor, crop: tuple[int, int, int, int], size: int) -> torch.Tensor:
    top, left, height, width = crop
    value = depth[top : top + height, left : left + width].view(1, 1, height, width)
    return F.interpolate(value, size=(size, size), mode="nearest")[0, 0]


def prepare_sunrgbd_split(
    name: SplitName,
    frames: Sequence[SUNRGBDFrame],
    *,
    clamp_max_depth_m: float | None,
    progress: Callable[[int, int], None] | None = None,
) -> PreparedSplit:
    """Decode and transform a manifest split without invoking a learned model."""
    if not frames:
        raise ValueError(f"SUN RGB-D {name} split is empty")
    paired = name == "train"
    image_rows, rgb_rows, intrinsics_rows, target_rows = [], [], [], []
    crop_rows, source_rows = [], []
    sample_ids, sensor_ids, group_ids = [], [], []
    expected_view_names: tuple[str, ...] | None = None
    for index, frame in enumerate(frames):
        if frame.split != name:
            raise ValueError(f"frame {frame.sample_id} belongs to {frame.split}, not {name}")
        if frame.intrinsics is None or frame.image_size_hw is None or frame.depth_size_hw is None:
            raise ValueError(f"frame {frame.sample_id} was not fully validated by the manifest adapter")
        if frame.image_size_hw != frame.depth_size_hw:
            raise ValueError(f"frame {frame.sample_id} has misaligned RGB/depth sizes")
        with Image.open(frame.image_path) as image:
            rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(decode_sunrgbd_depth(frame.depth_path, clamp_max_depth_m=clamp_max_depth_m)).float()
        source_size = tuple(int(value) for value in rgb.shape[-2:])
        if source_size != frame.image_size_hw or tuple(depth.shape) != source_size:
            raise ValueError(f"decoded source shape changed for {frame.sample_id}")
        views = deterministic_crop_views(source_size, paired=paired)
        view_names = tuple(view.name for view in views)
        if expected_view_names is None:
            expected_view_names = view_names
        elif expected_view_names != view_names:
            raise RuntimeError("SUN RGB-D view policy changed within one split")
        intrinsics = torch.as_tensor(frame.intrinsics, dtype=torch.float32)
        image_views, rgb_views, intrinsics_views, target_views = [], [], [], []
        for view in views:
            image_views.append(_resize_rgb(rgb, view.crop_box, 384))
            rgb_views.append(_resize_rgb(rgb, view.crop_box, 96))
            intrinsics_views.append(
                update_intrinsics_for_crop_resize(
                    intrinsics,
                    source_size,
                    (384, 384),
                    crop=view.crop_box,
                    half_pixel_centers=True,
                )
            )
            target = _resize_depth(depth, view.crop_box, 24)
            valid = torch.isfinite(target) & (target > 0.1) & (target < 10.0)
            if int(valid.sum()) < 100:
                raise ValueError(f"frame {frame.sample_id}/{view.name} has fewer than 100 valid 24x24 depth pixels")
            target_views.append(target)
        image_rows.append(torch.stack(image_views))
        rgb_rows.append(torch.stack(rgb_views))
        intrinsics_rows.append(torch.stack(intrinsics_views))
        target_rows.append(torch.stack(target_views))
        crop_rows.append(torch.tensor([view.crop_box for view in views], dtype=torch.int64))
        source_rows.append(torch.tensor([source_size for _ in views], dtype=torch.int64))
        sample_ids.append(frame.sample_id)
        sensor_ids.append(frame.sensor)
        group_ids.append(frame.group_id)
        if progress is not None:
            progress(index + 1, len(frames))
    assert expected_view_names is not None
    prepared = PreparedSplit(
        name=name,
        images_384=torch.stack(image_rows),
        rgb_96=torch.stack(rgb_rows),
        intrinsics_384=torch.stack(intrinsics_rows),
        targets_24=torch.stack(target_rows),
        sample_ids=sample_ids,
        sensor_ids=sensor_ids,
        group_ids=group_ids,
        crop_boxes=torch.stack(crop_rows),
        source_sizes=torch.stack(source_rows),
        view_names=expected_view_names,
    )
    _validate_prepared_split(prepared)
    return prepared


def _validate_prepared_split(split: PreparedSplit) -> None:
    expected_views = 2 if split.name == "train" else 1
    count = split.count
    if split.views != expected_views:
        raise ValueError(f"{split.name} must contain {expected_views} view(s)")
    if tuple(split.images_384.shape) != (count, expected_views, 3, 384, 384):
        raise ValueError(f"unexpected {split.name} V-JEPA image shape: {tuple(split.images_384.shape)}")
    if tuple(split.rgb_96.shape) != (count, expected_views, *RGB_SHAPE):
        raise ValueError(f"unexpected {split.name} RGB cache shape")
    if tuple(split.intrinsics_384.shape) != (count, expected_views, 3, 3):
        raise ValueError(f"unexpected {split.name} intrinsics shape")
    if tuple(split.targets_24.shape) != (count, expected_views, *TARGET_SHAPE):
        raise ValueError(f"unexpected {split.name} target shape")
    if tuple(split.crop_boxes.shape) != (count, expected_views, 4):
        raise ValueError(f"unexpected {split.name} crop metadata shape")
    if tuple(split.source_sizes.shape) != (count, expected_views, 2):
        raise ValueError(f"unexpected {split.name} source-size metadata shape")
    if any(len(values) != count for values in (split.sample_ids, split.sensor_ids, split.group_ids)):
        raise ValueError(f"{split.name} identity metadata count differs")
    if len(set(split.sample_ids)) != count or len(set(split.group_ids)) != count:
        raise ValueError(f"{split.name} sample/group IDs must be unique")
    tensors = (split.images_384, split.rgb_96, split.intrinsics_384, split.targets_24)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"{split.name} preprocessing produced non-finite values")


def normalize_final_features(
    train: torch.Tensor,
    validation: torch.Tensor,
    test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize all splits using channel statistics from both train views only."""
    if train.ndim != 5 or train.shape[1] != 2 or tuple(train.shape[2:]) != FEATURE_SHAPE:
        raise ValueError("train features must have shape [N,2,768,24,24]")
    for name, value in (("validation", validation), ("test", test)):
        if value.ndim != 5 or value.shape[1] != 1 or tuple(value.shape[2:]) != FEATURE_SHAPE:
            raise ValueError(f"{name} features must have shape [N,1,768,24,24]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} features are non-finite")
    if not torch.isfinite(train).all():
        raise ValueError("train features are non-finite")
    mean = train.float().mean(dim=(0, 1, 3, 4), keepdim=True)
    std = train.float().std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-4)

    def apply(value: torch.Tensor) -> torch.Tensor:
        normalized = ((value.float() - mean) / std).half()
        if not torch.isfinite(normalized).all():
            raise ValueError("normalized feature grid is non-finite")
        return normalized

    return apply(train), apply(validation), apply(test), {"mean": mean.cpu(), "std": std.cpu()}


def centered_log_depth_teacher(depth: torch.Tensor) -> torch.Tensor:
    """Remove each VGGT prediction's spatial log-depth mean (no metric scale)."""
    if depth.ndim != 4 or depth.shape[1] != 2 or tuple(depth.shape[-2:]) != TARGET_SHAPE:
        raise ValueError("train teacher depth must have shape [N,2,24,24]")
    if not torch.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("VGGT teacher depth must be finite and positive")
    log_depth = depth.float().clamp_min(1e-6).log()
    centered = log_depth - log_depth.mean(dim=(-2, -1), keepdim=True)
    return centered.half()


def _cache_split(
    prepared: PreparedSplit,
    features: torch.Tensor,
    teacher_centered_shape: torch.Tensor | None,
) -> dict[str, Any]:
    if prepared.name == "train":
        if features.ndim != 5 or features.shape[1] != 2:
            raise ValueError("train normalized features must retain both views")
        value: dict[str, Any] = {
            "features": features.half().contiguous(),
            "rgb": prepared.rgb_96.half().contiguous(),
            "intrinsics_384": prepared.intrinsics_384.float().contiguous(),
            "targets": prepared.targets_24.float().contiguous(),
            "sample_ids": list(prepared.sample_ids),
            "sensor_ids": list(prepared.sensor_ids),
        }
        if teacher_centered_shape is None:
            raise ValueError("train cache requires centered VGGT shape teacher")
        value["teacher_centered_shape"] = teacher_centered_shape.half().contiguous()
        return value
    if features.ndim != 5 or features.shape[1] != 1:
        raise ValueError(f"{prepared.name} normalized features must contain one original view")
    if teacher_centered_shape is not None:
        raise ValueError(f"{prepared.name} must not contain a VGGT teacher")
    return {
        "features": features[:, 0].half().contiguous(),
        "rgb": prepared.rgb_96[:, 0].half().contiguous(),
        "intrinsics_384": prepared.intrinsics_384[:, 0].float().contiguous(),
        "targets": prepared.targets_24[:, 0].float().contiguous(),
        "sample_ids": list(prepared.sample_ids),
        "sensor_ids": list(prepared.sensor_ids),
    }


def build_separate_cache_payloads(
    prepared: Mapping[str, PreparedSplit],
    normalized_features: Mapping[str, torch.Tensor],
    teacher_centered_shape: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a train/validation artifact and an isolated untouched-test artifact."""
    if set(prepared) != {"train", "validation", "test"} or set(normalized_features) != set(prepared):
        raise ValueError("cache builder requires exactly train, validation, and test inputs")
    train_validation = {
        "schema_version": CACHE_SCHEMA,
        "splits": {
            "train": _cache_split(prepared["train"], normalized_features["train"], teacher_centered_shape),
            "validation": _cache_split(prepared["validation"], normalized_features["validation"], None),
        },
    }
    test = {
        "schema_version": CACHE_SCHEMA,
        "splits": {"test": _cache_split(prepared["test"], normalized_features["test"], None)},
    }
    validate_cache_payload(train_validation, expected_splits={"train", "validation"})
    validate_cache_payload(test, expected_splits={"test"})
    train_ids = set(prepared["train"].sample_ids)
    validation_ids = set(prepared["validation"].sample_ids)
    test_ids = set(prepared["test"].sample_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("sample IDs overlap across cache split artifacts")
    return train_validation, test


def validate_cache_payload(payload: Mapping[str, Any], *, expected_splits: set[str]) -> None:
    if set(payload) != {"schema_version", "splits"} or payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"cache root must contain only schema_version={CACHE_SCHEMA!r} and splits")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != expected_splits:
        raise ValueError(f"cache must contain exactly splits {sorted(expected_splits)}")
    for split_name, value in splits.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"cache split {split_name} must be a mapping")
        required = {"features", "rgb", "intrinsics_384", "targets", "sample_ids", "sensor_ids"}
        allowed = required | {"teacher_centered_shape"}
        if set(value) - allowed or not required <= set(value):
            raise ValueError(f"cache split {split_name} has unexpected fields")
        paired = split_name == "train"
        prefix = (len(value["sample_ids"]), 2) if paired else (len(value["sample_ids"]),)
        expected_shapes = {
            "features": (*prefix, *FEATURE_SHAPE),
            "rgb": (*prefix, *RGB_SHAPE),
            "intrinsics_384": (*prefix, 3, 3),
            "targets": (*prefix, *TARGET_SHAPE),
        }
        for key, expected in expected_shapes.items():
            tensor = value[key]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
                raise ValueError(f"cache {split_name}.{key} must have shape {expected}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"cache {split_name}.{key} contains non-finite values")
        teacher = value.get("teacher_centered_shape")
        if paired:
            if not isinstance(teacher, torch.Tensor) or tuple(teacher.shape) != (*prefix, *TARGET_SHAPE):
                raise ValueError("train cache must contain paired teacher_centered_shape")
        elif teacher is not None:
            raise ValueError(f"cache split {split_name} must not contain teacher_centered_shape")
        for key in ("sample_ids", "sensor_ids"):
            if not isinstance(value[key], list) or len(value[key]) != prefix[0]:
                raise ValueError(f"cache {split_name}.{key} identity count differs")


def write_feature_cache(path: Path, payload: Mapping[str, Any]) -> Path:
    expected = set(payload.get("splits", {}))
    validate_cache_payload(payload, expected_splits=expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return path
