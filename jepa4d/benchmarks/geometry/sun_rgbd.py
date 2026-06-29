"""Official SUN RGB-D sensor-blocked dataset adapter and manifest builder.

The adapter follows the official toolbox depth decode exactly::

    bitor(bitshift(raw, -3), bitshift(raw, 13)) / 1000 metres

The official toolbox additionally clips depth above eight metres.  This module
only applies that clip when the manifest protocol explicitly declares
``clamp_max_depth_m``; ``None`` means no clamp.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

SCHEMA_VERSION = "jepa4d-sunrgbd-sensor-blocked-v1"
DATASET_ID = "sun-rgbd-official"
DATASET_VERSION = "SUNRGBD-v1"
SUPPORTED_SENSORS = ("kv1", "kv2", "realsense", "xtion")
SPLIT_SENSOR_POLICY = {
    "train": ("kv1", "xtion"),
    "validation": ("realsense",),
    "test": ("kv2",),
}
DEFAULT_TARGET_COUNTS = {"kv1": 192, "xtion": 192, "realsense": 128, "kv2": 128}
OFFICIAL_AVAILABLE_COUNTS = {"kv1": 2003, "kv2": 3784, "realsense": 1159, "xtion": 3389}
OFFICIAL_PROJECT_URL = "https://rgbd.cs.princeton.edu/"
OFFICIAL_ARCHIVE_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
OFFICIAL_ARCHIVE_BYTES = 6_885_481_608
OFFICIAL_ARCHIVE_SHA256 = "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c"
OFFICIAL_CITATION = {
    "authors": "S. Song, S. Lichtenberg, and J. Xiao",
    "title": "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite",
    "venue": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    "year": 2015,
    "url": "https://openaccess.thecvf.com/content_cvpr_2015/html/Song_SUN_RGB-D_A_2015_CVPR_paper.html",
}
REQUIRED_CITATIONS = [
    OFFICIAL_CITATION,
    {
        "authors": "N. Silberman, D. Hoiem, P. Kohli, and R. Fergus",
        "title": "Indoor Segmentation and Support Inference from RGBD Images",
        "venue": "European Conference on Computer Vision (ECCV)",
        "year": 2012,
    },
    {
        "authors": "A. Janoch, S. Karayev, Y. Jia, J. T. Barron, M. Fritz, K. Saenko, and T. Darrell",
        "title": "A Category-Level 3-D Object Dataset: Putting the Kinect to Work",
        "venue": "ICCV Workshop on Consumer Depth Cameras for Computer Vision",
        "year": 2011,
    },
    {
        "authors": "J. Xiao, A. Owens, and A. Torralba",
        "title": "SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels",
        "venue": "IEEE International Conference on Computer Vision (ICCV)",
        "year": 2013,
    },
]
USAGE_TERMS = {
    "citation_required": True,
    "required_citations": REQUIRED_CITATIONS,
    "citation_source": f"{OFFICIAL_PROJECT_URL} and {OFFICIAL_PROJECT_URL}data/README.txt",
    "license_status": "no explicit license text found",
    "license_and_use": "no explicit license text found; internal research/no redistribution",
    "intended_use": "internal research only",
    "redistribution": "no redistribution",
    "notice": (
        "The official toolbox README requires citation. No explicit dataset license text was found in the supplied "
        "archive/toolbox; retain locally for internal research and do not redistribute."
    ),
}


@dataclass(frozen=True, slots=True)
class SUNRGBDFrame:
    """One aligned RGB/depth frame rooted in one SUN RGB-D leaf directory."""

    sample_id: str
    group_id: str
    sensor: str
    leaf: Path
    image_path: Path
    depth_path: Path
    intrinsics_path: Path
    split: str | None = None
    intrinsics: np.ndarray | None = None
    image_size_hw: tuple[int, int] | None = None
    depth_size_hw: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class SUNRGBDBundle:
    """Verified manifest frames grouped by the frozen sensor-blocked split."""

    manifest_path: Path
    manifest: dict[str, Any]
    samples: tuple[SUNRGBDFrame, ...]
    splits: dict[str, tuple[SUNRGBDFrame, ...]]
    split_hash: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_depth_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def decode_sunrgbd_depth(
    raw_or_path: np.ndarray | Path,
    *,
    clamp_max_depth_m: float | None = None,
) -> np.ndarray:
    """Decode a SUN RGB-D ``depth_bfx`` uint16 PNG into float32 metres."""

    raw = _read_depth_png(raw_or_path) if isinstance(raw_or_path, Path) else np.asarray(raw_or_path)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"SUN RGB-D encoded depth must be a uint16 [H,W] array, got {raw.dtype} {raw.shape}")
    values = raw.astype(np.uint32)
    decoded_millimetres = np.bitwise_or(np.right_shift(values, 3), np.left_shift(values, 13)) & 0xFFFF
    depth_m = decoded_millimetres.astype(np.float32) / 1000.0
    if clamp_max_depth_m is not None:
        maximum = float(clamp_max_depth_m)
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("clamp_max_depth_m must be finite and positive")
        np.minimum(depth_m, maximum, out=depth_m)
    return depth_m


def load_intrinsics(path: Path) -> np.ndarray:
    """Load and validate a direct leaf ``intrinsics.txt`` matrix."""

    values = np.fromstring(path.read_text(), sep=" ", dtype=np.float64)
    if values.size != 9:
        raise ValueError(f"intrinsics must contain exactly nine numbers: {path}")
    matrix = values.reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError(f"intrinsics contain non-finite values: {path}")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"intrinsics focal lengths must be positive: {path}")
    if not np.allclose(matrix[2], np.asarray([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-8):
        raise ValueError(f"intrinsics final row is not [0,0,1]: {path}")
    if abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8:
        raise ValueError(f"intrinsics skew/off-diagonal terms are unsupported: {path}")
    return matrix


def _stable_id(relative_leaf: Path, image_stem: str) -> tuple[str, str]:
    group_id = relative_leaf.as_posix()
    sample_id = f"{group_id}/{image_stem}"
    return sample_id, group_id


def enumerate_sunrgbd_frames(root: Path) -> dict[str, list[SUNRGBDFrame]]:
    """Enumerate strict leaf frames under kv1/kv2/realsense/xtion.

    A valid leaf has a direct ``intrinsics.txt``, exactly one ``image/*.jpg``,
    and exactly one ``depth_bfx/*.png``. Nested ``fullres/intrinsics.txt`` files
    are intentionally ignored because they are not frame-leaf intrinsics.
    """

    resolved_root = root.resolve(strict=True)
    by_sensor: dict[str, list[SUNRGBDFrame]] = {}
    for sensor in SUPPORTED_SENSORS:
        sensor_root = resolved_root / sensor
        if not sensor_root.is_dir():
            raise FileNotFoundError(f"SUN RGB-D sensor directory is missing: {sensor_root}")
        frames: list[SUNRGBDFrame] = []
        for intrinsics_path in sorted(sensor_root.rglob("intrinsics.txt")):
            leaf = intrinsics_path.parent
            images = sorted((leaf / "image").glob("*.jpg")) if (leaf / "image").is_dir() else []
            depths = sorted((leaf / "depth_bfx").glob("*.png")) if (leaf / "depth_bfx").is_dir() else []
            if not images and not depths:
                continue
            if len(images) != 1 or len(depths) != 1:
                raise ValueError(
                    f"SUN RGB-D leaf must contain exactly one image JPG and one depth_bfx PNG: "
                    f"{leaf} (images={len(images)}, depths={len(depths)})"
                )
            for path in (leaf, images[0], depths[0], intrinsics_path):
                if not path.resolve(strict=True).is_relative_to(resolved_root):
                    raise ValueError(f"SUN RGB-D frame path escapes dataset root: {path}")
            relative_leaf = leaf.relative_to(resolved_root)
            sample_id, group_id = _stable_id(relative_leaf, images[0].stem)
            frames.append(
                SUNRGBDFrame(
                    sample_id=sample_id,
                    group_id=group_id,
                    sensor=sensor,
                    leaf=leaf,
                    image_path=images[0],
                    depth_path=depths[0],
                    intrinsics_path=intrinsics_path,
                )
            )
        frames.sort(key=lambda value: (value.group_id, value.sample_id))
        if len({frame.sample_id for frame in frames}) != len(frames):
            raise ValueError(f"duplicate SUN RGB-D sample ID under sensor {sensor}")
        if len({frame.group_id for frame in frames}) != len(frames):
            raise ValueError(f"multiple frame leaves share one group under sensor {sensor}")
        by_sensor[sensor] = frames
    return by_sensor


def validate_sunrgbd_frame(
    frame: SUNRGBDFrame,
    *,
    clamp_max_depth_m: float | None,
    minimum_valid_pixels: int = 100,
) -> tuple[SUNRGBDFrame, dict[str, Any]]:
    """Validate alignment, K, and decoded finite depth for one selected frame."""

    with Image.open(frame.image_path) as image:
        image_size = (image.height, image.width)
        image_mode = image.mode
    raw_depth = _read_depth_png(frame.depth_path)
    if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError(f"depth_bfx is not uint16 [H,W]: {frame.depth_path}")
    depth_size = tuple(int(value) for value in raw_depth.shape)
    if image_size != depth_size:
        raise ValueError(f"RGB/depth shapes differ for {frame.sample_id}: image={image_size}, depth={depth_size}")
    intrinsics = load_intrinsics(frame.intrinsics_path)
    height, width = image_size
    if not (-0.5 <= intrinsics[0, 2] <= width - 0.5 and -0.5 <= intrinsics[1, 2] <= height - 0.5):
        raise ValueError(f"principal point lies outside the aligned image for {frame.sample_id}")
    depth_m = decode_sunrgbd_depth(raw_depth, clamp_max_depth_m=clamp_max_depth_m)
    if not np.isfinite(depth_m).all():
        raise ValueError(f"decoded depth contains non-finite values for {frame.sample_id}")
    valid = depth_m > 0
    valid_count = int(valid.sum())
    if valid_count < minimum_valid_pixels:
        raise ValueError(f"fewer than {minimum_valid_pixels} valid depth pixels for {frame.sample_id}")
    positive = depth_m[valid]
    validated = replace(
        frame,
        intrinsics=intrinsics,
        image_size_hw=image_size,
        depth_size_hw=depth_size,
    )
    return validated, {
        "image_mode": image_mode,
        "image_size_hw": list(image_size),
        "depth_size_hw": list(depth_size),
        "rgb_depth_shape_aligned": True,
        "valid_depth_pixels": valid_count,
        "valid_depth_fraction": float(valid.mean()),
        "valid_depth_min_m": float(positive.min()),
        "valid_depth_max_m": float(positive.max()),
        "valid_depth_mean_m": float(positive.mean()),
        "all_decoded_depth_finite": True,
    }


def rank_midpoint_selection(frames: list[SUNRGBDFrame], count: int) -> list[SUNRGBDFrame]:
    """Select deterministic sorted rank-midpoint quantiles without replacement."""

    if count <= 0 or len(frames) < count:
        raise ValueError(f"need at least {count} frames, found {len(frames)}")
    indices = [math.floor((index + 0.5) * len(frames) / count) for index in range(count)]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise RuntimeError("rank-midpoint selection did not produce unique sorted indices")
    return [frames[index] for index in indices]


def split_for_sensor(sensor: str) -> str:
    matches = [split for split, sensors in SPLIT_SENSOR_POLICY.items() if sensor in sensors]
    if len(matches) != 1:
        raise ValueError(f"sensor {sensor!r} has no unique split assignment")
    return matches[0]


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"manifested file escapes SUN RGB-D root: {path}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _sample_manifest_row(
    frame: SUNRGBDFrame,
    validation: dict[str, Any],
    *,
    dataset_root: Path,
) -> dict[str, Any]:
    assert frame.intrinsics is not None
    assert frame.image_size_hw is not None
    assert frame.depth_size_hw is not None
    return {
        "sample_id": frame.sample_id,
        "group_id": frame.group_id,
        "sensor": frame.sensor,
        "split": frame.split,
        "files": {
            "image": _file_identity(frame.image_path, dataset_root),
            "depth_bfx": _file_identity(frame.depth_path, dataset_root),
            "intrinsics": _file_identity(frame.intrinsics_path, dataset_root),
        },
        "image_size_hw": list(frame.image_size_hw),
        "depth_size_hw": list(frame.depth_size_hw),
        "intrinsics": frame.intrinsics.tolist(),
        "validation": validation,
    }


def _canonical_split_hash(rows: list[dict[str, Any]]) -> str:
    identity = [
        {
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "sensor": row["sensor"],
            "split": row["split"],
            "files": row["files"],
        }
        for row in rows
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_sensor_blocked_manifest(
    dataset_root: Path,
    archive_path: Path,
    *,
    target_counts: dict[str, int] | None = None,
    clamp_max_depth_m: float | None = 8.0,
    expected_archive_bytes: int = OFFICIAL_ARCHIVE_BYTES,
    expected_archive_sha256: str = OFFICIAL_ARCHIVE_SHA256,
    verify_archive_hash: bool = True,
    expected_available_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the frozen sensor-blocked split and hash every selected asset."""

    root = dataset_root.resolve(strict=True)
    archive = archive_path.resolve(strict=True)
    counts = dict(DEFAULT_TARGET_COUNTS if target_counts is None else target_counts)
    if set(counts) != set(SUPPORTED_SENSORS) or any(int(value) <= 0 for value in counts.values()):
        raise ValueError(f"target_counts must contain positive counts for {SUPPORTED_SENSORS}")
    if counts["kv1"] != counts["xtion"]:
        raise ValueError("sensor-blocked training requires equal kv1 and xtion counts")
    if archive.stat().st_size != int(expected_archive_bytes):
        raise ValueError(f"SUN RGB-D archive byte mismatch: {archive.stat().st_size} != {expected_archive_bytes}")
    actual_archive_sha256 = sha256_file(archive) if verify_archive_hash else expected_archive_sha256
    if actual_archive_sha256 != expected_archive_sha256:
        raise ValueError(f"SUN RGB-D archive SHA-256 mismatch: {actual_archive_sha256}")
    if clamp_max_depth_m is not None and (not math.isfinite(clamp_max_depth_m) or clamp_max_depth_m <= 0):
        raise ValueError("clamp_max_depth_m must be finite and positive or None")

    available = enumerate_sunrgbd_frames(root)
    expected_available = (
        dict(OFFICIAL_AVAILABLE_COUNTS)
        if expected_available_counts is None
        else {sensor: int(value) for sensor, value in expected_available_counts.items()}
    )
    observed_available = {sensor: len(frames) for sensor, frames in available.items()}
    if set(expected_available) != set(SUPPORTED_SENSORS) or observed_available != expected_available:
        raise ValueError(
            "SUN RGB-D extraction is incomplete or contains an unexpected leaf inventory: "
            f"observed={observed_available}, expected={expected_available}"
        )
    selected: list[SUNRGBDFrame] = []
    for sensor in ("kv1", "xtion", "realsense", "kv2"):
        chosen = rank_midpoint_selection(available[sensor], int(counts[sensor]))
        selected.extend(replace(frame, split=split_for_sensor(sensor)) for frame in chosen)

    rows: list[dict[str, Any]] = []
    for frame in selected:
        validated, validation = validate_sunrgbd_frame(
            frame,
            clamp_max_depth_m=clamp_max_depth_m,
        )
        rows.append(_sample_manifest_row(validated, validation, dataset_root=root))

    sample_ids = [str(row["sample_id"]) for row in rows]
    group_ids = [str(row["group_id"]) for row in rows]
    selected_paths = [str(file_record["path"]) for row in rows for file_record in row["files"].values()]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("selected sample IDs overlap")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("selected group IDs overlap")
    if len(selected_paths) != len(set(selected_paths)):
        raise ValueError("selected file paths overlap")

    selected_counts = {sensor: sum(str(row["sensor"]) == sensor for row in rows) for sensor in SUPPORTED_SENSORS}
    if selected_counts != counts:
        raise RuntimeError(f"selected sensor counts differ from target: {selected_counts} != {counts}")
    split_counts = {split: sum(str(row["split"]) == split for row in rows) for split in SPLIT_SENSOR_POLICY}
    expected_split_counts = {
        "train": counts["kv1"] + counts["xtion"],
        "validation": counts["realsense"],
        "test": counts["kv2"],
    }
    if split_counts != expected_split_counts:
        raise RuntimeError(f"selected split counts differ from target: {split_counts}")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "official": True,
        "source": {
            "project_url": OFFICIAL_PROJECT_URL,
            "archive_url": OFFICIAL_ARCHIVE_URL,
            "archive": {
                "filename": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": actual_archive_sha256,
            },
        },
        "usage_terms": USAGE_TERMS,
        "protocol": {
            "name": "sunrgbd-sensor-blocked-v1",
            "split_policy": "train-kv1-plus-xtion-validation-realsense-untouched-test-kv2",
            "selection_policy": "sorted-leaf-rank-midpoint-quantiles-v1",
            "train_sensors": ["kv1", "xtion"],
            "validation_sensors": ["realsense"],
            "test_sensors": ["kv2"],
            "test_usage": "untouched_until_final_evaluation",
            "test_selection_inputs": "sorted leaf/sample identity only",
            "test_targets_used_for_model_or_checkpoint_selection": False,
            "test_depth_access_before_evaluation": "integrity/decode audit only",
            "target_counts_by_sensor": counts,
            "target_counts_by_split": expected_split_counts,
            "depth_decode": {
                "source": "official SUNRGBDtoolbox/readData/read3dPoints.m",
                "encoded_dtype": "uint16",
                "formula": "bitor(bitshift(raw,-3),bitshift(raw,13))/1000",
                "unit": "metres",
                "zero_is_invalid": True,
                "clamp_max_depth_m": clamp_max_depth_m,
                "clamp_policy": "apply_only_when_this_field_is_non-null",
            },
        },
        "dataset_root_name": root.name,
        "available_counts_by_sensor": observed_available,
        "selected_counts_by_sensor": selected_counts,
        "selected_counts_by_split": split_counts,
        "integrity": {
            "sample_ids_unique": True,
            "group_ids_unique": True,
            "selected_paths_unique": True,
            "no_path_overlap_across_splits": True,
            "split_hash": _canonical_split_hash(rows),
        },
        "samples": rows,
    }


def write_sensor_blocked_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False, width=120))


def _verify_file_identity(record: dict[str, Any], dataset_root: Path) -> Path:
    path = _resolve_manifest_path(record, dataset_root)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"manifest byte mismatch: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"manifest SHA-256 mismatch: {path}")
    return path


def _resolve_manifest_path(record: dict[str, Any], dataset_root: Path) -> Path:
    path = (dataset_root / str(record["path"])).resolve(strict=True)
    if not path.is_relative_to(dataset_root):
        raise ValueError(f"manifest path escapes dataset root: {path}")
    return path


def load_sensor_blocked_manifest(
    dataset_root: Path,
    manifest_path: Path,
    *,
    verify_file_hashes: bool = True,
    validate_depth: bool = True,
) -> SUNRGBDBundle:
    """Strictly load a generated manifest and reconstruct split frame objects."""

    root = dataset_root.resolve(strict=True)
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected SUN RGB-D manifest schema: {manifest.get('schema_version')}")
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("version") != DATASET_VERSION:
        raise ValueError("unexpected SUN RGB-D dataset identity")
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("SUN RGB-D source record is missing")
    archive_record = source.get("archive", {})
    if not isinstance(archive_record, dict):
        raise ValueError("SUN RGB-D archive record is missing")
    if source.get("project_url") != OFFICIAL_PROJECT_URL or source.get("archive_url") != OFFICIAL_ARCHIVE_URL:
        raise ValueError("SUN RGB-D official source URLs changed or are missing")
    if (
        str(archive_record.get("filename")) != "SUNRGBD.zip"
        or int(archive_record.get("bytes", 0)) <= 0
        or len(str(archive_record.get("sha256", ""))) != 64
    ):
        raise ValueError("SUN RGB-D archive identity is incomplete")
    if manifest.get("usage_terms") != USAGE_TERMS:
        raise ValueError("SUN RGB-D usage terms changed or are missing")
    protocol = manifest.get("protocol", {})
    expected_policy = {
        "train_sensors": ["kv1", "xtion"],
        "validation_sensors": ["realsense"],
        "test_sensors": ["kv2"],
        "test_usage": "untouched_until_final_evaluation",
        "test_selection_inputs": "sorted leaf/sample identity only",
        "test_targets_used_for_model_or_checkpoint_selection": False,
        "test_depth_access_before_evaluation": "integrity/decode audit only",
    }
    if any(protocol.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("SUN RGB-D sensor-blocked split policy changed")
    clamp_value = protocol.get("depth_decode", {}).get("clamp_max_depth_m")
    clamp_max_depth_m = None if clamp_value is None else float(clamp_value)

    rows = manifest.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SUN RGB-D manifest contains no samples")
    frames: list[SUNRGBDFrame] = []
    canonical_rows: list[dict[str, Any]] = []
    observed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SUN RGB-D sample row must be a mapping")
        sensor, split = str(row["sensor"]), str(row["split"])
        if sensor not in SUPPORTED_SENSORS or split != split_for_sensor(sensor):
            raise ValueError(f"sensor/split mismatch for {row.get('sample_id')}")
        files = row.get("files", {})
        if set(files) != {"image", "depth_bfx", "intrinsics"}:
            raise ValueError(f"sample file inventory is incomplete: {row.get('sample_id')}")
        if verify_file_hashes:
            image_path = _verify_file_identity(files["image"], root)
            depth_path = _verify_file_identity(files["depth_bfx"], root)
            intrinsics_path = _verify_file_identity(files["intrinsics"], root)
        else:
            image_path = _resolve_manifest_path(files["image"], root)
            depth_path = _resolve_manifest_path(files["depth_bfx"], root)
            intrinsics_path = _resolve_manifest_path(files["intrinsics"], root)
        observed_paths.extend(str(files[key]["path"]) for key in ("image", "depth_bfx", "intrinsics"))
        leaf = intrinsics_path.parent
        expected_sample_id, expected_group_id = _stable_id(leaf.relative_to(root), image_path.stem)
        if str(row["sample_id"]) != expected_sample_id or str(row["group_id"]) != expected_group_id:
            raise ValueError(f"sample/group identity does not match leaf paths: {row.get('sample_id')}")
        frame = SUNRGBDFrame(
            sample_id=str(row["sample_id"]),
            group_id=str(row["group_id"]),
            sensor=sensor,
            split=split,
            leaf=leaf,
            image_path=image_path,
            depth_path=depth_path,
            intrinsics_path=intrinsics_path,
        )
        if validate_depth:
            frame, observed_validation = validate_sunrgbd_frame(
                frame,
                clamp_max_depth_m=clamp_max_depth_m,
            )
            if observed_validation != row.get("validation"):
                raise ValueError(f"sample validation metadata changed: {frame.sample_id}")
            if frame.intrinsics is None or not np.array_equal(frame.intrinsics, np.asarray(row["intrinsics"])):
                raise ValueError(f"sample intrinsics changed: {frame.sample_id}")
        frames.append(frame)
        canonical_rows.append(row)

    sample_ids = [frame.sample_id for frame in frames]
    group_ids = [frame.group_id for frame in frames]
    if len(sample_ids) != len(set(sample_ids)) or len(group_ids) != len(set(group_ids)):
        raise ValueError("SUN RGB-D manifest sample/group IDs overlap")
    if len(observed_paths) != len(set(observed_paths)):
        raise ValueError("SUN RGB-D manifest paths overlap")
    split_hash = _canonical_split_hash(canonical_rows)
    if split_hash != str(manifest.get("integrity", {}).get("split_hash")):
        raise ValueError("SUN RGB-D split hash mismatch")
    splits = {split: tuple(frame for frame in frames if frame.split == split) for split in SPLIT_SENSOR_POLICY}
    selected_counts = {sensor: sum(frame.sensor == sensor for frame in frames) for sensor in SUPPORTED_SENSORS}
    if selected_counts != manifest.get("selected_counts_by_sensor"):
        raise ValueError("SUN RGB-D selected sensor counts changed")
    return SUNRGBDBundle(
        manifest_path=manifest_path.resolve(strict=True),
        manifest=manifest,
        samples=tuple(frames),
        splits=splits,
        split_hash=split_hash,
    )


def audit_manifest_summary(bundle: SUNRGBDBundle) -> dict[str, Any]:
    """Return a compact, JSON-safe receipt for login-node preparation."""

    clamp = bundle.manifest["protocol"]["depth_decode"]["clamp_max_depth_m"]
    return {
        "schema_version": "jepa4d-sunrgbd-login-audit-v1",
        "result": "pass",
        "manifest": str(bundle.manifest_path),
        "manifest_sha256": sha256_file(bundle.manifest_path),
        "split_hash": bundle.split_hash,
        "source": bundle.manifest["source"],
        "sample_count": len(bundle.samples),
        "available_counts_by_sensor": bundle.manifest["available_counts_by_sensor"],
        "split_counts": {split: len(values) for split, values in bundle.splits.items()},
        "sensor_counts": {
            sensor: sum(frame.sensor == sensor for frame in bundle.samples) for sensor in SUPPORTED_SENSORS
        },
        "depth_decode": {
            "formula": "bitor(bitshift(raw,-3),bitshift(raw,13))/1000",
            "clamp_max_depth_m": clamp,
        },
        "all_selected_files_hash_verified": True,
        "all_rgb_depth_shapes_aligned": True,
        "all_intrinsics_valid": True,
        "all_decoded_depth_finite_with_valid_pixels": True,
        "no_path_overlap_across_splits": True,
        "usage_terms": USAGE_TERMS,
    }
