"""Diagnostic-only camera calibration and metric-scale audits for Phase 2d.

This module deliberately does not load a model.  It audits the calibration
metadata and preprocessing geometry recorded by a TUM manifest, evaluates
increasingly flexible *oracle* corrections on already-persisted predictions,
and constructs reusable camera-intrinsics negative controls.

The oracles fit and evaluate on the same target pixels.  They are attribution
diagnostics, never deployable calibration methods or model-quality results.

Full prediction interchange schema
----------------------------------

``jepa4d-phase2d-depth-predictions-v1`` NPZ files contain:

* ``prediction_m``: ``[N,H,W]`` or ``[M,N,H,W]`` positive metric depth;
* ``target_m``: ``[N,H,W]`` or ``[M,N,H,W]`` metric ground truth;
* ``sample_ids`` (or legacy ``test_sample_ids``): ``[N]`` strings;
* optional ``sequence_ids``: ``[N]`` strings;
* optional ``variant_ids``: ``[M]`` strings (or scalar ``variant_id``);
* optional ``seeds``: ``[M]`` integers (or scalar ``seed``);
* optional scalar ``schema_version`` and ``audit_scope``.

Legacy compact Phase-2c diagnostic NPZ files are accepted, but the resulting
report is visibly marked ``compact_diagnostics_only``.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import html
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml
from PIL import Image

PREDICTION_SCHEMA_VERSION = "jepa4d-phase2d-depth-predictions-v1"
REPORT_SCHEMA_VERSION = "jepa4d-phase2d-calibration-scale-audit-v1"
DEFAULT_OUTPUT_SIZES = ((384, 384), (518, 518))
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class DepthPredictionSet:
    """One variant/seed prediction tensor and its immutable sample identity."""

    variant_id: str
    seed: int | None
    prediction_m: np.ndarray
    target_m: np.ndarray
    sample_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    selection_labels: tuple[str, ...]
    audit_scope: str
    source_path: str

    def __post_init__(self) -> None:
        prediction = np.asarray(self.prediction_m)
        target = np.asarray(self.target_m)
        if prediction.ndim != 3 or target.ndim != 3 or prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have the same [N,H,W] shape, got {prediction.shape} and {target.shape}"
            )
        if prediction.shape[0] == 0:
            raise ValueError("prediction set is empty")
        identity_lengths = {
            len(self.sample_ids),
            len(self.sequence_ids),
            len(self.selection_labels),
            prediction.shape[0],
        }
        if len(identity_lengths) != 1:
            raise ValueError("sample, sequence, selection-label, prediction, and target counts differ")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique within a prediction set")


def _json_scalar(value: np.ndarray | Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    raise ValueError(f"expected scalar value, got shape {array.shape}")


def _string_vector(value: np.ndarray | Sequence[Any], expected: int, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != expected:
        raise ValueError(f"{name} must have shape [{expected}], got {array.shape}")
    return tuple(str(item) for item in array.tolist())


def _derive_sequence_ids(sample_ids: Sequence[str], known_sequence_ids: Sequence[str]) -> tuple[str, ...]:
    ordered = sorted((str(value) for value in known_sequence_ids), key=len, reverse=True)
    result: list[str] = []
    for sample_id in sample_ids:
        matches = [sequence_id for sequence_id in ordered if sample_id.startswith(f"{sequence_id}_")]
        if not matches:
            raise ValueError(f"cannot derive sequence ID from sample {sample_id!r}")
        result.append(matches[0])
    return tuple(result)


def _variant_seed_from_filename(path: Path) -> tuple[str, int | None]:
    match = re.fullmatch(r"(.+)-seed(\d+)", path.stem)
    if match is None:
        return path.stem, None
    return match.group(1), int(match.group(2))


def _scope_from_identity(
    sample_ids: Sequence[str],
    sequence_ids: Sequence[str],
    expected_test_counts: Mapping[str, int],
    declared_scope: str | None,
) -> str:
    observed = {sequence_id: sequence_ids.count(sequence_id) for sequence_id in sorted(set(sequence_ids))}
    full = bool(expected_test_counts) and observed == dict(sorted(expected_test_counts.items()))
    inferred = "full_phase2c_test" if full else "compact_diagnostics_only"
    if declared_scope is None or declared_scope in {"", "auto"}:
        return inferred
    if declared_scope == "full_phase2c_test" and not full:
        raise ValueError(
            "NPZ declares full_phase2c_test but sample identity/counts do not match the manifest: "
            f"observed={observed}, expected={dict(expected_test_counts)}"
        )
    if declared_scope not in {"full_phase2c_test", "compact_diagnostics_only", "custom"}:
        raise ValueError(f"unsupported audit_scope: {declared_scope}")
    return declared_scope


def load_prediction_npz(
    path: Path,
    *,
    known_sequence_ids: Sequence[str],
    expected_test_counts: Mapping[str, int],
) -> list[DepthPredictionSet]:
    """Load either a full Phase-2d interchange file or legacy Phase-2c diagnostics."""

    with np.load(path, allow_pickle=False) as payload:
        if "prediction_m" not in payload or "target_m" not in payload:
            raise ValueError(f"{path} lacks prediction_m/target_m")
        prediction = np.asarray(payload["prediction_m"], dtype=np.float64)
        target = np.asarray(payload["target_m"], dtype=np.float64)
        if prediction.ndim not in {3, 4}:
            raise ValueError(f"prediction_m must be [N,H,W] or [M,N,H,W], got {prediction.shape}")
        model_count = 1 if prediction.ndim == 3 else prediction.shape[0]
        sample_count = prediction.shape[0] if prediction.ndim == 3 else prediction.shape[1]
        if target.ndim == 3:
            if target.shape != prediction.shape[-3:]:
                raise ValueError(f"shared target shape differs from predictions: {target.shape} vs {prediction.shape}")
            targets = np.broadcast_to(target, (model_count, *target.shape))
        elif target.ndim == 4 and target.shape == prediction.shape:
            targets = target
        else:
            raise ValueError(f"target_m shape is incompatible with predictions: {target.shape} vs {prediction.shape}")

        if "schema_version" in payload:
            schema = str(_json_scalar(payload["schema_version"]))
            if schema != PREDICTION_SCHEMA_VERSION:
                raise ValueError(f"unsupported prediction schema {schema!r}")

        id_key = "sample_ids" if "sample_ids" in payload else "test_sample_ids"
        if id_key not in payload:
            raise ValueError(f"{path} must contain sample_ids or test_sample_ids")
        sample_ids = _string_vector(payload[id_key], sample_count, id_key)
        sequence_ids = (
            _string_vector(payload["sequence_ids"], sample_count, "sequence_ids")
            if "sequence_ids" in payload
            else _derive_sequence_ids(sample_ids, known_sequence_ids)
        )
        selection_key = "selection_labels" if "selection_labels" in payload else "test_selection_labels"
        selection_labels = (
            _string_vector(payload[selection_key], sample_count, selection_key)
            if selection_key in payload
            else tuple("unspecified" for _ in range(sample_count))
        )
        declared_scope = str(_json_scalar(payload["audit_scope"])) if "audit_scope" in payload else None
        audit_scope = _scope_from_identity(sample_ids, sequence_ids, expected_test_counts, declared_scope)

        fallback_variant, fallback_seed = _variant_seed_from_filename(path)
        if "variant_ids" in payload:
            variants = _string_vector(payload["variant_ids"], model_count, "variant_ids")
        elif "variant_id" in payload:
            variants = tuple(str(_json_scalar(payload["variant_id"])) for _ in range(model_count))
        elif model_count == 1:
            variants = (fallback_variant,)
        else:
            raise ValueError("multi-variant predictions require variant_ids")
        if "seeds" in payload:
            seeds_array = np.asarray(payload["seeds"])
            if seeds_array.ndim != 1 or len(seeds_array) != model_count:
                raise ValueError(f"seeds must have shape [{model_count}], got {seeds_array.shape}")
            seeds = tuple(None if int(value) < 0 else int(value) for value in seeds_array.tolist())
        elif "seed" in payload:
            scalar_seed = int(_json_scalar(payload["seed"]))
            seeds = tuple(None if scalar_seed < 0 else scalar_seed for _ in range(model_count))
        else:
            seeds = tuple(fallback_seed for _ in range(model_count))

        prediction_models = prediction[None] if prediction.ndim == 3 else prediction
        return [
            DepthPredictionSet(
                variant_id=variants[index],
                seed=seeds[index],
                prediction_m=np.asarray(prediction_models[index], dtype=np.float64),
                target_m=np.asarray(targets[index], dtype=np.float64),
                sample_ids=sample_ids,
                sequence_ids=sequence_ids,
                selection_labels=selection_labels,
                audit_scope=audit_scope,
                source_path=str(path.resolve()),
            )
            for index in range(model_count)
        ]


def discover_phase2c_prediction_files(output: Path) -> list[Path]:
    diagnostics = output / "diagnostics"
    if not diagnostics.is_dir():
        raise FileNotFoundError(f"Phase-2c diagnostics directory is missing: {diagnostics}")
    paths = sorted(diagnostics.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no Phase-2c diagnostic NPZ files under {diagnostics}")
    return paths


def intrinsics_matrix(camera: Mapping[str, Any]) -> np.ndarray:
    required = ("fx", "fy", "cx", "cy")
    if any(key not in camera for key in required):
        raise ValueError(f"camera must contain {required}")
    fx, fy, cx, cy = (float(camera[key]) for key in required)
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError(f"invalid camera intrinsics: {(fx, fy, cx, cy)}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def center_crop_resize_intrinsics(
    intrinsics: np.ndarray,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
) -> dict[str, Any]:
    """Propagate K through Phase-2c center-square crop and resize.

    Sizes are ``(height, width)``.  The resize principal point follows the
    half-pixel mapping used by bilinear ``align_corners=False``:
    ``u_out = (u_in + 0.5) * scale - 0.5``.
    """

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    input_height, input_width = (int(value) for value in input_size)
    output_height, output_width = (int(value) for value in output_size)
    if min(input_height, input_width, output_height, output_width) <= 0:
        raise ValueError("image sizes must be positive")
    crop_size = min(input_height, input_width)
    top = (input_height - crop_size) // 2
    left = (input_width - crop_size) // 2
    crop = np.asarray([[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]])
    cropped = crop @ matrix
    scale_x = output_width / crop_size
    scale_y = output_height / crop_size
    resize = np.asarray(
        [
            [scale_x, 0.0, 0.5 * scale_x - 0.5],
            [0.0, scale_y, 0.5 * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    resized = resize @ cropped
    return {
        "input_size_hw": [input_height, input_width],
        "crop_box_xywh": [left, top, crop_size, crop_size],
        "crop_intrinsics": cropped.tolist(),
        "output_size_hw": [output_height, output_width],
        "resize_scale_xy": [scale_x, scale_y],
        "resize_coordinate_convention": "half-pixel-align-corners-false",
        "output_intrinsics": resized.tolist(),
    }


def field_of_view(intrinsics: np.ndarray, image_size: tuple[int, int]) -> dict[str, float]:
    """Compute asymmetric FoV from pixel-edge rays for a pinhole K."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    height, width = image_size
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if fx <= 0 or fy <= 0 or height <= 0 or width <= 0:
        raise ValueError("FoV inputs must be positive")
    left = math.degrees(math.atan2(cx + 0.5, fx))
    right = math.degrees(math.atan2(width - 0.5 - cx, fx))
    top = math.degrees(math.atan2(cy + 0.5, fy))
    bottom = math.degrees(math.atan2(height - 0.5 - cy, fy))
    return {
        "left_deg": left,
        "right_deg": right,
        "horizontal_deg": left + right,
        "top_deg": top,
        "bottom_deg": bottom,
        "vertical_deg": top + bottom,
    }


def _read_tum_index(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"unexpected {path.name} row: {line}")
        rows.append((float(fields[0]), fields[1]))
    return rows


def _inspect_image_paths(paths: Sequence[Path], *, status: str, root: Path) -> dict[str, Any]:
    sizes: dict[tuple[int, int], int] = {}
    modes: dict[str, int] = {}
    for path in paths:
        with Image.open(path) as image:
            size_hw = (image.height, image.width)
            mode = image.mode
        sizes[size_hw] = sizes.get(size_hw, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1
    ordered = sorted(sizes.items())
    return {
        "status": status,
        "root": str(root),
        "file_count": len(paths),
        "unique_sizes_hw": [{"size_hw": list(size), "count": count} for size, count in ordered],
        "image_modes": [{"mode": mode, "count": count} for mode, count in sorted(modes.items())],
        "consistent": len(ordered) == 1,
        "canonical_size_hw": list(ordered[0][0]) if len(ordered) == 1 else None,
    }


def _selected_image_sizes(dataset_parent: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    root = (dataset_parent / str(entry["root_name"])).resolve(strict=True)
    rgb_rows = _read_tum_index(root / "rgb.txt")
    selected_indices = [int(value) for value in entry.get("selected_indices", [])]
    if not selected_indices:
        raise ValueError(f"sequence {entry.get('sequence_id')} has no selected_indices")
    paths: list[Path] = []
    for index in selected_indices:
        if index < 0 or index >= len(rgb_rows):
            raise IndexError(f"selected RGB index out of range: {index}")
        paths.append(root / rgb_rows[index][1])
    return _inspect_image_paths(paths, status="inspected_selected_rgb_files", root=root)


def _greedy_timestamp_matches(
    left_timestamps: Sequence[float], right_timestamps: Sequence[float], maximum_delta: float
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, timestamp in enumerate(left_timestamps):
        start = bisect.bisect_left(right_timestamps, timestamp - maximum_delta)
        stop = bisect.bisect_right(right_timestamps, timestamp + maximum_delta)
        candidates.extend(
            (abs(timestamp - right_timestamps[right_index]), left_index, right_index)
            for right_index in range(start, stop)
            if abs(timestamp - right_timestamps[right_index]) < maximum_delta
        )
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: dict[int, int] = {}
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches[left_index] = right_index
    return matches


def _selected_depth_sizes(
    dataset_parent: Path,
    entry: Mapping[str, Any],
    *,
    association_max_delta_seconds: float,
) -> dict[str, Any]:
    root = (dataset_parent / str(entry["root_name"])).resolve(strict=True)
    rgb_rows = _read_tum_index(root / "rgb.txt")
    depth_rows = _read_tum_index(root / "depth.txt")
    matches = _greedy_timestamp_matches(
        [row[0] for row in rgb_rows],
        [row[0] for row in depth_rows],
        association_max_delta_seconds,
    )
    selected_indices = [int(value) for value in entry.get("selected_indices", [])]
    missing = sorted(set(selected_indices) - set(matches))
    if missing:
        raise ValueError(f"selected RGB indices lack formal-policy depth matches: {missing[:8]}")
    paths = [root / depth_rows[matches[index]][1] for index in selected_indices]
    return _inspect_image_paths(
        paths,
        status="inspected_selected_depth_files_via_global_greedy_association",
        root=root,
    )


def audit_depth_correction(
    entry: Mapping[str, Any],
    *,
    runtime_correction_factors: Sequence[float] = (),
) -> dict[str, Any]:
    """Audit declared depth correction without inferring absent provenance."""

    depth_scale = float(entry.get("depth_scale", 0.0))
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(f"invalid depth_scale: {depth_scale}")
    explicit_keys = (
        "depth_correction",
        "depth_correction_factor",
        "depth_correction_applied",
        "depth_correction_provenance",
    )
    declarations = {key: entry[key] for key in explicit_keys if key in entry}
    factors = [float(value) for value in runtime_correction_factors]
    if any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("runtime depth correction factors must be finite and positive")
    applied_in_asset = entry.get("depth_correction_applied")
    correction_factor = entry.get("depth_correction_factor")
    repeated_runtime = len([value for value in factors if not math.isclose(value, 1.0)]) > 1
    declared_duplicate = bool(applied_in_asset is True and any(not math.isclose(value, 1.0) for value in factors))
    if not declarations:
        duplicate_status = "unknown_not_declared"
        provenance_status = "unknown_not_declared"
    else:
        duplicate_status = "duplicate_detected" if declared_duplicate or repeated_runtime else "no_duplicate_detected"
        provenance_status = "declared"
    return {
        "png_integer_divisor": depth_scale,
        "depth_scale_semantics": "raw_png_integer_divisor_not_a_correction_factor",
        "phase2c_loader_operations": ["convert_uint16_to_float32", "divide_once_by_depth_scale"],
        "declared_fields": declarations,
        "declared_correction_factor": correction_factor,
        "asset_correction_applied": applied_in_asset,
        "runtime_correction_factors": factors,
        "provenance_status": provenance_status,
        "duplicate_correction_status": duplicate_status,
        "warning": (
            "The manifest does not declare whether an upstream TUM depth correction was already applied; "
            "the audit must not infer it."
            if not declarations
            else None
        ),
    }


def _audit_distortion(entry: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("distortion", "distortion_model", "distortion_coefficients", "images_undistorted")
    declarations = {key: entry[key] for key in keys if key in entry}
    if not declarations:
        return {
            "status": "unknown_not_declared",
            "declared_fields": {},
            "phase2c_undistortion_operation": "none",
            "warning": "The manifest pins K but not distortion coefficients/model or image undistortion state.",
        }
    return {
        "status": "declared",
        "declared_fields": declarations,
        "phase2c_undistortion_operation": "none",
        "warning": None,
    }


def audit_tum_manifest(
    manifest_path: Path,
    *,
    dataset_parent: Path | None = None,
    output_sizes: Sequence[tuple[int, int]] = DEFAULT_OUTPUT_SIZES,
) -> dict[str, Any]:
    """Audit original/transformed K, FoV, distortion, and depth provenance."""

    manifest = yaml.safe_load(manifest_path.read_text())
    entries = manifest.get("sequences")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a non-empty sequences list")
    association_max_delta_seconds = float(manifest.get("association_max_delta_seconds", 0.02))
    if not math.isfinite(association_max_delta_seconds) or association_max_delta_seconds <= 0:
        raise ValueError("association_max_delta_seconds must be finite and positive")
    sequences: list[dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("manifest sequence entries must be mappings")
        entry: dict[str, Any] = raw_entry
        matrix = intrinsics_matrix(entry.get("camera", {}))
        size_audit = (
            _selected_image_sizes(dataset_parent, entry)
            if dataset_parent is not None
            else {
                "status": "not_inspected_dataset_parent_not_provided",
                "root": None,
                "unique_sizes_hw": [],
                "consistent": None,
                "canonical_size_hw": None,
            }
        )
        depth_size_audit = (
            _selected_depth_sizes(
                dataset_parent,
                entry,
                association_max_delta_seconds=association_max_delta_seconds,
            )
            if dataset_parent is not None
            else {
                "status": "not_inspected_dataset_parent_not_provided",
                "root": None,
                "file_count": None,
                "unique_sizes_hw": [],
                "image_modes": [],
                "consistent": None,
                "canonical_size_hw": None,
            }
        )
        canonical = size_audit["canonical_size_hw"]
        transformed: list[dict[str, Any]] = []
        original_fov: dict[str, float] | None = None
        if canonical is not None:
            input_size = (int(canonical[0]), int(canonical[1]))
            original_fov = field_of_view(matrix, input_size)
            for output_size in output_sizes:
                geometry = center_crop_resize_intrinsics(matrix, input_size, output_size)
                crop_size = geometry["crop_box_xywh"][2]
                crop_matrix = np.asarray(geometry["crop_intrinsics"], dtype=np.float64)
                output_matrix = np.asarray(geometry["output_intrinsics"], dtype=np.float64)
                geometry["crop_fov"] = field_of_view(crop_matrix, (crop_size, crop_size))
                geometry["output_fov"] = field_of_view(output_matrix, output_size)
                transformed.append(geometry)
        sequences.append(
            {
                "sequence_id": str(entry.get("sequence_id")),
                "split": str(entry.get("split")),
                "camera_family": str(entry.get("camera_family")),
                "original_intrinsics": matrix.tolist(),
                "selected_rgb_size_audit": size_audit,
                "selected_depth_size_audit": depth_size_audit,
                "rgb_depth_dimensions_match": (
                    size_audit["canonical_size_hw"] == depth_size_audit["canonical_size_hw"]
                    if size_audit["canonical_size_hw"] is not None
                    and depth_size_audit["canonical_size_hw"] is not None
                    else None
                ),
                "rgb_depth_registration_status": "unknown_not_declared",
                "original_fov": original_fov,
                "center_crop_resize": transformed,
                "distortion": _audit_distortion(entry),
                "depth": audit_depth_correction(entry),
            }
        )
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "preprocessing_contract": {
            "crop": "center-square using integer floor offsets",
            "rgb_resize": "bilinear",
            "depth_resize": "nearest",
            "intrinsics_resize_convention": "half-pixel-align-corners-false",
            "undistortion": "none",
        },
        "sequences": sequences,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(matrix, dtype="<f8").tobytes()).hexdigest()


def build_intrinsics_controls(
    intrinsics_by_id: Mapping[str, np.ndarray],
    image_sizes_by_id: Mapping[str, tuple[int, int]],
    *,
    wrong_focal_scale: float = 0.75,
    wrong_principal_shift_fraction: tuple[float, float] = (0.08, -0.06),
) -> dict[str, Any]:
    """Build deterministic correct/wrong/shuffled-K negative controls.

    Shuffling searches for a cyclic assignment with a different K fingerprint
    for every item.  If the supplied collection contains only one unique K,
    the control remains available but is explicitly marked degenerate.
    """

    identifiers = sorted(intrinsics_by_id)
    if not identifiers or set(identifiers) != set(image_sizes_by_id):
        raise ValueError("intrinsics and image-size mappings must have the same non-empty IDs")
    if wrong_focal_scale <= 0 or not math.isfinite(wrong_focal_scale):
        raise ValueError("wrong_focal_scale must be finite and positive")
    matrices = {key: np.asarray(intrinsics_by_id[key], dtype=np.float64) for key in identifiers}
    for key, matrix in matrices.items():
        if matrix.shape != (3, 3):
            raise ValueError(f"intrinsics for {key} are not 3x3")
    fingerprints = {key: _matrix_fingerprint(matrix) for key, matrix in matrices.items()}

    shift = 0
    for candidate in range(1, len(identifiers)):
        if all(
            fingerprints[identifiers[index]] != fingerprints[identifiers[(index + candidate) % len(identifiers)]]
            for index in range(len(identifiers))
        ):
            shift = candidate
            break
    if shift == 0 and len(identifiers) > 1:
        shift = 1
    shuffled_sources = {
        identifier: identifiers[(index + shift) % len(identifiers)] for index, identifier in enumerate(identifiers)
    }

    controls: dict[str, list[dict[str, Any]]] = {"correct": [], "wrong": [], "shuffled": []}
    for identifier in identifiers:
        height, width = image_sizes_by_id[identifier]
        matrix = matrices[identifier]
        wrong = matrix.copy()
        wrong[0, 0] *= wrong_focal_scale
        wrong[1, 1] *= wrong_focal_scale
        wrong[0, 2] += wrong_principal_shift_fraction[0] * width
        wrong[1, 2] += wrong_principal_shift_fraction[1] * height
        source = shuffled_sources[identifier]
        controls["correct"].append(
            {
                "target_id": identifier,
                "source_id": identifier,
                "intrinsics": matrix.tolist(),
                "fingerprint": fingerprints[identifier],
            }
        )
        controls["wrong"].append(
            {
                "target_id": identifier,
                "source_id": f"perturbed:{identifier}",
                "intrinsics": wrong.tolist(),
                "fingerprint": _matrix_fingerprint(wrong),
            }
        )
        controls["shuffled"].append(
            {
                "target_id": identifier,
                "source_id": source,
                "intrinsics": matrices[source].tolist(),
                "fingerprint": fingerprints[source],
            }
        )
    degenerate = [
        row["target_id"] for row in controls["shuffled"] if row["fingerprint"] == fingerprints[str(row["target_id"])]
    ]
    return {
        "schema_version": "jepa4d-intrinsics-negative-controls-v1",
        "wrong_k_policy": {
            "focal_scale": wrong_focal_scale,
            "principal_shift_fraction_xy": list(wrong_principal_shift_fraction),
        },
        "shuffle_policy": "deterministic-cyclic-distinct-K-when-possible",
        "shuffle_shift": shift,
        "shuffled_degenerate_target_ids": degenerate,
        "controls": controls,
    }


def evaluate_intrinsics_controls(
    controls: Mapping[str, Any],
    evaluator: Callable[[Mapping[str, np.ndarray]], _T],
) -> dict[str, _T]:
    """Execute a caller-supplied K-conditioned evaluator for every control."""

    rows_by_mode = controls.get("controls")
    if not isinstance(rows_by_mode, Mapping):
        raise ValueError("controls payload lacks controls mapping")
    results: dict[str, _T] = {}
    for mode in ("correct", "wrong", "shuffled"):
        rows = rows_by_mode.get(mode)
        if not isinstance(rows, list):
            raise ValueError(f"controls payload lacks {mode} rows")
        mapping = {str(row["target_id"]): np.asarray(row["intrinsics"], dtype=np.float64) for row in rows}
        results[mode] = evaluator(mapping)
    return results


def _target_mask(target: np.ndarray) -> np.ndarray:
    return np.isfinite(target) & (target > 0.1) & (target < 10.0)


def _valid_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    target_valid = _target_mask(target)
    prediction_valid = np.isfinite(prediction) & (prediction > 0)
    valid = target_valid & prediction_valid
    target_count = int(target_valid.sum())
    if target_count < 100 or int(valid.sum()) < 100:
        raise ValueError("fewer than 100 valid depth pixels")
    coverage = float(valid.sum() / target_count)
    return prediction[valid].astype(np.float64), target[valid].astype(np.float64), coverage


def _single_frame_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted, truth, coverage = _valid_pair(prediction, target)
    error = predicted - truth
    ratio = np.maximum(predicted / truth, truth / predicted)
    alignment_scale = float(np.median(truth) / max(float(np.median(predicted)), 1e-12))
    return {
        "abs_rel": float(np.mean(np.abs(error) / truth)),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "delta_1": float(np.mean(ratio < 1.25)),
        "abs_log_scale_error": abs(math.log(max(alignment_scale, 1e-12))),
        "alignment_scale": alignment_scale,
        "prediction_coverage_on_valid_target": coverage,
    }


def _per_image_align(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[float]]:
    aligned = np.empty_like(prediction, dtype=np.float64)
    scales: list[float] = []
    for index in range(len(prediction)):
        predicted, truth, _ = _valid_pair(prediction[index], target[index])
        scale = float(np.median(truth) / max(float(np.median(predicted)), 1e-12))
        aligned[index] = prediction[index] * scale
        scales.append(scale)
    return aligned, scales


def _group_indices(values: Sequence[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        grouped.setdefault(value, []).append(index)
    return dict(sorted(grouped.items()))


def _pooled_scalar(prediction: np.ndarray, target: np.ndarray, indices: Sequence[int]) -> float:
    predicted_values: list[np.ndarray] = []
    truth_values: list[np.ndarray] = []
    for index in indices:
        predicted, truth, _ = _valid_pair(prediction[index], target[index])
        predicted_values.append(predicted)
        truth_values.append(truth)
    return float(
        np.median(np.concatenate(truth_values)) / max(float(np.median(np.concatenate(predicted_values))), 1e-12)
    )


def _per_sequence_scalar(
    prediction: np.ndarray, target: np.ndarray, sequence_ids: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.empty_like(prediction, dtype=np.float64)
    parameters: dict[str, Any] = {}
    for sequence_id, indices in _group_indices(sequence_ids).items():
        scale = _pooled_scalar(prediction, target, indices)
        output[indices] = prediction[indices] * scale
        parameters[sequence_id] = {"scale": scale, "frame_count": len(indices)}
    return output, parameters


def _per_image_scalar(
    prediction: np.ndarray, target: np.ndarray, sample_ids: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.empty_like(prediction, dtype=np.float64)
    parameters: dict[str, Any] = {}
    for index, sample_id in enumerate(sample_ids):
        predicted, truth, _ = _valid_pair(prediction[index], target[index])
        scale = float(np.median(truth) / max(float(np.median(predicted)), 1e-12))
        output[index] = prediction[index] * scale
        parameters[sample_id] = {"scale": scale}
    return output, parameters


def _least_squares_affine(prediction: np.ndarray, target: np.ndarray, indices: Sequence[int]) -> tuple[float, float]:
    count = 0
    sum_x = sum_y = sum_xx = sum_xy = 0.0
    for index in indices:
        x, y, _ = _valid_pair(prediction[index], target[index])
        count += len(x)
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_xx += float(np.dot(x, x))
        sum_xy += float(np.dot(x, y))
    denominator = sum_xx - sum_x * sum_x / count
    if count < 2 or denominator <= 1e-12:
        return _pooled_scalar(prediction, target, indices), 0.0
    slope = (sum_xy - sum_x * sum_y / count) / denominator
    intercept = (sum_y - slope * sum_x) / count
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        return _pooled_scalar(prediction, target, indices), 0.0
    return slope, intercept


def _per_sequence_affine(
    prediction: np.ndarray, target: np.ndarray, sequence_ids: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.empty_like(prediction, dtype=np.float64)
    parameters: dict[str, Any] = {}
    for sequence_id, indices in _group_indices(sequence_ids).items():
        slope, intercept = _least_squares_affine(prediction, target, indices)
        output[indices] = np.maximum(prediction[indices] * slope + intercept, 1e-6)
        parameters[sequence_id] = {"slope": slope, "intercept_m": intercept, "frame_count": len(indices)}
    return output, parameters


def _interpolate_grid(grid: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    grid_height, grid_width = grid.shape
    source_x = (np.arange(grid_width, dtype=np.float64) + 0.5) * width / grid_width - 0.5
    source_y = (np.arange(grid_height, dtype=np.float64) + 0.5) * height / grid_height - 0.5
    target_x = np.arange(width, dtype=np.float64)
    target_y = np.arange(height, dtype=np.float64)
    horizontal = np.stack([np.interp(target_x, source_x, row) for row in grid])
    return np.stack([np.interp(target_y, source_y, horizontal[:, column]) for column in range(width)], axis=1)


def _spatial_scale_grid(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_ids: Sequence[str],
    grid_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    grid_height, grid_width = grid_size
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("spatial scale grid dimensions must be positive")
    output = np.empty_like(prediction, dtype=np.float64)
    parameters: dict[str, Any] = {}
    height, width = prediction.shape[-2:]
    for image_index, sample_id in enumerate(sample_ids):
        predicted, truth, _ = _valid_pair(prediction[image_index], target[image_index])
        fallback = float(np.median(truth) / max(float(np.median(predicted)), 1e-12))
        grid = np.full((grid_height, grid_width), fallback, dtype=np.float64)
        valid = (
            _target_mask(target[image_index]) & np.isfinite(prediction[image_index]) & (prediction[image_index] > 0)
        )
        for row in range(grid_height):
            y0, y1 = row * height // grid_height, (row + 1) * height // grid_height
            for column in range(grid_width):
                x0, x1 = column * width // grid_width, (column + 1) * width // grid_width
                cell = valid[y0:y1, x0:x1]
                if int(cell.sum()) < 25:
                    continue
                cell_prediction = prediction[image_index, y0:y1, x0:x1][cell]
                cell_target = target[image_index, y0:y1, x0:x1][cell]
                grid[row, column] = float(np.median(cell_target) / max(float(np.median(cell_prediction)), 1e-12))
        scale_map = _interpolate_grid(grid, (height, width))
        output[image_index] = np.maximum(prediction[image_index] * scale_map, 1e-6)
        parameters[sample_id] = {
            "grid_size_hw": [grid_height, grid_width],
            "fallback_image_scale": fallback,
            "scale_grid": grid.tolist(),
            "scale_min": float(grid.min()),
            "scale_max": float(grid.max()),
        }
    return output, parameters


def _aggregate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_ids: Sequence[str],
    sequence_ids: Sequence[str],
) -> dict[str, Any]:
    aligned, alignment_scales = _per_image_align(prediction, target)
    frame_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        metric = _single_frame_metrics(prediction[index], target[index])
        aligned_metric = _single_frame_metrics(aligned[index], target[index])
        frame_rows.append(
            {
                "sample_id": sample_id,
                "sequence_id": sequence_ids[index],
                "metric": metric,
                "aligned": aligned_metric,
                "post_oracle_alignment_scale": alignment_scales[index],
            }
        )
    metric_keys = tuple(frame_rows[0]["metric"])
    sequence_rows: dict[str, Any] = {}
    for sequence_id, indices in _group_indices(sequence_ids).items():
        sequence_rows[sequence_id] = {
            "frame_count": len(indices),
            "metric": {
                key: float(np.mean([frame_rows[index]["metric"][key] for index in indices])) for key in metric_keys
            },
            "aligned": {
                key: float(np.mean([frame_rows[index]["aligned"][key] for index in indices])) for key in metric_keys
            },
        }
    sequence_values = list(sequence_rows.values())
    macro = {
        "metric": {key: float(np.mean([row["metric"][key] for row in sequence_values])) for key in metric_keys},
        "aligned": {key: float(np.mean([row["aligned"][key] for row in sequence_values])) for key in metric_keys},
        "sequence_count": len(sequence_values),
        "frame_count": len(frame_rows),
    }
    return {"macro_equal_sequence_weight": macro, "per_sequence": sequence_rows, "per_frame": frame_rows}


def run_scale_oracle_audit(
    prediction_set: DepthPredictionSet,
    *,
    spatial_grid_size: tuple[int, int] = (4, 4),
) -> dict[str, Any]:
    """Evaluate raw and four diagnostic-only scale/affine oracle families."""

    prediction = np.asarray(prediction_set.prediction_m, dtype=np.float64)
    target = np.asarray(prediction_set.target_m, dtype=np.float64)
    per_sequence, per_sequence_parameters = _per_sequence_scalar(prediction, target, prediction_set.sequence_ids)
    per_image, per_image_parameters = _per_image_scalar(prediction, target, prediction_set.sample_ids)
    affine, affine_parameters = _per_sequence_affine(prediction, target, prediction_set.sequence_ids)
    spatial, spatial_parameters = _spatial_scale_grid(prediction, target, prediction_set.sample_ids, spatial_grid_size)
    candidates: list[tuple[str, np.ndarray, dict[str, Any], str]] = [
        ("raw", prediction, {}, "none"),
        (
            "per_sequence_scalar",
            per_sequence,
            per_sequence_parameters,
            "pooled median(target)/median(prediction), fit on each evaluated sequence",
        ),
        (
            "per_image_scalar",
            per_image,
            per_image_parameters,
            "median(target)/median(prediction), fit on each evaluated image",
        ),
        (
            "per_sequence_affine",
            affine,
            affine_parameters,
            "least-squares positive slope and intercept, fit on each evaluated sequence",
        ),
        (
            "per_image_lowres_spatial_scale",
            spatial,
            spatial_parameters,
            f"{spatial_grid_size[0]}x{spatial_grid_size[1]} cellwise median scales with bilinear interpolation",
        ),
    ]
    oracles: dict[str, Any] = {}
    for name, values, parameters, fit_policy in candidates:
        oracles[name] = {
            "fit_policy": fit_policy,
            "diagnostic_only": name != "raw",
            "parameters": parameters,
            **_aggregate_metrics(
                values,
                target,
                prediction_set.sample_ids,
                prediction_set.sequence_ids,
            ),
        }
    return {
        "variant_id": prediction_set.variant_id,
        "seed": prediction_set.seed,
        "source_path": prediction_set.source_path,
        "audit_scope": prediction_set.audit_scope,
        "selection_labels": list(prediction_set.selection_labels),
        "sample_count": len(prediction_set.sample_ids),
        "image_size_hw": list(prediction.shape[-2:]),
        "warning": (
            "Oracle parameters use target pixels from the evaluated samples. They diagnose error structure and must not "
            "be reported as deployable model performance."
        ),
        "oracles": oracles,
    }


def _matrix_from_audit(sequence: Mapping[str, Any], output_size: tuple[int, int]) -> np.ndarray | None:
    for transform in sequence.get("center_crop_resize", []):
        if tuple(transform["output_size_hw"]) == output_size:
            return np.asarray(transform["output_intrinsics"], dtype=np.float64)
    return None


def build_manifest_intrinsics_controls(
    calibration_audit: Mapping[str, Any], *, output_size: tuple[int, int] = (518, 518)
) -> dict[str, Any]:
    matrices: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    unavailable: list[str] = []
    for sequence in calibration_audit.get("sequences", []):
        sequence_id = str(sequence["sequence_id"])
        matrix = _matrix_from_audit(sequence, output_size)
        if matrix is None:
            unavailable.append(sequence_id)
            continue
        matrices[sequence_id] = matrix
        sizes[sequence_id] = output_size
    if not matrices:
        return {
            "status": "not_available_without_inspected_image_sizes",
            "output_size_hw": list(output_size),
            "unavailable_sequence_ids": unavailable,
            "evaluation": "not_executed_no_K_conditioned_model_callback",
        }
    controls = build_intrinsics_controls(matrices, sizes)
    controls.update(
        {
            "status": "generated_not_executed",
            "output_size_hw": list(output_size),
            "unavailable_sequence_ids": unavailable,
            "evaluation": "not_executed_no_K_conditioned_model_callback",
        }
    )
    return controls


def assemble_phase2d_report(
    *,
    manifest_path: Path,
    calibration_audit: dict[str, Any],
    oracle_audits: Sequence[dict[str, Any]],
    intrinsics_controls: dict[str, Any],
    prediction_paths: Sequence[Path],
) -> dict[str, Any]:
    scopes = sorted({str(value["audit_scope"]) for value in oracle_audits})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnostic_only": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "prediction_sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in prediction_paths
        ],
        "audit_scopes": scopes,
        "scope_warning": (
            "At least one input contains only the bounded Phase-2c visualization subset; it is not the formal 128-frame "
            "test evaluation."
            if "compact_diagnostics_only" in scopes
            else None
        ),
        "calibration_audit": calibration_audit,
        "intrinsics_negative_controls": intrinsics_controls,
        "scale_oracle_audits": list(oracle_audits),
    }


def _flatten_oracle_summary(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in report.get("scale_oracle_audits", []):
        for oracle_name, oracle in audit["oracles"].items():
            macro = oracle["macro_equal_sequence_weight"]
            rows.append(
                {
                    "variant_id": audit["variant_id"],
                    "seed": audit["seed"],
                    "audit_scope": audit["audit_scope"],
                    "oracle": oracle_name,
                    "metric_abs_rel": macro["metric"]["abs_rel"],
                    "aligned_abs_rel": macro["aligned"]["abs_rel"],
                    "metric_rmse_m": macro["metric"]["rmse_m"],
                    "aligned_rmse_m": macro["aligned"]["rmse_m"],
                    "metric_delta_1": macro["metric"]["delta_1"],
                    "aligned_delta_1": macro["aligned"]["delta_1"],
                    "metric_abs_log_scale_error": macro["metric"]["abs_log_scale_error"],
                    "post_oracle_abs_log_scale_error": macro["metric"]["abs_log_scale_error"],
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _calibration_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in report["calibration_audit"]["sequences"]:
        raw = np.asarray(sequence["original_intrinsics"], dtype=np.float64)
        base = {
            "sequence_id": sequence["sequence_id"],
            "split": sequence["split"],
            "camera_family": sequence["camera_family"],
            "rgb_size_status": sequence["selected_rgb_size_audit"]["status"],
            "original_height": None,
            "original_width": None,
            "stage": "original",
            "output_height": None,
            "output_width": None,
            "fx": raw[0, 0],
            "fy": raw[1, 1],
            "cx": raw[0, 2],
            "cy": raw[1, 2],
            "horizontal_fov_deg": None,
            "vertical_fov_deg": None,
            "distortion_status": sequence["distortion"]["status"],
            "depth_correction_status": sequence["depth"]["duplicate_correction_status"],
        }
        canonical = sequence["selected_rgb_size_audit"]["canonical_size_hw"]
        if canonical is not None:
            base["original_height"], base["original_width"] = canonical
            base["horizontal_fov_deg"] = sequence["original_fov"]["horizontal_deg"]
            base["vertical_fov_deg"] = sequence["original_fov"]["vertical_deg"]
        rows.append(base)
        for transform in sequence["center_crop_resize"]:
            matrix = np.asarray(transform["output_intrinsics"], dtype=np.float64)
            row = dict(base)
            row.update(
                {
                    "stage": "center_crop_resize",
                    "output_height": transform["output_size_hw"][0],
                    "output_width": transform["output_size_hw"][1],
                    "fx": matrix[0, 0],
                    "fy": matrix[1, 1],
                    "cx": matrix[0, 2],
                    "cy": matrix[1, 2],
                    "horizontal_fov_deg": transform["output_fov"]["horizontal_deg"],
                    "vertical_fov_deg": transform["output_fov"]["vertical_deg"],
                }
            )
            rows.append(row)
    return rows


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_format_number(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def _oracle_chart(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>No prediction NPZ was supplied; calibration-only audit.</p>"
    maximum = max(float(row["metric_abs_rel"]) for row in rows) or 1.0
    bars = []
    for row in rows:
        value = float(row["metric_abs_rel"])
        width = max(1.0, 100.0 * value / maximum)
        label = f"{row['variant_id']} seed={row['seed']} · {row['oracle']}"
        bars.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{html.escape(label)}</div>"
            f"<div class='bar-track'><div class='bar' style='width:{width:.2f}%'></div></div>"
            f"<div class='bar-value'>{value:.4f}</div></div>"
        )
    return "".join(bars)


def _spatial_grid_panels(report: Mapping[str, Any], *, maximum_panels: int = 24) -> str:
    panels: list[str] = []
    for audit in report.get("scale_oracle_audits", []):
        parameters = audit["oracles"]["per_image_lowres_spatial_scale"]["parameters"]
        for sample_id, values in parameters.items():
            grid = np.asarray(values["scale_grid"], dtype=np.float64)
            minimum, maximum = float(grid.min()), float(grid.max())
            span = max(maximum - minimum, 1e-12)
            cells = []
            for value in grid.ravel():
                normalized = (float(value) - minimum) / span
                red = round(35 + 205 * normalized)
                green = round(210 - 105 * normalized)
                blue = round(245 - 195 * normalized)
                cells.append(
                    f"<div class='grid-cell' style='background:rgb({red},{green},{blue})' "
                    f"title='scale={float(value):.6f}'>{float(value):.2f}</div>"
                )
            label = f"{audit['variant_id']} seed={audit['seed']} · {sample_id}"
            panels.append(
                "<div class='grid-panel'>"
                f"<div class='grid-title'>{html.escape(label)}</div>"
                f"<div class='scale-grid' style='grid-template-columns:repeat({grid.shape[1]},1fr)'>"
                + "".join(cells)
                + "</div>"
                f"<div class='grid-range'>min {minimum:.3f} · max {maximum:.3f}</div></div>"
            )
            if len(panels) >= maximum_panels:
                break
        if len(panels) >= maximum_panels:
            break
    if not panels:
        return "<p>No spatial-oracle parameters are available.</p>"
    return "<div class='grid-gallery'>" + "".join(panels) + "</div>"


def render_phase2d_html(report: Mapping[str, Any]) -> str:
    """Render a dependency-free, self-contained audit dashboard."""

    oracle_rows = _flatten_oracle_summary(report)
    calibration_rows = _calibration_rows(report)
    warning = report.get("scope_warning")
    oracle_table = _html_table(
        ["Variant", "Seed", "Scope", "Oracle", "AbsRel", "Aligned AbsRel", "RMSE", "Delta-1", "Log-scale"],
        [
            (
                row["variant_id"],
                row["seed"],
                row["audit_scope"],
                row["oracle"],
                row["metric_abs_rel"],
                row["aligned_abs_rel"],
                row["metric_rmse_m"],
                row["metric_delta_1"],
                row["metric_abs_log_scale_error"],
            )
            for row in oracle_rows
        ],
    )
    calibration_table = _html_table(
        [
            "Sequence",
            "Split",
            "Stage",
            "Size",
            "fx",
            "fy",
            "cx",
            "cy",
            "HFoV",
            "VFoV",
            "Distortion",
            "Depth correction",
        ],
        [
            (
                row["sequence_id"],
                row["split"],
                row["stage"],
                (
                    f"{row['output_width']}×{row['output_height']}"
                    if row["output_width"] is not None
                    else (
                        f"{row['original_width']}×{row['original_height']}"
                        if row["original_width"] is not None
                        else "unknown"
                    )
                ),
                row["fx"],
                row["fy"],
                row["cx"],
                row["cy"],
                row["horizontal_fov_deg"],
                row["vertical_fov_deg"],
                row["distortion_status"],
                row["depth_correction_status"],
            )
            for row in calibration_rows
        ],
    )
    embedded = json.dumps(report, sort_keys=True, allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEPA-4D Phase 2d calibration and scale-oracle audit</title>
<style>
:root{{--bg:#0b1020;--panel:#151d32;--ink:#eaf0ff;--muted:#9facca;--accent:#6ee7b7;--warn:#fbbf24;--line:#2b3857}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:28px;margin:0 0 6px}}h2{{margin-top:28px}}
.subtitle{{color:var(--muted)}}.warning{{background:#3a2c0c;border:1px solid var(--warn);padding:12px;border-radius:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}}
.card,.section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}}
.value{{font-size:24px;font-weight:700;color:var(--accent)}}.table-wrap{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#202a43}}
.bar-row{{display:grid;grid-template-columns:minmax(240px,2fr) minmax(180px,5fr) 70px;gap:10px;align-items:center;margin:7px 0}}
.bar-label{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}}.bar-track{{height:13px;background:#24304c;border-radius:8px;overflow:hidden}}.bar{{height:100%;background:linear-gradient(90deg,#38bdf8,#6ee7b7)}}.bar-value{{font-variant-numeric:tabular-nums}}
.grid-gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}.grid-panel{{background:#10172a;border:1px solid var(--line);padding:10px;border-radius:8px}}.grid-title{{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:7px}}.scale-grid{{display:grid;gap:2px}}.grid-cell{{min-height:30px;display:grid;place-items:center;color:#07101b;font-size:11px;font-weight:700}}.grid-range{{font-size:11px;color:var(--muted);margin-top:5px}}
code{{color:#bfdbfe}}details{{margin-top:18px}}pre{{max-height:420px;overflow:auto;background:#080c17;padding:12px;border-radius:8px}}
</style></head><body><main>
<h1>Phase 2d calibration + scale-oracle audit</h1>
<p class="subtitle">Diagnostic only · no model inference · target-fitted oracles are not deployable performance.</p>
{f'<p class="warning">{html.escape(str(warning))}</p>' if warning else ""}
<div class="cards"><div class="card"><div>Prediction inputs</div><div class="value">{len(report.get("prediction_sources", []))}</div></div>
<div class="card"><div>Oracle audits</div><div class="value">{len(report.get("scale_oracle_audits", []))}</div></div>
<div class="card"><div>Sequences audited</div><div class="value">{len(report["calibration_audit"]["sequences"])}</div></div>
<div class="card"><div>Scopes</div><div class="value" style="font-size:14px">{html.escape(", ".join(report.get("audit_scopes", [])) or "calibration-only")}</div></div></div>
<section class="section"><h2>Metric AbsRel by oracle</h2>{_oracle_chart(oracle_rows)}</section>
<section class="section"><h2>Oracle summary</h2>{oracle_table}</section>
<section class="section"><h2>Low-resolution spatial scale factors</h2><p class="subtitle">Bounded to 24 panels; color is normalized within each image, while cells show the actual multiplicative scale.</p>{_spatial_grid_panels(report)}</section>
<section class="section"><h2>Camera/preprocessing audit</h2>{calibration_table}</section>
<section class="section"><h2>Negative-control status</h2><p><code>{html.escape(str(report["intrinsics_negative_controls"].get("status")))}</code>. K controls are generated for a future K-conditioned evaluator; this audit does not pretend they affected K-agnostic stored predictions.</p></section>
<details><summary>Embedded machine-readable report</summary><pre id="json"></pre></details>
<script type="application/json" id="report-data">{embedded}</script>
<script>document.getElementById('json').textContent=JSON.stringify(JSON.parse(document.getElementById('report-data').textContent),null,2);</script>
</main></body></html>"""


def write_phase2d_outputs(report: Mapping[str, Any], output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "phase2d_calibration_scale_audit.json"
    oracle_csv = output / "phase2d_oracle_summary.csv"
    calibration_csv = output / "phase2d_calibration_table.csv"
    html_path = output / "phase2d_calibration_scale_audit.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    oracle_rows = _flatten_oracle_summary(report)
    oracle_fields = (
        list(oracle_rows[0])
        if oracle_rows
        else [
            "variant_id",
            "seed",
            "audit_scope",
            "oracle",
            "metric_abs_rel",
            "aligned_abs_rel",
            "metric_rmse_m",
            "aligned_rmse_m",
            "metric_delta_1",
            "aligned_delta_1",
            "metric_abs_log_scale_error",
            "post_oracle_abs_log_scale_error",
        ]
    )
    _write_csv(oracle_csv, oracle_rows, oracle_fields)
    calibration_rows = _calibration_rows(report)
    _write_csv(calibration_csv, calibration_rows, list(calibration_rows[0]))
    html_path.write_text(render_phase2d_html(report))
    return {
        "json": str(json_path),
        "oracle_csv": str(oracle_csv),
        "calibration_csv": str(calibration_csv),
        "html": str(html_path),
    }


def manifest_sequence_contract(manifest_path: Path) -> tuple[list[str], dict[str, int]]:
    manifest = yaml.safe_load(manifest_path.read_text())
    entries = manifest.get("sequences")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain sequences")
    known = [str(entry["sequence_id"]) for entry in entries]
    expected_test = {
        str(entry["sequence_id"]): len(entry.get("selected_indices", []))
        for entry in entries
        if str(entry.get("split")) == "test"
    }
    return known, expected_test


def load_prediction_sets(paths: Iterable[Path], *, manifest_path: Path) -> list[DepthPredictionSet]:
    known, expected_test = manifest_sequence_contract(manifest_path)
    result: list[DepthPredictionSet] = []
    seen: set[tuple[str, str, int | None]] = set()
    for path in paths:
        for prediction_set in load_prediction_npz(
            path,
            known_sequence_ids=known,
            expected_test_counts=expected_test,
        ):
            key = (str(Path(prediction_set.source_path)), prediction_set.variant_id, prediction_set.seed)
            if key in seen:
                continue
            seen.add(key)
            result.append(prediction_set)
    return result
