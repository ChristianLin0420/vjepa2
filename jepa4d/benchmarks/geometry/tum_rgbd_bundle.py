"""Sequence-disjoint, content-bound TUM RGB-D bundles for Phase 2c."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jepa4d.benchmarks.geometry.tum_rgbd import TUMSample, _read_index

SPLITS = ("train", "validation", "test")
ASSOCIATION_POLICY = "global-greedy-one-to-one-rgb-depth-and-rgb-pose-v1"
SELECTION_POLICY = "rank-midpoint-quantiles-over-association-valid-rgb-v1"
FORMAL_SEQUENCE_CONTRACT: dict[str, dict[str, Any]] = {
    "freiburg1_xyz": {
        "split": "train",
        "camera_family": "freiburg1",
        "root_name": "rgbd_dataset_freiburg1_xyz",
        "source_url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
        "archive_filename": "rgbd_dataset_freiburg1_xyz.tgz",
        "archive_bytes": 448204271,
        "archive_sha256": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
        "camera": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    },
    "freiburg1_floor": {
        "split": "train",
        "camera_family": "freiburg1",
        "root_name": "rgbd_dataset_freiburg1_floor",
        "source_url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz",
        "archive_filename": "rgbd_dataset_freiburg1_floor.tgz",
        "archive_bytes": 734599281,
        "archive_sha256": "16f8e7f13e0fc3983711b5d9db8ee663c73a5e29225b64dbe5ccaa9257e082bb",
        "camera": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    },
    "freiburg2_xyz": {
        "split": "validation",
        "camera_family": "freiburg2",
        "root_name": "rgbd_dataset_freiburg2_xyz",
        "source_url": "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz",
        "archive_filename": "rgbd_dataset_freiburg2_xyz.tgz",
        "archive_bytes": 2201854648,
        "archive_sha256": "a4903027a7fe6bc6573a1ce96a8afa5689f15d74f4a3f0627e91b5799f62e40b",
        "camera": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    },
    "freiburg3_long_office_household": {
        "split": "test",
        "camera_family": "freiburg3",
        "root_name": "rgbd_dataset_freiburg3_long_office_household",
        "source_url": (
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
        ),
        "archive_filename": "rgbd_dataset_freiburg3_long_office_household.tgz",
        "archive_bytes": 1483556251,
        "archive_sha256": "c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c",
        "camera": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
    },
    "freiburg3_structure_texture_far": {
        "split": "test",
        "camera_family": "freiburg3",
        "root_name": "rgbd_dataset_freiburg3_structure_texture_far",
        "source_url": (
            "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz"
        ),
        "archive_filename": "rgbd_dataset_freiburg3_structure_texture_far.tgz",
        "archive_bytes": 520806553,
        "archive_sha256": "3f58c707f54c93b68fecd77293630e96a76d7b0f2703eb8ad9cff45ba4bbb81a",
        "camera": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
    },
}


@dataclass(frozen=True, slots=True)
class TUMSequenceSelection:
    sequence_id: str
    camera_family: str
    split: str
    root: Path
    archive: Path
    samples: tuple[TUMSample, ...]
    maximum_rgb_depth_delta_seconds: float
    maximum_rgb_pose_delta_seconds: float
    archive_sha256: str
    association_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class TUMCrossSequenceBundle:
    manifest: dict[str, Any]
    manifest_path: Path
    selections: tuple[TUMSequenceSelection, ...]
    splits: dict[str, list[TUMSample]]
    fingerprint: dict[str, Any]
    split_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _greedy_one_to_one(
    left: list[tuple[float, list[str]]],
    right: list[tuple[float, list[str]]],
    maximum_delta: float,
) -> dict[int, tuple[int, float]]:
    """Globally minimize timestamp deltas while using every row at most once."""
    if maximum_delta <= 0:
        raise ValueError("association maximum delta must be positive")
    right_timestamps = [value[0] for value in right]
    candidates: list[tuple[float, int, int]] = []
    for left_index, (timestamp, _) in enumerate(left):
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
    associations: dict[int, tuple[int, float]] = {}
    for delta, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        associations[left_index] = (right_index, delta)
    return associations


def association_valid_indices(root: Path, maximum_delta: float) -> tuple[list[int], dict[int, Any], dict[int, Any]]:
    rgb = _read_index(root / "rgb.txt", 2)
    depth = _read_index(root / "depth.txt", 2)
    groundtruth = _read_index(root / "groundtruth.txt", 8)
    depth_matches = _greedy_one_to_one(rgb, depth, maximum_delta)
    pose_matches = _greedy_one_to_one(rgb, groundtruth, maximum_delta)
    valid = sorted(set(depth_matches) & set(pose_matches))
    return valid, depth_matches, pose_matches


def rank_midpoint_indices(valid_indices: list[int], frame_count: int) -> list[int]:
    if frame_count <= 0 or len(valid_indices) < frame_count:
        raise ValueError(f"need at least {frame_count} association-valid frames, found {len(valid_indices)}")
    selected = [
        valid_indices[math.floor((index + 0.5) * len(valid_indices) / frame_count)] for index in range(frame_count)
    ]
    if len(selected) != len(set(selected)) or selected != sorted(selected):
        raise RuntimeError("rank-midpoint selection did not produce unique chronological indices")
    return selected


def _load_selection(
    dataset_parent: Path,
    entry: dict[str, Any],
    frame_count: int,
    maximum_delta: float,
) -> TUMSequenceSelection:
    sequence_id = str(entry["sequence_id"])
    split = str(entry["split"])
    if split not in SPLITS:
        raise ValueError(f"invalid split for {sequence_id}: {split}")
    root = (dataset_parent / str(entry["root_name"])).resolve(strict=True)
    archive = (dataset_parent / str(entry["archive"]["filename"])).resolve(strict=True)
    if archive.stat().st_size != int(entry["archive"]["bytes"]):
        raise ValueError(f"archive byte count mismatch for {sequence_id}")
    expected_sha = str(entry["archive"]["sha256"])
    actual_sha = _sha256(archive)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ValueError(f"archive SHA-256 mismatch for {sequence_id}")
    valid, depth_matches, pose_matches = association_valid_indices(root, maximum_delta)
    expected_indices = rank_midpoint_indices(valid, frame_count)
    selected_indices = [int(value) for value in entry["selected_indices"]]
    if selected_indices != expected_indices:
        raise ValueError(f"frozen selection differs from deterministic association-valid selection for {sequence_id}")

    rgb = _read_index(root / "rgb.txt", 2)
    depth = _read_index(root / "depth.txt", 2)
    groundtruth = _read_index(root / "groundtruth.txt", 8)
    samples: list[TUMSample] = []
    association_rows: list[dict[str, Any]] = []
    depth_deltas = []
    pose_deltas = []
    for rgb_index in selected_indices:
        depth_index, depth_delta = depth_matches[rgb_index]
        pose_index, pose_delta = pose_matches[rgb_index]
        timestamp, rgb_values = rgb[rgb_index]
        depth_timestamp, depth_values = depth[depth_index]
        pose_timestamp, pose_values = groundtruth[pose_index]
        depth_deltas.append(depth_delta)
        pose_deltas.append(pose_delta)
        samples.append(
            TUMSample(
                sample_id=f"{sequence_id}_{rgb_index:06d}",
                timestamp=timestamp,
                rgb_path=root / rgb_values[0],
                depth_path=root / depth_values[0],
                translation=np.asarray(pose_values[:3], dtype=np.float64),
                quaternion_xyzw=np.asarray(pose_values[3:], dtype=np.float64),
                sequence_id=sequence_id,
                depth_scale=float(entry.get("depth_scale", 5000.0)),
            )
        )
        association_rows.append(
            {
                "rgb_index": rgb_index,
                "rgb_timestamp": timestamp,
                "rgb_path": rgb_values[0],
                "depth_index": depth_index,
                "depth_timestamp": depth_timestamp,
                "depth_path": depth_values[0],
                "rgb_depth_delta_seconds": depth_delta,
                "pose_index": pose_index,
                "pose_timestamp": pose_timestamp,
                "rgb_pose_delta_seconds": pose_delta,
            }
        )
    return TUMSequenceSelection(
        sequence_id=sequence_id,
        camera_family=str(entry["camera_family"]),
        split=split,
        root=root,
        archive=archive,
        samples=tuple(samples),
        maximum_rgb_depth_delta_seconds=max(depth_deltas),
        maximum_rgb_pose_delta_seconds=max(pose_deltas),
        archive_sha256=actual_sha,
        association_rows=tuple(association_rows),
    )


def _selection_fingerprint(selection: TUMSequenceSelection) -> dict[str, Any]:
    root = selection.root
    required = [root / name for name in ("rgb.txt", "depth.txt", "groundtruth.txt")]
    selected = sorted(
        {sample.rgb_path.resolve() for sample in selection.samples}
        | {sample.depth_path.resolve() for sample in selection.samples}
    )
    for path in required + selected:
        if not path.is_file() or not path.is_relative_to(root):
            raise FileNotFoundError(f"invalid extracted dataset file: {path}")
    extracted = {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in required + selected
    }
    wanted_members = {f"{root.name}/{relative}": relative for relative in extracted}
    archive_members: dict[str, dict[str, Any]] = {}
    with tarfile.open(selection.archive, mode="r:gz") as bundle:
        for member in bundle:
            relative = wanted_members.get(member.name.removeprefix("./"))
            if relative is None:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"unable to read required archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            archive_members[relative] = {"bytes": size, "sha256": digest.hexdigest()}
    if archive_members != extracted:
        missing = sorted(set(extracted) - set(archive_members))
        mismatched = sorted(key for key in archive_members if archive_members[key] != extracted.get(key))
        raise ValueError(f"dataset root differs from archive: missing={missing}, mismatched={mismatched}")
    return {
        "sequence_id": selection.sequence_id,
        "camera_family": selection.camera_family,
        "split": selection.split,
        "root": str(root),
        "archive": {
            "path": str(selection.archive),
            "bytes": selection.archive.stat().st_size,
            "sha256": selection.archive_sha256,
        },
        "index_files": {path.name: extracted[str(path.relative_to(root))] for path in required},
        "selected_files": [
            {"path": str(path.relative_to(root)), **extracted[str(path.relative_to(root))]} for path in selected
        ],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "timestamp": sample.timestamp,
                "rgb": str(sample.rgb_path.relative_to(root)),
                "depth": str(sample.depth_path.relative_to(root)),
            }
            for sample in selection.samples
        ],
        "associations": list(selection.association_rows),
        "maximum_rgb_depth_delta_seconds": selection.maximum_rgb_depth_delta_seconds,
        "maximum_rgb_pose_delta_seconds": selection.maximum_rgb_pose_delta_seconds,
        "archive_extraction_verified": True,
    }


def _validate_formal_entry(entry: dict[str, Any]) -> None:
    sequence_id = str(entry.get("sequence_id"))
    if sequence_id not in FORMAL_SEQUENCE_CONTRACT:
        raise ValueError(f"unexpected formal Phase 2c sequence: {sequence_id}")
    expected = FORMAL_SEQUENCE_CONTRACT[sequence_id]
    observed = {
        "split": entry.get("split"),
        "camera_family": entry.get("camera_family"),
        "root_name": entry.get("root_name"),
        "source_url": entry.get("source_url"),
        "archive_filename": entry.get("archive", {}).get("filename"),
        "archive_bytes": entry.get("archive", {}).get("bytes"),
        "archive_sha256": entry.get("archive", {}).get("sha256"),
        "camera": entry.get("camera"),
    }
    if observed != expected:
        raise ValueError(f"formal sequence contract changed for {sequence_id}: {observed}")
    if entry.get("public_ground_truth") is not True or float(entry.get("depth_scale", 0.0)) != 5000.0:
        raise ValueError(f"public ground truth and depth scale are not pinned for {sequence_id}")


def load_cross_sequence_bundle(dataset_parent: Path, manifest_path: Path) -> TUMCrossSequenceBundle:
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("schema_version") != "jepa4d-tum-cross-sequence-v1":
        raise ValueError("unexpected cross-sequence manifest schema")
    expected_manifest_fields = {
        "dataset_id": "tum-rgbd-phase2c-camera-family-blocked",
        "version": "1.0.0",
        "license": "CC-BY-4.0",
        "official": "True",
        "split_policy": "sequence-disjoint-camera-family-blocked-2-train-1-validation-2-test",
        "primary_metric": "equal-weight-macro-per-test-sequence-metric-absrel",
        "checkpoint_selection": "validation-sequence-metric-absrel-only",
        "teacher_scale_policy": "one-global-scale-fitted-on-pooled-training-sequences-and-frozen",
        "comparison_schema": "jepa4d-phase2c-cross-sequence-comparison-v1",
    }
    observed_manifest_fields = {key: str(manifest.get(key)) for key in expected_manifest_fields}
    if observed_manifest_fields != expected_manifest_fields:
        raise ValueError(f"formal Phase 2c manifest policy changed: {observed_manifest_fields}")
    entries = manifest.get("sequences")
    if not isinstance(entries, list) or len(entries) != 5:
        raise ValueError("formal Phase 2c requires exactly five sequences")
    sequence_ids = [str(entry["sequence_id"]) for entry in entries]
    root_names = [str(entry["root_name"]) for entry in entries]
    if len(set(sequence_ids)) != len(sequence_ids) or len(set(root_names)) != len(root_names):
        raise ValueError("sequence IDs and dataset roots must be unique")
    if sequence_ids != list(FORMAL_SEQUENCE_CONTRACT):
        raise ValueError(f"formal Phase 2c sequence order/set changed: {sequence_ids}")
    for entry in entries:
        _validate_formal_entry(entry)
    split_counts = {split: sum(str(entry["split"]) == split for entry in entries) for split in SPLITS}
    if split_counts != {"train": 2, "validation": 1, "test": 2}:
        raise ValueError(f"formal camera-family-blocked roles must be 2/1/2, found {split_counts}")
    camera_by_split = {
        split: {str(entry["camera_family"]) for entry in entries if str(entry["split"]) == split} for split in SPLITS
    }
    if camera_by_split != {"train": {"freiburg1"}, "validation": {"freiburg2"}, "test": {"freiburg3"}}:
        raise ValueError(f"camera families must be blocked by split, found {camera_by_split}")
    frame_count = int(manifest.get("frames_per_sequence", 0))
    maximum_delta = float(manifest.get("association_max_delta_seconds", 0.0))
    if frame_count != 64 or maximum_delta != 0.02:
        raise ValueError("formal Phase 2c requires 64 frames/sequence and a 20 ms association bound")
    if (
        manifest.get("association_policy") != ASSOCIATION_POLICY
        or manifest.get("selection_policy") != SELECTION_POLICY
    ):
        raise ValueError("unexpected association or selection policy")

    selections = tuple(_load_selection(dataset_parent, entry, frame_count, maximum_delta) for entry in entries)
    resolved_roots = [selection.root for selection in selections]
    resolved_archives = [selection.archive for selection in selections]
    if len(set(resolved_roots)) != len(resolved_roots) or len(set(resolved_archives)) != len(resolved_archives):
        raise ValueError("formal Phase 2c roots or archives alias across sequence roles")
    splits = {
        split: [sample for selection in selections if selection.split == split for sample in selection.samples]
        for split in SPLITS
    }
    if {key: len(value) for key, value in splits.items()} != {"train": 128, "validation": 64, "test": 128}:
        raise ValueError("formal Phase 2c split must contain exactly 128/64/128 frames")
    if len({sample.sample_id for values in splits.values() for sample in values}) != 320:
        raise ValueError("cross-sequence sample IDs are not globally unique")
    fingerprints = [_selection_fingerprint(selection) for selection in selections]
    manifest_sha = _sha256(manifest_path)
    fingerprint = {
        "schema_version": "jepa4d-tum-cross-sequence-fingerprint-v1",
        "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
        "sequences": fingerprints,
        "split_counts": {key: len(value) for key, value in splits.items()},
    }
    split_contract = {
        "manifest_sha256": manifest_sha,
        "sequences": [
            {
                "sequence_id": selection.sequence_id,
                "split": selection.split,
                "sample_ids": [sample.sample_id for sample in selection.samples],
            }
            for selection in selections
        ],
    }
    split_hash = hashlib.sha256(json.dumps(split_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return TUMCrossSequenceBundle(manifest, manifest_path.resolve(), selections, splits, fingerprint, split_hash)
