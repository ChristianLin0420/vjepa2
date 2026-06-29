from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from jepa4d.benchmarks.geometry.sun_rgbd import (
    SCHEMA_VERSION,
    SUPPORTED_SENSORS,
    USAGE_TERMS,
    audit_manifest_summary,
    build_sensor_blocked_manifest,
    decode_sunrgbd_depth,
    enumerate_sunrgbd_frames,
    load_sensor_blocked_manifest,
    rank_midpoint_selection,
    validate_sunrgbd_frame,
    write_sensor_blocked_manifest,
)

FIXTURE_AVAILABLE_COUNTS = dict.fromkeys(SUPPORTED_SENSORS, 5)


def _encode_depth_millimetres(decoded: np.ndarray) -> np.ndarray:
    values = decoded.astype(np.uint32)
    return np.bitwise_or(np.left_shift(values, 3), np.right_shift(values, 13)).astype(np.uint16)


def test_official_depth_bit_rotation_roundtrip_and_protocol_clamp() -> None:
    millimetres = np.asarray([[0, 1, 999, 1000], [7999, 8000, 8001, 9000]], dtype=np.uint16)
    encoded = _encode_depth_millimetres(millimetres)
    decoded = decode_sunrgbd_depth(encoded)
    assert decoded == pytest.approx(millimetres.astype(np.float32) / 1000.0)
    clamped = decode_sunrgbd_depth(encoded, clamp_max_depth_m=8.0)
    assert clamped.max() == 8.0
    assert clamped[1, 2] == 8.0
    assert decoded[1, 3] == 9.0


def _write_frame(root: Path, sensor: str, index: int, *, shape: tuple[int, int] = (12, 14)) -> None:
    leaf = root / sensor / "collection" / f"frame_{index:03d}"
    (leaf / "image").mkdir(parents=True)
    (leaf / "depth_bfx").mkdir()
    height, width = shape
    rgb = np.full((height, width, 3), 20 + index, dtype=np.uint8)
    Image.fromarray(rgb).save(leaf / "image" / f"rgb_{index:03d}.jpg")
    depth_mm = np.full((height, width), 1000 + index * 100, dtype=np.uint16)
    depth_mm[0, 0] = 0
    Image.fromarray(_encode_depth_millimetres(depth_mm)).save(leaf / "depth_bfx" / f"depth_{index:03d}.png")
    intrinsics = np.asarray([[500.0, 0.0, width / 2], [0.0, 505.0, height / 2], [0.0, 0.0, 1.0]])
    np.savetxt(leaf / "intrinsics.txt", intrinsics, fmt="%.8f")
    (leaf / "fullres").mkdir()
    np.savetxt(leaf / "fullres" / "intrinsics.txt", intrinsics, fmt="%.8f")


def _fixture(tmp_path: Path, frames_per_sensor: int = 5) -> tuple[Path, Path, int, str]:
    root = tmp_path / "SUNRGBD"
    for sensor in SUPPORTED_SENSORS:
        for index in range(frames_per_sensor):
            _write_frame(root, sensor, index)
    archive = tmp_path / "SUNRGBD.zip"
    archive.write_bytes(b"synthetic SUN RGB-D archive identity")
    content = archive.read_bytes()
    return root, archive, len(content), hashlib.sha256(content).hexdigest()


def test_enumerator_uses_strict_leaf_intrinsics_and_rank_midpoints(tmp_path: Path) -> None:
    root, _, _, _ = _fixture(tmp_path)
    frames = enumerate_sunrgbd_frames(root)
    assert {sensor: len(values) for sensor, values in frames.items()} == dict.fromkeys(SUPPORTED_SENSORS, 5)
    assert all("/fullres/" not in str(frame.intrinsics_path) for values in frames.values() for frame in values)
    selected = rank_midpoint_selection(frames["kv1"], 2)
    assert [frame.group_id for frame in selected] == [
        "kv1/collection/frame_001",
        "kv1/collection/frame_003",
    ]


def test_sensor_blocked_manifest_is_deterministic_hashed_and_loadable(tmp_path: Path) -> None:
    root, archive, archive_bytes, archive_sha = _fixture(tmp_path)
    options = {
        "target_counts": {"kv1": 2, "xtion": 2, "realsense": 2, "kv2": 2},
        "clamp_max_depth_m": 8.0,
        "expected_archive_bytes": archive_bytes,
        "expected_archive_sha256": archive_sha,
        "expected_available_counts": FIXTURE_AVAILABLE_COUNTS,
    }
    first = build_sensor_blocked_manifest(root, archive, **options)
    second = build_sensor_blocked_manifest(root, archive, **options)
    assert first == second
    assert first["schema_version"] == SCHEMA_VERSION
    assert first["usage_terms"] == USAGE_TERMS
    assert first["usage_terms"]["license_status"] == "no explicit license text found"
    assert first["usage_terms"]["license_and_use"] == (
        "no explicit license text found; internal research/no redistribution"
    )
    assert first["usage_terms"]["redistribution"] == "no redistribution"
    assert first["selected_counts_by_split"] == {"train": 4, "validation": 2, "test": 2}
    assert first["selected_counts_by_sensor"] == {"kv1": 2, "kv2": 2, "realsense": 2, "xtion": 2}
    assert first["protocol"]["test_sensors"] == ["kv2"]
    assert first["protocol"]["test_usage"] == "untouched_until_final_evaluation"
    assert first["integrity"] == {
        "sample_ids_unique": True,
        "group_ids_unique": True,
        "selected_paths_unique": True,
        "no_path_overlap_across_splits": True,
        "split_hash": first["integrity"]["split_hash"],
    }
    assert all(row["validation"]["rgb_depth_shape_aligned"] for row in first["samples"])
    assert all(len(row["files"]["image"]["sha256"]) == 64 for row in first["samples"])

    manifest_path = tmp_path / "manifest.yaml"
    write_sensor_blocked_manifest(first, manifest_path)
    bundle = load_sensor_blocked_manifest(root, manifest_path)
    assert len(bundle.samples) == 8
    assert {split: len(values) for split, values in bundle.splits.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert {frame.sensor for frame in bundle.splits["train"]} == {"kv1", "xtion"}
    assert {frame.sensor for frame in bundle.splits["validation"]} == {"realsense"}
    assert {frame.sensor for frame in bundle.splits["test"]} == {"kv2"}
    receipt = audit_manifest_summary(bundle)
    assert receipt["result"] == "pass"
    assert receipt["all_selected_files_hash_verified"] is True


def test_manifest_requires_equal_training_sensor_counts(tmp_path: Path) -> None:
    root, archive, archive_bytes, archive_sha = _fixture(tmp_path)
    with pytest.raises(ValueError, match="equal kv1 and xtion"):
        build_sensor_blocked_manifest(
            root,
            archive,
            target_counts={"kv1": 2, "xtion": 1, "realsense": 2, "kv2": 2},
            expected_archive_bytes=archive_bytes,
            expected_archive_sha256=archive_sha,
            expected_available_counts=FIXTURE_AVAILABLE_COUNTS,
        )


def test_manifest_rejects_incomplete_extraction_inventory(tmp_path: Path) -> None:
    root, archive, archive_bytes, archive_sha = _fixture(tmp_path)
    with pytest.raises(ValueError, match="extraction is incomplete"):
        build_sensor_blocked_manifest(
            root,
            archive,
            target_counts={"kv1": 2, "xtion": 2, "realsense": 2, "kv2": 2},
            expected_archive_bytes=archive_bytes,
            expected_archive_sha256=archive_sha,
            expected_available_counts=dict.fromkeys(SUPPORTED_SENSORS, 6),
        )


def test_loader_rejects_hash_and_split_mutations(tmp_path: Path) -> None:
    root, archive, archive_bytes, archive_sha = _fixture(tmp_path)
    manifest = build_sensor_blocked_manifest(
        root,
        archive,
        target_counts={"kv1": 2, "xtion": 2, "realsense": 2, "kv2": 2},
        expected_archive_bytes=archive_bytes,
        expected_archive_sha256=archive_sha,
        expected_available_counts=FIXTURE_AVAILABLE_COUNTS,
    )
    manifest["samples"][0]["files"]["image"]["sha256"] = "0" * 64
    path = tmp_path / "bad-hash.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_sensor_blocked_manifest(root, path)

    manifest = build_sensor_blocked_manifest(
        root,
        archive,
        target_counts={"kv1": 2, "xtion": 2, "realsense": 2, "kv2": 2},
        expected_archive_bytes=archive_bytes,
        expected_archive_sha256=archive_sha,
        expected_available_counts=FIXTURE_AVAILABLE_COUNTS,
    )
    manifest["samples"][0]["split"] = "test"
    path = tmp_path / "bad-split.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="sensor/split mismatch"):
        load_sensor_blocked_manifest(root, path, verify_file_hashes=False, validate_depth=False)


def test_frame_validation_rejects_shape_mismatch_and_bad_intrinsics(tmp_path: Path) -> None:
    root, _, _, _ = _fixture(tmp_path, frames_per_sensor=1)
    frame = enumerate_sunrgbd_frames(root)["kv1"][0]
    valid, metrics = validate_sunrgbd_frame(frame, clamp_max_depth_m=None)
    assert valid.intrinsics is not None
    assert metrics["all_decoded_depth_finite"] is True

    Image.fromarray(np.zeros((10, 10), dtype=np.uint16)).save(frame.depth_path)
    with pytest.raises(ValueError, match="shapes differ"):
        validate_sunrgbd_frame(frame, clamp_max_depth_m=None)

    _write_frame(root, "kv1", 9)
    bad = enumerate_sunrgbd_frames(root)["kv1"][-1]
    bad.intrinsics_path.write_text("1 2 3")
    with pytest.raises(ValueError, match="nine numbers"):
        validate_sunrgbd_frame(bad, clamp_max_depth_m=None)


def test_archive_identity_is_verified_before_dataset_selection(tmp_path: Path) -> None:
    root, archive, archive_bytes, archive_sha = _fixture(tmp_path)
    with pytest.raises(ValueError, match="archive byte mismatch"):
        build_sensor_blocked_manifest(
            root,
            archive,
            target_counts={"kv1": 1, "xtion": 1, "realsense": 1, "kv2": 1},
            expected_archive_bytes=archive_bytes + 1,
            expected_archive_sha256=archive_sha,
            expected_available_counts=FIXTURE_AVAILABLE_COUNTS,
        )
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        build_sensor_blocked_manifest(
            root,
            archive,
            target_counts={"kv1": 1, "xtion": 1, "realsense": 1, "kv2": 1},
            expected_archive_bytes=archive_bytes,
            expected_archive_sha256="0" * 64,
            expected_available_counts=FIXTURE_AVAILABLE_COUNTS,
        )
