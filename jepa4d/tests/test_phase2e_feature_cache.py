from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from jepa4d.benchmarks.geometry.sun_rgbd import SUNRGBDFrame
from jepa4d.data.camera_geometry import update_intrinsics_for_crop_resize
from jepa4d.evaluation.phase2e_feature_cache import (
    CACHE_SCHEMA,
    RECEIPT_SCHEMA,
    PreparedSplit,
    build_separate_cache_payloads,
    centered_log_depth_teacher,
    deterministic_crop_views,
    normalize_final_features,
    prepare_sunrgbd_split,
    sha256_file,
    validate_cache_payload,
    write_feature_cache,
)
from jepa4d.visualization.phase2e_cache_report import build_phase2e_cache_report
from scripts.build_phase2e_sunrgbd_feature_cache import _previews, _split_summary


def _encode_depth_millimetres(decoded: np.ndarray) -> np.ndarray:
    values = decoded.astype(np.uint32)
    return np.bitwise_or(np.left_shift(values, 3), np.right_shift(values, 13)).astype(np.uint16)


def _frame(tmp_path: Path, *, split: str = "train", shape: tuple[int, int] = (100, 140)) -> SUNRGBDFrame:
    height, width = shape
    leaf = tmp_path / "kv1" / "collection" / "frame_000"
    (leaf / "image").mkdir(parents=True)
    (leaf / "depth_bfx").mkdir()
    x = np.linspace(0, 255, width, dtype=np.uint8)[None].repeat(height, axis=0)
    rgb = np.stack((x, np.flip(x, axis=1), np.full_like(x, 80)), axis=-1)
    image_path = leaf / "image" / "rgb.jpg"
    Image.fromarray(rgb).save(image_path)
    depth_mm = np.full((height, width), 2000, dtype=np.uint16)
    depth_mm[:, : width // 4] = 1000
    depth_path = leaf / "depth_bfx" / "depth.png"
    Image.fromarray(_encode_depth_millimetres(depth_mm)).save(depth_path)
    intrinsics = np.asarray([[500.0, 0.0, 69.5], [0.0, 510.0, 49.5], [0.0, 0.0, 1.0]])
    intrinsics_path = leaf / "intrinsics.txt"
    np.savetxt(intrinsics_path, intrinsics)
    return SUNRGBDFrame(
        sample_id="kv1/collection/frame_000/rgb",
        group_id="kv1/collection/frame_000",
        sensor="kv1",
        split=split,
        leaf=leaf,
        image_path=image_path,
        depth_path=depth_path,
        intrinsics_path=intrinsics_path,
        intrinsics=intrinsics,
        image_size_hw=shape,
        depth_size_hw=shape,
    )


def _prepared(name: str, count: int = 1) -> PreparedSplit:
    views = 2 if name == "train" else 1
    return PreparedSplit(
        name=name,  # type: ignore[arg-type]
        images_384=torch.rand(count, views, 3, 384, 384),
        rgb_96=torch.rand(count, views, 3, 96, 96),
        intrinsics_384=torch.tensor([[[[300.0, 0.0, 191.5], [0.0, 305.0, 191.5], [0.0, 0.0, 1.0]]] * views] * count),
        targets_24=torch.full((count, views, 24, 24), 2.0),
        sample_ids=[f"{name}-{index}" for index in range(count)],
        sensor_ids=[{"train": "kv1", "validation": "realsense", "test": "kv2"}[name] for _ in range(count)],
        group_ids=[f"{name}-group-{index}" for index in range(count)],
        crop_boxes=torch.tensor([[[0, 0, 100, 100]] * views] * count),
        source_sizes=torch.tensor([[[100, 140]] * views] * count),
        view_names=("center_square", "center_crop_0.85") if views == 2 else ("center_square",),
    )


def test_deterministic_crop_policy_is_nested_and_repeatable() -> None:
    first = deterministic_crop_views((100, 140), paired=True)
    second = deterministic_crop_views((100, 140), paired=True)
    assert first == second
    assert first[0].crop_box == (0, 20, 100, 100)
    assert first[1].crop_box == (7, 27, 85, 85)
    assert deterministic_crop_views((100, 140), paired=False) == (first[0],)


def test_prepared_views_keep_crop_and_intrinsics_exactly_aligned(tmp_path: Path) -> None:
    frame = _frame(tmp_path)
    prepared = prepare_sunrgbd_split("train", [frame], clamp_max_depth_m=8.0)
    assert prepared.images_384.shape == (1, 2, 3, 384, 384)
    assert prepared.rgb_96.shape == (1, 2, 3, 96, 96)
    assert prepared.targets_24.shape == (1, 2, 24, 24)
    assert prepared.crop_boxes.tolist() == [[[0, 20, 100, 100], [7, 27, 85, 85]]]
    intrinsics = torch.as_tensor(frame.intrinsics, dtype=torch.float32)
    for index, crop in enumerate(((0, 20, 100, 100), (7, 27, 85, 85))):
        expected = update_intrinsics_for_crop_resize(intrinsics, (100, 140), (384, 384), crop=crop)
        assert torch.equal(prepared.intrinsics_384[0, index], expected)
    metadata = prepared.metadata_rows()[0]
    assert metadata["group_id"] == frame.group_id
    assert metadata["views"][1]["source_size_height_width"] == [100, 140]


def test_train_only_normalization_and_centered_teacher() -> None:
    train = torch.stack(
        (
            torch.zeros(768, 24, 24),
            torch.full((768, 24, 24), 2.0),
        ),
        dim=0,
    ).unsqueeze(0)
    validation = torch.full((1, 1, 768, 24, 24), 100.0)
    test = torch.full((1, 1, 768, 24, 24), -100.0)
    normalized_train, _, _, normalizer = normalize_final_features(train, validation, test)
    assert float(normalizer["mean"].mean()) == pytest.approx(1.0)
    assert float(normalized_train.float().mean()) == pytest.approx(0.0, abs=1e-5)
    teacher = centered_log_depth_teacher(torch.rand(2, 2, 24, 24) + 0.5)
    assert teacher.shape == (2, 2, 24, 24)
    assert teacher.float().mean(dim=(-2, -1)) == pytest.approx(torch.zeros(2, 2), abs=2e-4)


def test_cache_schema_keeps_test_physically_separate(tmp_path: Path) -> None:
    prepared = {name: _prepared(name) for name in ("train", "validation", "test")}
    features = {
        "train": torch.zeros(1, 2, 768, 24, 24),
        "validation": torch.zeros(1, 1, 768, 24, 24),
        "test": torch.zeros(1, 1, 768, 24, 24),
    }
    teacher = torch.zeros(1, 2, 24, 24)
    train_validation, test = build_separate_cache_payloads(prepared, features, teacher)
    assert train_validation["schema_version"] == CACHE_SCHEMA
    assert set(train_validation["splits"]) == {"train", "validation"}
    assert set(test["splits"]) == {"test"}
    assert "test" not in train_validation["splits"]
    assert "teacher_centered_shape" in train_validation["splits"]["train"]
    assert "teacher_centered_shape" not in test["splits"]["test"]
    train_path = write_feature_cache(tmp_path / "train_validation_cache.pt", train_validation)
    test_path = write_feature_cache(tmp_path / "test_cache.pt", test)
    assert sha256_file(train_path) != sha256_file(test_path)
    validate_cache_payload(torch.load(train_path, weights_only=True), expected_splits={"train", "validation"})
    validate_cache_payload(torch.load(test_path, weights_only=True), expected_splits={"test"})


def test_cache_stage_never_summarizes_or_previews_test_targets() -> None:
    prepared = {name: _prepared(name) for name in ("train", "validation", "test")}
    raw = torch.zeros(1, 1, 768, 24, 24)
    test_summary = _split_summary(prepared["test"], raw, raw)
    assert "target_depth_m" not in test_summary
    assert not ({"intrinsics_384", "raw_features", "normalized_features"} & set(test_summary))
    assert test_summary["input_tensors"] == {
        "access": "opaque_until_final_evaluation",
        "schema_validation": "pass",
        "statistics_computed": False,
        "preview_generated": False,
    }
    assert test_summary["target_tensor"] == {
        "access": "opaque_until_final_evaluation",
        "schema_validation": "pass",
        "shape": [1, 24, 24],
        "dtype": "float32",
        "statistics_computed": False,
        "preview_generated": False,
    }

    previews = _previews(prepared)
    assert previews
    assert all(not str(preview["label"]).startswith("test/") for preview in previews)
    assert all(preview["sample_id"] != "test-0" for preview in previews)


def test_cache_report_is_self_contained_and_explicitly_non_metric(tmp_path: Path) -> None:
    summaries = {}
    for name, sensor in (("train", "kv1"), ("validation", "realsense"), ("test", "kv2")):
        summary = {
            "samples": 1,
            "views": 2 if name == "train" else 1,
            "sensor_counts": {sensor: 1},
            "shapes": {
                "features": [1, 2, 768, 24, 24] if name == "train" else [1, 768, 24, 24],
                "rgb": [1, 2, 3, 96, 96] if name == "train" else [1, 3, 96, 96],
                "targets": [1, 2, 24, 24] if name == "train" else [1, 24, 24],
            },
        }
        if name == "test":
            summary["input_tensors"] = {
                "access": "opaque_until_final_evaluation",
                "schema_validation": "pass",
                "statistics_computed": False,
                "preview_generated": False,
            }
            summary["target_tensor"] = {
                "access": "opaque_until_final_evaluation",
                "schema_validation": "pass",
                "shape": [1, 24, 24],
                "dtype": "float32",
                "statistics_computed": False,
                "preview_generated": False,
            }
        else:
            summary["intrinsics_384"] = {
                "fx_mean": 300.0,
                "fy_mean": 305.0,
                "cx_mean": 191.5,
                "cy_mean": 191.5,
            }
            summary["normalized_features"] = {"mean": 0.0, "std": 1.0}
            summary["target_depth_m"] = {
                "valid_fraction": 1.0,
                "valid_min_m": 1.0,
                "valid_max_m": 3.0,
                "valid_mean_m": 2.0,
                "valid_std_m": 0.2,
            }
        summaries[name] = summary
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "dataset": {"dataset_id": "SUN-RGBD", "split_hash": "abc"},
        "view_policy": {"name": "test"},
        "feature_normalization": {"policy": "train-only"},
        "teacher_policy": {"split": "train-only"},
        "caches": {"train_validation": {"sha256": "a"}, "test": {"sha256": "b"}},
        "split_summaries": summaries,
        "test_samples": 1,
    }
    previews = [{"label": "validation/realsense", "rgb": np.zeros((8, 8, 3)), "depth": np.ones((4, 4))}]
    report = build_phase2e_cache_report(receipt, previews, tmp_path / "report.html")
    content = report.read_text()
    assert "computes no model-quality metric" in content
    assert "no test-target statistic, preview, or W&amp;B media" in content
    assert "plotly.js" in content.lower()
    assert "<script src=" not in content.lower()

    summaries["test"]["target_depth_m"] = {
        "valid_fraction": 1.0,
        "valid_min_m": 1.0,
        "valid_max_m": 3.0,
        "valid_mean_m": 2.0,
        "valid_std_m": 0.2,
    }
    with pytest.raises(ValueError, match="test target statistics are forbidden"):
        build_phase2e_cache_report(receipt, previews, tmp_path / "leaky-statistics.html")
    del summaries["test"]["target_depth_m"]
    with pytest.raises(ValueError, match="test target previews are forbidden"):
        build_phase2e_cache_report(
            receipt,
            [{"label": "test/kv2", "rgb": np.zeros((8, 8, 3)), "depth": np.ones((4, 4))}],
            tmp_path / "leaky-preview.html",
        )
    summaries["test"]["normalized_features"] = {"mean": 0.0, "std": 1.0}
    with pytest.raises(ValueError, match="test input statistics are forbidden"):
        build_phase2e_cache_report(receipt, previews, tmp_path / "leaky-input-statistics.html")
