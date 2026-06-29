from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from jepa4d.evaluation.phase2d_calibration_audit import (
    PREDICTION_SCHEMA_VERSION,
    DepthPredictionSet,
    assemble_phase2d_report,
    audit_depth_correction,
    audit_tum_manifest,
    build_intrinsics_controls,
    center_crop_resize_intrinsics,
    evaluate_intrinsics_controls,
    field_of_view,
    load_prediction_npz,
    run_scale_oracle_audit,
    write_phase2d_outputs,
)


def test_center_crop_resize_updates_intrinsics_and_preserves_cropped_fov() -> None:
    intrinsics = np.asarray([[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]])
    transformed = center_crop_resize_intrinsics(intrinsics, (480, 640), (384, 384))
    assert transformed["crop_box_xywh"] == [80, 0, 480, 480]
    cropped = np.asarray(transformed["crop_intrinsics"])
    output = np.asarray(transformed["output_intrinsics"])
    assert cropped[0, 2] == pytest.approx(240.0)
    assert cropped[1, 2] == pytest.approx(240.0)
    assert output[0, 0] == pytest.approx(400.0)
    assert output[1, 1] == pytest.approx(408.0)
    assert output[0, 2] == pytest.approx((240.0 + 0.5) * 0.8 - 0.5)
    crop_fov = field_of_view(cropped, (480, 480))
    output_fov = field_of_view(output, (384, 384))
    assert output_fov["horizontal_deg"] == pytest.approx(crop_fov["horizontal_deg"])
    assert output_fov["vertical_deg"] == pytest.approx(crop_fov["vertical_deg"])
    assert crop_fov["horizontal_deg"] < field_of_view(intrinsics, (480, 640))["horizontal_deg"]


def _write_manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "datasets"
    root = parent / "sequence-root"
    rgb = root / "rgb"
    depth = root / "depth"
    rgb.mkdir(parents=True)
    depth.mkdir()
    Image.new("RGB", (640, 480), color=(20, 40, 60)).save(rgb / "0000.png")
    Image.fromarray(np.full((480, 640), 5000, dtype=np.uint16)).save(depth / "0000.png")
    (root / "rgb.txt").write_text("# timestamp path\n1.0 rgb/0000.png\n")
    (root / "depth.txt").write_text("# timestamp path\n1.0 depth/0000.png\n")
    manifest = {
        "schema_version": "fixture",
        "sequences": [
            {
                "sequence_id": "sequence_a",
                "split": "test",
                "camera_family": "camera_a",
                "root_name": "sequence-root",
                "depth_scale": 5000.0,
                "camera": {"fx": 500.0, "fy": 510.0, "cx": 320.0, "cy": 240.0},
                "selected_indices": [0],
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path, parent


def test_manifest_audit_reports_unknown_provenance_and_transformed_k(tmp_path: Path) -> None:
    manifest, parent = _write_manifest_fixture(tmp_path)
    audit = audit_tum_manifest(manifest, dataset_parent=parent)
    sequence = audit["sequences"][0]
    assert sequence["selected_rgb_size_audit"]["canonical_size_hw"] == [480, 640]
    assert sequence["selected_depth_size_audit"]["canonical_size_hw"] == [480, 640]
    assert sequence["rgb_depth_dimensions_match"] is True
    assert sequence["rgb_depth_registration_status"] == "unknown_not_declared"
    assert sequence["distortion"]["status"] == "unknown_not_declared"
    assert sequence["depth"]["provenance_status"] == "unknown_not_declared"
    assert sequence["depth"]["duplicate_correction_status"] == "unknown_not_declared"
    assert [row["output_size_hw"] for row in sequence["center_crop_resize"]] == [[384, 384], [518, 518]]


def test_depth_correction_detects_declared_double_application() -> None:
    audit = audit_depth_correction(
        {
            "depth_scale": 5000.0,
            "depth_correction_applied": True,
            "depth_correction_factor": 1.035,
            "depth_correction_provenance": "fixture",
        },
        runtime_correction_factors=(1.035,),
    )
    assert audit["duplicate_correction_status"] == "duplicate_detected"
    assert audit["png_integer_divisor"] == 5000.0
    assert "not_a_correction_factor" in audit["depth_scale_semantics"]


def test_intrinsics_negative_controls_are_reusable_and_non_degenerate() -> None:
    matrices = {
        "a": np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        "b": np.asarray([[530.0, 0.0, 315.0], [0.0, 535.0, 245.0], [0.0, 0.0, 1.0]]),
    }
    sizes = {"a": (480, 640), "b": (480, 640)}
    controls = build_intrinsics_controls(matrices, sizes)
    assert controls["shuffled_degenerate_target_ids"] == []
    rows = controls["controls"]
    assert rows["correct"][0]["fingerprint"] != rows["wrong"][0]["fingerprint"]
    assert rows["shuffled"][0]["source_id"] == "b"

    evaluated = evaluate_intrinsics_controls(
        controls,
        lambda values: {key: float(value[0, 0]) for key, value in values.items()},
    )
    assert set(evaluated) == {"correct", "wrong", "shuffled"}
    assert evaluated["correct"]["a"] == 500.0
    assert evaluated["shuffled"]["a"] == 530.0


def _prediction_set(prediction: np.ndarray, target: np.ndarray) -> DepthPredictionSet:
    count = len(prediction)
    sample_ids = tuple(f"seq_{index // 2}_{index:03d}" for index in range(count))
    sequence_ids = tuple(f"seq_{index // 2}" for index in range(count))
    return DepthPredictionSet(
        variant_id="fixture",
        seed=0,
        prediction_m=prediction,
        target_m=target,
        sample_ids=sample_ids,
        sequence_ids=sequence_ids,
        selection_labels=tuple("fixture" for _ in range(count)),
        audit_scope="custom",
        source_path="fixture.npz",
    )


def test_sequence_and_image_scalar_oracles_recover_group_scale() -> None:
    target = np.stack([np.linspace(1.0, 4.0, 400).reshape(20, 20) for _ in range(4)])
    prediction = target.copy()
    prediction[:2] /= 2.0
    prediction[2:] /= 4.0
    result = run_scale_oracle_audit(_prediction_set(prediction, target), spatial_grid_size=(2, 2))
    raw = result["oracles"]["raw"]["macro_equal_sequence_weight"]["metric"]["abs_rel"]
    sequence = result["oracles"]["per_sequence_scalar"]["macro_equal_sequence_weight"]["metric"]["abs_rel"]
    image = result["oracles"]["per_image_scalar"]["macro_equal_sequence_weight"]["metric"]["abs_rel"]
    assert raw > 0.5
    assert sequence < 1e-12
    assert image < 1e-12
    assert result["oracles"]["per_sequence_scalar"]["parameters"]["seq_0"]["scale"] == pytest.approx(2.0)
    assert result["oracles"]["per_sequence_scalar"]["parameters"]["seq_1"]["scale"] == pytest.approx(4.0)


def test_low_resolution_spatial_oracle_exposes_spatial_scale_error() -> None:
    target = np.full((2, 40, 40), 4.0)
    prediction = np.empty_like(target)
    prediction[:, :, :20] = target[:, :, :20] / 2.0
    prediction[:, :, 20:] = target[:, :, 20:] / 4.0
    result = run_scale_oracle_audit(_prediction_set(prediction, target), spatial_grid_size=(1, 2))
    scalar = result["oracles"]["per_image_scalar"]["macro_equal_sequence_weight"]["metric"]["abs_rel"]
    spatial = result["oracles"]["per_image_lowres_spatial_scale"]["macro_equal_sequence_weight"]["metric"]["abs_rel"]
    assert spatial < scalar
    grids = result["oracles"]["per_image_lowres_spatial_scale"]["parameters"]
    assert np.asarray(next(iter(grids.values()))["scale_grid"]) == pytest.approx(np.asarray([[2.0, 4.0]]))


def test_prediction_loader_accepts_full_multivariant_schema_and_marks_compact(tmp_path: Path) -> None:
    target = np.ones((4, 12, 12), dtype=np.float32)
    predictions = np.stack((target * 0.5, target * 0.25))
    sample_ids = np.asarray(["test_a_000", "test_a_001", "test_b_000", "test_b_001"])
    full = tmp_path / "full.npz"
    np.savez_compressed(
        full,
        schema_version=np.asarray(PREDICTION_SCHEMA_VERSION),
        prediction_m=predictions,
        target_m=target,
        sample_ids=sample_ids,
        sequence_ids=np.asarray(["test_a", "test_a", "test_b", "test_b"]),
        variant_ids=np.asarray(["final", "fixed"]),
        seeds=np.asarray([0, 1]),
        audit_scope=np.asarray("full_phase2c_test"),
    )
    loaded = load_prediction_npz(
        full,
        known_sequence_ids=("test_a", "test_b"),
        expected_test_counts={"test_a": 2, "test_b": 2},
    )
    assert [value.variant_id for value in loaded] == ["final", "fixed"]
    assert all(value.audit_scope == "full_phase2c_test" for value in loaded)

    compact = tmp_path / "vjepa_final-seed2.npz"
    np.savez_compressed(
        compact,
        prediction_m=target[:2],
        target_m=target[:2],
        test_sample_ids=sample_ids[:2],
        test_selection_labels=np.asarray(["midpoint", "worst"]),
    )
    compact_loaded = load_prediction_npz(
        compact,
        known_sequence_ids=("test_a", "test_b"),
        expected_test_counts={"test_a": 2, "test_b": 2},
    )
    assert compact_loaded[0].variant_id == "vjepa_final"
    assert compact_loaded[0].seed == 2
    assert compact_loaded[0].audit_scope == "compact_diagnostics_only"


def test_outputs_include_json_csv_and_self_contained_html(tmp_path: Path) -> None:
    manifest, parent = _write_manifest_fixture(tmp_path)
    calibration = audit_tum_manifest(manifest, dataset_parent=parent)
    target = np.ones((1, 12, 12), dtype=np.float64)
    prediction_set = DepthPredictionSet(
        variant_id="fixture",
        seed=None,
        prediction_m=target * 0.5,
        target_m=target,
        sample_ids=("sequence_a_000",),
        sequence_ids=("sequence_a",),
        selection_labels=("fixture",),
        audit_scope="compact_diagnostics_only",
        source_path="fixture.npz",
    )
    oracle = run_scale_oracle_audit(prediction_set, spatial_grid_size=(2, 2))
    report = assemble_phase2d_report(
        manifest_path=manifest,
        calibration_audit=calibration,
        oracle_audits=[oracle],
        intrinsics_controls={"status": "fixture"},
        prediction_paths=[],
    )
    paths = write_phase2d_outputs(report, tmp_path / "report")
    assert set(paths) == {"json", "oracle_csv", "calibration_csv", "html"}
    html_text = Path(paths["html"]).read_text()
    assert "Phase 2d calibration + scale-oracle audit" in html_text
    assert "Low-resolution spatial scale factors" in html_text
    assert "<script src=" not in html_text
    assert "compact_diagnostics_only" in html_text
    assert json.loads(Path(paths["json"]).read_text())["diagnostic_only"] is True
