from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from jepa4d.models.geometry_student import DenseGeometryProbe, geometry_probe_loss
from scripts import run_phase2b_geometry_distillation as phase2b
from slurm.phase2b_preflight import _assert_close


def _sample(root: Path, stem: str = "0000") -> SimpleNamespace:
    rgb = root / "rgb" / f"{stem}.png"
    depth = root / "depth" / f"{stem}.png"
    rgb.parent.mkdir(parents=True, exist_ok=True)
    depth.parent.mkdir(parents=True, exist_ok=True)
    columns = np.tile(np.arange(6, dtype=np.uint8), (4, 1))
    Image.fromarray(np.stack((columns, columns, columns), axis=-1)).save(rgb)
    Image.fromarray(columns.astype(np.uint16) * 1000 + 1000).save(depth)
    return SimpleNamespace(rgb_path=rgb, depth_path=depth, sample_id=stem, timestamp=float(int(stem)))


def test_shared_crop_and_single_image_batch_keep_batch_axis(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "0000"), _sample(tmp_path, "0001")]
    images = phase2b._images(samples, size=4)
    depths = phase2b._targets(samples, (4, 4))
    # A 4x6 input must drop columns 0 and 5 for both modalities.
    assert torch.allclose(images[0, 0, 0] * 255, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(depths[0, 0], torch.tensor([0.4, 0.6, 0.8, 1.0]))
    batch = phase2b._single_image_batch(samples)
    assert batch.images.shape == (2, 1, 1, 3, 384, 384)
    assert batch.valid_mask.all()


def test_training_fitted_scale_is_independent_of_test_targets() -> None:
    train_prediction = torch.full((2, 12, 12), 2.0)
    train_target = torch.full((2, 12, 12), 6.0)
    frozen_scale = phase2b._fit_metric_scale(train_prediction, train_target)
    test_prediction = torch.full((1, 12, 12), 1.0)
    scaled_before_seeing_test = test_prediction * frozen_scale
    first_test_target = torch.full((1, 12, 12), 3.0)
    second_test_target = torch.full((1, 12, 12), 9.0)
    assert frozen_scale == 3.0
    assert torch.equal(scaled_before_seeing_test, torch.full_like(test_prediction, 3.0))
    assert phase2b._evaluate_depths(scaled_before_seeing_test, first_test_target)["metric_abs_rel"] == 0.0
    assert phase2b._evaluate_depths(scaled_before_seeing_test, second_test_target)["metric_abs_rel"] > 0.6


def test_metrics_reject_invalid_predictions_on_valid_target_pixels() -> None:
    target = torch.ones(1, 12, 12)
    prediction = torch.ones_like(target)
    prediction[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        phase2b._evaluate_depths(prediction, target)
    prediction[0, 0, 0] = 0
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        phase2b._evaluate_depths(prediction, target)


def test_multilayer_average_is_parameter_matched_and_train_normalized() -> None:
    train = {f"vjepa_layer_{key}": torch.randn(3, 8, 4, 4) + key for key in (2, 5, 8, 11)}
    validation = {key: value[:2] + 100 for key, value in train.items()}
    test = {key: value[:1] - 100 for key, value in train.items()}
    normalized = phase2b._normalize_multilayer(train, validation, test)
    changed_validation = {key: value + 10_000 for key, value in validation.items()}
    changed = phase2b._normalize_multilayer(train, changed_validation, test)
    assert normalized[0].shape == (3, 8, 4, 4)
    assert set(normalized[3]) == set(train)
    for key in train:
        assert torch.equal(normalized[3][key]["mean"], changed[3][key]["mean"])
        assert torch.equal(normalized[3][key]["std"], changed[3][key]["std"])
    assert abs(float(normalized[0].float().mean())) < 1e-3
    final_probe = DenseGeometryProbe(8)
    multilayer_probe = DenseGeometryProbe(normalized[0].shape[1])
    assert sum(value.numel() for value in final_probe.parameters()) == sum(
        value.numel() for value in multilayer_probe.parameters()
    )


def test_geometry_loss_rejects_empty_or_invalid_teacher_pixels() -> None:
    prediction = torch.zeros(1, 4, 4)
    logvar = torch.zeros_like(prediction)
    target = torch.ones_like(prediction)
    with pytest.raises(ValueError, match="no finite valid pixels"):
        geometry_probe_loss(prediction, logvar, target, torch.zeros_like(target, dtype=torch.bool))
    teacher = torch.ones_like(target)
    teacher[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="teacher depth"):
        geometry_probe_loss(
            prediction, logvar, target, torch.ones_like(target, dtype=torch.bool), teacher_depth=teacher
        )
    bad_prediction = prediction.clone()
    bad_prediction[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="predicted log depth"):
        geometry_probe_loss(bad_prediction, logvar, target, torch.ones_like(target, dtype=torch.bool))


def test_dataset_root_must_match_verified_archive_members(tmp_path: Path) -> None:
    root = tmp_path / "rgbd_dataset_freiburg1_xyz"
    sample = _sample(root)
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        (root / name).write_text(f"fixture {name}\n")
    archive = tmp_path / "dataset.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname=root.name)
    splits = {"train": [sample], "validation": [], "test": []}
    record = phase2b._dataset_fingerprint(root, splits, archive)
    assert record["archive_extraction_verified"] is True
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(sample.rgb_path)
    with pytest.raises(ValueError, match="does not match verified archive"):
        phase2b._dataset_fingerprint(root, splits, archive)


def test_atomic_json_rejects_nan(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    with pytest.raises(ValueError):
        phase2b._write_json(destination, {"metric": float("nan")})
    assert not destination.exists()


def test_environment_snapshot_is_json_serializable_on_current_device() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    snapshot = phase2b._environment_snapshot(device)
    json.dumps(snapshot, allow_nan=False)
    if torch.cuda.is_available():
        assert isinstance(snapshot["gpu"]["uuid"], str)


def test_runner_records_failures_during_pre_wandb_initialization(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = tmp_path / "failed-run"
    with pytest.raises(FileNotFoundError):
        phase2b.main(
            dataset_root=tmp_path / "dataset",
            archive=tmp_path / "missing.tgz",
            output=output,
            manifest_path=tmp_path / "manifest.yaml",
            vjepa_checkpoint=tmp_path / "vjepa",
            vjepa_implementation=tmp_path / "implementation",
            vggt_checkpoint=tmp_path / "vggt",
            device="cuda:0",
            epochs=1,
            wandb_enabled=False,
            wandb_project="test",
            wandb_entity=None,
            run_name="test",
        )
    failure = json.loads((output / "run_failure.json").read_text())
    assert failure["error"].startswith("FileNotFoundError:")
    assert '"stage": "run_failed"' in (output / "events.jsonl").read_text()
    assert "run_failure.json" in json.loads((output / "artifact_manifest.json").read_text())


def test_reported_preprocessing_policy_is_json_serializable(tmp_path: Path) -> None:
    destination = tmp_path / "policy.json"
    phase2b._write_json(destination, {"crop": "center-square", "rgb": "bilinear", "depth": "nearest"})
    assert json.loads(destination.read_text())["depth"] == "nearest"


def test_chunk_invariance_accepts_reduction_drift_but_rejects_content_changes() -> None:
    separate = torch.linspace(-2, 2, 4096).reshape(2, 32, 64)
    reduction_drift = separate + 0.002
    statistics = _assert_close(
        "test features",
        reduction_drift,
        separate,
        rtol=1e-2,
        atol=3e-3,
        max_outlier_fraction=1e-3,
        max_relative_rmse=5e-3,
        min_cosine_similarity=0.9999,
    )
    assert statistics["max_abs"] < 0.003
    assert statistics["cosine_similarity"] > 0.999
    sparse_outlier = separate.clone()
    sparse_outlier.view(-1)[0] += 0.1
    sparse_statistics = _assert_close(
        "test sparse outlier",
        sparse_outlier,
        separate,
        rtol=1e-2,
        atol=3e-3,
        max_outlier_fraction=1e-3,
        max_relative_rmse=5e-3,
        min_cosine_similarity=0.9999,
    )
    assert sparse_statistics["outlier_count"] == 1
    with pytest.raises(RuntimeError, match="changes with chunk size"):
        _assert_close(
            "test features",
            separate.roll(1, dims=0),
            separate,
            rtol=1e-2,
            atol=3e-3,
            max_outlier_fraction=1e-3,
            max_relative_rmse=5e-3,
            min_cosine_similarity=0.9999,
        )
