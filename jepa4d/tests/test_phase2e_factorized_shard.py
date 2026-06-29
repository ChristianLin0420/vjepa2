from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from jepa4d.models.factorized_geometry import FactorizedGeometryOutput
from jepa4d.models.geometry_student import geometry_probe_loss
from scripts.run_phase2e_factorized_shard import (
    CACHE_SCHEMA,
    DEFAULT_VARIANTS,
    load_feature_cache,
    phase2e_loss,
    run_training_shard,
    snapshot_gpu_telemetry,
    upload_wandb_artifact,
    validate_shard_artifacts,
    variant_spec,
)


def test_gpu_telemetry_snapshot_is_numeric_and_hash_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_root = tmp_path / "logs"
    output = tmp_path / "output"
    log_root.mkdir()
    output.mkdir()
    (log_root / "gpu-telemetry.csv").write_text(
        "timestamp, index, uuid, name, pstate, temperature.gpu, utilization.gpu [%], "
        "utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], "
        "clocks.current.sm [MHz]\n"
        "2026/06/29 00:00:00, 0, GPU-test, NVIDIA A100, P0, 40, 75 %, 12 %, 2048 MiB, "
        "81920 MiB, 250 W, 1410 MHz\n"
    )
    monkeypatch.setenv("JEPA4D_JOB_LOG_DIR", str(log_root))
    summary, rows = snapshot_gpu_telemetry(output)
    assert summary["available"] is True
    assert summary["statistics"]["utilization_gpu"]["mean"] == 75.0
    assert summary["statistics"]["memory_used_mib"]["max"] == 2048.0
    assert rows[0]["utilization_gpu_pct"] == 75.0
    assert rows[0]["power_w"] == 250.0
    assert (output / "gpu_telemetry_summary.json").is_file()


def _split(name: str, count: int, *, paired: bool, teacher: bool) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(13 if name == "train" else 17)
    prefix = (count, 2) if paired else (count,)
    features = torch.randn(*prefix, 768, 24, 24, generator=generator, dtype=torch.float16)
    rgb = torch.rand(*prefix, 3, 16, 16, generator=generator)
    targets = 0.75 + 2.0 * torch.rand(*prefix, 24, 24, generator=generator)
    intrinsics = torch.zeros(*prefix, 3, 3)
    intrinsics[..., 0, 0] = 320.0
    intrinsics[..., 1, 1] = 315.0
    intrinsics[..., 0, 2] = 191.5
    intrinsics[..., 1, 2] = 191.5
    intrinsics[..., 2, 2] = 1.0
    value: dict[str, Any] = {
        "features": features,
        "rgb": rgb,
        "intrinsics_384": intrinsics,
        "targets": targets,
        "sample_ids": [f"{name}-{index}" for index in range(count)],
        "sensor_ids": [f"sensor-{index % 2}" for index in range(count)],
    }
    if teacher:
        log_target = targets.log()
        value["teacher_centered_shape"] = log_target - log_target.mean(dim=(-2, -1), keepdim=True)
    return value


def _cache(path: Path, *, paired: bool = True, teacher: bool = True, include_test: bool = False) -> Path:
    splits = {
        "train": _split("train", 2, paired=paired, teacher=teacher),
        "validation": _split("validation", 2, paired=False, teacher=False),
    }
    if include_test:
        splits["test"] = _split("forbidden", 1, paired=False, teacher=False)
    torch.save({"schema_version": CACHE_SCHEMA, "splits": splits}, path)
    return path


@pytest.fixture
def one_cpu_thread() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_cache_loader_accepts_paired_train_and_unpaired_validation(tmp_path: Path) -> None:
    loaded = load_feature_cache(_cache(tmp_path / "cache.pt"))
    assert loaded.train.paired
    assert loaded.train.views == 2
    assert loaded.train.features.shape == (2, 2, 768, 24, 24)
    assert not loaded.validation.paired
    assert loaded.validation.features.shape == (2, 768, 24, 24)
    assert loaded.train.teacher_centered_shape is not None
    assert len(loaded.sha256) == 64


def test_cache_loader_rejects_any_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no test split"):
        load_feature_cache(_cache(tmp_path / "forbidden.pt", include_test=True))


def test_named_variant_matrix_is_exact_and_small() -> None:
    assert len(DEFAULT_VARIANTS) == 8
    specs = {name: variant_spec(name, hidden_dim=8) for name in DEFAULT_VARIANTS}
    assert specs["monolithic_final"].config.mode == "monolithic"
    assert specs["factorized_bias"].config.scale_inputs == ()
    assert specs["factorized_vjepa"].config.scale_inputs == ("vjepa",)
    assert specs["factorized_rgb"].config.scale_inputs == ("rgb",)
    assert specs["factorized_vjepa_rgb"].config.scale_inputs == ("vjepa", "rgb")
    assert specs["factorized_vjepa_k"].config.camera_mode == "known_rays"
    assert specs["factorized_full"].config.scale_inputs == ("vjepa", "rgb", "intrinsics", "ray_summary")
    assert specs["factorized_full_teacher"].use_teacher
    with pytest.raises(ValueError, match="unknown Phase2e variant"):
        variant_spec("not-a-variant")


def test_phase2e_loss_contains_scale_shape_teacher_and_pair_consistency() -> None:
    targets = torch.ones(4, 24, 24)
    scales = torch.tensor([0.0, 0.5, 0.0, -0.5]).view(4, 1, 1)
    centered = torch.zeros_like(targets, requires_grad=True)
    output = FactorizedGeometryOutput(
        log_depth=centered + scales,
        log_variance=torch.zeros_like(targets, requires_grad=True),
        centered_shape=centered,
        global_log_scale=scales.requires_grad_(),
        effective_intrinsics=None,
        camera_rays=None,
    )
    teacher = torch.linspace(-0.3, 0.3, 24).view(1, 1, 24).expand_as(targets)
    total, parts = phase2e_loss(
        output,
        targets,
        teacher_centered_shape=teacher,
        use_teacher=True,
        group_count=2,
        views=2,
    )
    assert torch.isfinite(total)
    assert parts["global_log_scale"] > 0
    assert parts["paired_scale_consistency"] > 0
    assert "centered_gt_shape" in parts
    assert parts["centered_teacher"] > 0
    total.backward()
    assert centered.grad is not None


def test_monolithic_phase2e_loss_is_exactly_the_comparable_base_geometry_loss() -> None:
    targets = 0.5 + torch.rand(2, 24, 24)
    log_depth = torch.randn(2, 24, 24, requires_grad=True)
    log_variance = torch.randn(2, 24, 24, requires_grad=True)
    output = FactorizedGeometryOutput(
        log_depth=log_depth,
        log_variance=log_variance,
        centered_shape=None,
        global_log_scale=None,
        effective_intrinsics=None,
        camera_rays=None,
    )
    valid = torch.isfinite(targets) & (targets > 0.1) & (targets < 10.0)
    expected, _ = geometry_probe_loss(log_depth, log_variance, targets, valid)
    actual, parts = phase2e_loss(
        output,
        targets,
        teacher_centered_shape=None,
        use_teacher=False,
        group_count=1,
        views=2,
    )
    assert torch.equal(actual, expected)
    for key in ("global_log_scale", "centered_gt_shape", "centered_teacher", "paired_scale_consistency"):
        assert float(parts[key]) == 0.0


def test_training_shard_runs_strict_reload_and_writes_self_contained_evidence(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    cache = _cache(tmp_path / "cache.pt")
    output = tmp_path / "output"
    shard = run_training_shard(
        cache,
        output,
        ("monolithic_final", "factorized_full_teacher"),
        (0,),
        epochs=2,
        batch_size=2,
        hidden_dim=8,
        device_name="cpu",
        wandb_enabled=False,
        run_name="synthetic-phase2e",
    )
    assert shard["status"] == "success"
    assert shard["selection_split"] == "validation"
    assert len(shard["config_sha256"]) == 64
    assert len(shard["resolved_config_file_sha256"]) == 64
    assert len(shard["results"]) == 2
    for result in shard["results"]:
        assert result["checkpoint_reload"] == "strict-prediction-equality-pass"
        history = [json.loads(line) for line in Path(result["history"]).read_text().splitlines()]
        assert len(history) == 2
        assert {
            "validation_metric_abs_rel",
            "validation_aligned_abs_rel",
            "validation_abs_log_scale_error",
            "validation_log_depth_nll",
        } <= set(history[0])
        prediction = torch.load(result["validation_predictions"], map_location="cpu", weights_only=True)
        assert prediction["schema_version"] == "jepa4d-phase2e-validation-predictions-v1"
        assert prediction["prediction_m"].shape == (2, 24, 24)
        assert Path(result["checkpoint"]).is_file()
        checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
        assert checkpoint["run_config_sha256"] == shard["config_sha256"]
    resolved = json.loads((output / "resolved_config.json").read_text())
    assert set(resolved["data_splits"]) == {"train", "validation"}
    assert resolved["checkpoint_selection"] == "minimum validation raw metric_abs_rel only"
    report = (output / "phase2e_report.html").read_text()
    assert "data:image/png;base64," in report
    assert "<svg" in report
    assert "<script" not in report
    assert "Fixed first-validation-sample diagnostics" in report

    manifest = validate_shard_artifacts(output, shard)
    assert manifest["selection_split"] == "validation"
    assert {item["role"].split(":", maxsplit=1)[0] for item in manifest["files"]} >= {
        "shard",
        "resolved_config",
        "html_report",
        "checkpoint",
        "history",
        "validation_predictions",
        "validation_metrics_path",
    }
    assert json.loads((output / "artifact_manifest.json").read_text()) == manifest

    class FakeArtifact:
        def __init__(self, *, name: str, type: str, metadata: dict[str, Any]) -> None:
            self.name = name
            self.type = type
            self.metadata = metadata
            self.added_directory: tuple[str, str] | None = None

        def add_dir(self, path: str, *, name: str) -> None:
            self.added_directory = (path, name)

    class FakeUploadedArtifact:
        id = "artifact-id-123"
        name = "run-id-123-phase2e-factorized-shard"
        qualified_name = "entity/project/run-id-123-phase2e-factorized-shard:v0"
        version = "v0"
        digest = "artifact-digest-123"

        def __init__(self) -> None:
            self.wait_timeout: int | None = None

        def wait(self, *, timeout: int) -> FakeUploadedArtifact:
            self.wait_timeout = timeout
            return self

    class FakeRun:
        id = "run-id-123"
        url = "https://wandb.invalid/run-id-123"
        path = "entity/project/run-id-123"

        def __init__(self) -> None:
            self.logged_artifact: FakeArtifact | None = None
            self.uploaded = FakeUploadedArtifact()

        def log_artifact(self, artifact: FakeArtifact) -> FakeUploadedArtifact:
            self.logged_artifact = artifact
            return self.uploaded

    class FakeWandb:
        Artifact = FakeArtifact

    fake_run = FakeRun()
    receipt = upload_wandb_artifact(fake_run, output, shard, wandb_module=FakeWandb)
    assert fake_run.uploaded.wait_timeout == 900
    assert fake_run.logged_artifact is not None
    assert fake_run.logged_artifact.added_directory == (str(output.resolve()), "phase2e")
    assert receipt["run_id"] == "run-id-123"
    assert receipt["artifact_id"] == "artifact-id-123"
    assert json.loads((output / "wandb_receipt.json").read_text()) == receipt

    (output / "phase2e_report.html").write_text(report + "tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_shard_artifacts(output, shard)
    with pytest.raises(ValueError, match="new and empty"):
        run_training_shard(
            cache,
            output,
            ("factorized_bias",),
            (0,),
            epochs=1,
            hidden_dim=8,
            device_name="cpu",
            wandb_enabled=False,
        )


def test_teacher_variant_requires_teacher_cache_before_output_creation(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache.pt", teacher=False)
    output = tmp_path / "unused-output"
    with pytest.raises(ValueError, match="teacher_centered_shape is absent"):
        run_training_shard(
            cache,
            output,
            ("factorized_full_teacher",),
            (0,),
            epochs=1,
            hidden_dim=8,
            device_name="cpu",
            wandb_enabled=False,
        )
    assert not output.exists()
