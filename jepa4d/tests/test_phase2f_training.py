from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from jepa4d.models.phase2f_scale_geometry import Phase2fGeometryConfig, Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_training import (
    PHASE2F_CHECKPOINT_SCHEMA,
    assert_strict_phase2f_reload,
    load_phase2f_checkpoint,
    profile_phase2f_latency,
    save_phase2f_checkpoint,
    train_phase2f_step,
)


@pytest.fixture
def one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _model(arm: str = "M3") -> Phase2fScaleGeometryProbe:
    return Phase2fScaleGeometryProbe(
        Phase2fGeometryConfig(
            input_dim=8,
            arm=arm,  # type: ignore[arg-type]
            hidden_dim=16,
            group_norm_groups=4,
            scale_feature_dim=4,
            scale_hidden_dim=8,
            coarse_field_size=(3, 3),
            maximum_scale_field_amplitude=0.1,
        )
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(44)
    features = torch.randn(3, 8, 6, 6, generator=generator)
    target = torch.rand(3, 6, 6, generator=generator) * 2.0 + 0.4
    intrinsics = torch.tensor(
        [
            [[400.0, 0.0, 300.0], [0.0, 420.0, 220.0], [0.0, 0.0, 1.0]],
            [[520.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            [[650.0, 0.0, 280.0], [0.0, 630.0, 210.0], [0.0, 0.0, 1.0]],
        ]
    )
    return features, target, intrinsics


def _snapshots(model: Phase2fScaleGeometryProbe) -> dict[str, list[torch.Tensor]]:
    return {
        "shape": [parameter.detach().clone() for parameter in model.shape_parameters()],
        "scale": [parameter.detach().clone() for parameter in model.scale_parameters()],
        "field": [parameter.detach().clone() for parameter in model.field_parameters()],
    }


def _group_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_training_step_updates_owned_groups_and_logs_firewall_metrics(one_cpu_thread: None) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features, target, intrinsics = _batch()
    before = _snapshots(model)
    result = train_phase2f_step(
        model,
        optimizer,
        features,
        target,
        intrinsics=intrinsics,
        intrinsics_image_size=(480, 640),
    )
    after = _snapshots(model)

    assert result.firewall is not None and result.firewall.passed
    assert result.metrics["gradient_firewall_max_forbidden_norm"] == 0.0
    assert result.metrics["gradient_firewall_shape_to_scale"] == 0.0
    assert result.metrics["gradient_firewall_scale_to_shape"] == 0.0
    assert result.metrics["gradient_firewall_field_to_shape"] == 0.0
    assert result.metrics["gradient_firewall_shape_to_shape"] > 0
    assert result.metrics["gradient_firewall_scale_to_scale"] > 0
    assert result.metrics["gradient_firewall_field_to_field"] > 0
    assert result.metrics["gradient_norm_shape"] > 0
    assert result.metrics["gradient_norm_scale"] > 0
    assert result.metrics["gradient_norm_field"] > 0
    assert result.metrics["gradient_norm_total_before_clip"] > 0
    assert result.metrics["scale_field_fit"] > 0
    assert result.parameter_counts["total"] == model.trainable_parameter_count
    assert _group_changed(before["shape"], after["shape"])
    assert _group_changed(before["scale"], after["scale"])
    assert _group_changed(before["field"], after["field"])


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "M3"])
def test_training_step_is_finite_for_every_default_arm(arm: str, one_cpu_thread: None) -> None:
    model = _model(arm)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    features, target, intrinsics = _batch()
    camera = {} if arm in {"M0", "M1"} else {"intrinsics": intrinsics, "intrinsics_image_size": (480, 640)}
    result = train_phase2f_step(model, optimizer, features, target, **camera)
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())
    assert result.firewall is not None and result.firewall.passed


def test_training_step_accepts_source_grouped_paired_views(one_cpu_thread: None) -> None:
    model = _model("M1")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    features, target, _ = _batch()
    paired_features = torch.stack((features[:2], features[:2] + 0.1), dim=1).flatten(0, 1)
    paired_target = torch.stack((target[:2], target[:2] * 1.05), dim=1).flatten(0, 1)
    result = train_phase2f_step(
        model,
        optimizer,
        paired_features,
        paired_target,
        group_count=2,
        views=2,
    )
    assert result.metrics["paired_scale_consistency"] >= 0
    assert result.firewall is not None and result.firewall.passed


def test_latency_report_separates_components_and_exact_parameters(one_cpu_thread: None) -> None:
    model = _model("M2")
    model.train()
    features, _, intrinsics = _batch()
    report = profile_phase2f_latency(
        model,
        features[:1],
        intrinsics=intrinsics[:1],
        intrinsics_image_size=(480, 640),
        warmups=1,
        repeats=2,
    )
    assert model.training
    assert report.arm == "M2"
    assert report.warmups == 1
    assert report.repeats == 2
    assert report.parameter_counts == model.parameter_counts()
    assert report.end_to_end_ms["mean"] > 0
    for name in ("camera_transform", "dense_shape_decoder", "pooling", "scale_head", "composition"):
        assert report.component_ms[name]["count"] == 2
        assert report.component_ms[name]["mean"] is not None
    assert report.component_ms["ray_construction"]["count"] == 0
    assert report.component_ms["coarse_scale_field"]["count"] == 0


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "M3"])
def test_strict_checkpoint_reload_preserves_every_output_factor(
    arm: str,
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    model = _model(arm)
    features, _, intrinsics = _batch()
    camera = {} if arm in {"M0", "M1"} else {"intrinsics": intrinsics, "intrinsics_image_size": (480, 640)}
    path = save_phase2f_checkpoint(model, tmp_path / f"{arm}.pt")
    reloaded, payload = load_phase2f_checkpoint(path)
    assert payload["schema_version"] == PHASE2F_CHECKPOINT_SCHEMA
    assert payload["parameter_counts"] == model.parameter_counts()
    assert_strict_phase2f_reload(model, reloaded, features, **camera)


def test_latency_profile_validates_repetition_counts(one_cpu_thread: None) -> None:
    model = _model("M0")
    features, _, _ = _batch()
    with pytest.raises(ValueError, match="repeats"):
        profile_phase2f_latency(model, features[:1], repeats=0)
