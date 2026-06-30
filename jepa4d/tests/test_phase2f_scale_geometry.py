from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest
import torch
import torch.nn.functional as F

from jepa4d.data.camera_geometry import resize_intrinsics
from jepa4d.models.geometry_student import geometry_probe_loss
from jepa4d.models.phase2f_scale_geometry import (
    DEFAULT_PHASE2F_ARMS,
    Phase2fGeometryConfig,
    Phase2fScaleGeometryProbe,
    canonical_camera_features,
)
from jepa4d.training.phase2f_losses import (
    Phase2fLossConfig,
    paired_view_scale_consistency,
    phase2f_loss,
    robust_optimal_log_scale_target,
    valid_depth_mask,
)
from jepa4d.training.phase2f_training import audit_gradient_firewall, phase2f_arm_configs

_T = TypeVar("_T")


def _features(*, requires_grad: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(12)
    return torch.randn(2, 8, 6, 7, generator=generator, requires_grad=requires_grad)


def _target() -> torch.Tensor:
    y, x = torch.meshgrid(torch.linspace(-0.3, 0.4, 6), torch.linspace(-0.2, 0.5, 7), indexing="ij")
    first = (0.5 * x + 0.25 * y + 0.8).exp()
    second = (-0.2 * x + 0.35 * y + 1.1).exp()
    return torch.stack((first, second))


def _intrinsics() -> torch.Tensor:
    return torch.tensor(
        [
            [[500.0, 0.0, 319.5], [0.0, 510.0, 239.5], [0.0, 0.0, 1.0]],
            [[620.0, 0.0, 300.0], [0.0, 600.0, 225.0], [0.0, 0.0, 1.0]],
        ]
    )


def _model(arm: str) -> Phase2fScaleGeometryProbe:
    return Phase2fScaleGeometryProbe(
        Phase2fGeometryConfig(
            input_dim=8,
            arm=arm,  # type: ignore[arg-type]
            hidden_dim=16,
            group_norm_groups=4,
            scale_feature_dim=4,
            scale_hidden_dim=8,
            camera_prompt_dim=4,
            coarse_field_size=(3, 4),
            maximum_scale_field_amplitude=0.05,
        )
    )


def test_default_arm_registry_excludes_optional_m4_and_exact_counts_cover_each_model() -> None:
    default = phase2f_arm_configs(768)
    optional = phase2f_arm_configs(768, include_optional_m4=True)
    assert tuple(default) == DEFAULT_PHASE2F_ARMS == ("M0", "M1", "M2", "M3")
    assert tuple(optional) == (*DEFAULT_PHASE2F_ARMS, "M4")

    counts = {arm: Phase2fScaleGeometryProbe(config).parameter_counts() for arm, config in optional.items()}
    assert {arm: values["total"] for arm, values in counts.items()} == {
        "M0": 86_402,
        "M1": 92_820,
        "M2": 92_916,
        "M3": 93_685,
        "M4": 93_052,
    }
    assert default["M3"].coarse_field_size == (4, 4)
    for arm, values in counts.items():
        assert values["total"] == sum(value for key, value in values.items() if key != "total")
        assert values["total"] <= 95_042, arm
    assert counts["M1"]["total"] / counts["M0"]["total"] == pytest.approx(1.0742806879470383)
    assert counts["M2"]["total"] / counts["M0"]["total"] == pytest.approx(1.07539177333858)
    assert counts["M3"]["total"] / counts["M0"]["total"] == pytest.approx(1.0842920302770769)
    assert counts["M4"]["camera_prompt"] > 0
    assert counts["M3"]["coarse_scale_field"] == 769


def test_m0_is_the_existing_monolithic_loss_contract() -> None:
    model = _model("M0")
    output = model(_features())
    assert output.centered_shape is None
    assert output.global_log_scale is None
    assert output.shape_log_variance is None
    assert output.global_scale_log_variance is None
    assert output.scale_field is None

    target = _target()
    valid = valid_depth_mask(target)
    expected, expected_parts = geometry_probe_loss(output.log_depth, output.log_variance, target, valid)
    actual = phase2f_loss(output, target)
    assert torch.equal(actual.total, expected)
    assert actual.optimal_log_scale_target is None
    for name, value in expected_parts.items():
        assert torch.equal(actual.components[f"monolithic_{name}"], value)


def test_m1_obeys_exact_factorization_and_independent_uncertainty_composition() -> None:
    model = _model("M1")
    output = model(_features())
    assert output.centered_shape is not None
    assert output.global_log_scale is not None
    assert output.shape_log_variance is not None
    assert output.global_scale_log_variance is not None
    assert torch.allclose(output.centered_shape.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-7)
    assert torch.equal(output.log_depth, output.centered_shape + output.global_log_scale)
    expected_variance = torch.logaddexp(
        output.shape_log_variance,
        output.global_scale_log_variance.expand_as(output.shape_log_variance),
    )
    assert torch.equal(output.log_variance, expected_variance)


def test_robust_scale_target_is_l1_optimal_and_stops_shape_gradients() -> None:
    target_log = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 100.0]]])
    shape = torch.zeros_like(target_log, requires_grad=True)
    target = robust_optimal_log_scale_target(target_log, shape, torch.ones_like(target_log, dtype=torch.bool))
    assert torch.equal(target, torch.tensor([2.0]))
    assert not target.requires_grad

    shifted_shape = torch.tensor([[[0.0, 0.0, 0.5, 0.0, 0.0]]], requires_grad=True)
    shifted = robust_optimal_log_scale_target(
        target_log,
        shifted_shape,
        torch.ones_like(target_log, dtype=torch.bool),
    )
    assert torch.equal(shifted, torch.tensor([1.5]))


def test_frozen_loss_defaults_and_paired_view_grouping_are_exact() -> None:
    config = Phase2fLossConfig()
    assert config.smooth_l1_beta == 0.1
    assert config.centered_shape_weight == 1.0
    assert config.shape_gradient_weight == 0.25
    assert config.shape_nll_weight == 0.10
    assert config.global_scale_weight == 1.0
    assert config.scale_nll_weight == 0.10
    assert config.paired_scale_consistency_weight == 0.10
    assert config.scale_field_fit_weight == 0.25
    assert config.scale_field_tv_weight == 0.01

    scales = torch.tensor([0.0, 1.0, 2.0, 4.0]).view(4, 1, 1)
    actual = paired_view_scale_consistency(scales, group_count=2, views=2)
    expected = F.smooth_l1_loss(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 4.0]), beta=0.1)
    assert torch.equal(actual, expected)
    assert paired_view_scale_consistency(scales, group_count=4, views=1) == 0
    with pytest.raises(ValueError, match="does not match"):
        paired_view_scale_consistency(scales, group_count=3, views=2)


def test_centered_shape_target_uses_valid_median_and_exact_weighted_objective() -> None:
    model = _model("M1")
    output = model(_features())
    target = _target()
    result = phase2f_loss(output, target)
    assert output.centered_shape is not None
    assert output.shape_log_variance is not None
    target_log = target.log()
    medians = torch.stack([sample.median() for sample in target_log])
    target_centered = target_log - medians[:, None, None]
    residual = output.centered_shape - target_centered
    shape_l1 = F.smooth_l1_loss(output.centered_shape, target_centered, beta=0.1)
    horizontal = F.smooth_l1_loss(
        torch.diff(output.centered_shape, dim=-1),
        torch.diff(target_centered, dim=-1),
        beta=0.1,
    )
    vertical = F.smooth_l1_loss(
        torch.diff(output.centered_shape, dim=-2),
        torch.diff(target_centered, dim=-2),
        beta=0.1,
    )
    shape_gradient = torch.stack((horizontal, vertical)).mean()
    shape_nll = 0.5 * (torch.exp(-output.shape_log_variance) * residual.square() + output.shape_log_variance).mean()
    expected = shape_l1 + 0.25 * shape_gradient + 0.10 * shape_nll
    assert torch.allclose(result.shape_objective, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("arm", ["M1", "M2", "M3", "M4"])
def test_factorized_objectives_enforce_the_gradient_firewall(arm: str) -> None:
    model = _model(arm)
    features = _features(requires_grad=True)
    camera = {} if arm == "M1" else {"intrinsics": _intrinsics(), "intrinsics_image_size": (480, 640)}
    output = model(features, **camera)
    loss = phase2f_loss(output, _target())
    report = audit_gradient_firewall(model, loss)
    assert report.passed
    assert report.tolerance == 0.0
    assert report.maximum_forbidden_norm == 0.0
    assert report.norms["shape"]["shape"] > 0
    assert report.norms["scale"]["scale"] > 0
    assert report.norms["shape"]["scale"] == 0
    assert report.norms["scale"]["shape"] == 0
    if arm == "M3":
        assert report.norms["field"]["field"] > 0
        assert report.norms["field"]["shape"] == 0
        assert report.norms["field"]["scale"] == 0

    scale_feature_gradient = torch.autograd.grad(
        loss.scale_objective,
        features,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert scale_feature_gradient is None or torch.count_nonzero(scale_feature_gradient) == 0


def test_canonical_camera_features_are_resize_invariant_and_camera_arms_are_strict() -> None:
    intrinsics = _intrinsics()
    resized = resize_intrinsics(intrinsics, (480, 640), (240, 320))
    original_features = canonical_camera_features(intrinsics, (480, 640))
    resized_features = canonical_camera_features(resized, (240, 320))
    assert torch.allclose(original_features, resized_features, rtol=1e-6, atol=1e-6)

    m2 = _model("M2")
    with pytest.raises(ValueError, match="requires intrinsics"):
        m2(_features())
    output = m2(_features(), intrinsics=intrinsics, intrinsics_image_size=(480, 640))
    assert torch.allclose(output.canonical_camera_features, original_features)

    m1 = _model("M1")
    with pytest.raises(ValueError, match="does not consume"):
        m1(_features(), intrinsics=intrinsics, intrinsics_image_size=(480, 640))


def test_m3_scale_field_is_zero_mean_bounded_and_regularized() -> None:
    model = _model("M3")
    output = model(_features(), intrinsics=_intrinsics(), intrinsics_image_size=(480, 640))
    assert output.coarse_scale_field is not None
    assert output.scale_field is not None
    assert output.centered_shape is not None
    assert output.global_log_scale is not None
    assert output.coarse_scale_field.shape == (2, 3, 4)
    assert output.scale_field.shape == (2, 6, 7)
    assert torch.allclose(output.coarse_scale_field.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-8)
    assert torch.allclose(output.scale_field.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-8)
    assert float(output.coarse_scale_field.abs().max()) <= 0.05 + 1e-7
    assert float(output.scale_field.abs().max()) <= 0.05 + 1e-7
    assert torch.equal(
        output.log_depth,
        output.centered_shape + output.global_log_scale + output.scale_field,
    )

    loss = phase2f_loss(output, _target())
    assert torch.isfinite(loss.total)
    assert loss.components["scale_field_fit"] > 0
    assert loss.components["scale_field_tv"] >= 0
    assert loss.components["scale_field_zero_mean_error"] < 1e-8
    assert loss.components["scale_field_max_abs"] <= 0.05 + 1e-7
    assert "scale_field_amplitude" not in loss.components


class _RecordingHook:
    def __init__(self) -> None:
        self.names: list[str] = []

    def __call__(self, name: str, operation: Callable[[], _T]) -> _T:
        self.names.append(name)
        return operation()


def test_component_hooks_have_stable_semantic_boundaries() -> None:
    hook = _RecordingHook()
    model = _model("M3")
    model(
        _features(),
        intrinsics=_intrinsics(),
        intrinsics_image_size=(480, 640),
        timing_hook=hook,
    )
    assert hook.names == [
        "camera_transform",
        "dense_shape_decoder",
        "pooling",
        "scale_head",
        "coarse_scale_field",
        "composition",
    ]


def test_loss_configuration_rejects_disabled_core_uncertainty_terms() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Phase2fLossConfig(centered_shape_weight=0)
