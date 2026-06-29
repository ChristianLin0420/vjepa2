from __future__ import annotations

import pytest
import torch

from jepa4d.data.camera_geometry import (
    apply_intrinsics_control,
    camera_ray_summary,
    normalized_camera_rays,
    normalized_intrinsics_summary,
    resize_intrinsics,
    update_intrinsics_for_crop_resize,
)
from jepa4d.models.factorized_geometry import (
    FactorizedGeometryConfig,
    FactorizedShapeScaleGeometryProbe,
)


def _intrinsics(fx: float = 500.0, fy: float = 510.0, cx: float = 319.5, cy: float = 239.5) -> torch.Tensor:
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_crop_resize_intrinsics_follow_half_pixel_pixel_center_mapping() -> None:
    source = _intrinsics()
    updated = update_intrinsics_for_crop_resize(
        source,
        input_size=(480, 640),
        crop=(0, 80, 480, 480),
        output_size=(384, 384),
    )
    expected = torch.tensor([[400.0, 0.0, 191.5], [0.0, 408.0, 191.5], [0.0, 0.0, 1.0]])
    assert torch.allclose(updated, expected)

    source_pixel = torch.tensor([219.5, 139.5, 1.0])
    resized_pixel = torch.tensor([0.8 * (source_pixel[0] - 80 + 0.5) - 0.5, 0.8 * (source_pixel[1] + 0.5) - 0.5, 1.0])
    source_direction = torch.linalg.solve(source, source_pixel)
    resized_direction = torch.linalg.solve(updated, resized_pixel)
    assert torch.allclose(source_direction, resized_direction, rtol=1e-6, atol=1e-6)


def test_resize_preserves_normalized_intrinsics_summary() -> None:
    source = _intrinsics().repeat(2, 1, 1)
    source[1, 0, 0] = 620.0
    resized = resize_intrinsics(source, (480, 640), (240, 320))
    assert torch.allclose(
        normalized_intrinsics_summary(source, (480, 640)),
        normalized_intrinsics_summary(resized, (240, 320)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalized_camera_rays_are_unit_length_and_use_pixel_centres() -> None:
    intrinsics = _intrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    rays = normalized_camera_rays(intrinsics, (3, 3))
    assert rays.shape == (3, 3, 3)
    assert torch.allclose(torch.linalg.vector_norm(rays, dim=0), torch.ones(3, 3), atol=1e-6)
    assert torch.allclose(rays[:, 1, 1], torch.tensor([0.0, 0.0, 1.0]))
    summary = camera_ray_summary(rays)
    assert summary.shape == (6,)
    assert torch.isfinite(summary).all()


def test_wrong_and_shuffled_intrinsics_are_explicit_deterministic_controls() -> None:
    first = _intrinsics()
    second = _intrinsics(fx=700.0, fy=710.0, cx=300.0, cy=220.0)
    batch = torch.stack((first, second))
    wrong = apply_intrinsics_control(
        batch,
        "wrong",
        wrong_focal_scale=1.5,
        wrong_principal_shift=(4.0, -3.0),
    )
    assert torch.allclose(wrong[:, 0, 0], batch[:, 0, 0] * 1.5)
    assert torch.allclose(wrong[:, 1, 1], batch[:, 1, 1] * 1.5)
    assert torch.allclose(wrong[:, 0, 2], batch[:, 0, 2] + 4.0)
    assert torch.allclose(wrong[:, 1, 2], batch[:, 1, 2] - 3.0)

    shuffled = apply_intrinsics_control(batch, "shuffled")
    assert torch.equal(shuffled[0], batch[1])
    assert torch.equal(shuffled[1], batch[0])
    explicit = apply_intrinsics_control(batch, "shuffled", permutation=torch.tensor([1, 0]))
    assert torch.equal(explicit, shuffled)
    with pytest.raises(ValueError, match="at least two"):
        apply_intrinsics_control(first, "shuffled")
    with pytest.raises(ValueError, match="every batch index"):
        apply_intrinsics_control(batch, "shuffled", permutation=torch.tensor([0, 0]))


def test_factorized_bias_only_probe_obeys_exact_shape_plus_scale_identity() -> None:
    torch.manual_seed(3)
    model = FactorizedShapeScaleGeometryProbe(
        FactorizedGeometryConfig(input_dim=8, hidden_dim=16, scale_inputs=(), group_norm_groups=4)
    )
    features = torch.randn(2, 8, 5, 7, requires_grad=True)
    output = model(features)
    assert output.log_depth.shape == output.log_variance.shape == (2, 5, 7)
    assert output.centered_shape is not None
    assert output.global_log_scale is not None
    assert output.global_log_scale.shape == (2, 1, 1)
    assert torch.allclose(output.centered_shape.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-7)
    assert torch.equal(output.log_depth, output.centered_shape + output.global_log_scale)
    assert output.effective_intrinsics is None
    assert output.camera_rays is None
    (output.log_depth.square().mean() + output.log_variance.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.ablation_signature() == "factorized-camera_none-scale_bias-K_correct"


def test_all_scale_inputs_and_known_rays_form_a_small_camera_aware_probe() -> None:
    torch.manual_seed(5)
    config = FactorizedGeometryConfig(
        input_dim=32,
        hidden_dim=32,
        camera_mode="known_rays",
        scale_inputs=("vjepa", "rgb", "intrinsics", "ray_summary"),
        group_norm_groups=8,
    )
    model = FactorizedShapeScaleGeometryProbe(config)
    features = torch.randn(2, 32, 4, 6)
    rgb = torch.rand(2, 3, 32, 48)
    intrinsics = torch.stack(
        (
            _intrinsics(fx=40.0, fy=38.0, cx=23.5, cy=15.5),
            _intrinsics(fx=32.0, fy=30.0, cx=21.5, cy=14.5),
        )
    )
    output = model(features, rgb=rgb, intrinsics=intrinsics, intrinsics_image_size=(32, 48))
    assert output.centered_shape is not None
    assert output.global_log_scale is not None
    assert output.effective_intrinsics is not None
    assert output.effective_intrinsics.shape == (2, 3, 3)
    assert output.camera_rays is not None
    assert output.camera_rays.shape == (2, 3, 4, 6)
    assert torch.isfinite(output.log_depth).all()
    assert model.scale_feature_dim == 34
    assert model.trainable_parameter_count < 50_000
    assert model.ablation_signature("wrong") == (
        "factorized-camera_known_rays-scale_vjepa+rgb+intrinsics+ray_summary-K_wrong"
    )
    output.global_log_scale.sum().backward()
    assert model.vjepa_scale_projection is not None
    assert model.rgb_scale_encoder is not None
    assert all(parameter.grad is not None for parameter in model.vjepa_scale_projection.parameters())
    assert all(parameter.grad is not None for parameter in model.rgb_scale_encoder.parameters())
    model.zero_grad(set_to_none=True)

    wrong = model(
        features,
        rgb=rgb,
        intrinsics=intrinsics,
        intrinsics_image_size=(32, 48),
        intrinsics_control="wrong",
        wrong_focal_scale=1.8,
    )
    assert wrong.camera_rays is not None
    assert not torch.allclose(output.camera_rays, wrong.camera_rays)
    assert not torch.allclose(output.log_depth, wrong.log_depth)

    shuffled = model(
        features,
        rgb=rgb,
        intrinsics=intrinsics,
        intrinsics_image_size=(32, 48),
        intrinsics_control="shuffled",
        camera_permutation=torch.tensor([1, 0]),
    )
    assert shuffled.effective_intrinsics is not None
    assert torch.equal(shuffled.effective_intrinsics[0], output.effective_intrinsics[1])
    assert torch.equal(shuffled.effective_intrinsics[1], output.effective_intrinsics[0])


@pytest.mark.parametrize(
    "scale_inputs",
    [
        (),
        ("vjepa",),
        ("rgb",),
        ("intrinsics",),
        ("ray_summary",),
        ("vjepa", "rgb"),
    ],
)
def test_scale_input_ablations_have_a_uniform_output_api(scale_inputs: tuple[str, ...]) -> None:
    torch.manual_seed(11)
    config = FactorizedGeometryConfig(
        input_dim=8,
        hidden_dim=16,
        scale_inputs=scale_inputs,  # type: ignore[arg-type]
        group_norm_groups=4,
    )
    model = FactorizedShapeScaleGeometryProbe(config)
    features = torch.randn(2, 8, 4, 4)
    kwargs: dict[str, object] = {}
    if "rgb" in scale_inputs:
        kwargs["rgb"] = torch.rand(2, 3, 16, 16)
    if "intrinsics" in scale_inputs or "ray_summary" in scale_inputs:
        kwargs["intrinsics"] = _intrinsics(fx=12.0, fy=12.0, cx=7.5, cy=7.5).repeat(2, 1, 1)
        kwargs["intrinsics_image_size"] = (16, 16)
    output = model(features, **kwargs)  # type: ignore[arg-type]
    assert output.log_depth.shape == (2, 4, 4)
    assert output.global_log_scale is not None
    assert torch.isfinite(output.global_log_scale).all()


def test_monolithic_and_camera_none_controls_are_explicit() -> None:
    config = FactorizedGeometryConfig(
        input_dim=8,
        hidden_dim=16,
        mode="monolithic",
        camera_mode="none",
        scale_inputs=(),
        group_norm_groups=4,
    )
    model = FactorizedShapeScaleGeometryProbe(config)
    output = model(torch.randn(2, 8, 4, 4))
    assert output.centered_shape is None
    assert output.global_log_scale is None
    assert output.log_depth.shape == output.log_variance.shape == (2, 4, 4)
    assert model.ablation_signature() == "monolithic-camera_none-scale_none-K_correct"
    with pytest.raises(ValueError, match="does not consume intrinsics"):
        model(torch.randn(2, 8, 4, 4), intrinsics_control="wrong")
    with pytest.raises(ValueError, match="does not use a separate scale head"):
        FactorizedGeometryConfig(input_dim=8, mode="monolithic", scale_inputs=("vjepa",))

    known_camera = FactorizedShapeScaleGeometryProbe(
        FactorizedGeometryConfig(
            input_dim=8,
            hidden_dim=16,
            mode="monolithic",
            camera_mode="known_rays",
            scale_inputs=(),
            group_norm_groups=4,
        )
    )
    known_output = known_camera(
        torch.randn(2, 8, 4, 4),
        intrinsics=_intrinsics(fx=4.0, fy=4.0, cx=1.5, cy=1.5).repeat(2, 1, 1),
    )
    assert known_output.camera_rays is not None
    assert known_output.camera_rays.shape == (2, 3, 4, 4)


def test_missing_configured_inputs_fail_loudly() -> None:
    features = torch.randn(2, 8, 4, 4)
    rgb_model = FactorizedShapeScaleGeometryProbe(
        FactorizedGeometryConfig(input_dim=8, hidden_dim=16, scale_inputs=("rgb",), group_norm_groups=4)
    )
    with pytest.raises(ValueError, match="requires rgb"):
        rgb_model(features)

    camera_model = FactorizedShapeScaleGeometryProbe(
        FactorizedGeometryConfig(
            input_dim=8,
            hidden_dim=16,
            camera_mode="known_rays",
            scale_inputs=(),
            group_norm_groups=4,
        )
    )
    with pytest.raises(ValueError, match="requires intrinsics"):
        camera_model(features)


def test_full_width_vjepa_configuration_remains_compact() -> None:
    candidate = FactorizedShapeScaleGeometryProbe(
        FactorizedGeometryConfig(
            input_dim=768,
            camera_mode="known_rays",
            scale_inputs=("vjepa", "rgb", "intrinsics", "ray_summary"),
        )
    )
    baseline = FactorizedShapeScaleGeometryProbe(
        FactorizedGeometryConfig(input_dim=768, mode="monolithic", scale_inputs=())
    )
    assert candidate.trainable_parameter_count == 95_003
    assert baseline.trainable_parameter_count == 86_402
    assert candidate.trainable_parameter_count <= 1.10 * baseline.trainable_parameter_count
