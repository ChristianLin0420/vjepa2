from __future__ import annotations

import pytest
import torch

from jepa4d.evaluation.phase2f_camera_controls import (
    PROFILE_IDS,
    PROFILE_PERMUTATION,
    WRONG_FOCAL_SCALE,
    WRONG_PRINCIPAL_SHIFT,
    apply_profile_to_intrinsics,
    apply_profile_to_rgb,
    build_paired_camera_controls,
    frozen_camera_profiles,
    transform_and_reduce_depth,
    validate_camera_output_delta,
    validate_profile_permutation,
)


def _base_k() -> torch.Tensor:
    return torch.tensor([[300.0, 0.0, 191.5], [0.0, 310.0, 191.5], [0.0, 0.0, 1.0]])


def test_frozen_profiles_match_preregistration_exactly() -> None:
    profiles = frozen_camera_profiles()
    assert tuple(profile.profile_id for profile in profiles) == PROFILE_IDS
    assert [(profile.crop, profile.resized_size, profile.padding_tlbr) for profile in profiles] == [
        ((0, 0, 384, 384), (384, 384), (0, 0, 0, 0)),
        ((29, 29, 326, 326), (384, 384), (0, 0, 0, 0)),
        ((0, 29, 326, 326), (384, 384), (0, 0, 0, 0)),
        ((58, 29, 326, 326), (384, 384), (0, 0, 0, 0)),
        ((29, 0, 326, 326), (384, 384), (0, 0, 0, 0)),
        ((29, 58, 326, 326), (384, 384), (0, 0, 0, 0)),
        ((0, 0, 384, 384), (326, 326), (29, 29, 29, 29)),
        ((0, 0, 384, 384), (326, 384), (29, 0, 29, 0)),
    ]
    image = torch.linspace(0, 1, 384 * 384).reshape(1, 384, 384).repeat(3, 1, 1)
    outputs = [apply_profile_to_rgb(image, profile) for profile in profiles]
    assert all(output.shape == (3, 384, 384) for output in outputs)
    assert torch.equal(outputs[0], image)


def test_profile_intrinsics_use_half_pixel_crop_resize_and_padding() -> None:
    base = _base_k()
    profiles = frozen_camera_profiles()
    p0 = apply_profile_to_intrinsics(base, profiles[0])
    p1 = apply_profile_to_intrinsics(base, profiles[1])
    p6 = apply_profile_to_intrinsics(base, profiles[6])
    p7 = apply_profile_to_intrinsics(base, profiles[7])
    assert torch.equal(p0, base)

    crop_scale = 384 / 326
    assert float(p1[0, 0]) == pytest.approx(300 * crop_scale)
    assert float(p1[1, 1]) == pytest.approx(310 * crop_scale)
    assert float(p1[0, 2]) == pytest.approx(crop_scale * (191.5 + 0.5 - 29) - 0.5)
    assert float(p1[1, 2]) == pytest.approx(crop_scale * (191.5 + 0.5 - 29) - 0.5)

    pad_scale = 326 / 384
    assert float(p6[0, 0]) == pytest.approx(300 * pad_scale)
    assert float(p6[0, 2]) == pytest.approx(pad_scale * (191.5 + 0.5) - 0.5 + 29)
    assert float(p7[0, 0]) == pytest.approx(300.0)
    assert float(p7[1, 1]) == pytest.approx(310 * pad_scale)
    assert float(p7[1, 2]) == pytest.approx(pad_scale * (191.5 + 0.5) - 0.5 + 29)


def test_controls_use_exact_wrong_k_and_within_source_derangement() -> None:
    second = _base_k()
    second[0, 0] *= 1.01
    second[1, 1] *= 1.01
    second[0, 2] += 1.0
    second[1, 2] -= 1.0
    controls = build_paired_camera_controls(torch.stack((_base_k(), second)))
    assert controls.updated.shape == (2, 8, 3, 3)
    assert controls.distinct_updated_per_source == (8, 8)
    assert controls.permutation.tolist() == list(PROFILE_PERMUTATION)
    assert controls.permutation_assignment_change_fraction == 1.0
    assert controls.permutation_matrix_change_fraction == 1.0
    assert torch.equal(controls.permuted, controls.updated[:, list(PROFILE_PERMUTATION)])
    assert torch.equal(controls.stale[:, 0], controls.stale[:, 7])
    assert torch.allclose(controls.wrong[..., 0, 0], controls.updated[..., 0, 0] * WRONG_FOCAL_SCALE)
    assert torch.allclose(controls.wrong[..., 1, 1], controls.updated[..., 1, 1] * WRONG_FOCAL_SCALE)
    assert torch.allclose(controls.wrong[..., 0, 2], controls.updated[..., 0, 2] + WRONG_PRINCIPAL_SHIFT[0])
    assert torch.allclose(controls.wrong[..., 1, 2], controls.updated[..., 1, 2] + WRONG_PRINCIPAL_SHIFT[1])


def test_permutation_fails_closed_below_95_percent() -> None:
    controls = build_paired_camera_controls(_base_k())
    identity = torch.arange(8, dtype=torch.long)
    with pytest.raises(ValueError, match="changes only 0.000000 of assignments"):
        validate_profile_permutation(controls.updated, identity)


def test_mask_weighted_target_reduction_respects_padding() -> None:
    depth = torch.full((384, 384), 2.0)
    valid = torch.ones((384, 384), dtype=torch.bool)
    depth[:, :64] = 0
    valid[:, :64] = False
    p0_depth, p0_valid = transform_and_reduce_depth(depth, valid, frozen_camera_profiles()[0])
    p6_depth, p6_valid = transform_and_reduce_depth(depth, valid, frozen_camera_profiles()[6])
    assert p0_depth.shape == (24, 24)
    assert p0_valid.dtype == torch.bool
    assert torch.allclose(p0_depth[p0_valid], torch.full_like(p0_depth[p0_valid], 2.0))
    assert not p6_valid[0].any()
    assert torch.allclose(p6_depth[p6_valid], torch.full_like(p6_depth[p6_valid], 2.0))


def test_camera_output_delta_must_strictly_exceed_one_micrometre() -> None:
    updated = torch.ones(2, 8, 24, 24)
    with pytest.raises(ValueError, match="does not exceed"):
        validate_camera_output_delta(updated, updated + 1e-7)
    assert validate_camera_output_delta(updated, updated + 2e-6) > 1e-6
