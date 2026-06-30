"""Gradient-separated shape, scale, field, and uncertainty losses for Phase 2f."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from jepa4d.models.geometry_student import geometry_probe_loss
from jepa4d.models.phase2f_scale_geometry import Phase2fGeometryOutput


@dataclass(frozen=True, slots=True)
class Phase2fLossConfig:
    """Frozen weights and Smooth-L1 transition for the registered objective."""

    smooth_l1_beta: float = 0.1
    centered_shape_weight: float = 1.0
    shape_gradient_weight: float = 0.25
    shape_nll_weight: float = 0.10
    global_scale_weight: float = 1.0
    scale_nll_weight: float = 0.10
    paired_scale_consistency_weight: float = 0.10
    scale_field_fit_weight: float = 0.25
    scale_field_tv_weight: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.smooth_l1_beta,
            self.centered_shape_weight,
            self.shape_gradient_weight,
            self.shape_nll_weight,
            self.global_scale_weight,
            self.scale_nll_weight,
            self.paired_scale_consistency_weight,
            self.scale_field_fit_weight,
            self.scale_field_tv_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Phase 2f loss weights must be finite and non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if self.centered_shape_weight == 0 or self.global_scale_weight == 0:
            raise ValueError("centered_shape_weight and global_scale_weight must be positive")


@dataclass(slots=True)
class Phase2fLossResult:
    """Separate differentiable objectives plus detached logging diagnostics."""

    total: torch.Tensor
    shape_objective: torch.Tensor
    scale_objective: torch.Tensor
    field_objective: torch.Tensor
    optimal_log_scale_target: torch.Tensor | None
    components: dict[str, torch.Tensor]


def valid_depth_mask(target_depth: torch.Tensor) -> torch.Tensor:
    if target_depth.ndim != 3 or not torch.is_floating_point(target_depth):
        raise ValueError(f"expected floating target depth [B,H,W], got {tuple(target_depth.shape)}")
    return torch.isfinite(target_depth) & (target_depth > 0)


def _validate_mask(values: torch.Tensor, valid: torch.Tensor) -> None:
    if valid.shape != values.shape or valid.dtype != torch.bool:
        raise ValueError(
            f"valid mask shape/dtype {tuple(valid.shape)}/{valid.dtype} does not match {tuple(values.shape)}"
        )
    counts = valid.flatten(1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every sample must contain at least one valid depth pixel")


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    _validate_mask(values, valid)
    counts = valid.flatten(1).sum(dim=1)
    sums = torch.where(valid, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    return sums / counts


def _masked_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    _validate_mask(values, valid)
    return torch.stack([values[index][valid[index]].median() for index in range(values.shape[0])])


def robust_optimal_log_scale_target(
    target_log_depth: torch.Tensor,
    centered_shape: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the exact per-sample L1-optimal scale after stopping shape gradients.

    A median of ``log(depth_gt) - stopgrad(shape)`` minimizes the absolute
    log-residual and has a 50% breakdown point. The returned tensor is always
    detached, making the intended shape-to-scale firewall explicit even when
    the caller passes a differentiable shape tensor.
    """

    if target_log_depth.shape != centered_shape.shape:
        raise ValueError(
            f"target log-depth shape {tuple(target_log_depth.shape)} != shape {tuple(centered_shape.shape)}"
        )
    _validate_mask(target_log_depth, valid_mask)
    residual = target_log_depth.detach() - centered_shape.detach()
    targets = [residual[index][valid_mask[index]].median() for index in range(residual.shape[0])]
    result = torch.stack(targets)
    if not torch.isfinite(result).all():
        raise ValueError("optimal log-scale target is non-finite")
    return result.detach()


def _gradient_smooth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    horizontal_valid = valid[..., 1:] & valid[..., :-1]
    if bool(horizontal_valid.any()):
        horizontal_residual = torch.diff(predicted, dim=-1) - torch.diff(target, dim=-1)
        losses.append(
            F.smooth_l1_loss(
                horizontal_residual[horizontal_valid],
                torch.zeros_like(horizontal_residual[horizontal_valid]),
                beta=beta,
            )
        )
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    if bool(vertical_valid.any()):
        vertical_residual = torch.diff(predicted, dim=-2) - torch.diff(target, dim=-2)
        losses.append(
            F.smooth_l1_loss(
                vertical_residual[vertical_valid],
                torch.zeros_like(vertical_residual[vertical_valid]),
                beta=beta,
            )
        )
    if not losses:
        raise ValueError("shape gradient loss has no neighboring valid pixels")
    return torch.stack(losses).mean()


def _total_variation(field: torch.Tensor) -> torch.Tensor:
    if field.ndim != 3:
        raise ValueError(f"expected coarse field [B,H,W], got {tuple(field.shape)}")
    losses: list[torch.Tensor] = []
    if field.shape[-1] > 1:
        losses.append(torch.diff(field, dim=-1).abs().mean())
    if field.shape[-2] > 1:
        losses.append(torch.diff(field, dim=-2).abs().mean())
    return field.new_zeros(()) if not losses else torch.stack(losses).mean()


def _gaussian_nll(residual: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-log_variance) * residual.square() + log_variance)


def paired_view_scale_consistency(
    global_log_scale: torch.Tensor,
    *,
    group_count: int | None,
    views: int,
    beta: float = 0.1,
) -> torch.Tensor:
    """Return the registered two-view Smooth-L1 scale consistency term."""

    if isinstance(views, bool) or not isinstance(views, int) or views not in {1, 2}:
        raise ValueError("Phase 2f views must be one or two")
    flattened = global_log_scale.flatten()
    if views == 1:
        if group_count is not None and group_count != len(flattened):
            raise ValueError("group_count does not match the single-view scale batch")
        return flattened.new_zeros(())
    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count <= 0:
        raise ValueError("two-view scale consistency requires a positive group_count")
    if len(flattened) != group_count * views:
        raise ValueError(f"scale batch {len(flattened)} does not match group_count={group_count}, views={views}")
    scales = flattened.reshape(group_count, views)
    return F.smooth_l1_loss(scales[:, 0], scales[:, 1], beta=beta)


def phase2f_loss(
    output: Phase2fGeometryOutput,
    target_depth: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    config: Phase2fLossConfig | None = None,
    group_count: int | None = None,
    views: int = 1,
) -> Phase2fLossResult:
    """Build objectives whose gradient ownership follows the Phase 2f firewall."""

    if config is None:
        config = Phase2fLossConfig()
    if target_depth.shape != output.log_depth.shape:
        raise ValueError(f"target shape {tuple(target_depth.shape)} != output {tuple(output.log_depth.shape)}")
    valid = valid_depth_mask(target_depth) if valid_mask is None else valid_mask
    _validate_mask(target_depth, valid)
    target_log = target_depth.clamp_min(1e-4).log()
    if not torch.isfinite(target_log[valid]).all():
        raise ValueError("target log-depth is non-finite on valid pixels")

    if output.arm == "M0":
        if any(
            value is not None
            for value in (
                output.centered_shape,
                output.global_log_scale,
                output.coarse_scale_field,
                output.scale_field,
                output.shape_log_variance,
                output.global_scale_log_variance,
            )
        ):
            raise ValueError("M0 must not expose factorized outputs")
        base, base_parts = geometry_probe_loss(output.log_depth, output.log_variance, target_depth, valid)
        zero = base.new_zeros(())
        return Phase2fLossResult(
            total=base,
            shape_objective=base,
            scale_objective=zero,
            field_objective=zero,
            optimal_log_scale_target=None,
            components={
                "total": base.detach(),
                "shape_objective": base.detach(),
                "scale_objective": zero,
                "field_objective": zero,
                **{f"monolithic_{key}": value for key, value in base_parts.items()},
            },
        )

    required = (
        output.centered_shape,
        output.global_log_scale,
        output.shape_log_variance,
        output.global_scale_log_variance,
    )
    if any(value is None for value in required):
        raise ValueError(f"factorized arm {output.arm} is missing shape, scale, or uncertainty factors")
    assert output.centered_shape is not None
    assert output.global_log_scale is not None
    assert output.shape_log_variance is not None
    assert output.global_scale_log_variance is not None

    target_center = _masked_median(target_log, valid)
    target_shape = target_log - target_center[:, None, None]
    predicted_shape = output.centered_shape
    shape_residual = predicted_shape - target_shape
    shape_nll = _gaussian_nll(shape_residual, output.shape_log_variance)[valid].mean()
    shape_l1 = F.smooth_l1_loss(
        predicted_shape[valid],
        target_shape[valid],
        beta=config.smooth_l1_beta,
    )
    shape_gradient = _gradient_smooth_l1(
        predicted_shape,
        target_shape,
        valid,
        beta=config.smooth_l1_beta,
    )
    shape_objective = (
        config.centered_shape_weight * shape_l1
        + config.shape_gradient_weight * shape_gradient
        + config.shape_nll_weight * shape_nll
    )

    optimal_scale = robust_optimal_log_scale_target(target_log, output.centered_shape, valid)
    predicted_scale = output.global_log_scale.flatten()
    scale_log_variance = output.global_scale_log_variance.flatten()
    if predicted_scale.shape != optimal_scale.shape or scale_log_variance.shape != optimal_scale.shape:
        raise ValueError("global scale mean/variance must provide one scalar per sample")
    scale_residual = predicted_scale - optimal_scale
    scale_nll = _gaussian_nll(scale_residual, scale_log_variance).mean()
    scale_l1 = F.smooth_l1_loss(predicted_scale, optimal_scale, beta=config.smooth_l1_beta)
    paired_consistency = paired_view_scale_consistency(
        output.global_log_scale,
        group_count=group_count,
        views=views,
        beta=config.smooth_l1_beta,
    )
    scale_objective = (
        config.global_scale_weight * scale_l1
        + config.scale_nll_weight * scale_nll
        + config.paired_scale_consistency_weight * paired_consistency
    )

    zero = output.log_depth.new_zeros(())
    field_fit = zero
    field_tv = zero
    field_zero_mean = zero
    field_max_abs = zero
    field_objective = zero
    if output.arm == "M3":
        if output.scale_field is None or output.coarse_scale_field is None:
            raise ValueError("M3 requires dense and coarse scale fields")
        field_target = target_log.detach() - output.centered_shape.detach() - optimal_scale[:, None, None]
        field_target = field_target - _masked_mean(field_target, valid)[:, None, None]
        field_fit = F.smooth_l1_loss(
            output.scale_field[valid],
            field_target[valid],
            beta=config.smooth_l1_beta,
        )
        field_tv = _total_variation(output.coarse_scale_field)
        field_zero_mean = output.coarse_scale_field.mean(dim=(-2, -1)).abs().mean()
        field_max_abs = output.coarse_scale_field.abs().amax()
        field_objective = config.scale_field_fit_weight * field_fit + config.scale_field_tv_weight * field_tv
    elif output.scale_field is not None or output.coarse_scale_field is not None:
        raise ValueError(f"arm {output.arm} must not expose a coarse scale field")

    joint_residual = output.log_depth - target_log
    joint_nll = _gaussian_nll(joint_residual, output.log_variance)[valid].mean()
    total = shape_objective + scale_objective + field_objective
    components = {
        "total": total.detach(),
        "shape_objective": shape_objective.detach(),
        "shape_nll": shape_nll.detach(),
        "shape_l1": shape_l1.detach(),
        "shape_gradient": shape_gradient.detach(),
        "scale_objective": scale_objective.detach(),
        "scale_nll": scale_nll.detach(),
        "scale_l1": scale_l1.detach(),
        "paired_scale_consistency": paired_consistency.detach(),
        "scale_field_objective": field_objective.detach(),
        "scale_field_fit": field_fit.detach(),
        "scale_field_tv": field_tv.detach(),
        "scale_field_zero_mean_error": field_zero_mean.detach(),
        "scale_field_max_abs": field_max_abs.detach(),
        "joint_nll_diagnostic_only": joint_nll.detach(),
        "optimal_log_scale_mean": optimal_scale.mean().detach(),
    }
    return Phase2fLossResult(
        total=total,
        shape_objective=shape_objective,
        scale_objective=scale_objective,
        field_objective=field_objective,
        optimal_log_scale_target=optimal_scale,
        components=components,
    )
