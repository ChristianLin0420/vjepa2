"""Compact dense geometry probes for Phase 2b representation comparisons."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DenseGeometryProbe(nn.Module):
    """Predict metric log-depth and log-variance from a frozen feature grid."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != self.input_dim:
            raise ValueError(f"expected [B,{self.input_dim},H,W] features, got {tuple(features.shape)}")
        output = self.network(features.float())
        log_depth = output[:, 0].clamp(-4.0, 4.0)
        log_variance = output[:, 1].clamp(-8.0, 6.0)
        return log_depth, log_variance


def geometry_probe_loss(
    predicted_log_depth: torch.Tensor,
    predicted_log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    teacher_depth: torch.Tensor | None = None,
    teacher_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_log_depth = target_depth.clamp_min(1e-4).log()
    residual = predicted_log_depth - target_log_depth
    target_valid = valid_mask & torch.isfinite(target_log_depth)
    if not torch.isfinite(predicted_log_depth[target_valid]).all():
        raise ValueError("predicted log depth is non-finite on valid target pixels")
    if not torch.isfinite(predicted_log_variance[target_valid]).all():
        raise ValueError("predicted log variance is non-finite on valid target pixels")
    valid = target_valid
    if int(valid.sum()) == 0:
        raise ValueError("geometry probe loss has no finite valid pixels")
    nll = 0.5 * (torch.exp(-predicted_log_variance) * residual.square() + predicted_log_variance)
    nll_loss = nll[valid].mean()
    centered = residual[valid] - residual[valid].mean()
    scale_invariant = centered.square().mean()
    gradient_loss = _gradient_loss(predicted_log_depth, target_log_depth, valid)
    distillation = torch.zeros((), device=predicted_log_depth.device)
    if teacher_depth is not None:
        if teacher_depth.shape != target_depth.shape:
            raise ValueError(f"teacher depth shape {tuple(teacher_depth.shape)} != target {tuple(target_depth.shape)}")
        if not torch.isfinite(teacher_depth[valid]).all() or not (teacher_depth[valid] > 0).all():
            raise ValueError("teacher depth is non-finite or non-positive on valid pixels")
        teacher_log_depth = teacher_depth.clamp_min(1e-4).log()
        distillation = F.smooth_l1_loss(predicted_log_depth[valid], teacher_log_depth[valid])
    total = nll_loss + 0.25 * scale_invariant + 0.1 * gradient_loss + teacher_weight * distillation
    return total, {
        "nll": nll_loss.detach(),
        "scale_invariant": scale_invariant.detach(),
        "gradient": gradient_loss.detach(),
        "distillation": distillation.detach(),
    }


def _gradient_loss(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (-1, -2):
        predicted_delta = torch.diff(predicted, dim=dimension)
        target_delta = torch.diff(target, dim=dimension)
        valid_delta = torch.diff(valid.float(), dim=dimension).eq(0)
        if dimension == -1:
            valid_delta &= valid[..., 1:] & valid[..., :-1]
        else:
            valid_delta &= valid[..., 1:, :] & valid[..., :-1, :]
        if valid_delta.any():
            losses.append((predicted_delta - target_delta).abs()[valid_delta].mean())
    if not losses:
        raise ValueError("geometry gradient loss has no valid neighboring pixels")
    return torch.stack(losses).mean()


def rgb_grid_features(images: torch.Tensor, size: int = 24) -> torch.Tensor:
    """Create a no-pretraining RGB+coordinate baseline on the probe grid."""
    pooled = F.adaptive_avg_pool2d(images.float(), (size, size))
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=images.device),
        torch.linspace(-1, 1, size, device=images.device),
        indexing="ij",
    )
    coordinates = torch.stack((x, y)).unsqueeze(0).expand(images.shape[0], -1, -1, -1)
    return torch.cat((pooled, coordinates), dim=1)
