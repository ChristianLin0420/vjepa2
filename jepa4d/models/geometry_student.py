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


class BoundedResidualLayerFusion(nn.Module):
    """Fuse three standardized intermediate layers into a canonical final layer.

    For final feature ``F`` and intermediate features ``I_l`` the output is

    ``F + sum(tanh(g_l) / 3 * (I_l - F))``.

    The three scalar gates start at zero, so initialization is exactly the
    final-layer baseline while every gate still receives a first-step
    gradient.  Coefficients are signed and each is bounded to ``[-1/3, 1/3]``.
    """

    def __init__(self, input_dim: int, layer_order: tuple[int, int, int] = (2, 5, 8)) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if len(layer_order) != 3 or len(set(layer_order)) != 3:
            raise ValueError("layer_order must contain exactly three unique layers")
        self.input_dim = input_dim
        self.layer_order = layer_order
        self.raw_gates = nn.Parameter(torch.zeros(len(layer_order)))

    def effective_coefficients(self) -> torch.Tensor:
        return torch.tanh(self.raw_gates) / len(self.layer_order)

    def forward(self, final: torch.Tensor, intermediates: torch.Tensor) -> torch.Tensor:
        expected_intermediate_shape = (final.shape[0], len(self.layer_order), *final.shape[1:])
        if final.ndim != 4 or final.shape[1] != self.input_dim:
            raise ValueError(f"expected final [B,{self.input_dim},H,W], got {tuple(final.shape)}")
        if tuple(intermediates.shape) != expected_intermediate_shape:
            raise ValueError(f"expected intermediates {expected_intermediate_shape}, got {tuple(intermediates.shape)}")
        final_float = final.float()
        intermediate_float = intermediates.float()
        if not torch.isfinite(final_float).all() or not torch.isfinite(intermediate_float).all():
            raise ValueError("fusion features must be finite")
        coefficients = self.effective_coefficients().view(1, -1, 1, 1, 1)
        return final_float + (coefficients * (intermediate_float - final_float.unsqueeze(1))).sum(dim=1)


class ResidualFusionGeometryProbe(nn.Module):
    """A three-parameter residual layer fuser followed by the shared probe."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        layer_order: tuple[int, int, int] = (2, 5, 8),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_order = layer_order
        self.fusion = BoundedResidualLayerFusion(input_dim, layer_order)
        self.probe = DenseGeometryProbe(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 5 or features.shape[1] != 1 + len(self.layer_order):
            raise ValueError(
                f"expected [B,{1 + len(self.layer_order)},C,H,W] final-plus-intermediate features, "
                f"got {tuple(features.shape)}"
            )
        fused = self.fusion(features[:, 0], features[:, 1:])
        return self.probe(fused)

    def fusion_state(self) -> dict[str, float | list[int]]:
        coefficients = self.fusion.effective_coefficients().detach().cpu()
        values: dict[str, float | list[int]] = {
            "layer_order": list(self.layer_order),
            "final_coefficient": float(1.0 - coefficients.sum()),
        }
        for layer, raw, effective in zip(
            self.layer_order,
            self.fusion.raw_gates.detach().cpu(),
            coefficients,
            strict=True,
        ):
            values[f"raw_gate_layer_{layer}"] = float(raw)
            values[f"coefficient_layer_{layer}"] = float(effective)
        return values


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
