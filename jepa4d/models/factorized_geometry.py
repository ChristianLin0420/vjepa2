"""Small camera-aware geometry probes with explicit shape/scale factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from jepa4d.data.camera_geometry import (
    IntrinsicsControl,
    apply_intrinsics_control,
    camera_ray_summary,
    normalized_camera_rays,
    normalized_intrinsics_summary,
    resize_intrinsics,
)

ProbeMode = Literal["monolithic", "factorized"]
CameraMode = Literal["none", "known_rays"]
ScaleInput = Literal["vjepa", "rgb", "intrinsics", "ray_summary"]

_PROBE_MODES = {"monolithic", "factorized"}
_CAMERA_MODES = {"none", "known_rays"}
_SCALE_INPUTS = {"vjepa", "rgb", "intrinsics", "ray_summary"}


@dataclass(frozen=True, slots=True)
class FactorizedGeometryConfig:
    """Architecture and causal-ablation choices for a compact geometry probe."""

    input_dim: int
    hidden_dim: int = 64
    mode: ProbeMode = "factorized"
    camera_mode: CameraMode = "none"
    scale_inputs: tuple[ScaleInput, ...] = ("vjepa",)
    scale_hidden_dim: int = 24
    vjepa_scale_dim: int = 8
    rgb_scale_dim: int = 16
    group_norm_groups: int = 8
    minimum_log_variance: float = -8.0
    maximum_log_variance: float = 6.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if self.mode not in _PROBE_MODES:
            raise ValueError(f"unknown probe mode: {self.mode}")
        if self.camera_mode not in _CAMERA_MODES:
            raise ValueError(f"unknown camera mode: {self.camera_mode}")
        if len(set(self.scale_inputs)) != len(self.scale_inputs):
            raise ValueError("scale_inputs must not contain duplicates")
        unknown = set(self.scale_inputs) - _SCALE_INPUTS
        if unknown:
            raise ValueError(f"unknown scale inputs: {sorted(unknown)}")
        if self.mode == "monolithic" and self.scale_inputs:
            raise ValueError("monolithic mode does not use a separate scale head; set scale_inputs=()")
        if self.scale_hidden_dim <= 0 or self.vjepa_scale_dim <= 0 or self.rgb_scale_dim <= 0:
            raise ValueError("scale branch dimensions must be positive")
        if self.group_norm_groups <= 0 or self.hidden_dim % self.group_norm_groups:
            raise ValueError("hidden_dim must be divisible by group_norm_groups")
        if self.minimum_log_variance >= self.maximum_log_variance:
            raise ValueError("minimum_log_variance must be below maximum_log_variance")


@dataclass(slots=True)
class FactorizedGeometryOutput:
    """Dense prediction plus inspectable shape, scale, and camera factors."""

    log_depth: torch.Tensor
    log_variance: torch.Tensor
    centered_shape: torch.Tensor | None
    global_log_scale: torch.Tensor | None
    effective_intrinsics: torch.Tensor | None
    camera_rays: torch.Tensor | None


class TinyRGBScaleEncoder(nn.Module):
    """A deliberately small appearance branch for global metric-scale cues."""

    def __init__(self, output_dim: int = 16) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        stem_dim = max(8, output_dim // 2)
        self.network = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected RGB [B,3,H,W], got {tuple(rgb.shape)}")
        if not torch.is_floating_point(rgb) or not torch.isfinite(rgb).all():
            raise ValueError("RGB scale inputs must be finite floating-point tensors")
        return self.network(rgb.float()).flatten(1)


class FactorizedShapeScaleGeometryProbe(nn.Module):
    """Predict dense log-depth while exposing camera, shape, and scale roles.

    In ``factorized`` mode the returned prediction is exactly

    ``log_depth = centered_shape + global_log_scale``.

    ``centered_shape`` is produced from dense V-JEPA features (and optionally
    known unit-ray channels), then zero-centred over the full spatial grid of
    each sample. The scalar scale branch can independently consume pooled
    V-JEPA features, a tiny RGB encoder, normalized intrinsics, and/or a ray
    summary. Empty ``scale_inputs`` gives a learned bias-only scale control.

    Intrinsics passed to :meth:`forward` must describe ``intrinsics_image_size``.
    If that differs from the dense grid, the matrix is resized with the
    half-pixel convention. Use ``update_intrinsics_for_crop_resize`` before the
    model when the RGB image was cropped.
    """

    def __init__(self, config: FactorizedGeometryConfig) -> None:
        super().__init__()
        self.config = config
        shape_input_dim = config.input_dim + (3 if config.camera_mode == "known_rays" else 0)
        self.shape_trunk = nn.Sequential(
            nn.Conv2d(shape_input_dim, config.hidden_dim, kernel_size=1),
            nn.GroupNorm(config.group_norm_groups, config.hidden_dim),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.geometry_output = nn.Conv2d(config.hidden_dim, 2, kernel_size=1)

        self.vjepa_scale_projection: nn.Module | None = None
        self.rgb_scale_encoder: nn.Module | None = None
        scale_feature_dim = 0
        if "vjepa" in config.scale_inputs:
            self.vjepa_scale_projection = nn.Sequential(
                nn.Linear(config.input_dim, config.vjepa_scale_dim),
                nn.GELU(),
            )
            scale_feature_dim += config.vjepa_scale_dim
        if "rgb" in config.scale_inputs:
            self.rgb_scale_encoder = TinyRGBScaleEncoder(config.rgb_scale_dim)
            scale_feature_dim += config.rgb_scale_dim
        if "intrinsics" in config.scale_inputs:
            scale_feature_dim += 4
        if "ray_summary" in config.scale_inputs:
            scale_feature_dim += 6

        self.scale_feature_dim = scale_feature_dim
        self.scale_head: nn.Module | None = None
        self.global_log_scale_bias: nn.Parameter | None = None
        if config.mode == "factorized":
            if scale_feature_dim:
                self.scale_head = nn.Sequential(
                    nn.Linear(scale_feature_dim, config.scale_hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.scale_hidden_dim, 1),
                )
                final = self.scale_head[-1]
                assert isinstance(final, nn.Linear)
                nn.init.normal_(final.weight, mean=0.0, std=0.01)
                nn.init.zeros_(final.bias)
            else:
                self.global_log_scale_bias = nn.Parameter(torch.zeros(1))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def ablation_signature(self, intrinsics_control: IntrinsicsControl = "correct") -> str:
        if self.config.mode == "monolithic":
            scale = "none"
        else:
            scale = "+".join(self.config.scale_inputs) if self.config.scale_inputs else "bias"
        return f"{self.config.mode}-camera_{self.config.camera_mode}-scale_{scale}-K_{intrinsics_control}"

    def _requires_intrinsics(self) -> bool:
        return self.config.camera_mode == "known_rays" or bool(
            {"intrinsics", "ray_summary"} & set(self.config.scale_inputs)
        )

    def _prepare_camera(
        self,
        features: torch.Tensor,
        rgb: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        intrinsics_image_size: tuple[int, int] | None,
        intrinsics_control: IntrinsicsControl,
        camera_permutation: torch.Tensor | None,
        wrong_focal_scale: float,
        wrong_principal_shift: tuple[float, float],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        needs_intrinsics = self._requires_intrinsics()
        if not needs_intrinsics:
            if intrinsics_control != "correct" or camera_permutation is not None:
                raise ValueError("camera control was requested by a configuration that does not consume intrinsics")
            return None, None
        if intrinsics is None:
            raise ValueError("known camera conditioning requires intrinsics")
        if intrinsics.ndim == 2:
            if intrinsics_control == "shuffled":
                raise ValueError("shuffled-camera control requires one intrinsics matrix per batch example")
            values = intrinsics.unsqueeze(0).expand(features.shape[0], -1, -1).clone()
        elif intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
            if intrinsics.shape[0] == 1 and features.shape[0] != 1:
                if intrinsics_control == "shuffled":
                    raise ValueError("shuffled-camera control requires one intrinsics matrix per batch example")
                values = intrinsics.expand(features.shape[0], -1, -1).clone()
            elif intrinsics.shape[0] == features.shape[0]:
                values = intrinsics
            else:
                raise ValueError(
                    f"intrinsics batch {intrinsics.shape[0]} does not match feature batch {features.shape[0]}"
                )
        else:
            raise ValueError(f"expected intrinsics [3,3] or [B,3,3], got {tuple(intrinsics.shape)}")
        values = values.to(device=features.device, dtype=torch.float32)
        controlled = apply_intrinsics_control(
            values,
            intrinsics_control,
            permutation=camera_permutation,
            wrong_focal_scale=wrong_focal_scale,
            wrong_principal_shift=wrong_principal_shift,
        )
        if intrinsics_image_size is None:
            source_size = (
                (int(rgb.shape[-2]), int(rgb.shape[-1]))
                if rgb is not None
                else (int(features.shape[-2]), int(features.shape[-1]))
            )
        else:
            source_size = intrinsics_image_size
        grid_size = (features.shape[-2], features.shape[-1])
        effective = resize_intrinsics(controlled, source_size, grid_size)
        needs_rays = self.config.camera_mode == "known_rays" or "ray_summary" in self.config.scale_inputs
        rays = normalized_camera_rays(effective, grid_size) if needs_rays else None
        assert effective.ndim == 3
        if rays is not None:
            assert rays.ndim == 4
        return effective, rays

    def _global_log_scale(
        self,
        features: torch.Tensor,
        rgb: torch.Tensor | None,
        effective_intrinsics: torch.Tensor | None,
        rays: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = features.shape[0]
        if not self.scale_feature_dim:
            assert self.global_log_scale_bias is not None
            return self.global_log_scale_bias.view(1, 1, 1).expand(batch, -1, -1)
        values: list[torch.Tensor] = []
        if "vjepa" in self.config.scale_inputs:
            assert self.vjepa_scale_projection is not None
            pooled = features.float().mean(dim=(-2, -1))
            values.append(self.vjepa_scale_projection(pooled))
        if "rgb" in self.config.scale_inputs:
            if rgb is None:
                raise ValueError("the configured RGB scale branch requires rgb")
            if rgb.shape[0] != batch:
                raise ValueError(f"RGB batch {rgb.shape[0]} does not match feature batch {batch}")
            if rgb.device != features.device:
                raise ValueError("RGB and dense features must be on the same device")
            assert self.rgb_scale_encoder is not None
            values.append(self.rgb_scale_encoder(rgb))
        grid_size = (features.shape[-2], features.shape[-1])
        if "intrinsics" in self.config.scale_inputs:
            assert effective_intrinsics is not None
            summary = normalized_intrinsics_summary(effective_intrinsics, grid_size)
            assert summary.ndim == 2
            values.append(summary)
        if "ray_summary" in self.config.scale_inputs:
            assert rays is not None
            summary = camera_ray_summary(rays)
            assert summary.ndim == 2
            values.append(summary)
        concatenated = torch.cat(values, dim=-1)
        if concatenated.shape != (batch, self.scale_feature_dim):
            raise RuntimeError(
                f"scale feature shape {tuple(concatenated.shape)} does not match {(batch, self.scale_feature_dim)}"
            )
        assert self.scale_head is not None
        return self.scale_head(concatenated).view(batch, 1, 1)

    def forward(
        self,
        features: torch.Tensor,
        *,
        rgb: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        intrinsics_image_size: tuple[int, int] | None = None,
        intrinsics_control: IntrinsicsControl = "correct",
        camera_permutation: torch.Tensor | None = None,
        wrong_focal_scale: float = 1.25,
        wrong_principal_shift: tuple[float, float] = (0.0, 0.0),
    ) -> FactorizedGeometryOutput:
        if features.ndim != 4 or features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"expected dense V-JEPA features [B,{self.config.input_dim},H,W], got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("dense V-JEPA features must be finite floating-point tensors")
        if rgb is not None and (rgb.ndim != 4 or rgb.shape[1] != 3):
            raise ValueError(f"expected RGB [B,3,H,W], got {tuple(rgb.shape)}")

        effective_intrinsics, rays = self._prepare_camera(
            features,
            rgb,
            intrinsics,
            intrinsics_image_size,
            intrinsics_control,
            camera_permutation,
            wrong_focal_scale,
            wrong_principal_shift,
        )
        shape_input = features.float()
        if self.config.camera_mode == "known_rays":
            assert rays is not None
            shape_input = torch.cat((shape_input, rays), dim=1)
        dense_output = self.geometry_output(self.shape_trunk(shape_input))
        raw_depth_or_shape = dense_output[:, 0]
        log_variance = dense_output[:, 1].clamp(
            self.config.minimum_log_variance,
            self.config.maximum_log_variance,
        )

        if self.config.mode == "monolithic":
            return FactorizedGeometryOutput(
                log_depth=raw_depth_or_shape.clamp(-8.0, 8.0),
                log_variance=log_variance,
                centered_shape=None,
                global_log_scale=None,
                effective_intrinsics=effective_intrinsics,
                camera_rays=rays,
            )

        centered_shape = raw_depth_or_shape - raw_depth_or_shape.mean(dim=(-2, -1), keepdim=True)
        global_log_scale = self._global_log_scale(features, rgb, effective_intrinsics, rays)
        log_depth = centered_shape + global_log_scale
        return FactorizedGeometryOutput(
            log_depth=log_depth,
            log_variance=log_variance,
            centered_shape=centered_shape,
            global_log_scale=global_log_scale,
            effective_intrinsics=effective_intrinsics,
            camera_rays=rays,
        )
