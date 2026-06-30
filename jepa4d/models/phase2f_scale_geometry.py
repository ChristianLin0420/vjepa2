"""Detached shape/scale geometry probes for the Phase 2f ablation matrix.

The module deliberately accepts feature tensors directly. Dataset transforms,
feature caches, experiment orchestration, and evaluation policy live outside
this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import torch
import torch.nn.functional as F
from torch import nn

from jepa4d.data.camera_geometry import normalized_intrinsics_summary

Phase2fArm = Literal["M0", "M1", "M2", "M3", "M4"]

DEFAULT_PHASE2F_ARMS: tuple[Phase2fArm, ...] = ("M0", "M1", "M2", "M3")
OPTIONAL_PHASE2F_ARMS: tuple[Phase2fArm, ...] = ("M4",)
PHASE2F_COMPONENTS: tuple[str, ...] = (
    "camera_transform",
    "ray_construction",
    "dense_shape_decoder",
    "pooling",
    "scale_head",
    "coarse_scale_field",
    "composition",
)

_ARMS = set(DEFAULT_PHASE2F_ARMS + OPTIONAL_PHASE2F_ARMS)
_T = TypeVar("_T")


class ComponentTimingHook(Protocol):
    """Execute one named component while recording externally defined timing."""

    def __call__(self, name: str, operation: Callable[[], _T]) -> _T: ...


def _timed(hook: ComponentTimingHook | None, name: str, operation: Callable[[], _T]) -> _T:
    return operation() if hook is None else hook(name, operation)


@dataclass(frozen=True, slots=True)
class Phase2fGeometryConfig:
    """Architecture choices shared by the Phase 2f scale/camera arms."""

    input_dim: int
    arm: Phase2fArm
    hidden_dim: int = 64
    group_norm_groups: int = 8
    scale_feature_dim: int = 8
    scale_hidden_dim: int = 24
    camera_prompt_dim: int = 8
    canonical_normalized_focal: tuple[float, float] = (1.0, 1.0)
    coarse_field_size: tuple[int, int] = (4, 4)
    maximum_scale_field_amplitude: float = 0.25
    minimum_log_variance: float = -8.0
    maximum_log_variance: float = 6.0

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise ValueError(f"unknown Phase 2f arm: {self.arm}")
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if self.group_norm_groups <= 0 or self.hidden_dim % self.group_norm_groups:
            raise ValueError("hidden_dim must be divisible by group_norm_groups")
        if self.scale_feature_dim <= 0 or self.scale_hidden_dim <= 0 or self.camera_prompt_dim <= 0:
            raise ValueError("scale and prompt dimensions must be positive")
        if len(self.coarse_field_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.coarse_field_size
        ):
            raise ValueError("coarse_field_size must contain two positive integers")
        if not math.isfinite(self.maximum_scale_field_amplitude) or self.maximum_scale_field_amplitude <= 0:
            raise ValueError("maximum_scale_field_amplitude must be finite and positive")
        if len(self.canonical_normalized_focal) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in self.canonical_normalized_focal
        ):
            raise ValueError("canonical_normalized_focal must contain two finite positive values")
        if self.minimum_log_variance >= self.maximum_log_variance:
            raise ValueError("minimum_log_variance must be below maximum_log_variance")

    @property
    def factorized(self) -> bool:
        return self.arm != "M0"

    @property
    def consumes_intrinsics(self) -> bool:
        return self.arm in {"M2", "M3", "M4"}

    @property
    def predicts_coarse_scale_field(self) -> bool:
        return self.arm == "M3"


@dataclass(slots=True)
class Phase2fGeometryOutput:
    """Inspectable mean and uncertainty factors for one Phase 2f arm."""

    arm: Phase2fArm
    log_depth: torch.Tensor
    log_variance: torch.Tensor
    centered_shape: torch.Tensor | None
    global_log_scale: torch.Tensor | None
    coarse_scale_field: torch.Tensor | None
    scale_field: torch.Tensor | None
    shape_log_variance: torch.Tensor | None
    global_scale_log_variance: torch.Tensor | None
    canonical_camera_features: torch.Tensor | None


def canonical_camera_features(
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    *,
    canonical_normalized_focal: tuple[float, float] = (1.0, 1.0),
) -> torch.Tensor:
    """Return an interpretable four-value offset from a canonical pinhole camera.

    The values are ``log(fx / width)``, ``log(fy / height)``, and centered
    principal-point offsets, with the configured canonical focal subtracted.
    Because the representation is normalized, equivalent full-frame resizes
    produce the same camera features under the half-pixel convention.
    """

    if len(canonical_normalized_focal) != 2 or any(
        not math.isfinite(value) or value <= 0 for value in canonical_normalized_focal
    ):
        raise ValueError("canonical_normalized_focal must contain two finite positive values")
    summary = normalized_intrinsics_summary(intrinsics, image_size).float()
    canonical = summary.new_tensor(
        (
            math.log(canonical_normalized_focal[0]),
            math.log(canonical_normalized_focal[1]),
            0.0,
            0.0,
        )
    )
    return summary - canonical


def _project_zero_mean_amplitude(values: torch.Tensor, maximum_amplitude: float) -> torch.Tensor:
    centered = values - values.mean(dim=(-2, -1), keepdim=True)
    maximum = centered.abs().amax(dim=(-2, -1), keepdim=True)
    scale = (maximum / maximum_amplitude).clamp_min(1.0)
    return centered / scale


def _bounded_zero_mean(values: torch.Tensor, maximum_amplitude: float) -> torch.Tensor:
    """Apply the registered tanh bound, then preserve exact zero spatial mean."""

    bounded = maximum_amplitude * torch.tanh(values)
    return _project_zero_mean_amplitude(bounded, maximum_amplitude)


class Phase2fScaleGeometryProbe(nn.Module):
    """Implement M0-M4 with explicit branch ownership and cheap camera cues.

    M1-M4 detach the frozen feature input before every scale-side operation.
    Their learned shape decoder and learned scale components are disjoint, so
    the matching Phase 2f loss can enforce a strict gradient firewall.
    """

    def __init__(self, config: Phase2fGeometryConfig) -> None:
        super().__init__()
        self.config = config
        self.shape_decoder = nn.Sequential(
            nn.Conv2d(config.input_dim, config.hidden_dim, kernel_size=1),
            nn.GroupNorm(config.group_norm_groups, config.hidden_dim),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, 2, kernel_size=1),
        )

        self.scale_projection: nn.Module | None = None
        self.scale_head: nn.Module | None = None
        self.camera_prompt: nn.Module | None = None
        self.coarse_scale_field_head: nn.Module | None = None
        if config.factorized:
            self.scale_projection = nn.Sequential(
                nn.Linear(config.input_dim, config.scale_feature_dim),
                nn.GELU(),
            )
            camera_dim = 0
            if config.arm in {"M2", "M3"}:
                camera_dim = 4
            elif config.arm == "M4":
                self.camera_prompt = nn.Sequential(
                    nn.Linear(4, config.camera_prompt_dim),
                    nn.GELU(),
                )
                camera_dim = config.camera_prompt_dim
            self.scale_head = nn.Sequential(
                nn.Linear(config.scale_feature_dim + camera_dim, config.scale_hidden_dim),
                nn.GELU(),
                nn.Linear(config.scale_hidden_dim, 2),
            )
            final_scale = self.scale_head[-1]
            assert isinstance(final_scale, nn.Linear)
            nn.init.normal_(final_scale.weight, mean=0.0, std=0.01)
            if final_scale.bias is not None:
                nn.init.zeros_(final_scale.bias)
        if config.predicts_coarse_scale_field:
            self.coarse_scale_field_head = nn.Conv2d(config.input_dim, 1, kernel_size=1)
            nn.init.normal_(self.coarse_scale_field_head.weight, mean=0.0, std=0.001)
            if self.coarse_scale_field_head.bias is not None:
                nn.init.zeros_(self.coarse_scale_field_head.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def shape_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.shape_decoder.parameters()

    def scale_parameters(self) -> Iterator[nn.Parameter]:
        for module in (self.scale_projection, self.scale_head, self.camera_prompt):
            if module is not None:
                yield from module.parameters()

    def field_parameters(self) -> Iterator[nn.Parameter]:
        if self.coarse_scale_field_head is not None:
            yield from self.coarse_scale_field_head.parameters()

    @staticmethod
    def _module_parameter_count(module: nn.Module | None) -> int:
        if module is None:
            return 0
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        """Return a non-overlapping, exact count for each timed component."""

        counts = {
            "shape_decoder": self._module_parameter_count(self.shape_decoder),
            "scale_projection": self._module_parameter_count(self.scale_projection),
            "scale_head": self._module_parameter_count(self.scale_head),
            "camera_prompt": self._module_parameter_count(self.camera_prompt),
            "coarse_scale_field": self._module_parameter_count(self.coarse_scale_field_head),
        }
        counts["total"] = sum(counts.values())
        if counts["total"] != self.trainable_parameter_count:
            raise RuntimeError("component parameter counts do not cover the trainable model exactly")
        return counts

    def _prepare_camera(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor | None,
        intrinsics_image_size: tuple[int, int] | None,
    ) -> torch.Tensor | None:
        if not self.config.consumes_intrinsics:
            if intrinsics is not None or intrinsics_image_size is not None:
                raise ValueError(f"arm {self.config.arm} does not consume camera intrinsics")
            return None
        if intrinsics is None or intrinsics_image_size is None:
            raise ValueError(f"arm {self.config.arm} requires intrinsics and intrinsics_image_size")
        if intrinsics.ndim == 2:
            if intrinsics.shape != (3, 3):
                raise ValueError(f"expected intrinsics [3,3] or [B,3,3], got {tuple(intrinsics.shape)}")
            values = intrinsics.unsqueeze(0).expand(features.shape[0], -1, -1)
        elif intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
            if intrinsics.shape[0] == 1 and features.shape[0] != 1:
                values = intrinsics.expand(features.shape[0], -1, -1)
            elif intrinsics.shape[0] == features.shape[0]:
                values = intrinsics
            else:
                raise ValueError(
                    f"intrinsics batch {intrinsics.shape[0]} does not match feature batch {features.shape[0]}"
                )
        else:
            raise ValueError(f"expected intrinsics [3,3] or [B,3,3], got {tuple(intrinsics.shape)}")
        values = values.detach().to(device=features.device, dtype=torch.float32)
        camera = canonical_camera_features(
            values,
            intrinsics_image_size,
            canonical_normalized_focal=self.config.canonical_normalized_focal,
        )
        if camera.shape != (features.shape[0], 4):
            raise RuntimeError(f"canonical camera features have unexpected shape {tuple(camera.shape)}")
        return camera

    def _decode_shape(self, features: torch.Tensor) -> torch.Tensor:
        return self.shape_decoder(features.float())

    def _pool_scale_features(self, features: torch.Tensor) -> torch.Tensor:
        return features.detach().float().mean(dim=(-2, -1))

    def _predict_global_scale(
        self,
        pooled_features: torch.Tensor,
        camera: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.scale_projection is not None and self.scale_head is not None
        values = [self.scale_projection(pooled_features)]
        if self.config.arm in {"M2", "M3"}:
            assert camera is not None
            values.append(camera)
        elif self.config.arm == "M4":
            assert camera is not None and self.camera_prompt is not None
            values.append(self.camera_prompt(camera))
        output = self.scale_head(torch.cat(values, dim=-1))
        global_scale = output[:, 0].view(-1, 1, 1)
        global_log_variance = (
            output[:, 1]
            .clamp(
                self.config.minimum_log_variance,
                self.config.maximum_log_variance,
            )
            .view(-1, 1, 1)
        )
        return global_scale, global_log_variance

    def _predict_scale_field(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.coarse_scale_field_head is not None
        pooled = F.adaptive_avg_pool2d(features.detach().float(), self.config.coarse_field_size)
        raw_coarse = self.coarse_scale_field_head(pooled)[:, 0]
        coarse = _bounded_zero_mean(raw_coarse, self.config.maximum_scale_field_amplitude)
        dense = F.interpolate(
            coarse.unsqueeze(1),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        dense = _project_zero_mean_amplitude(dense, self.config.maximum_scale_field_amplitude)
        return coarse, dense

    def forward(
        self,
        features: torch.Tensor,
        *,
        intrinsics: torch.Tensor | None = None,
        intrinsics_image_size: tuple[int, int] | None = None,
        timing_hook: ComponentTimingHook | None = None,
    ) -> Phase2fGeometryOutput:
        if features.ndim != 4 or features.shape[1] != self.config.input_dim:
            raise ValueError(f"expected features [B,{self.config.input_dim},H,W], got {tuple(features.shape)}")
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError("features must be finite floating-point tensors")

        if self.config.consumes_intrinsics:
            camera = _timed(
                timing_hook,
                "camera_transform",
                lambda: self._prepare_camera(features, intrinsics, intrinsics_image_size),
            )
        else:
            camera = self._prepare_camera(features, intrinsics, intrinsics_image_size)
        dense = _timed(timing_hook, "dense_shape_decoder", lambda: self._decode_shape(features))

        if not self.config.factorized:

            def compose_monolithic() -> Phase2fGeometryOutput:
                log_depth = dense[:, 0].clamp(-8.0, 8.0)
                log_variance = dense[:, 1].clamp(
                    self.config.minimum_log_variance,
                    self.config.maximum_log_variance,
                )
                return Phase2fGeometryOutput(
                    arm=self.config.arm,
                    log_depth=log_depth,
                    log_variance=log_variance,
                    centered_shape=None,
                    global_log_scale=None,
                    coarse_scale_field=None,
                    scale_field=None,
                    shape_log_variance=None,
                    global_scale_log_variance=None,
                    canonical_camera_features=None,
                )

            return _timed(timing_hook, "composition", compose_monolithic)

        pooled = _timed(timing_hook, "pooling", lambda: self._pool_scale_features(features))
        global_scale, global_log_variance = _timed(
            timing_hook,
            "scale_head",
            lambda: self._predict_global_scale(pooled, camera),
        )
        coarse_field: torch.Tensor | None = None
        scale_field: torch.Tensor | None = None
        if self.config.predicts_coarse_scale_field:
            coarse_field, scale_field = _timed(
                timing_hook,
                "coarse_scale_field",
                lambda: self._predict_scale_field(features),
            )

        def compose_factorized() -> Phase2fGeometryOutput:
            raw_shape = dense[:, 0]
            centered_shape = raw_shape - raw_shape.mean(dim=(-2, -1), keepdim=True)
            shape_log_variance = dense[:, 1].clamp(
                self.config.minimum_log_variance,
                self.config.maximum_log_variance,
            )
            correction: torch.Tensor | float = 0.0 if scale_field is None else scale_field
            log_depth = centered_shape + global_scale + correction
            expanded_scale_log_variance = global_log_variance.expand_as(shape_log_variance)
            log_variance = torch.logaddexp(shape_log_variance, expanded_scale_log_variance)
            return Phase2fGeometryOutput(
                arm=self.config.arm,
                log_depth=log_depth,
                log_variance=log_variance,
                centered_shape=centered_shape,
                global_log_scale=global_scale,
                coarse_scale_field=coarse_field,
                scale_field=scale_field,
                shape_log_variance=shape_log_variance,
                global_scale_log_variance=global_log_variance,
                canonical_camera_features=camera,
            )

        return _timed(timing_hook, "composition", compose_factorized)
