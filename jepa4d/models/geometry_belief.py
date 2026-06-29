"""Metric-geometry belief adapter with deterministic and optional VGGT backends.

The module intentionally calls its output a *belief*: even when a dense depth or
point map is available, scale, pose, and reconstruction confidence remain
explicit. In particular, an uncalibrated single image never receives high scale
confidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from jepa4d.data.camera import default_pinhole_intrinsics
from jepa4d.data.schemas import RGBInputBatch

GeometryBackend = Literal["mock", "vggt"]


def _tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    finite = torch.isfinite(value)
    finite_values = value[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite_fraction": finite.float().mean().item(),
        "mean": finite_values.float().mean().item() if finite_values.numel() else None,
        "std": finite_values.float().std().item() if finite_values.numel() > 1 else 0.0,
    }


@dataclass(slots=True)
class GeometryBelief:
    """Probabilistic camera and scene geometry aligned to `[B,V,T]`."""

    camera_extrinsics: torch.Tensor | None
    camera_intrinsics: torch.Tensor | None
    depth_mean: torch.Tensor | None
    depth_logvar: torch.Tensor | None
    pointmap_mean: torch.Tensor | None
    pointmap_logvar: torch.Tensor | None
    tracks_2d: torch.Tensor | None
    tracks_3d: torch.Tensor | None
    scale_confidence: torch.Tensor
    pose_confidence: torch.Tensor
    reconstruction_confidence: torch.Tensor
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch = self.scale_confidence.shape[0]
        for name in ("pose_confidence", "reconstruction_confidence"):
            value = getattr(self, name)
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [B], got {tuple(value.shape)}")
        for name in ("scale_confidence", "pose_confidence", "reconstruction_confidence"):
            value = getattr(self, name)
            if not torch.all((value >= 0) & (value <= 1)):
                raise ValueError(f"{name} must stay within [0,1]")
        if (
            self.depth_mean is not None
            and self.depth_logvar is not None
            and self.depth_mean.shape != self.depth_logvar.shape
        ):
            raise ValueError("depth mean and log-variance shapes must match")
        if self.pointmap_mean is not None and self.pointmap_mean.shape[-1] != 3:
            raise ValueError("point maps must end in XYZ")

    def to_serializable(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "camera_extrinsics": _tensor_summary(self.camera_extrinsics),
            "camera_intrinsics": _tensor_summary(self.camera_intrinsics),
            "depth_mean": _tensor_summary(self.depth_mean),
            "depth_logvar": _tensor_summary(self.depth_logvar),
            "pointmap_mean": _tensor_summary(self.pointmap_mean),
            "pointmap_logvar": _tensor_summary(self.pointmap_logvar),
            "tracks_2d": _tensor_summary(self.tracks_2d),
            "tracks_3d": _tensor_summary(self.tracks_3d),
            "scale_confidence": self.scale_confidence.detach().cpu().tolist(),
            "pose_confidence": self.pose_confidence.detach().cpu().tolist(),
            "reconstruction_confidence": self.reconstruction_confidence.detach().cpu().tolist(),
            "metadata": self.metadata,
        }

    def save_npz(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, np.ndarray] = {}
        for name in (
            "camera_extrinsics",
            "camera_intrinsics",
            "depth_mean",
            "depth_logvar",
            "pointmap_mean",
            "pointmap_logvar",
            "tracks_2d",
            "tracks_3d",
            "scale_confidence",
            "pose_confidence",
            "reconstruction_confidence",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = value.detach().cpu().numpy()
        values["mode"] = np.asarray(self.mode)
        np.savez_compressed(target, **values)  # type: ignore[arg-type]
        return target


def _confidence_to_probability(confidence: torch.Tensor) -> torch.Tensor:
    """VGGT confidences are positive scores, not calibrated probabilities."""
    confidence = confidence.float().clamp_min(0)
    return (confidence / (confidence + 1.0)).clamp(1e-4, 1 - 1e-4)


def _unproject(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject `[B,S,H,W]` depth into world-frame points."""
    batch, sequence, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    fx = intrinsics[..., 0, 0, None, None]
    fy = intrinsics[..., 1, 1, None, None]
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    camera_points = torch.stack(((x - cx) * depth / fx, (y - cy) * depth / fy, depth), dim=-1)
    homogeneous = torch.cat((camera_points, torch.ones_like(camera_points[..., :1])), dim=-1)
    camera_from_world = extrinsics
    world_from_camera = torch.linalg.inv(camera_from_world)
    world = torch.einsum("bsij,bshwj->bshwi", world_from_camera, homogeneous)
    return world[..., :3]


class GeometryBeliefHead(nn.Module):
    """Geometry adapter supporting a deterministic mock and official VGGT.

    The mock backend is intended for CI, API development, and uncertainty tests;
    its geometry is not a learned prediction. The VGGT backend lazily imports the
    official package and loads `facebook/VGGT-1B` unless a local model is passed.
    """

    def __init__(
        self,
        *,
        backend: str = "mock",
        device: str | torch.device = "cpu",
        model_id: str = "facebook/VGGT-1B",
        model: nn.Module | None = None,
        output_size: int = 112,
        query_grid_size: int = 8,
        known_scale_prior: bool = False,
    ) -> None:
        super().__init__()
        if backend not in {"mock", "vggt"}:
            raise ValueError(f"unknown geometry backend: {backend}")
        self.backend = backend
        self.device_name = str(device)
        self.model_id = model_id
        self.output_size = output_size
        self.query_grid_size = query_grid_size
        self.known_scale_prior = known_scale_prior
        self.model = model
        self._load_seconds = 0.0
        if backend == "vggt" and self.model is None:
            started = time.perf_counter()
            try:
                from vggt.models.vggt import VGGT
            except ImportError as error:
                raise ImportError(
                    "VGGT backend requested but the official package is unavailable. "
                    "Install with `pip install 'git+https://github.com/facebookresearch/vggt.git'`."
                ) from error
            self.model = VGGT.from_pretrained(model_id)
            self._load_seconds = time.perf_counter() - started
        if self.model is not None:
            self.model.to(self.device_name).eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def forward(self, batch: RGBInputBatch) -> GeometryBelief:
        started = time.perf_counter()
        belief = self._forward_mock(batch) if self.backend == "mock" else self._forward_vggt(batch)
        belief.metadata.update(
            {
                "backend": self.backend,
                "model_id": "deterministic_geometry_mock" if self.backend == "mock" else self.model_id,
                "runtime_seconds": time.perf_counter() - started,
                "model_load_seconds": self._load_seconds,
                "input_mode": batch.mode,
                "input_shape": list(batch.images.shape),
                "known_intrinsics": batch.intrinsics is not None,
                "known_extrinsics": batch.extrinsics is not None,
                "known_scale_prior": self.known_scale_prior,
                "confidence_semantics": "heuristic belief confidence; benchmark calibration required",
            }
        )
        return belief

    def _belief_confidences(
        self, batch: RGBInputBatch, dense_confidence: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, views, steps = batch.images.shape[:3]
        observations = views * steps
        calibrated = batch.intrinsics is not None
        scale = 0.08 + 0.08 * min(observations - 1, 4)
        if calibrated:
            scale += 0.15
        if self.known_scale_prior:
            scale += 0.35
        if observations == 1 and not calibrated and not self.known_scale_prior:
            scale = min(scale, 0.15)
        pose = 0.05 if observations == 1 else min(0.8, 0.2 + 0.1 * observations)
        if batch.extrinsics is not None:
            pose = 0.95
        reconstruction = min(0.85, 0.15 + 0.08 * observations)
        if dense_confidence is not None:
            reconstruction *= float(dense_confidence.float().mean().clamp(0, 1))
        output_device = dense_confidence.device if dense_confidence is not None else batch.images.device
        return (
            torch.full((batch_size,), scale, device=output_device).clamp(0, 1),
            torch.full((batch_size,), pose, device=output_device).clamp(0, 1),
            torch.full((batch_size,), reconstruction, device=output_device).clamp(0, 1),
        )

    def _forward_mock(self, batch: RGBInputBatch) -> GeometryBelief:
        images = batch.images.to(self.device_name)
        batch_size, views, steps, _, input_height, input_width = images.shape
        sequence = views * steps
        flat = images.reshape(batch_size * sequence, 3, input_height, input_width)
        resized = F.interpolate(flat, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        luminance = 0.299 * resized[:, 0] + 0.587 * resized[:, 1] + 0.114 * resized[:, 2]
        y = torch.linspace(0, 1, self.output_size, device=images.device).view(1, self.output_size, 1)
        depth = (0.8 + 1.7 * (1 - luminance) + 0.35 * y).reshape(
            batch_size, views, steps, self.output_size, self.output_size
        )
        border_x = torch.linspace(-1, 1, self.output_size, device=images.device).abs().view(1, 1, 1, 1, -1)
        border_y = torch.linspace(-1, 1, self.output_size, device=images.device).abs().view(1, 1, 1, -1, 1)
        epistemic = 1.2 if batch.mode == "single_image" and batch.intrinsics is None else 0.45
        depth_logvar = torch.full_like(depth, epistemic) + 0.25 * torch.maximum(border_x, border_y)
        if batch.intrinsics is None:
            intrinsics_view = default_pinhole_intrinsics(
                batch_size, views, self.output_size, self.output_size, focal_ratio=1.2
            ).to(images.device)
        else:
            intrinsics_view = batch.intrinsics.to(images.device).clone()
            intrinsics_view[..., 0, :] *= self.output_size / input_width
            intrinsics_view[..., 1, :] *= self.output_size / input_height
        intrinsics = intrinsics_view[:, :, None].expand(-1, -1, steps, -1, -1).contiguous()
        if batch.extrinsics is None:
            extrinsics = torch.eye(4, device=images.device).view(1, 1, 1, 4, 4).repeat(batch_size, views, steps, 1, 1)
            view_offsets = torch.arange(views, device=images.device, dtype=images.dtype).view(1, views, 1)
            time_offsets = torch.arange(steps, device=images.device, dtype=images.dtype).view(1, 1, steps)
            extrinsics[..., 0, 3] = -0.12 * view_offsets
            extrinsics[..., 2, 3] = -0.03 * time_offsets
        else:
            extrinsics = batch.extrinsics.to(images.device)[:, :, None].expand(-1, -1, steps, -1, -1).contiguous()
        pointmap = _unproject(
            depth.reshape(batch_size, sequence, self.output_size, self.output_size),
            intrinsics.reshape(batch_size, sequence, 3, 3),
            extrinsics.reshape(batch_size, sequence, 4, 4),
        ).reshape(batch_size, views, steps, self.output_size, self.output_size, 3)
        point_logvar = depth_logvar.unsqueeze(-1).expand_as(pointmap)
        tracks_2d, tracks_3d = self._make_tracks(pointmap)
        scale, pose, reconstruction = self._belief_confidences(batch, torch.exp(-depth_logvar).clamp(0, 1))
        return GeometryBelief(
            camera_extrinsics=extrinsics,
            camera_intrinsics=intrinsics,
            depth_mean=depth,
            depth_logvar=depth_logvar,
            pointmap_mean=pointmap,
            pointmap_logvar=point_logvar,
            tracks_2d=tracks_2d,
            tracks_3d=tracks_3d,
            scale_confidence=scale,
            pose_confidence=pose,
            reconstruction_confidence=reconstruction,
            mode=f"mock_{batch.mode}",
            metadata={"synthetic_geometry": True, "output_resolution": [self.output_size, self.output_size]},
        )

    def _query_points(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        margin_x, margin_y = width / (self.query_grid_size + 1), height / (self.query_grid_size + 1)
        y, x = torch.meshgrid(
            torch.arange(1, self.query_grid_size + 1, device=device) * margin_y,
            torch.arange(1, self.query_grid_size + 1, device=device) * margin_x,
            indexing="ij",
        )
        return torch.stack((x.flatten(), y.flatten()), dim=-1)

    def _make_tracks(self, pointmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, steps, height, width, _ = pointmap.shape
        sequence = views * steps
        points = self._query_points(height, width, pointmap.device)
        x = points[:, 0].round().long().clamp(0, width - 1)
        y = points[:, 1].round().long().clamp(0, height - 1)
        tracks_2d = points.view(1, 1, -1, 2).expand(batch, sequence, -1, -1).clone()
        flattened = pointmap.reshape(batch, sequence, height, width, 3)
        tracks_3d = flattened[:, :, y, x]
        return tracks_2d, tracks_3d

    def _forward_vggt(self, batch: RGBInputBatch) -> GeometryBelief:
        assert self.model is not None
        batch_size, views, steps, _, _, _ = batch.images.shape
        sequence = views * steps
        images = batch.images.to(self.device_name).reshape(batch_size, sequence, 3, *batch.images.shape[-2:])
        images = F.interpolate(images.flatten(0, 1), size=(518, 518), mode="bilinear", align_corners=False).reshape(
            batch_size, sequence, 3, 518, 518
        )
        query = self._query_points(518, 518, images.device).unsqueeze(0).expand(batch_size, -1, -1)
        with torch.inference_mode():
            predictions = self.model(images, query_points=query)
        try:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ImportError as error:  # pragma: no cover - requires optional dependency
            raise ImportError("installed VGGT package lacks pose conversion utilities") from error
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        homogeneous_extrinsics = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).view(1, 1, 4, 4)
        homogeneous_extrinsics = homogeneous_extrinsics.repeat(batch_size, sequence, 1, 1)
        homogeneous_extrinsics[..., :3, :4] = extrinsics
        depth = predictions["depth"].squeeze(-1).float()
        depth_probability = _confidence_to_probability(predictions["depth_conf"])
        points = predictions["world_points"].float()
        point_probability = _confidence_to_probability(predictions["world_points_conf"])
        depth_logvar = -torch.log(depth_probability)
        point_logvar = -torch.log(point_probability).unsqueeze(-1).expand_as(points)
        tracks_2d = predictions.get("track")
        tracks_3d = None
        if tracks_2d is not None:
            x = tracks_2d[..., 0].round().long().clamp(0, points.shape[-2] - 1)
            y = tracks_2d[..., 1].round().long().clamp(0, points.shape[-3] - 1)
            point_batches = []
            for batch_index in range(batch_size):
                sequence_points = []
                for sequence_index in range(sequence):
                    sequence_points.append(
                        points[
                            batch_index, sequence_index, y[batch_index, sequence_index], x[batch_index, sequence_index]
                        ]
                    )
                point_batches.append(torch.stack(sequence_points))
            tracks_3d = torch.stack(point_batches)
        scale, pose, reconstruction = self._belief_confidences(batch, (depth_probability + point_probability) / 2)

        def reshape(value: torch.Tensor, *tail: int) -> torch.Tensor:
            return value.reshape(batch_size, views, steps, *tail)

        return GeometryBelief(
            camera_extrinsics=reshape(homogeneous_extrinsics.float(), 4, 4),
            camera_intrinsics=reshape(intrinsics.float(), 3, 3),
            depth_mean=reshape(depth, *depth.shape[-2:]),
            depth_logvar=reshape(depth_logvar, *depth_logvar.shape[-2:]),
            pointmap_mean=reshape(points, *points.shape[-3:]),
            pointmap_logvar=reshape(point_logvar, *point_logvar.shape[-3:]),
            tracks_2d=tracks_2d,
            tracks_3d=tracks_3d,
            scale_confidence=scale,
            pose_confidence=pose,
            reconstruction_confidence=reconstruction,
            mode=f"vggt_{batch.mode}",
            metadata={"synthetic_geometry": False, "output_resolution": list(depth.shape[-2:])},
        )
