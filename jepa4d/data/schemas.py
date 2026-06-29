"""Tensor-aware, serializable contracts shared by JEPA-4D modules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

InputMode = Literal["single_image", "multi_view", "video", "multiview_video"]


def _tensor_metadata(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


@dataclass(slots=True)
class RobotState:
    """Robot state aligned with an RGB observation."""

    timestamp: float
    joint_positions: torch.Tensor | None = None
    joint_velocities: torch.Tensor | None = None
    base_pose: torch.Tensor | None = None
    frame_id: str = "base_link"

    def to_serializable(self, include_tensors: bool = False) -> dict[str, Any]:
        def encode(value: torch.Tensor | None) -> Any:
            if value is None:
                return None
            return value.detach().cpu().tolist() if include_tensors else _tensor_metadata(value)

        return {
            "timestamp": self.timestamp,
            "joint_positions": encode(self.joint_positions),
            "joint_velocities": encode(self.joint_velocities),
            "base_pose": encode(self.base_pose),
            "frame_id": self.frame_id,
        }


@dataclass(slots=True)
class RGBInputBatch:
    """Canonical RGB input with explicit batch, view, and time dimensions."""

    images: torch.Tensor
    timestamps: torch.Tensor
    camera_ids: list[list[str]]
    mode: InputMode
    intrinsics: torch.Tensor | None = None
    extrinsics: torch.Tensor | None = None
    robot_state: RobotState | None = None
    action_history: torch.Tensor | None = None
    valid_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))
    source_refs: list[list[list[str]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.images.ndim != 6:
            raise ValueError(f"images must have shape [B,V,T,3,H,W], got {tuple(self.images.shape)}")
        batch, views, steps, channels, _, _ = self.images.shape
        if channels != 3:
            raise ValueError(f"RGB inputs require 3 channels, got {channels}")
        if tuple(self.timestamps.shape) != (batch, views, steps):
            raise ValueError("timestamps must have shape [B,V,T]")
        if len(self.camera_ids) != batch or any(len(ids) != views for ids in self.camera_ids):
            raise ValueError("camera_ids must contain one camera ID per batch/view")
        expected_mode: InputMode
        if views == 1 and steps == 1:
            expected_mode = "single_image"
        elif views > 1 and steps == 1:
            expected_mode = "multi_view"
        elif views == 1:
            expected_mode = "video"
        else:
            expected_mode = "multiview_video"
        if self.mode != expected_mode:
            raise ValueError(f"mode={self.mode!r} disagrees with V={views}, T={steps}; expected {expected_mode!r}")
        if self.intrinsics is not None and tuple(self.intrinsics.shape) != (batch, views, 3, 3):
            raise ValueError("intrinsics must have shape [B,V,3,3]")
        if self.extrinsics is not None and tuple(self.extrinsics.shape) != (batch, views, 4, 4):
            raise ValueError("extrinsics must have shape [B,V,4,4]")
        if self.valid_mask.numel() == 0:
            self.valid_mask = torch.ones((batch, views, steps), dtype=torch.bool, device=self.images.device)
        elif tuple(self.valid_mask.shape) != (batch, views, steps):
            raise ValueError("valid_mask must have shape [B,V,T]")

    @property
    def shape(self) -> tuple[int, int, int, int, int, int]:
        return tuple(self.images.shape)  # type: ignore[return-value]

    def to(self, device: torch.device | str) -> RGBInputBatch:
        return RGBInputBatch(
            images=self.images.to(device),
            timestamps=self.timestamps.to(device),
            camera_ids=self.camera_ids,
            mode=self.mode,
            intrinsics=None if self.intrinsics is None else self.intrinsics.to(device),
            extrinsics=None if self.extrinsics is None else self.extrinsics.to(device),
            robot_state=self.robot_state,
            action_history=None if self.action_history is None else self.action_history.to(device),
            valid_mask=self.valid_mask.to(device),
            source_refs=self.source_refs,
        )

    def to_serializable(self, include_tensors: bool = False) -> dict[str, Any]:
        def tensor(value: torch.Tensor | None) -> Any:
            if value is None:
                return None
            return value.detach().cpu().tolist() if include_tensors else _tensor_metadata(value)

        return {
            "images": tensor(self.images),
            "timestamps": tensor(self.timestamps),
            "camera_ids": self.camera_ids,
            "mode": self.mode,
            "intrinsics": tensor(self.intrinsics),
            "extrinsics": tensor(self.extrinsics),
            "robot_state": None if self.robot_state is None else self.robot_state.to_serializable(include_tensors),
            "action_history": tensor(self.action_history),
            "valid_mask": tensor(self.valid_mask),
            "source_refs": self.source_refs,
        }


@dataclass(slots=True)
class JEPATokenBundle:
    """Dense and pooled V-JEPA features with preserved view/time identity."""

    dense_tokens: torch.Tensor
    global_tokens: torch.Tensor
    layer_tokens: dict[int, torch.Tensor]
    patch_grid: tuple[int, int]
    feature_scale: int
    modality: Literal["image", "video"]
    valid_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dense_tokens.ndim != 5:
            raise ValueError("dense_tokens must have shape [B,V,T,N,C]")
        batch, views, steps, _, channels = self.dense_tokens.shape
        if tuple(self.global_tokens.shape) != (batch, views, steps, channels):
            raise ValueError("global_tokens must have shape [B,V,T,C]")
        if tuple(self.valid_mask.shape) != (batch, views, steps):
            raise ValueError("valid_mask must have shape [B,V,T]")
        expected = self.patch_grid[0] * self.patch_grid[1]
        if self.dense_tokens.shape[-2] != expected:
            raise ValueError(f"token count {self.dense_tokens.shape[-2]} does not match patch grid {self.patch_grid}")

    def to_serializable(self, include_tensors: bool = False) -> dict[str, Any]:
        def encode(value: torch.Tensor) -> Any:
            return value.detach().cpu().tolist() if include_tensors else _tensor_metadata(value)

        return {
            "dense_tokens": encode(self.dense_tokens),
            "global_tokens": encode(self.global_tokens),
            "layer_tokens": {str(k): encode(v) for k, v in self.layer_tokens.items()},
            "patch_grid": list(self.patch_grid),
            "feature_scale": self.feature_scale,
            "modality": self.modality,
            "valid_mask": encode(self.valid_mask),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dense_tokens": self.dense_tokens.cpu(),
                "global_tokens": self.global_tokens.cpu(),
                "layer_tokens": {k: value.cpu() for k, value in self.layer_tokens.items()},
                "patch_grid": self.patch_grid,
                "feature_scale": self.feature_scale,
                "modality": self.modality,
                "valid_mask": self.valid_mask.cpu(),
                "metadata": self.metadata,
            },
            target,
        )
        return target

    def write_metadata(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_serializable(), indent=2) + "\n")
        return target
