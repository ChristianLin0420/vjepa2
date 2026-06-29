"""Unified view/time identity encoding for heterogeneous RGB observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from jepa4d.data.rgb_input import collate_rgb_inputs
from jepa4d.data.schemas import RGBInputBatch


@dataclass(slots=True)
class ViewSetEncoding:
    images: torch.Tensor
    identity_tokens: torch.Tensor
    valid_mask: torch.Tensor
    mode_index: torch.Tensor


class ViewSetTokenizer(nn.Module):
    """Attach learnable mode/view identity and continuous temporal encodings."""

    MODES = ("single_image", "multi_view", "video", "multiview_video")

    def __init__(self, embed_dim: int = 128, max_views: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_views = max_views
        self.mode_embedding = nn.Embedding(len(self.MODES), embed_dim)
        self.view_embedding = nn.Embedding(max_views, embed_dim)
        self.camera_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    @staticmethod
    def _time_encoding(timestamps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        frequency = torch.exp(
            torch.arange(half, device=timestamps.device, dtype=timestamps.dtype)
            * (-torch.log(torch.tensor(10_000.0, device=timestamps.device, dtype=timestamps.dtype)) / max(half - 1, 1))
        )
        angles = timestamps.unsqueeze(-1) * frequency
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return torch.nn.functional.pad(encoded, (0, dim - encoded.shape[-1]))

    def forward(self, batch: RGBInputBatch) -> ViewSetEncoding:
        batch_size, views, steps = batch.images.shape[:3]
        if views > self.max_views:
            raise ValueError(f"received {views} views, maximum is {self.max_views}")
        mode_id = self.MODES.index(batch.mode)
        mode_index = torch.full((batch_size,), mode_id, dtype=torch.long, device=batch.images.device)
        mode = self.mode_embedding(mode_index)[:, None, None, :]
        view_ids = torch.arange(views, device=batch.images.device)
        view = self.view_embedding(view_ids)[None, :, None, :]
        time = self._time_encoding(batch.timestamps.to(batch.images.dtype), self.embed_dim)
        identity = mode + view + time
        identity = identity * batch.valid_mask.unsqueeze(-1)
        return ViewSetEncoding(batch.images, identity, batch.valid_mask, mode_index)

    collate_rgb_inputs = staticmethod(collate_rgb_inputs)
