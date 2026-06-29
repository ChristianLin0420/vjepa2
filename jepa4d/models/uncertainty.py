"""Uncertainty summaries shared by adapters."""

from __future__ import annotations

import torch


def feature_dispersion(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.float().var(dim=-2).mean(dim=-1)
