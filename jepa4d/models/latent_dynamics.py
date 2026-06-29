"""Typed latent-dynamics interface; trainable dynamics are deferred to Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LatentDynamicsOutput:
    next_tokens: torch.Tensor
    uncertainty: torch.Tensor
    value: torch.Tensor


class ActionConditionedLatentDynamics:
    def __call__(
        self, tokens: torch.Tensor, actions: torch.Tensor, proprioception: torch.Tensor | None = None
    ) -> LatentDynamicsOutput:
        del proprioception
        action_delta = actions.float().mean(dim=-1, keepdim=True).unsqueeze(-1)
        return LatentDynamicsOutput(
            tokens + 0.0 * action_delta, torch.ones(tokens.shape[:-1]), torch.zeros(tokens.shape[0])
        )
