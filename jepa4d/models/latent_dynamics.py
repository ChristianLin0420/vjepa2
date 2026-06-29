"""Action-conditioned latent dynamics with deterministic and trainable backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(slots=True)
class LatentDynamicsOutput:
    next_tokens: torch.Tensor
    uncertainty: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class LatentRollout:
    token_trajectory: torch.Tensor
    uncertainty: torch.Tensor
    value: torch.Tensor


class ActionConditionedLatentDynamics(nn.Module):
    """Predict one latent step without exposing latents to the symbolic planner.

    ``deterministic`` is the offline contract backend. ``learned`` provides a
    compact residual MLP boundary for V-JEPA 2-AC/JEPA-WM distillation; it is
    intentionally not represented as a pretrained dynamics model.
    """

    def __init__(
        self,
        token_dim: int | None = None,
        action_dim: int | None = None,
        proprioception_dim: int = 0,
        hidden_dim: int = 256,
        *,
        backend: Literal["deterministic", "learned"] = "deterministic",
        uncertainty_floor: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 <= uncertainty_floor <= 1.0:
            raise ValueError("uncertainty_floor must be in [0, 1]")
        self.backend = backend
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.proprioception_dim = proprioception_dim
        self.uncertainty_floor = uncertainty_floor
        if backend == "learned":
            if token_dim is None or action_dim is None:
                raise ValueError("learned dynamics require token_dim and action_dim")
            input_dim = token_dim + action_dim + proprioception_dim
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
            )
            self.delta_head = nn.Linear(hidden_dim, token_dim)
            self.uncertainty_head = nn.Linear(hidden_dim, 1)
            self.value_head = nn.Linear(hidden_dim, 1)
        elif backend != "deterministic":
            raise ValueError(f"unknown dynamics backend: {backend}")

    @staticmethod
    def _condition(value: torch.Tensor, token_count: int) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("actions and proprioception must have shape [B,D]")
        return value[:, None, :].expand(-1, token_count, -1)

    def forward(
        self, tokens: torch.Tensor, actions: torch.Tensor, proprioception: torch.Tensor | None = None
    ) -> LatentDynamicsOutput:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,C]")
        if actions.ndim != 2 or actions.shape[0] != tokens.shape[0]:
            raise ValueError("actions must have shape [B,A] and match the token batch")
        if not torch.isfinite(tokens).all() or not torch.isfinite(actions).all():
            raise ValueError("dynamics inputs must be finite")
        if self.backend == "deterministic":
            # A transparent control-sensitive transition for tests and planning
            # integration. It is not a learned prediction-quality baseline.
            action_signal = actions.float().mean(dim=-1, keepdim=True)
            basis = torch.linspace(-1.0, 1.0, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
            delta = 0.05 * action_signal[:, None, :] * basis[None, None, :]
            if proprioception is not None:
                if proprioception.ndim != 2 or proprioception.shape[0] != tokens.shape[0]:
                    raise ValueError("proprioception must have shape [B,P]")
                delta = delta + 0.01 * proprioception.float().mean(dim=-1)[:, None, None]
            next_tokens = tokens + delta
            uncertainty = (
                (self.uncertainty_floor + 0.05 * actions.float().square().mean(dim=-1).sqrt())
                .clamp(max=1.0)[:, None]
                .expand(-1, tokens.shape[1])
            )
            value = -next_tokens.float().square().mean(dim=(1, 2))
            return LatentDynamicsOutput(next_tokens, uncertainty, value)

        assert self.token_dim is not None and self.action_dim is not None
        if tokens.shape[-1] != self.token_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("input dimensions do not match learned dynamics configuration")
        conditions = [self._condition(actions.float(), tokens.shape[1])]
        if self.proprioception_dim:
            if proprioception is None or proprioception.shape != (tokens.shape[0], self.proprioception_dim):
                raise ValueError(f"proprioception must have shape [B,{self.proprioception_dim}]")
            conditions.append(self._condition(proprioception.float(), tokens.shape[1]))
        hidden = self.trunk(torch.cat([tokens.float(), *conditions], dim=-1))
        next_tokens = tokens + self.delta_head(hidden).to(tokens.dtype)
        uncertainty = (
            self.uncertainty_floor + torch.nn.functional.softplus(self.uncertainty_head(hidden).squeeze(-1))
        ).clamp(max=1.0)
        value = self.value_head(hidden.mean(dim=1)).squeeze(-1)
        return LatentDynamicsOutput(next_tokens, uncertainty, value)

    def rollout(
        self, tokens: torch.Tensor, action_sequence: torch.Tensor, proprioception: torch.Tensor | None = None
    ) -> LatentRollout:
        if action_sequence.ndim != 3 or action_sequence.shape[0] != tokens.shape[0]:
            raise ValueError("action_sequence must have shape [B,H,A]")
        current = tokens
        states, uncertainties, values = [tokens], [], []
        for step in range(action_sequence.shape[1]):
            output = self(current, action_sequence[:, step], proprioception)
            current = output.next_tokens
            states.append(current)
            uncertainties.append(output.uncertainty)
            values.append(output.value)
        return LatentRollout(
            token_trajectory=torch.stack(states, dim=1),
            uncertainty=torch.stack(uncertainties, dim=1),
            value=torch.stack(values, dim=1),
        )
