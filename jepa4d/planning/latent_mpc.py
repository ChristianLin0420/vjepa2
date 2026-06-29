"""Cross-entropy model-predictive control over action-conditioned latents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from jepa4d.models.latent_dynamics import ActionConditionedLatentDynamics, LatentRollout


@dataclass(frozen=True, slots=True)
class CEMConfig:
    horizon: int = 4
    population: int = 128
    elite_fraction: float = 0.1
    iterations: int = 4
    action_low: float = -1.0
    action_high: float = 1.0
    action_cost: float = 0.01
    uncertainty_cost: float = 1.0
    minimum_std: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.population < 2 or self.iterations < 1:
            raise ValueError("horizon, population, and iterations must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be smaller than action_high")


@dataclass(slots=True)
class MPCPlan:
    actions: torch.Tensor
    score: float
    predicted_uncertainty: float
    iterations: int

    @property
    def first_action(self) -> torch.Tensor:
        return self.actions[0]


class CEMPlanner:
    def __init__(self, action_dim: int, config: CEMConfig | None = None) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.config = config or CEMConfig()

    def _scores(
        self,
        rollout: LatentRollout,
        actions: torch.Tensor,
        objective: Callable[[LatentRollout], torch.Tensor] | None,
    ) -> torch.Tensor:
        score = rollout.value[:, -1] if objective is None else objective(rollout)
        if score.shape != (actions.shape[0],):
            raise ValueError("objective must return one score per candidate")
        return (
            score
            - self.config.uncertainty_cost * rollout.uncertainty.mean(dim=(1, 2))
            - self.config.action_cost * actions.square().mean(dim=(1, 2))
        )

    @torch.no_grad()
    def plan(
        self,
        tokens: torch.Tensor,
        dynamics: ActionConditionedLatentDynamics,
        *,
        proprioception: torch.Tensor | None = None,
        objective: Callable[[LatentRollout], torch.Tensor] | None = None,
    ) -> MPCPlan:
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError("CEM planning currently accepts one [1,N,C] belief at a time")
        cfg = self.config
        mean = torch.zeros((cfg.horizon, self.action_dim), device=tokens.device)
        std = torch.full_like(mean, (cfg.action_high - cfg.action_low) / 2)
        generator = torch.Generator(device=tokens.device).manual_seed(cfg.seed)
        elite_count = max(1, round(cfg.population * cfg.elite_fraction))
        best_actions, best_score, best_uncertainty = mean, float("-inf"), 1.0
        for _ in range(cfg.iterations):
            noise = torch.randn(
                (cfg.population, cfg.horizon, self.action_dim), generator=generator, device=tokens.device
            )
            candidates = (mean[None] + std[None] * noise).clamp(cfg.action_low, cfg.action_high)
            expanded_tokens = tokens.expand(cfg.population, -1, -1)
            expanded_proprio = None if proprioception is None else proprioception.expand(cfg.population, -1)
            rollout = dynamics.rollout(expanded_tokens, candidates, expanded_proprio)
            scores = self._scores(rollout, candidates, objective)
            elite_indices = scores.topk(elite_count).indices
            elite = candidates[elite_indices]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, unbiased=False).clamp_min(cfg.minimum_std)
            index = int(scores.argmax())
            if float(scores[index]) > best_score:
                best_actions = candidates[index].clone()
                best_score = float(scores[index])
                best_uncertainty = float(rollout.uncertainty[index].mean())
        return MPCPlan(best_actions, best_score, best_uncertainty, cfg.iterations)
