"""Bounded recovery policy for failed execution or verification."""

from __future__ import annotations

from dataclasses import dataclass

from jepa4d.planning.task_graph import Subgoal, SubgoalStatus


@dataclass(frozen=True, slots=True)
class ReplanningDecision:
    retry: bool
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReplanningPolicy:
    max_attempts_per_subgoal: int = 2
    max_replans: int = 4

    def __post_init__(self) -> None:
        if self.max_attempts_per_subgoal < 1 or self.max_replans < 0:
            raise ValueError("replanning limits must be non-negative")

    def decide(self, subgoal: Subgoal, reason: str, replans_so_far: int) -> ReplanningDecision:
        retry = subgoal.attempts < self.max_attempts_per_subgoal and replans_so_far < self.max_replans
        subgoal.status = SubgoalStatus.PENDING if retry else SubgoalStatus.FAILED
        subgoal.failure_reason = reason
        return ReplanningDecision(retry, "retry_subgoal" if retry else "abort", reason)
