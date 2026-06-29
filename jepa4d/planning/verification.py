"""Observation-driven subgoal verification with explicit uncertainty gates."""

from __future__ import annotations

from dataclasses import dataclass

from jepa4d.planning.task_graph import Subgoal
from jepa4d.robotics.robot_interfaces import RobotObservation


@dataclass(frozen=True, slots=True)
class VerificationResult:
    satisfied: bool
    confidence: float
    condition: str
    reason: str

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    confidence_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")

    def verify(self, subgoal: Subgoal, observation: RobotObservation) -> VerificationResult:
        condition = subgoal.verification_condition
        confidence = float(observation.facts.get(condition, 0.0)) if condition else 1.0
        satisfied = confidence >= self.confidence_threshold
        reason = (
            "verified" if satisfied else ("condition_absent" if confidence == 0.0 else "uncertainty_above_threshold")
        )
        return VerificationResult(satisfied, confidence, condition, reason)
