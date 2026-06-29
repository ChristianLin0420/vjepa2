"""Robot execution contracts shared by mocks and optional hardware adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RobotAction:
    skill: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RobotObservation:
    timestamp: float
    facts: dict[str, float]
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    action: RobotAction
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RobotInterface(Protocol):
    def observe(self) -> RobotObservation: ...
    def execute(self, action: RobotAction) -> ExecutionResult: ...
