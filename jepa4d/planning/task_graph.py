"""Symbolic task graph contract."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class TaskGraph:
    instruction: str
    subgoals: list[str] = field(default_factory=list)
