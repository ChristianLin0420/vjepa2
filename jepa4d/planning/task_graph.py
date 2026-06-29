"""Typed symbolic task graph with guarded state transitions and evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class SubgoalStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(slots=True)
class Subgoal:
    subgoal_id: str
    action: str
    target: str = ""
    depends_on: list[str] = field(default_factory=list)
    verification_condition: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    status: SubgoalStatus = SubgoalStatus.PENDING
    attempts: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None


@dataclass(slots=True)
class TaskGraph:
    instruction: str
    subgoals: list[Subgoal] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = [value.subgoal_id for value in self.subgoals]
        if len(ids) != len(set(ids)):
            raise ValueError("subgoal IDs must be unique")
        known: set[str] = set()
        for value in self.subgoals:
            if any(dependency not in known for dependency in value.depends_on):
                raise ValueError("dependencies must refer to earlier subgoals")
            known.add(value.subgoal_id)

    @property
    def complete(self) -> bool:
        return bool(self.subgoals) and all(value.status == SubgoalStatus.VERIFIED for value in self.subgoals)

    def ready(self) -> list[Subgoal]:
        verified = {value.subgoal_id for value in self.subgoals if value.status == SubgoalStatus.VERIFIED}
        return [
            value
            for value in self.subgoals
            if value.status == SubgoalStatus.PENDING and all(dependency in verified for dependency in value.depends_on)
        ]

    def get(self, subgoal_id: str) -> Subgoal:
        try:
            return next(value for value in self.subgoals if value.subgoal_id == subgoal_id)
        except StopIteration as error:
            raise KeyError(subgoal_id) from error

    def to_serializable(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "complete": self.complete,
            "subgoals": [asdict(value) for value in self.subgoals],
        }

    @classmethod
    def pick_and_place(cls, object_name: str, destination: str) -> TaskGraph:
        return cls(
            instruction=f"pick {object_name} and place it at {destination}",
            subgoals=[
                Subgoal("find-object", "search", object_name, verification_condition=f"visible:{object_name}"),
                Subgoal("pick-object", "pick", object_name, ["find-object"], f"holding:{object_name}"),
                Subgoal(
                    "place-object",
                    "place",
                    object_name,
                    ["pick-object"],
                    f"at:{object_name}:{destination}",
                    {"destination": destination},
                ),
            ],
        )
