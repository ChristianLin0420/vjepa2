"""Deterministic stateful robot used to gate verified planning offline."""

from __future__ import annotations

from dataclasses import dataclass, field

from jepa4d.robotics.robot_interfaces import ExecutionResult, RobotAction, RobotObservation


@dataclass
class MockRobot:
    objects: dict[str, str] = field(default_factory=lambda: {"mug": "counter"})
    confidence: float = 0.95
    fail_once: set[str] = field(default_factory=set)
    holding: str | None = None
    timestamp: float = 0.0
    action_history: list[RobotAction] = field(default_factory=list)
    _failed: set[str] = field(default_factory=set)

    def execute(self, action: RobotAction) -> ExecutionResult:
        self.timestamp += 1.0
        self.action_history.append(action)
        key = f"{action.skill}:{action.target}"
        if key in self.fail_once and key not in self._failed:
            self._failed.add(key)
            return ExecutionResult(False, action, "injected_control_failure")
        if action.skill in {"observe", "search", "navigate"}:
            success = action.skill != "search" or action.target in self.objects
            return ExecutionResult(success, action, "" if success else "object_not_found")
        if action.skill == "pick":
            if action.target not in self.objects:
                return ExecutionResult(False, action, "object_not_found")
            if self.holding is not None:
                return ExecutionResult(False, action, "gripper_occupied")
            self.holding = action.target
            return ExecutionResult(True, action)
        if action.skill == "place":
            if self.holding != action.target:
                return ExecutionResult(False, action, "target_not_held")
            destination = str(action.parameters.get("destination", "unknown"))
            self.objects[action.target] = destination
            self.holding = None
            return ExecutionResult(True, action)
        return ExecutionResult(False, action, "unsupported_skill")

    def observe(self) -> RobotObservation:
        facts = {f"visible:{name}": self.confidence for name in self.objects}
        if self.holding is not None:
            facts[f"holding:{self.holding}"] = self.confidence
        facts.update(
            {
                f"at:{name}:{location}": self.confidence
                for name, location in self.objects.items()
                if name != self.holding
            }
        )
        return RobotObservation(self.timestamp, facts, {"holding": self.holding, "objects": dict(self.objects)})
