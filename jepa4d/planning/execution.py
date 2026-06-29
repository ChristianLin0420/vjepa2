"""Verified task-graph execution over a robot interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from jepa4d.planning.query_api import WorldModelQueryAPI
from jepa4d.planning.replanning import ReplanningPolicy
from jepa4d.planning.task_graph import SubgoalStatus, TaskGraph
from jepa4d.planning.verification import VerificationPolicy
from jepa4d.robotics.robot_interfaces import RobotAction, RobotInterface


@dataclass(frozen=True, slots=True)
class PlanningEvent:
    step: int
    event: str
    subgoal_id: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionTrace:
    instruction: str
    success: bool
    events: list[PlanningEvent]
    replans: int
    verification_actions: int
    failure_reason: str | None
    task_graph: TaskGraph

    def to_serializable(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "success": self.success,
            "events": [asdict(value) for value in self.events],
            "replans": self.replans,
            "verification_actions": self.verification_actions,
            "failure_reason": self.failure_reason,
            "task_graph": self.task_graph.to_serializable(),
        }


class VerifiedTaskPlanner:
    """Run ``propose → execute → observe → verify → replan`` deterministically."""

    def __init__(
        self,
        *,
        verification: VerificationPolicy | None = None,
        replanning: ReplanningPolicy | None = None,
        query_api: WorldModelQueryAPI | None = None,
    ) -> None:
        self.verification = verification or VerificationPolicy()
        self.replanning = replanning or ReplanningPolicy()
        self.query_api = query_api

    def execute(self, graph: TaskGraph, robot: RobotInterface) -> ExecutionTrace:
        events: list[PlanningEvent] = []
        replans = 0
        verification_actions = 0
        step = 0
        failure_reason: str | None = None
        while not graph.complete:
            ready = graph.ready()
            if not ready:
                failure_reason = failure_reason or "no_runnable_subgoal"
                break
            subgoal = ready[0]
            subgoal.status = SubgoalStatus.RUNNING
            subgoal.attempts += 1
            action = RobotAction(subgoal.action, subgoal.target, dict(subgoal.parameters))
            result = robot.execute(action)
            step += 1
            events.append(
                PlanningEvent(
                    step,
                    "execution",
                    subgoal.subgoal_id,
                    {"success": result.success, "reason": result.reason, "action": asdict(action)},
                )
            )
            if not result.success:
                decision = self.replanning.decide(subgoal, result.reason or "execution_failure", replans)
                events.append(
                    PlanningEvent(
                        step,
                        "failure_attribution",
                        subgoal.subgoal_id,
                        {"stage": "control", "reason": decision.reason},
                    )
                )
                if decision.retry:
                    replans += 1
                    events.append(PlanningEvent(step, "replan", subgoal.subgoal_id, {"action": decision.action}))
                    continue
                failure_reason = decision.reason
                break

            observation = robot.observe()
            verification_actions += 1
            verification = self.verification.verify(subgoal, observation)
            evidence = {
                "condition": verification.condition,
                "confidence": verification.confidence,
                "uncertainty": verification.uncertainty,
                "timestamp": observation.timestamp,
            }
            subgoal.evidence.append(evidence)
            events.append(
                PlanningEvent(
                    step,
                    "verification",
                    subgoal.subgoal_id,
                    {**evidence, "satisfied": verification.satisfied, "reason": verification.reason},
                )
            )
            if verification.satisfied:
                subgoal.status = SubgoalStatus.VERIFIED
                subgoal.failure_reason = None
                if self.query_api is not None:
                    self.query_api.mark_task_state(subgoal.subgoal_id, "verified", evidence)
                continue
            decision = self.replanning.decide(subgoal, verification.reason, replans)
            events.append(
                PlanningEvent(
                    step,
                    "failure_attribution",
                    subgoal.subgoal_id,
                    {"stage": "verification", "reason": verification.reason},
                )
            )
            if decision.retry:
                replans += 1
                events.append(PlanningEvent(step, "replan", subgoal.subgoal_id, {"action": decision.action}))
                continue
            failure_reason = decision.reason
            break
        return ExecutionTrace(
            graph.instruction, graph.complete, events, replans, verification_actions, failure_reason, graph
        )
