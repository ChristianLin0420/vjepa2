"""Deterministic verified-planning recovery benchmark."""

from __future__ import annotations

from typing import Any

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.planning.execution import VerifiedTaskPlanner
from jepa4d.planning.task_graph import TaskGraph
from jepa4d.planning.verification import VerificationPolicy
from jepa4d.robotics.mock_robot import MockRobot


class PlanningSmokeBenchmark(BenchmarkAdapter):
    name = "planning-smoke"
    stage = "planning"
    supports_single_image = True
    supports_multiview = True
    supports_video = True

    def run(self, model_or_system: Any = None, split: str = "tiny") -> list[dict[str, Any]]:
        del model_or_system, split
        robot = MockRobot(fail_once={"pick:mug"})
        graph = TaskGraph.pick_and_place("mug", "table")
        trace = VerifiedTaskPlanner(verification=VerificationPolicy(0.8)).execute(graph, robot)
        return [trace.to_serializable()]

    def compute_metrics(self, predictions: Any, ground_truth: object = None) -> dict[str, float]:
        del ground_truth
        trace = predictions[0]
        events = trace["events"]
        verified = sum(event["event"] == "verification" and event["evidence"]["satisfied"] for event in events)
        attribution = sum(event["event"] == "failure_attribution" for event in events)
        return {
            "task_success": float(trace["success"]),
            "subgoal_progress": verified / max(len(trace["task_graph"]["subgoals"]), 1),
            "failure_attribution": float(attribution > 0),
            "recovery_success": float(trace["success"] and trace["replans"] > 0),
            "replans": float(trace["replans"]),
            "verification_actions": float(trace["verification_actions"]),
        }

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "status": "ready",
            "quality_claim": "deterministic-simulation-only",
        }
