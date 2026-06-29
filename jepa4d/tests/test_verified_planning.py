import pytest

from jepa4d.benchmarks.planning.smoke import PlanningSmokeBenchmark
from jepa4d.planning.execution import VerifiedTaskPlanner
from jepa4d.planning.task_graph import SubgoalStatus, TaskGraph
from jepa4d.planning.verification import VerificationPolicy
from jepa4d.robotics.mock_robot import MockRobot
from jepa4d.visualization.observability import ExperimentLogger


def test_task_graph_rejects_forward_dependencies() -> None:
    graph = TaskGraph.pick_and_place("mug", "table")
    assert [item.subgoal_id for item in graph.ready()] == ["find-object"]
    with pytest.raises(ValueError):
        TaskGraph("bad", [graph.subgoals[1], graph.subgoals[0]])


def test_verified_planner_attributes_failure_and_recovers() -> None:
    robot = MockRobot(fail_once={"pick:mug"})
    trace = VerifiedTaskPlanner().execute(TaskGraph.pick_and_place("mug", "table"), robot)
    assert trace.success
    assert trace.replans == 1
    assert trace.verification_actions == 3
    assert all(item.status == SubgoalStatus.VERIFIED for item in trace.task_graph.subgoals)
    assert any(event.event == "failure_attribution" and event.evidence["stage"] == "control" for event in trace.events)
    assert robot.objects["mug"] == "table"


def test_safe_threshold_prevents_false_verification() -> None:
    robot = MockRobot(confidence=0.6)
    trace = VerifiedTaskPlanner(verification=VerificationPolicy(0.8)).execute(
        TaskGraph.pick_and_place("mug", "table"), robot
    )
    assert not trace.success
    assert trace.task_graph.subgoals[0].status == SubgoalStatus.FAILED
    assert trace.task_graph.subgoals[0].failure_reason == "uncertainty_above_threshold"
    assert all(event.evidence.get("confidence", 1.0) < 0.8 for event in trace.events if event.event == "verification")


def test_planning_smoke_gate() -> None:
    benchmark = PlanningSmokeBenchmark()
    metrics = benchmark.compute_metrics(benchmark.run())
    assert metrics["task_success"] == 1.0
    assert metrics["subgoal_progress"] == 1.0
    assert metrics["failure_attribution"] == 1.0
    assert metrics["recovery_success"] == 1.0


def test_disabled_planning_logger_accepts_trace() -> None:
    trace = VerifiedTaskPlanner().execute(TaskGraph.pick_and_place("mug", "table"), MockRobot())
    ExperimentLogger(enabled=False).log_planning_trace(trace, mpc=None)
