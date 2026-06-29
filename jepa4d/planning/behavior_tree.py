"""Small behavior-tree runtime used by the verified task executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class NodeStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass(slots=True)
class BehaviorNode:
    target: str = ""

    def tick(self) -> NodeStatus:
        return NodeStatus.SUCCESS


@dataclass(slots=True)
class CallbackNode(BehaviorNode):
    callback: Callable[[], NodeStatus] = lambda: NodeStatus.SUCCESS

    def tick(self) -> NodeStatus:
        return self.callback()


@dataclass(slots=True)
class SequenceNode(BehaviorNode):
    children: list[BehaviorNode] = field(default_factory=list)
    cursor: int = 0

    def tick(self) -> NodeStatus:
        while self.cursor < len(self.children):
            status = self.children[self.cursor].tick()
            if status != NodeStatus.SUCCESS:
                return status
            self.cursor += 1
        return NodeStatus.SUCCESS


class NavigateToRegion(BehaviorNode):
    pass


class SearchObject(BehaviorNode):
    pass


class ApproachObject(BehaviorNode):
    pass


class PickObject(BehaviorNode):
    pass


class PlaceObject(BehaviorNode):
    pass


class OpenContainer(BehaviorNode):
    pass


class VerifyCondition(BehaviorNode):
    pass


class ReplanOnFailure(BehaviorNode):
    pass


class AskForHelpOrClarification(BehaviorNode):
    pass
