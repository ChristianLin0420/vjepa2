"""Behavior-tree node names reserved for structured Phase 5 execution."""

from dataclasses import dataclass


@dataclass(slots=True)
class BehaviorNode:
    target: str = ""

    def tick(self) -> str:
        return "mock_success"


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
