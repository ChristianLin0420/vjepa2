"""Abstract robot interface for future execution adapters."""

from typing import Protocol


class RobotInterface(Protocol):
    def observe(self) -> object: ...
    def execute(self, action: object) -> object: ...
