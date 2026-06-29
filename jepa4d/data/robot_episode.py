"""Minimal episode contract used by future robotics adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from jepa4d.data.schemas import RGBInputBatch, RobotState


@dataclass(slots=True)
class RobotEpisode:
    episode_id: str
    observations: list[RGBInputBatch] = field(default_factory=list)
    states: list[RobotState] = field(default_factory=list)
    task: str = ""
