"""Synchronize active, graph, and episodic memory views."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from jepa4d.memory.active_local_map import ActiveLocalMap
from jepa4d.memory.episodic_memory import EpisodicMemory
from jepa4d.memory.scene_graph import SceneGraph


@dataclass(slots=True)
class FourDMemoryCore:
    active_local_map: ActiveLocalMap = field(default_factory=ActiveLocalMap)
    scene_graph: SceneGraph = field(default_factory=SceneGraph)
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    task_state: dict[str, dict] = field(default_factory=dict)


@dataclass(slots=True)
class FourDMemorySnapshot:
    map_frame_id: str
    robot_frame_id: str
    timestamp: float
    active_local_map: ActiveLocalMap
    scene_graph: SceneGraph
    episodic_events: list
    uncertainty_summary: dict[str, float]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "map_frame_id": self.map_frame_id,
            "robot_frame_id": self.robot_frame_id,
            "timestamp": self.timestamp,
            "active_local_map": asdict(self.active_local_map),
            "scene_graph": self.scene_graph.to_serializable(),
            "episodic_events": [asdict(event) for event in self.episodic_events],
            "uncertainty_summary": self.uncertainty_summary,
        }
