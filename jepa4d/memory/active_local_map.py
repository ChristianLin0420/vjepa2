"""Robot-centric active-map summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ActiveLocalMap:
    radius_m: float = 5.0
    frame_id: str = "base_link"
    observations: list[dict[str, Any]] = field(default_factory=list)

    def update(self, geometry: Any, objects: list[Any], robot_state: Any) -> None:
        self.observations.append(
            {
                "geometry_mode": getattr(geometry, "mode", "unknown"),
                "object_count": len(objects),
                "robot_frame": getattr(robot_state, "frame_id", self.frame_id),
            }
        )
