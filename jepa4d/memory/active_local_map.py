"""Bounded robot-centric working memory for nearby objects and observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class LocalMapObject:
    object_id: str
    category: str
    position: list[float] | None
    distance_m: float | None
    confidence: float
    last_seen_time: float
    dynamic: bool = False


@dataclass(slots=True)
class ActiveLocalMap:
    radius_m: float = 5.0
    frame_id: str = "base_link"
    max_observations: int = 256
    stale_after_s: float = 30.0
    observations: list[dict[str, Any]] = field(default_factory=list)
    objects: dict[str, LocalMapObject] = field(default_factory=dict)
    updated_at: float = 0.0

    def update(
        self,
        geometry: Any,
        objects: list[Any],
        robot_state: Any,
        *,
        timestamp: float | None = None,
    ) -> dict[str, int]:
        now = float(
            timestamp
            if timestamp is not None
            else getattr(
                robot_state,
                "timestamp",
                max([getattr(value, "last_seen_time", 0.0) for value in objects], default=0.0),
            )
        )
        robot_origin = self._robot_origin(robot_state)
        accepted = 0
        rejected = 0
        for value in objects:
            position = getattr(value, "pose_robot", None) or getattr(value, "pose_map", None)
            distance = None
            if position is not None:
                point = np.asarray(position[:3], dtype=np.float32)
                distance = float(np.linalg.norm(point - robot_origin))
            if distance is not None and distance > self.radius_m:
                rejected += 1
                continue
            object_id = str(value.object_id)
            states = getattr(value, "states", {})
            self.objects[object_id] = LocalMapObject(
                object_id=object_id,
                category=str(getattr(value, "category", "unknown")),
                position=None if position is None else [float(item) for item in position[:3]],
                distance_m=distance,
                confidence=float(getattr(value, "confidence", {}).get("overall", 0.0)),
                last_seen_time=now,
                dynamic=bool(states.get("dynamic", 0.0) >= 0.5),
            )
            accepted += 1
        removed = self.prune(now)
        self.observations.append(
            {
                "timestamp": now,
                "geometry_mode": getattr(geometry, "mode", "unknown"),
                "object_count": len(objects),
                "accepted_local_objects": accepted,
                "rejected_outside_radius": rejected,
                "pruned_stale_objects": removed,
                "robot_frame": getattr(robot_state, "frame_id", self.frame_id),
            }
        )
        if len(self.observations) > self.max_observations:
            del self.observations[: len(self.observations) - self.max_observations]
        self.updated_at = now
        return {"accepted": accepted, "outside_radius": rejected, "pruned": removed}

    def prune(self, timestamp: float) -> int:
        stale = [
            object_id
            for object_id, value in self.objects.items()
            if timestamp - value.last_seen_time > self.stale_after_s
        ]
        for object_id in stale:
            del self.objects[object_id]
        return len(stale)

    @staticmethod
    def _robot_origin(robot_state: Any) -> np.ndarray:
        pose = getattr(robot_state, "base_pose", None)
        if pose is None:
            return np.zeros(3, dtype=np.float32)
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        values = np.asarray(pose, dtype=np.float32).reshape(-1)
        return values[:3] if values.size >= 3 else np.zeros(3, dtype=np.float32)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "radius_m": self.radius_m,
            "frame_id": self.frame_id,
            "max_observations": self.max_observations,
            "stale_after_s": self.stale_after_s,
            "observations": self.observations,
            "objects": {key: asdict(value) for key, value in self.objects.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_serializable(cls, value: dict[str, Any]) -> ActiveLocalMap:
        result = cls(
            radius_m=float(value.get("radius_m", 5.0)),
            frame_id=str(value.get("frame_id", "base_link")),
            max_observations=int(value.get("max_observations", 256)),
            stale_after_s=float(value.get("stale_after_s", 30.0)),
            observations=list(value.get("observations", [])),
            updated_at=float(value.get("updated_at", 0.0)),
        )
        result.objects = {key: LocalMapObject(**item) for key, item in value.get("objects", {}).items()}
        return result
