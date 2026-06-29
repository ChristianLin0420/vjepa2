"""Small serializable graph API used by Phase 0 queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SceneObject:
    object_id: str
    category: str
    description: str
    region_id: str = "unknown"
    confidence: float = 0.0
    last_seen_time: float = 0.0
    affordances: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneGraph:
    objects: dict[str, SceneObject] = field(default_factory=dict)
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)

    def upsert_object(self, object_slot: Any, geometry: Any = None, timestamp: float | None = None) -> SceneObject:
        del geometry
        value = SceneObject(
            object_id=object_slot.object_id,
            category=object_slot.category,
            description=object_slot.description,
            confidence=float(object_slot.confidence.get("overall", 0.0)),
            last_seen_time=timestamp if timestamp is not None else object_slot.last_seen_time,
            affordances=dict(object_slot.affordances),
            metadata={
                "bbox_2d": object_slot.bbox_2d,
                "bbox_3d": object_slot.bbox_3d,
                "pose_map": object_slot.pose_map,
                "pose_robot": object_slot.pose_robot,
                "observation_refs": list(object_slot.observation_refs),
                "states": dict(object_slot.states),
            },
        )
        self.objects[value.object_id] = value
        return value

    def upsert_region(self, region_candidate: dict[str, Any]) -> None:
        self.regions[str(region_candidate["region_id"])] = dict(region_candidate)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "objects": {key: asdict(value) for key, value in self.objects.items()},
            "regions": self.regions,
            "edges": self.edges,
        }
