"""Hierarchical scene graph with durable evidence and temporal object history."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ObjectHistoryEntry:
    timestamp: float
    confidence: float
    pose_map: list[float] | None
    states: dict[str, float]
    observation_refs: list[str]


@dataclass(slots=True)
class SceneObject:
    object_id: str
    category: str
    description: str
    region_id: str = "unknown"
    confidence: float = 0.0
    confidence_timestamp: float = 0.0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    observation_count: int = 0
    affordances: dict[str, float] = field(default_factory=dict)
    states: dict[str, float] = field(default_factory=dict)
    observation_refs: list[str] = field(default_factory=list)
    history: list[ObjectHistoryEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneGraph:
    objects: dict[str, SceneObject] = field(default_factory=dict)
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)
    max_history_per_object: int = 512

    def upsert_object(self, object_slot: Any, geometry: Any = None, timestamp: float | None = None) -> SceneObject:
        del geometry
        now = float(timestamp if timestamp is not None else object_slot.last_seen_time)
        confidence = float(object_slot.confidence.get("overall", 0.0))
        new_refs = list(dict.fromkeys(object_slot.observation_refs))
        entry = ObjectHistoryEntry(
            timestamp=now,
            confidence=confidence,
            pose_map=object_slot.pose_map,
            states=dict(object_slot.states),
            observation_refs=new_refs,
        )
        previous = self.objects.get(object_slot.object_id)
        if previous is None:
            value = SceneObject(
                object_id=object_slot.object_id,
                category=object_slot.category,
                description=object_slot.description,
                confidence=confidence,
                confidence_timestamp=now,
                first_seen_time=now,
                last_seen_time=now,
                observation_count=len(new_refs),
                affordances=dict(object_slot.affordances),
                states=dict(object_slot.states),
                observation_refs=new_refs,
                history=[entry],
            )
        else:
            previous.category = object_slot.category or previous.category
            previous.description = object_slot.description or previous.description
            # Consecutive views are correlated evidence; an exponential moving
            # average avoids the false near-certainty produced by noisy-OR.
            previous.confidence = 0.7 * previous.confidence + 0.3 * confidence
            previous.confidence_timestamp = now
            previous.last_seen_time = max(previous.last_seen_time, now)
            previous.affordances.update(object_slot.affordances)
            previous.states.update(object_slot.states)
            previous.observation_refs = list(dict.fromkeys([*previous.observation_refs, *new_refs]))
            previous.observation_count = len(previous.observation_refs)
            previous.history.append(entry)
            if len(previous.history) > self.max_history_per_object:
                del previous.history[: len(previous.history) - self.max_history_per_object]
            value = previous
        value.metadata.update(
            {
                "bbox_2d": object_slot.bbox_2d,
                "bbox_3d": object_slot.bbox_3d,
                "pose_map": object_slot.pose_map,
                "pose_robot": object_slot.pose_robot,
            }
        )
        self.objects[value.object_id] = value
        return value

    def decay_confidence(self, timestamp: float, half_life_s: float = 3600.0) -> int:
        if half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        changed = 0
        for value in self.objects.values():
            age = max(0.0, timestamp - value.confidence_timestamp)
            if age > 0:
                value.confidence *= math.exp(-math.log(2.0) * age / half_life_s)
                value.confidence_timestamp = timestamp
                changed += 1
        return changed

    def upsert_region(self, region_candidate: dict[str, Any]) -> None:
        self.regions[str(region_candidate["region_id"])] = dict(region_candidate)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "objects": {key: asdict(value) for key, value in self.objects.items()},
            "regions": self.regions,
            "edges": self.edges,
            "max_history_per_object": self.max_history_per_object,
        }

    @classmethod
    def from_serializable(cls, value: dict[str, Any]) -> SceneGraph:
        graph = cls(
            regions=dict(value.get("regions", {})),
            edges=list(value.get("edges", [])),
            max_history_per_object=int(value.get("max_history_per_object", 512)),
        )
        for object_id, raw in value.get("objects", {}).items():
            item = dict(raw)
            item["history"] = [ObjectHistoryEntry(**entry) for entry in item.get("history", [])]
            item.setdefault("first_seen_time", item.get("last_seen_time", 0.0))
            item.setdefault("confidence_timestamp", item.get("last_seen_time", 0.0))
            item.setdefault("observation_count", len(item.get("observation_refs", [])))
            item.setdefault("states", {})
            item.setdefault("observation_refs", item.get("metadata", {}).get("observation_refs", []))
            graph.objects[object_id] = SceneObject(**item)
        return graph
