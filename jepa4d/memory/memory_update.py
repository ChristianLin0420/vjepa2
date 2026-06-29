"""Atomic updates across active, graph, episodic, vector, and durable memory."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from jepa4d.memory.active_local_map import ActiveLocalMap
from jepa4d.memory.episodic_memory import EpisodicEvent, EpisodicMemory
from jepa4d.memory.persistence import MemoryPersistence
from jepa4d.memory.scene_graph import SceneGraph
from jepa4d.memory.vector_index import VectorIndex


@dataclass(slots=True)
class MemoryUpdateResult:
    revision: int
    timestamp: float
    inserted_objects: int
    updated_objects: int
    local_objects: int
    episodic_events: int
    persistence_records: int

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FourDMemoryCore:
    active_local_map: ActiveLocalMap = field(default_factory=ActiveLocalMap)
    scene_graph: SceneGraph = field(default_factory=SceneGraph)
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    vector_index: VectorIndex = field(default_factory=VectorIndex)
    task_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    map_frame_id: str = "map"
    robot_frame_id: str = "base_link"
    revision: int = 0
    last_update_time: float = 0.0

    def update(
        self,
        geometry: Any,
        objects: list[Any],
        robot_state: Any = None,
        *,
        timestamp: float,
        persistence: MemoryPersistence | None = None,
        snapshot: bool = True,
    ) -> MemoryUpdateResult:
        if timestamp < self.last_update_time:
            raise ValueError(f"memory updates must be monotonic: timestamp={timestamp} < last={self.last_update_time}")
        previous_ids = set(self.scene_graph.objects)
        local = self.active_local_map.update(geometry, objects, robot_state, timestamp=timestamp)
        events: list[EpisodicEvent] = []
        for slot in objects:
            graph_object = self.scene_graph.upsert_object(slot, geometry=geometry, timestamp=timestamp)
            embedding = getattr(slot, "visual_embedding", None)
            if embedding is not None:
                self.vector_index.add(slot.object_id, embedding)
            event_type = "object_reobserved" if slot.object_id in previous_ids else "object_discovered"
            event = EpisodicEvent(
                event_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"jepa4d:{self.revision + 1}:{timestamp:.9f}:{slot.object_id}:{event_type}",
                    )
                ),
                timestamp=timestamp,
                event_type=event_type,
                description=f"{event_type.replace('_', ' ')}: {graph_object.category}",
                entity_ids=[slot.object_id],
                evidence={
                    "observation_refs": list(slot.observation_refs),
                    "confidence": graph_object.confidence,
                    "pose_map": graph_object.metadata.get("pose_map"),
                },
            )
            self.episodic_memory.add_event(event)
            events.append(event)
        self.revision += 1
        self.last_update_time = timestamp
        records_written = 0
        if persistence is not None:
            records: list[tuple[str, str, dict[str, Any]]] = [
                ("object", object_id, asdict(self.scene_graph.objects[object_id]))
                for object_id in {slot.object_id for slot in objects}
            ]
            records.extend(("event", event.event_id, asdict(event)) for event in events)
            records.append(("active_local_map", self.robot_frame_id, self.active_local_map.to_serializable()))
            snapshot_payload = self.snapshot(timestamp).to_serializable() if snapshot else None
            records_written = persistence.commit_update(
                records,
                timestamp=timestamp,
                revision=self.revision if snapshot else None,
                snapshot_payload=snapshot_payload,
            )
        return MemoryUpdateResult(
            revision=self.revision,
            timestamp=timestamp,
            inserted_objects=sum(slot.object_id not in previous_ids for slot in objects),
            updated_objects=sum(slot.object_id in previous_ids for slot in objects),
            local_objects=local["accepted"],
            episodic_events=len(events),
            persistence_records=records_written,
        )

    def decay_confidence(self, timestamp: float, half_life_s: float = 3600.0) -> int:
        changed = self.scene_graph.decay_confidence(timestamp, half_life_s)
        self.active_local_map.prune(timestamp)
        return changed

    def snapshot(self, timestamp: float | None = None) -> FourDMemorySnapshot:
        now = self.last_update_time if timestamp is None else timestamp
        confidences = [value.confidence for value in self.scene_graph.objects.values()]
        uncertainty = {
            "mean_object_uncertainty": 1.0 - sum(confidences) / max(len(confidences), 1),
            "max_object_uncertainty": 1.0 - min(confidences, default=0.0),
            "object_count": float(len(confidences)),
        }
        return FourDMemorySnapshot(
            map_frame_id=self.map_frame_id,
            robot_frame_id=self.robot_frame_id,
            timestamp=now,
            revision=self.revision,
            active_local_map=self.active_local_map,
            scene_graph=self.scene_graph,
            episodic_events=list(self.episodic_memory.events),
            task_state=self.task_state,
            uncertainty_summary=uncertainty,
        )

    @classmethod
    def load(cls, persistence: MemoryPersistence) -> FourDMemoryCore:
        latest = persistence.load_latest_snapshot()
        if latest is None:
            return cls()
        return cls.from_serializable(latest["payload"])

    @classmethod
    def replay(cls, persistence: MemoryPersistence) -> FourDMemoryCore:
        """Reconstruct current state solely from the append-only event log."""
        memory = cls()
        event_ids: set[str] = set()
        for event in persistence.list_events():
            payload = event["payload"]
            if event["kind"] == "object":
                graph = SceneGraph.from_serializable({"objects": {event["record_id"]: payload}})
                memory.scene_graph.objects[event["record_id"]] = graph.objects[event["record_id"]]
            elif event["kind"] == "event" and event["record_id"] not in event_ids:
                memory.episodic_memory.add_event(EpisodicEvent(**payload))
                event_ids.add(event["record_id"])
            elif event["kind"] == "active_local_map":
                memory.active_local_map = ActiveLocalMap.from_serializable(payload)
                memory.revision += 1
            memory.last_update_time = max(memory.last_update_time, float(event["timestamp"]))
        return memory

    @classmethod
    def from_serializable(cls, value: dict[str, Any]) -> FourDMemoryCore:
        memory = cls(
            active_local_map=ActiveLocalMap.from_serializable(value["active_local_map"]),
            scene_graph=SceneGraph.from_serializable(value["scene_graph"]),
            episodic_memory=EpisodicMemory([EpisodicEvent(**event) for event in value.get("episodic_events", [])]),
            task_state=dict(value.get("task_state", {})),
            map_frame_id=str(value.get("map_frame_id", "map")),
            robot_frame_id=str(value.get("robot_frame_id", "base_link")),
            revision=int(value.get("revision", 0)),
            last_update_time=float(value.get("timestamp", 0.0)),
        )
        for object_id, graph_object in memory.scene_graph.objects.items():
            embedding = persistence_embedding(graph_object.metadata)
            if embedding is not None:
                memory.vector_index.add(object_id, embedding)
        return memory


def persistence_embedding(metadata: dict[str, Any]) -> Any:
    """Compatibility hook for future snapshots that include compact embeddings."""
    return metadata.get("visual_embedding")


@dataclass(slots=True)
class FourDMemorySnapshot:
    map_frame_id: str
    robot_frame_id: str
    timestamp: float
    revision: int
    active_local_map: ActiveLocalMap
    scene_graph: SceneGraph
    episodic_events: list[EpisodicEvent]
    task_state: dict[str, dict[str, Any]]
    uncertainty_summary: dict[str, float]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "map_frame_id": self.map_frame_id,
            "robot_frame_id": self.robot_frame_id,
            "timestamp": self.timestamp,
            "revision": self.revision,
            "active_local_map": self.active_local_map.to_serializable(),
            "scene_graph": self.scene_graph.to_serializable(),
            "episodic_events": [asdict(event) for event in self.episodic_events],
            "task_state": self.task_state,
            "uncertainty_summary": self.uncertainty_summary,
        }
