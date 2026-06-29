"""Task-aware, deterministic snapshot compression without mutating live memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jepa4d.memory.memory_update import FourDMemoryCore, FourDMemorySnapshot


@dataclass(slots=True)
class LODPolicy:
    max_object_history: int = 32
    max_events: int = 256
    max_local_observations: int = 64

    def compress(
        self, snapshot: FourDMemorySnapshot, task_context: dict[str, Any] | None = None
    ) -> FourDMemorySnapshot:
        payload = snapshot.to_serializable()
        protected = set((task_context or {}).get("entity_ids", []))
        removed_history = 0
        for object_id, value in payload["scene_graph"]["objects"].items():
            history = value.get("history", [])
            limit = self.max_object_history * 2 if object_id in protected else self.max_object_history
            removed_history += max(0, len(history) - limit)
            value["history"] = history[-limit:]
        events = payload["episodic_events"]
        protected_events = [event for event in events if protected.intersection(event.get("entity_ids", []))]
        recent_events = events[-self.max_events :]
        by_id = {event["event_id"]: event for event in [*protected_events, *recent_events]}
        payload["episodic_events"] = sorted(by_id.values(), key=lambda value: value["timestamp"])
        observations = payload["active_local_map"]["observations"]
        payload["active_local_map"]["observations"] = observations[-self.max_local_observations :]
        payload["uncertainty_summary"]["lod_removed_history_entries"] = float(removed_history)
        payload["uncertainty_summary"]["lod_removed_events"] = float(len(events) - len(by_id))
        memory = FourDMemoryCore.from_serializable(payload)
        compressed = memory.snapshot(snapshot.timestamp)
        compressed.uncertainty_summary.update(payload["uncertainty_summary"])
        return compressed
