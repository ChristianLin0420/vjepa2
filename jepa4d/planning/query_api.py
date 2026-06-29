"""Typed query facade: planners never consume raw feature tensors directly."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from jepa4d.memory.memory_update import FourDMemoryCore


class WorldModelQueryAPI:
    def __init__(self, memory: FourDMemoryCore | None = None) -> None:
        self.memory = memory or FourDMemoryCore()

    def find_object(self, query: str, region: str | None = None, time: str | None = None) -> list[dict[str, Any]]:
        del time
        terms = query.lower().split()
        return [
            asdict(value)
            for value in self.memory.scene_graph.objects.values()
            if (region is None or value.region_id == region)
            and any(term in f"{value.category} {value.description}".lower() for term in terms)
        ]

    def get_region_summary(self, region_id: str) -> dict[str, Any]:
        objects = [asdict(value) for value in self.memory.scene_graph.objects.values() if value.region_id == region_id]
        return {"region_id": region_id, "region": self.memory.scene_graph.regions.get(region_id), "objects": objects}

    def get_route(self, start_region: str, goal_region: str) -> dict[str, Any]:
        if start_region == goal_region:
            return {"start": start_region, "goal": goal_region, "regions": [start_region], "reachable": True}
        adjacency: dict[str, set[str]] = {}
        for edge in self.memory.scene_graph.edges:
            source, target = edge.get("source"), edge.get("target")
            if source and target and edge.get("relation", "connected_to") == "connected_to":
                adjacency.setdefault(source, set()).add(target)
                adjacency.setdefault(target, set()).add(source)
        queue: list[tuple[str, list[str]]] = [(start_region, [start_region])]
        visited = {start_region}
        while queue:
            region, route = queue.pop(0)
            for neighbor in adjacency.get(region, set()):
                if neighbor == goal_region:
                    return {
                        "start": start_region,
                        "goal": goal_region,
                        "regions": route + [neighbor],
                        "reachable": True,
                    }
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, route + [neighbor]))
        return {"start": start_region, "goal": goal_region, "regions": [], "reachable": False}

    def get_local_context(self, radius_m: float, frame: str = "base_link") -> dict[str, Any]:
        return {"radius_m": radius_m, "frame": frame, "observations": self.memory.active_local_map.observations}

    def get_objects_in_region(self, region_id: str, query: str | None = None) -> list[dict[str, Any]]:
        return (
            self.find_object(query or "", region=region_id) if query else self.get_region_summary(region_id)["objects"]
        )

    def get_observation_history(self, entity_id: str) -> dict[str, Any]:
        events = [asdict(event) for event in self.memory.episodic_memory.events if entity_id in event.entity_ids]
        return {"entity_id": entity_id, "events": events}

    def verify_condition(self, condition: str) -> dict[str, Any]:
        matches = self.find_object(condition)
        return {
            "condition": condition,
            "satisfied": bool(matches),
            "confidence": max([x["confidence"] for x in matches], default=0.0),
            "evidence": matches,
        }

    def get_affordances(self, object_id: str) -> dict[str, Any]:
        value = self.memory.scene_graph.objects.get(object_id)
        return {"object_id": object_id, "affordances": {} if value is None else value.affordances}

    def get_uncertainty(self, entity_id: str) -> dict[str, Any]:
        value = self.memory.scene_graph.objects.get(entity_id)
        confidence = 0.0 if value is None else value.confidence
        return {"entity_id": entity_id, "uncertainty": 1.0 - confidence}

    def suggest_verification_action(self, entity_id: str) -> dict[str, Any]:
        uncertainty = self.get_uncertainty(entity_id)["uncertainty"]
        return {"entity_id": entity_id, "action": "acquire_next_best_view", "priority": uncertainty}

    def mark_task_state(self, subgoal_id: str, status: str, evidence: dict) -> None:
        self.memory.task_state[subgoal_id] = {"status": status, "evidence": evidence}
