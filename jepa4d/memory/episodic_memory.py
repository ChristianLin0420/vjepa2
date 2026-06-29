"""Time-indexed events for long-horizon memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EpisodicEvent:
    event_id: str
    timestamp: float
    event_type: str
    description: str
    entity_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodicMemory:
    events: list[EpisodicEvent] = field(default_factory=list)
    max_events: int = 100_000

    def add_event(self, event: EpisodicEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda value: (value.timestamp, value.event_id))
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]

    def query(
        self,
        *,
        entity_id: str | None = None,
        event_type: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[EpisodicEvent]:
        return [
            event
            for event in self.events
            if (entity_id is None or entity_id in event.entity_ids)
            and (event_type is None or event.event_type == event_type)
            and (start_time is None or event.timestamp >= start_time)
            and (end_time is None or event.timestamp <= end_time)
        ]
