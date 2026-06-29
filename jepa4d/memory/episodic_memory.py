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

    def add_event(self, event: EpisodicEvent) -> None:
        self.events.append(event)
