"""Object-slot interface placeholder for Phase 3 teacher adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ObjectSlot:
    object_id: str
    category: str = "unknown"
    description: str = ""
    mask: np.ndarray | None = None
    bbox_2d: list[float] | None = None
    bbox_3d: list[float] | None = None
    pose_map: list[float] | None = None
    pose_robot: list[float] | None = None
    visual_embedding: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    language_embedding: np.ndarray | None = None
    affordances: dict[str, float] = field(default_factory=dict)
    states: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    last_seen_time: float = 0.0
    observation_refs: list[str] = field(default_factory=list)

    def to_serializable(self) -> dict:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "description": self.description,
            "mask": None if self.mask is None else self.mask.tolist(),
            "bbox_2d": self.bbox_2d,
            "bbox_3d": self.bbox_3d,
            "pose_map": self.pose_map,
            "pose_robot": self.pose_robot,
            "visual_embedding": self.visual_embedding.tolist(),
            "language_embedding": None if self.language_embedding is None else self.language_embedding.tolist(),
            "affordances": self.affordances,
            "states": self.states,
            "confidence": self.confidence,
            "last_seen_time": self.last_seen_time,
            "observation_refs": self.observation_refs,
        }
