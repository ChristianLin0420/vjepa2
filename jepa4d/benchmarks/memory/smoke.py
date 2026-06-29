"""Deterministic persistence, replay, history, and query-latency benchmark."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.memory.persistence import MemoryPersistence
from jepa4d.models.object_slot_grounder import ObjectSlot
from jepa4d.planning.query_api import WorldModelQueryAPI


class MemorySmokeBenchmark(BenchmarkAdapter):
    name = "memory-smoke"
    stage = "memory"
    requires_runtime_depth = False
    supports_single_image = True
    supports_multiview = True
    supports_video = True

    def run(self, model_or_system: Any = None, split: str = "tiny") -> list[dict[str, Any]]:
        del model_or_system, split
        with tempfile.TemporaryDirectory() as directory:
            persistence = MemoryPersistence(Path(directory) / "memory.db")
            memory = FourDMemoryCore()
            for step in range(5):
                memory.update(
                    None,
                    [
                        ObjectSlot(
                            "tracked-object",
                            category="mug",
                            description="benchmark mug",
                            pose_map=[float(step), 0.0, 0.0],
                            visual_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                            confidence={"overall": 0.6},
                            observation_refs=[f"frame-{step}"],
                        )
                    ],
                    timestamp=float(step),
                    persistence=persistence,
                )
            started = time.perf_counter()
            matches = WorldModelQueryAPI(memory).find_object("mug")
            latency_ms = (time.perf_counter() - started) * 1000
            loaded = FourDMemoryCore.load(persistence)
            replayed = FourDMemoryCore.replay(persistence)
            return [
                {
                    "history_entries": len(memory.scene_graph.objects["tracked-object"].history),
                    "observation_refs": memory.scene_graph.objects["tracked-object"].observation_count,
                    "query_matches": len(matches),
                    "query_latency_ms": latency_ms,
                    "reload_equal": loaded.snapshot().to_serializable() == memory.snapshot().to_serializable(),
                    "replay_equal": replayed.snapshot().to_serializable() == memory.snapshot().to_serializable(),
                    "event_count": len(memory.episodic_memory.events),
                }
            ]

    def compute_metrics(self, predictions: Any, ground_truth: object = None) -> dict[str, float]:
        del ground_truth
        value = predictions[0]
        return {
            "history_recall": min(1.0, value["history_entries"] / 5),
            "observation_reference_recall": min(1.0, value["observation_refs"] / 5),
            "query_recall": min(1.0, float(value["query_matches"])),
            "reload_parity": float(value["reload_equal"]),
            "replay_parity": float(value["replay_equal"]),
            "query_latency_ms": value["query_latency_ms"],
        }

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "status": "ready", "quality_claim": "contract-only"}
