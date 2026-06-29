from pathlib import Path

import numpy as np
import pytest
import torch

from jepa4d.data.schemas import RobotState
from jepa4d.memory.lod_policy import LODPolicy
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.memory.persistence import MemoryPersistence
from jepa4d.models.object_slot_grounder import ObjectSlot


def slot(
    object_id: str = "mug-1",
    *,
    confidence: float = 0.8,
    pose: list[float] | None = None,
    observation: str = "obs-1",
) -> ObjectSlot:
    return ObjectSlot(
        object_id,
        category="mug",
        description="red mug",
        pose_map=pose,
        visual_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        confidence={"overall": confidence},
        states={"visible": 1.0},
        observation_refs=[observation],
    )


def test_incremental_update_merges_history_and_emits_events() -> None:
    memory = FourDMemoryCore()
    first = memory.update(None, [slot(pose=[1.0, 0.0, 0.0])], timestamp=1.0)
    second = memory.update(
        None,
        [slot(confidence=0.5, pose=[1.2, 0.0, 0.0], observation="obs-2")],
        timestamp=2.0,
    )

    value = memory.scene_graph.objects["mug-1"]
    assert first.inserted_objects == 1 and first.updated_objects == 0
    assert second.inserted_objects == 0 and second.updated_objects == 1
    assert value.observation_refs == ["obs-1", "obs-2"]
    assert value.observation_count == 2
    assert len(value.history) == 2
    assert value.confidence == pytest.approx(0.71)
    assert [event.event_type for event in memory.episodic_memory.events] == [
        "object_discovered",
        "object_reobserved",
    ]
    assert memory.vector_index.search(np.asarray([1.0, 0.0]))[0][0] == "mug-1"


def test_active_map_radius_staleness_and_robot_origin() -> None:
    memory = FourDMemoryCore()
    robot = RobotState(timestamp=1.0, base_pose=torch.tensor([10.0, 0.0, 0.0]))
    result = memory.update(
        None,
        [slot("near", pose=[12.0, 0.0, 0.0]), slot("far", pose=[20.0, 0.0, 0.0])],
        robot,
        timestamp=1.0,
    )
    assert result.local_objects == 1
    assert set(memory.active_local_map.objects) == {"near"}
    assert memory.active_local_map.objects["near"].distance_m == pytest.approx(2.0)
    assert memory.active_local_map.prune(32.0) == 1


def test_persistence_snapshot_reload_and_event_replay(tmp_path: Path) -> None:
    persistence = MemoryPersistence(tmp_path / "memory.db")
    memory = FourDMemoryCore()
    memory.update(None, [slot()], timestamp=1.0, persistence=persistence)
    memory.update(None, [slot(observation="obs-2")], timestamp=2.0, persistence=persistence)

    assert persistence.stats() == {"records": 4, "events": 6, "snapshots": 2, "schema_version": 2}
    assert [event["sequence"] for event in persistence.list_events()] == list(range(1, 7))
    loaded = FourDMemoryCore.load(persistence)
    replayed = FourDMemoryCore.replay(persistence)
    assert loaded.revision == replayed.revision == 2
    assert loaded.scene_graph.to_serializable() == replayed.scene_graph.to_serializable()
    assert len(loaded.episodic_memory.events) == len(replayed.episodic_memory.events) == 2
    assert loaded.active_local_map.updated_at == replayed.active_local_map.updated_at == 2.0


def test_persistence_batch_rolls_back_on_invalid_json(tmp_path: Path) -> None:
    persistence = MemoryPersistence(tmp_path / "memory.db")
    with pytest.raises(ValueError):
        persistence.apply_batch(
            [("object", "valid", {"value": 1}), ("object", "invalid", {"value": float("nan")})],
            timestamp=1.0,
        )
    assert persistence.get("valid") is None
    assert persistence.stats()["events"] == 0


def test_monotonic_updates_and_nonduplicating_decay() -> None:
    memory = FourDMemoryCore()
    memory.update(None, [slot()], timestamp=10.0)
    with pytest.raises(ValueError, match="monotonic"):
        memory.update(None, [slot()], timestamp=9.0)
    memory.decay_confidence(3610.0, half_life_s=3600.0)
    first_decay = memory.scene_graph.objects["mug-1"].confidence
    memory.decay_confidence(3610.0, half_life_s=3600.0)
    assert first_decay == pytest.approx(0.4)
    assert memory.scene_graph.objects["mug-1"].confidence == first_decay


def test_lod_compression_is_bounded_and_non_mutating() -> None:
    memory = FourDMemoryCore()
    for timestamp in range(1, 8):
        memory.update(None, [slot(observation=f"obs-{timestamp}")], timestamp=float(timestamp))
    original = memory.snapshot()
    compressed = LODPolicy(max_object_history=2, max_events=3, max_local_observations=2).compress(original)
    assert len(original.scene_graph.objects["mug-1"].history) == 7
    assert len(compressed.scene_graph.objects["mug-1"].history) == 2
    assert len(compressed.episodic_events) == 3
    assert len(compressed.active_local_map.observations) == 2
    assert compressed.uncertainty_summary["lod_removed_history_entries"] == 5.0
