from jepa4d.memory.episodic_memory import EpisodicEvent
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.models.object_slot_grounder import ObjectSlot


def test_synchronized_memory_updates() -> None:
    memory = FourDMemoryCore()
    slot = ObjectSlot("mug-1", category="mug", description="red mug", confidence={"overall": 0.8})
    memory.scene_graph.upsert_object(slot, timestamp=2.0)
    memory.episodic_memory.add_event(EpisodicEvent("seen-1", 2.0, "observation", "saw mug", ["mug-1"]))
    assert memory.scene_graph.objects["mug-1"].last_seen_time == 2.0
    assert memory.episodic_memory.events[0].entity_ids == ["mug-1"]
