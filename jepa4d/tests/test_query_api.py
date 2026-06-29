from fastapi.testclient import TestClient

from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.models.object_slot_grounder import ObjectSlot
from jepa4d.planning.query_api import WorldModelQueryAPI
from jepa4d.server.app import app


def test_query_api_never_exposes_raw_tensors() -> None:
    memory = FourDMemoryCore()
    memory.scene_graph.upsert_object(
        ObjectSlot("mug-1", category="mug", description="red ceramic mug", confidence={"overall": 0.7})
    )
    api = WorldModelQueryAPI(memory)
    result = api.find_object("red mug")
    assert result[0]["object_id"] == "mug-1"
    assert api.suggest_verification_action("mug-1")["action"] == "acquire_next_best_view"
    memory.scene_graph.edges.append({"source": "hall", "target": "kitchen", "relation": "connected_to"})
    assert api.get_route("hall", "kitchen")["regions"] == ["hall", "kitchen"]


def test_query_api_exposes_history_and_local_belief() -> None:
    memory = FourDMemoryCore()
    memory.update(
        None,
        [
            ObjectSlot(
                "mug-1",
                category="mug",
                description="red mug",
                pose_map=[1.0, 0.0, 0.0],
                confidence={"overall": 0.8},
                observation_refs=["frame-1"],
            )
        ],
        timestamp=1.0,
    )
    api = WorldModelQueryAPI(memory)
    assert api.get_local_context(2.0)["objects"][0]["object_id"] == "mug-1"
    history = api.get_observation_history("mug-1")
    assert len(history["events"]) == len(history["observations"]) == 1
    assert api.get_uncertainty("mug-1")["observation_count"] == 1


def test_server_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["phase"] == 4
    assert "vggt_optional" in response.json()["geometry_backends"]
