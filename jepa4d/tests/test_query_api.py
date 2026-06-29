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


def test_server_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["phase"] == 2
    assert "vggt_optional" in response.json()["geometry_backends"]
