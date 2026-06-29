"""Phase 0 query server with explicit mock planning responses."""

from fastapi import FastAPI

from jepa4d.planning.query_api import WorldModelQueryAPI
from jepa4d.server.schemas import FindObjectRequest, MemoryUpdateRequest, PlanRequest, VerifyRequest

app = FastAPI(title="JEPA-4D WorldModel API", version="0.1.0")
query_api = WorldModelQueryAPI()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "phase": 4,
        "geometry_backends": ["mock", "vggt_optional"],
        "memory_revision": query_api.memory.revision,
    }


@app.post("/memory/update")
def memory_update(request: MemoryUpdateRequest) -> dict:
    query_api.memory.active_local_map.observations.append(request.observation)
    return {"updated": True, "observation_count": len(query_api.memory.active_local_map.observations)}


@app.post("/query/find_object")
def find_object(request: FindObjectRequest) -> list[dict]:
    return query_api.find_object(request.query, request.region)


@app.get("/query/region/{region_id}")
def region(region_id: str) -> dict:
    return query_api.get_region_summary(region_id)


@app.post("/query/verify")
def verify(request: VerifyRequest) -> dict:
    return query_api.verify_condition(request.condition)


@app.post("/planner/plan")
def plan(request: PlanRequest) -> dict:
    return {"instruction": request.instruction, "status": "mock", "task_graph": ["ground", "execute", "verify"]}


@app.post("/planner/replan")
def replan(request: PlanRequest) -> dict:
    return {"instruction": request.instruction, "status": "mock_replanned", "reason": "verification requested"}
