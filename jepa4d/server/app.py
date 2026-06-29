"""Queryable memory and deterministic Phase-5 planning service."""

from fastapi import FastAPI

from jepa4d.planning.execution import VerifiedTaskPlanner
from jepa4d.planning.query_api import WorldModelQueryAPI
from jepa4d.planning.task_graph import TaskGraph
from jepa4d.robotics.mock_robot import MockRobot
from jepa4d.server.schemas import FindObjectRequest, MemoryUpdateRequest, PlanRequest, VerifyRequest

app = FastAPI(title="JEPA-4D WorldModel API", version="0.1.0")
query_api = WorldModelQueryAPI()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "phase": 5,
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
    graph = TaskGraph.pick_and_place(request.object_name, request.destination)
    trace = VerifiedTaskPlanner(query_api=query_api).execute(
        graph, MockRobot(objects={request.object_name: "counter"})
    )
    return trace.to_serializable()


@app.post("/planner/replan")
def replan(request: PlanRequest) -> dict:
    graph = TaskGraph.pick_and_place(request.object_name, request.destination)
    robot = MockRobot(objects={request.object_name: "counter"}, fail_once={f"pick:{request.object_name}"})
    return VerifiedTaskPlanner(query_api=query_api).execute(graph, robot).to_serializable()
