"""HTTP schemas."""

from typing import Any

from pydantic import BaseModel


class FindObjectRequest(BaseModel):
    query: str
    region: str | None = None


class VerifyRequest(BaseModel):
    condition: str


class MemoryUpdateRequest(BaseModel):
    observation: dict[str, Any]


class PlanRequest(BaseModel):
    instruction: str
    object_name: str = "mug"
    destination: str = "table"
