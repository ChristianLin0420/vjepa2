"""Stable machine-readable comparison schema for current and future variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VariantResult:
    variant_id: str
    family: str
    role: str
    seed: int | None
    metrics: dict[str, float]
    runtime: dict[str, float]
    parameters: int
    checkpoint: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    experiment_id: str
    schema_version: str
    dataset_manifest: str
    split_hash: str
    metric_policy: dict[str, Any]
    variants: list[VariantResult]
    failures: list[dict[str, str]]
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)
    wandb_url: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)
