"""Benchmark adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BenchmarkAdapter(ABC):
    name: str
    stage: str
    requires_runtime_depth: bool = False
    supports_single_image: bool = False
    supports_multiview: bool = False
    supports_video: bool = False

    def prepare(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def run(self, model_or_system: object, split: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def compute_metrics(self, predictions: object, ground_truth: object) -> dict[str, float]: ...

    @abstractmethod
    def report(self) -> dict[str, Any]: ...
