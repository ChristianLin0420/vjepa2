"""Discoverable benchmark registry with JSON-ready reports."""

from __future__ import annotations

from typing import Any

from jepa4d.benchmarks.base import BenchmarkAdapter


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BenchmarkAdapter] = {}

    def register(self, adapter: BenchmarkAdapter) -> BenchmarkAdapter:
        if adapter.name in self._adapters:
            raise ValueError(f"benchmark already registered: {adapter.name}")
        self._adapters[adapter.name] = adapter
        return adapter

    def get(self, name: str) -> BenchmarkAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise KeyError(f"unknown benchmark {name!r}; available={sorted(self._adapters)}") from error

    def report(self) -> dict[str, Any]:
        return {
            "benchmarks": [
                {
                    "name": adapter.name,
                    "stage": adapter.stage,
                    "requires_runtime_depth": adapter.requires_runtime_depth,
                    "supports_single_image": adapter.supports_single_image,
                    "supports_multiview": adapter.supports_multiview,
                    "supports_video": adapter.supports_video,
                }
                for adapter in self._adapters.values()
            ]
        }


registry = BenchmarkRegistry()
