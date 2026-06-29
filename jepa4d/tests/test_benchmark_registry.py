import pytest

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.benchmarks.registry import BenchmarkRegistry


class TinyBenchmark(BenchmarkAdapter):
    name = "tiny"
    stage = "representation"
    supports_single_image = True

    def run(self, model_or_system: object, split: str) -> list[dict]:
        return [{"split": split}]

    def compute_metrics(self, predictions: object, ground_truth: object) -> dict[str, float]:
        return {"ok": 1.0}

    def report(self) -> dict:
        return {"name": self.name}


def test_registry_is_explicit() -> None:
    registry = BenchmarkRegistry()
    registry.register(TinyBenchmark())
    assert registry.get("tiny").stage == "representation"
    assert registry.report()["benchmarks"][0]["supports_single_image"]
    with pytest.raises(ValueError):
        registry.register(TinyBenchmark())
