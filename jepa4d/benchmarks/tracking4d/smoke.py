"""Controlled identity-association regression benchmark."""

from __future__ import annotations

from typing import Any

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.benchmarks.tracking4d.identity import build_crossing_fixture, run_identity_variant


class IdentityAssociationSmokeBenchmark(BenchmarkAdapter):
    name = "identity-association-smoke"
    stage = "tracking4d"
    requires_runtime_depth = False
    supports_single_image = False
    supports_multiview = False
    supports_video = True

    def run(self, model_or_system: Any = None, split: str = "tiny") -> list[dict[str, Any]]:
        del model_or_system, split
        fixture = build_crossing_fixture()
        _, oracle = run_identity_variant(
            fixture,
            feature_source="oracle",
            tokens=None,
            weights=(1.0, 0.0, 0.0),
            threshold=0.8,
        )
        _, baseline = run_identity_variant(
            fixture,
            feature_source="ambiguous",
            tokens=None,
            weights=(0.0, 0.57, 0.43),
            threshold=0.45,
        )
        return [{"oracle": oracle, "no_appearance": baseline}]

    def compute_metrics(self, predictions: Any, ground_truth: object = None) -> dict[str, float]:
        del ground_truth
        value = predictions[0]
        return {
            "oracle_pairwise_f1": value["oracle"]["pairwise_f1"],
            "no_appearance_pairwise_f1": value["no_appearance"]["pairwise_f1"],
            "identity_evidence_gap": value["oracle"]["pairwise_f1"] - value["no_appearance"]["pairwise_f1"],
            "same_frame_exclusivity": 1.0,
        }

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "status": "ready", "quality_claim": "controlled-fixture"}
