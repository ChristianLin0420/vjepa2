"""Tiny deterministic geometry benchmark for CI and API regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.data.rgb_input import from_view_sequences


class GeometrySmokeBenchmark(BenchmarkAdapter):
    name = "geometry-smoke"
    stage = "geometry"
    requires_runtime_depth = False
    supports_single_image = True
    supports_multiview = True
    supports_video = True

    def run(self, model_or_system: Any, split: str = "tiny") -> list[dict[str, Any]]:
        del split
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        predictions = []
        for views in (1, 2):
            batch = from_view_sequences([[image] for _ in range(views)])
            belief = model_or_system(batch)
            predictions.append(
                {
                    "views": views,
                    "finite_fraction": torch.isfinite(belief.depth_mean).float().mean().item(),
                    "scale_confidence": belief.scale_confidence.item(),
                    "point_count": belief.pointmap_mean.numel() // 3,
                }
            )
        return predictions

    def compute_metrics(self, predictions: Any, ground_truth: object = None) -> dict[str, float]:
        del ground_truth
        values = predictions
        return {
            "finite_fraction": min(item["finite_fraction"] for item in values),
            "multiview_confidence_gain": values[1]["scale_confidence"] - values[0]["scale_confidence"],
            "point_count": float(sum(item["point_count"] for item in values)),
        }

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "status": "ready"}
