"""Deterministic feature-contract benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.data.rgb_input import from_view_sequences


class RepresentationSmokeBenchmark(BenchmarkAdapter):
    name = "representation-smoke"
    stage = "representation"
    supports_single_image = True
    supports_multiview = True
    supports_video = True

    def run(self, model_or_system: Any, split: str = "tiny") -> list[dict[str, Any]]:
        del split
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        samples = {
            "single_image": [[image]],
            "multi_view": [[image], [image]],
            "video": [[image, image, image, image]],
        }
        predictions = []
        for mode, views in samples.items():
            bundle = model_or_system(from_view_sequences(views))
            predictions.append(
                {
                    "mode": mode,
                    "shape": list(bundle.dense_tokens.shape),
                    "finite_fraction": torch.isfinite(bundle.dense_tokens).float().mean().item(),
                    "feature_std": bundle.dense_tokens.float().std().item(),
                }
            )
        return predictions

    def compute_metrics(self, predictions: Any, ground_truth: object = None) -> dict[str, float]:
        del ground_truth
        return {
            "finite_fraction": min(value["finite_fraction"] for value in predictions),
            "minimum_feature_std": min(value["feature_std"] for value in predictions),
            "modes_completed": float(len(predictions)),
        }

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "status": "ready"}
