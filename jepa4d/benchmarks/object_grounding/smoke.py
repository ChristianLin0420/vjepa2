"""Deterministic object-slot association smoke benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.data.rgb_input import from_view_sequences


class ObjectGroundingSmokeBenchmark(BenchmarkAdapter):
    name = "object-grounding-smoke"
    stage = "object_grounding"
    requires_runtime_depth = False
    supports_single_image = True
    supports_multiview = True
    supports_video = True

    def run(self, model_or_system: Any, split: str = "tiny") -> list[dict[str, Any]]:
        del split
        y, x = np.mgrid[:48, :64]
        image = np.stack((x, y, x + y), axis=-1).astype(np.uint8)
        batch = from_view_sequences([[image], [np.roll(image, 2, axis=1)]])
        result = model_or_system(batch, ["mug", "table"])
        return [
            {
                "observations": len(result.observations),
                "slots": len(result.slots),
                "mean_track_length": len(result.observations) / max(len(result.slots), 1),
                "valid_mask_fraction": float(np.mean([observation.mask.any() for observation in result.observations])),
                "unique_id_fraction": len({slot.object_id for slot in result.slots}) / max(len(result.slots), 1),
            }
        ]

    def compute_metrics(self, predictions: Any, ground_truth: object = None) -> dict[str, float]:
        del ground_truth
        prediction = predictions[0]
        return {
            "association_recall": min(1.0, prediction["mean_track_length"] / 2.0),
            "valid_mask_fraction": prediction["valid_mask_fraction"],
            "unique_id_fraction": prediction["unique_id_fraction"],
            "slot_count": float(prediction["slots"]),
        }

    def report(self) -> dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "status": "ready", "quality_claim": "contract-only"}
