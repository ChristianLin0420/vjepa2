"""W&B logging with rich feature and training diagnostics."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jepa4d.data.schemas import JEPATokenBundle, RGBInputBatch
from jepa4d.models.geometry_belief import GeometryBelief
from jepa4d.models.object_slot_grounder import ObjectGroundingResult


def pca_rgb(tokens: torch.Tensor, patch_grid: tuple[int, int]) -> np.ndarray:
    """Project one token grid onto three principal components for visualization."""
    values = tokens.detach().float().cpu()
    values = values - values.mean(dim=0, keepdim=True)
    _, _, vectors = torch.pca_lowrank(values, q=min(3, values.shape[0], values.shape[1]))
    projected = values @ vectors[:, :3]
    if projected.shape[1] < 3:
        projected = torch.nn.functional.pad(projected, (0, 3 - projected.shape[1]))
    low = projected.quantile(0.02, dim=0)
    high = projected.quantile(0.98, dim=0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    return projected.reshape(*patch_grid, 3).numpy()


def temporal_cosine(bundle: JEPATokenBundle) -> list[float]:
    features = bundle.global_tokens[0, 0].float()
    if features.shape[0] < 2:
        return []
    return torch.nn.functional.cosine_similarity(features[:-1], features[1:], dim=-1).cpu().tolist()


class ExperimentLogger:
    """Optional W&B run plus locally useful scalar histories.

    W&B is imported only when enabled. Tests and offline demos therefore do not
    need credentials or a network connection.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        project: str = "jepa4d-worldmodel",
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.run: Any = None
        self.started = time.perf_counter()
        if enabled:
            import wandb

            self.run = wandb.init(
                project=project,
                name=name,
                config=config or {},
                tags=tags or [],
                mode=mode or os.getenv("WANDB_MODE", "online"),  # type: ignore[arg-type]
            )
            wandb.define_metric("inference/step")
            wandb.define_metric("inference/*", step_metric="inference/step")
            wandb.define_metric("training/global_step")
            wandb.define_metric("training/*", step_metric="training/global_step")

    @property
    def url(self) -> str | None:
        return None if self.run is None else self.run.url

    def log_feature_bundle(self, batch: RGBInputBatch, bundle: JEPATokenBundle, runtime: dict[str, float]) -> None:
        if not self.enabled:
            return
        import wandb

        dense = bundle.dense_tokens.detach().float().cpu()
        global_tokens = bundle.global_tokens.detach().float().cpu()
        cosine = temporal_cosine(bundle)
        norms = dense.norm(dim=-1)
        scalars: dict[str, Any] = {
            "inference/step": 0,
            "inference/runtime_total_s": runtime.get("total_s", 0.0),
            "inference/runtime_model_s": bundle.metadata.get("forward_seconds", 0.0),
            "inference/throughput_frames_per_s": batch.valid_mask.sum().item()
            / max(runtime.get("total_s", 1e-9), 1e-9),
            "inference/input_views": batch.images.shape[1],
            "inference/input_steps": batch.images.shape[2],
            "inference/output_temporal_bins": dense.shape[2],
            "inference/token_count": dense.shape[-2],
            "inference/feature_dim": dense.shape[-1],
            "features/mean": dense.mean().item(),
            "features/std": dense.std().item(),
            "features/min": dense.min().item(),
            "features/max": dense.max().item(),
            "features/l2_norm_mean": norms.mean().item(),
            "features/l2_norm_std": norms.std().item(),
            "features/finite_fraction": torch.isfinite(dense).float().mean().item(),
            "features/global_token_std": global_tokens.std().item(),
            "features/temporal_cosine_mean": float(np.mean(cosine)) if cosine else 1.0,
            "features/temporal_cosine_min": float(np.min(cosine)) if cosine else 1.0,
            "system/python": platform.python_version(),
            "system/torch": torch.__version__,
            "system/cuda_available": int(torch.cuda.is_available()),
        }
        scalars["features/value_histogram"] = wandb.Histogram(
            dense.flatten().numpy()[:: max(1, dense.numel() // 100_000)].tolist()
        )
        scalars["features/norm_histogram"] = wandb.Histogram(norms.flatten().numpy().tolist())
        scalars["visualizations/pca_rgb"] = wandb.Image(
            pca_rgb(dense[0, 0, 0], bundle.patch_grid), caption="Dense-token PCA"
        )
        input_image = batch.images[0, 0, 0].permute(1, 2, 0).cpu().numpy()
        scalars["visualizations/input"] = wandb.Image(input_image, caption=f"{batch.mode}: view 0, time 0")
        if cosine:
            table = wandb.Table(
                data=[[i, value] for i, value in enumerate(cosine)], columns=["temporal_pair", "cosine"]
            )
            scalars["visualizations/temporal_consistency"] = wandb.plot.line(
                table, "temporal_pair", "cosine", title="Adjacent temporal-bin cosine similarity"
            )
        layer_table = wandb.Table(columns=["layer", "shape", "mean", "std"])
        for layer, values in sorted(bundle.layer_tokens.items()):
            layer_table.add_data(
                layer, str(list(values.shape)), values.float().mean().item(), values.float().std().item()
            )
            scalars[f"features/layers/{layer}_mean"] = values.float().mean().item()
            scalars[f"features/layers/{layer}_std"] = values.float().std().item()
            scalars[f"features/layers/{layer}_norm"] = values.float().norm(dim=-1).mean().item()
        scalars["features/layer_summary"] = layer_table
        self.run.log(scalars)

    def log_training_step(
        self,
        step: int,
        *,
        losses: dict[str, float | torch.Tensor],
        learning_rate: float,
        grad_norm: float | None = None,
        weight_norm: float | None = None,
        throughput: float | None = None,
        memory_gb: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Standard detailed scalar schema for future JEPA-4D training loops."""
        if not self.enabled:
            return
        payload: dict[str, Any] = {"training/global_step": step, "training/learning_rate": learning_rate}
        payload.update(
            {
                f"training/loss/{name}": float(value.detach()) if torch.is_tensor(value) else value
                for name, value in losses.items()
            }
        )
        for key, value in {
            "training/grad_norm": grad_norm,
            "training/weight_norm": weight_norm,
            "training/samples_per_s": throughput,
            "training/memory_gb": memory_gb,
        }.items():
            if value is not None:
                payload[key] = value
        payload.update({f"training/{key}": value for key, value in (extra or {}).items()})
        self.run.log(payload)

    def log_geometry_belief(self, batch: RGBInputBatch, belief: GeometryBelief) -> None:
        """Log calibrated-belief diagnostics without conflating confidence with accuracy."""
        if not self.enabled:
            return
        import wandb

        payload: dict[str, Any] = {
            "geometry/runtime_s": belief.metadata.get("runtime_seconds", 0.0),
            "geometry/input_views": batch.images.shape[1],
            "geometry/input_steps": batch.images.shape[2],
            "geometry/scale_confidence": belief.scale_confidence.float().mean().item(),
            "geometry/pose_confidence": belief.pose_confidence.float().mean().item(),
            "geometry/reconstruction_confidence": belief.reconstruction_confidence.float().mean().item(),
        }
        if belief.depth_mean is not None:
            depth = belief.depth_mean.detach().float().cpu()
            payload.update(
                {
                    "geometry/depth_mean": depth.mean().item(),
                    "geometry/depth_std": depth.std().item(),
                    "geometry/depth_min": depth.min().item(),
                    "geometry/depth_max": depth.max().item(),
                    "geometry/depth_histogram": wandb.Histogram(
                        depth.flatten().numpy()[:: max(1, depth.numel() // 100_000)].tolist()
                    ),
                    "geometry/depth_map": wandb.Image(depth[0, 0, 0].numpy(), caption="Depth mean"),
                }
            )
        if belief.depth_logvar is not None:
            logvar = belief.depth_logvar.detach().float().cpu()
            payload.update(
                {
                    "geometry/depth_logvar_mean": logvar.mean().item(),
                    "geometry/depth_logvar_p95": torch.quantile(logvar, 0.95).item(),
                    "geometry/depth_uncertainty": wandb.Image(logvar[0, 0, 0].numpy(), caption="Depth log-variance"),
                }
            )
        if belief.pointmap_mean is not None:
            points = belief.pointmap_mean.detach().float().cpu()
            payload["geometry/point_finite_fraction"] = torch.isfinite(points).float().mean().item()
            payload["geometry/point_extent_xyz"] = wandb.Table(
                columns=["axis", "min", "max", "span"],
                data=[
                    [
                        axis,
                        points[..., index].min().item(),
                        points[..., index].max().item(),
                        (points[..., index].max() - points[..., index].min()).item(),
                    ]
                    for index, axis in enumerate("xyz")
                ],
            )
        if belief.tracks_2d is not None:
            payload["geometry/track_count"] = belief.tracks_2d.shape[-2]
        self.run.log(payload)

    def log_artifact(self, path: str | Path, artifact_type: str = "inference-output") -> None:
        if self.enabled:
            import wandb

            artifact = wandb.Artifact(Path(path).stem, type=artifact_type)
            if Path(path).is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self.run.log_artifact(artifact)

    def log_object_grounding(self, batch: RGBInputBatch, result: ObjectGroundingResult) -> None:
        if not self.enabled:
            return
        import wandb

        scores = [observation.score for observation in result.observations]
        table = wandb.Table(
            columns=["object_id", "category", "observations", "detection", "association", "has_pose"],
            data=[
                [
                    slot.object_id,
                    slot.category,
                    len(slot.observations),
                    slot.confidence.get("detection", 0.0),
                    slot.confidence.get("association", 0.0),
                    slot.pose_map is not None,
                ]
                for slot in result.slots
            ],
        )
        image = batch.images[0, 0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        visible_observations = [
            value for value in result.observations if value.view_index == 0 and value.time_index == 0
        ]
        semantic_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        class_labels: dict[int, str] = {}
        for class_id, observation in enumerate(visible_observations, start=1):
            semantic_mask[observation.mask] = class_id
            class_labels[class_id] = observation.category
        mask_payload: dict[str, Any] = {"predictions": {"mask_data": semantic_mask, "class_labels": class_labels}}
        payload: dict[str, Any] = {
            "objects/runtime_s": result.metadata.get("runtime_seconds", 0.0),
            "objects/query_count": len(result.queries),
            "objects/detection_count": len(result.observations),
            "objects/slot_count": len(result.slots),
            "objects/mean_observations_per_slot": len(result.observations) / max(len(result.slots), 1),
            "objects/with_geometry_fraction": sum(slot.pose_map is not None for slot in result.slots)
            / max(len(result.slots), 1),
            "objects/slot_table": table,
        }
        if scores:
            payload["objects/detection_score_mean"] = float(np.mean(scores))
            payload["objects/detection_score_min"] = float(np.min(scores))
            payload["objects/detection_score_histogram"] = wandb.Histogram(scores)
        if visible_observations:
            payload["objects/mask_overlay"] = wandb.Image(image, masks=mask_payload, caption="Grounded object masks")
        self.run.log(payload)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if self.run is not None:
            if summary:
                self.run.summary.update(summary)
            self.run.finish()
