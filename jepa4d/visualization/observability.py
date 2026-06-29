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
from jepa4d.memory.memory_update import FourDMemoryCore, MemoryUpdateResult
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
            wandb.define_metric("pipeline/step")
            wandb.define_metric("pipeline/*", step_metric="pipeline/step")
            wandb.define_metric("pipeline/timing_index")
            wandb.define_metric("pipeline/stage_latency_s", step_metric="pipeline/timing_index")
            wandb.define_metric("pipeline/cumulative_latency_s", step_metric="pipeline/timing_index")
            wandb.define_metric("training/global_step")
            wandb.define_metric("training/*", step_metric="training/global_step")
            wandb.define_metric("memory/revision")
            wandb.define_metric("memory/*", step_metric="memory/revision")
            wandb.define_metric("identity/variant_index")
            wandb.define_metric("identity/*", step_metric="identity/variant_index")
            wandb.define_metric("planning/step")
            wandb.define_metric("planning/*", step_metric="planning/step")
            wandb.define_metric("benchmark/stage_index")
            wandb.define_metric("benchmark/*", step_metric="benchmark/stage_index")

    @property
    def url(self) -> str | None:
        return None if self.run is None else self.run.url

    def log_feature_bundle(
        self, batch: RGBInputBatch, bundle: JEPATokenBundle, runtime: dict[str, float], *, step: int = 0
    ) -> None:
        if not self.enabled:
            return
        import wandb

        dense = bundle.dense_tokens.detach().float().cpu()
        global_tokens = bundle.global_tokens.detach().float().cpu()
        cosine = temporal_cosine(bundle)
        norms = dense.norm(dim=-1)
        scalars: dict[str, Any] = {
            "inference/step": step,
            "pipeline/step": step,
            "pipeline/stage": "vjepa_features",
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

    def log_geometry_belief(self, batch: RGBInputBatch, belief: GeometryBelief, *, step: int = 1) -> None:
        """Log calibrated-belief diagnostics without conflating confidence with accuracy."""
        if not self.enabled:
            return
        import wandb

        payload: dict[str, Any] = {
            "inference/step": step,
            "pipeline/step": step,
            "pipeline/stage": "geometry_belief",
            "geometry/runtime_s": belief.metadata.get("runtime_seconds", 0.0),
            "geometry/cuda_peak_memory_gb": (
                0.0
                if belief.metadata.get("cuda_peak_memory_bytes") is None
                else belief.metadata["cuda_peak_memory_bytes"] / 1024**3
            ),
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

            artifact_name = f"{self.run.id}-{Path(path).name}".replace("_", "-").replace(".", "-")
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            if Path(path).is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self.run.log_artifact(artifact)

    def log_object_grounding(self, batch: RGBInputBatch, result: ObjectGroundingResult, *, step: int = 2) -> None:
        if not self.enabled:
            return
        import wandb

        scores = [observation.score for observation in result.observations]
        mask_area_ratios = [float(observation.mask.mean()) for observation in result.observations]
        image_area = batch.images.shape[-2] * batch.images.shape[-1]
        box_area_ratios = [
            max(0.0, observation.bbox_2d[2] - observation.bbox_2d[0])
            * max(0.0, observation.bbox_2d[3] - observation.bbox_2d[1])
            / image_area
            for observation in result.observations
        ]
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
            "inference/step": step,
            "pipeline/step": step,
            "pipeline/stage": "object_grounding",
            "objects/runtime_s": result.metadata.get("runtime_seconds", 0.0),
            "objects/query_count": len(result.queries),
            "objects/detection_count": len(result.observations),
            "objects/slot_count": len(result.slots),
            "objects/mean_observations_per_slot": len(result.observations) / max(len(result.slots), 1),
            "objects/with_geometry_fraction": sum(slot.pose_map is not None for slot in result.slots)
            / max(len(result.slots), 1),
            "objects/slot_table": table,
            "objects/mean_mask_area_ratio": float(np.mean(mask_area_ratios)) if mask_area_ratios else 0.0,
            "objects/mean_box_area_ratio": float(np.mean(box_area_ratios)) if box_area_ratios else 0.0,
            "objects/query_detection_coverage": len({value.category for value in result.observations})
            / max(len(result.queries), 1),
            "objects/model_load_s": result.metadata.get("model_load_seconds", 0.0),
        }
        if scores:
            payload["objects/detection_score_mean"] = float(np.mean(scores))
            payload["objects/detection_score_min"] = float(np.min(scores))
            payload["objects/detection_score_histogram"] = wandb.Histogram(scores)
            payload["objects/mask_area_histogram"] = wandb.Histogram(mask_area_ratios)
            payload["objects/box_area_histogram"] = wandb.Histogram(box_area_ratios)
        observation_table = wandb.Table(
            columns=[
                "observation_id",
                "view",
                "time",
                "camera",
                "category",
                "score",
                "mask_area_ratio",
                "box_area_ratio",
                "has_pose",
            ]
        )
        for observation, mask_ratio, box_ratio in zip(
            result.observations, mask_area_ratios, box_area_ratios, strict=True
        ):
            observation_table.add_data(
                observation.observation_id,
                observation.view_index,
                observation.time_index,
                observation.camera_id,
                observation.category,
                observation.score,
                mask_ratio,
                box_ratio,
                observation.pose_map is not None,
            )
        payload["objects/observation_table"] = observation_table
        query_table = wandb.Table(columns=["query", "detections", "slots", "mean_score"])
        for query in result.queries:
            query_observations = [value for value in result.observations if value.category == query]
            query_table.add_data(
                query,
                len(query_observations),
                sum(slot.category == query for slot in result.slots),
                float(np.mean([value.score for value in query_observations])) if query_observations else 0.0,
            )
        payload["objects/query_coverage"] = query_table
        view_table = wandb.Table(columns=["view", "time", "detections"])
        for view in range(batch.images.shape[1]):
            for time_index in range(batch.images.shape[2]):
                count = sum(
                    value.view_index == view and value.time_index == time_index for value in result.observations
                )
                view_table.add_data(view, time_index, count)
        payload["objects/detections_by_view_time"] = wandb.plot.bar(
            view_table, "view", "detections", title="Detections by view (one bar per time entry)"
        )
        track_lengths = [len(slot.observations) for slot in result.slots]
        if track_lengths:
            payload["objects/track_length_histogram"] = wandb.Histogram(track_lengths)
        if visible_observations:
            box_payload = {
                "predictions": {
                    "box_data": [
                        {
                            "position": {
                                "minX": observation.bbox_2d[0],
                                "minY": observation.bbox_2d[1],
                                "maxX": observation.bbox_2d[2],
                                "maxY": observation.bbox_2d[3],
                            },
                            "class_id": class_id,
                            "box_caption": f"{observation.category} {observation.score:.3f}",
                            "scores": {"detector": observation.score},
                        }
                        for class_id, observation in enumerate(visible_observations, start=1)
                    ],
                    "class_labels": class_labels,
                }
            }
            payload["objects/mask_and_box_overlay"] = wandb.Image(
                image, masks=mask_payload, boxes=box_payload, caption="Grounded masks and detector boxes"
            )
        self.run.log(payload)

    def log_pipeline_summary(self, timings: dict[str, float], *, step: int = 3) -> None:
        """Log a stage timeline and an explicit hardware/runtime snapshot."""
        if not self.enabled:
            return
        import wandb

        timing_table = wandb.Table(
            columns=["stage", "seconds"], data=[[name, value] for name, value in timings.items()]
        )
        cumulative = 0.0
        for index, (name, value) in enumerate(timings.items()):
            cumulative += value
            self.run.log(
                {
                    "pipeline/timing_index": index,
                    "pipeline/timing_stage": name,
                    "pipeline/stage_latency_s": value,
                    "pipeline/cumulative_latency_s": cumulative,
                }
            )
        payload: dict[str, Any] = {
            "inference/step": step,
            "pipeline/step": step,
            "pipeline/stage": "complete",
            "pipeline/total_s": sum(timings.values()),
            "pipeline/stage_timing_table": timing_table,
            "pipeline/stage_timing_chart": wandb.plot.bar(
                timing_table, "stage", "seconds", title="End-to-end stage latency"
            ),
            "system/cuda_available": int(torch.cuda.is_available()),
            "system/device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "system/torch_cuda_build": torch.version.cuda or "none",
        }
        payload.update({f"pipeline/timing/{name}_s": value for name, value in timings.items()})
        if torch.cuda.is_available():
            payload.update(
                {
                    "system/gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "system/gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
        self.run.log(payload)

    def log_memory_update(self, result: MemoryUpdateResult, memory: FourDMemoryCore) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "memory/revision": result.revision,
            "memory/timestamp": result.timestamp,
            "memory/inserted_objects": result.inserted_objects,
            "memory/updated_objects": result.updated_objects,
            "memory/local_objects": len(memory.active_local_map.objects),
            "memory/global_objects": len(memory.scene_graph.objects),
            "memory/episodic_events": len(memory.episodic_memory.events),
            "memory/persistence_records": result.persistence_records,
            "memory/mean_confidence": (
                float(np.mean([value.confidence for value in memory.scene_graph.objects.values()]))
                if memory.scene_graph.objects
                else 0.0
            ),
            "memory/history_entries": sum(len(value.history) for value in memory.scene_graph.objects.values()),
        }
        self.run.log(payload)

    def log_planning_trace(self, trace: Any, *, mpc: dict[str, Any] | None = None) -> None:
        """Log closed-loop evidence, failures, replans, and optional latent-MPC diagnostics."""
        if not self.enabled:
            return
        import wandb

        event_table = wandb.Table(
            columns=[
                "step",
                "event",
                "subgoal",
                "stage",
                "success",
                "condition",
                "confidence",
                "uncertainty",
                "reason",
            ]
        )
        confidence_values: list[float] = []
        uncertainty_values: list[float] = []
        for event in trace.events:
            evidence = event.evidence
            confidence = evidence.get("confidence")
            uncertainty = evidence.get("uncertainty")
            if confidence is not None:
                confidence_values.append(float(confidence))
            if uncertainty is not None:
                uncertainty_values.append(float(uncertainty))
            event_table.add_data(
                event.step,
                event.event,
                event.subgoal_id,
                evidence.get("stage", ""),
                evidence.get("success", evidence.get("satisfied")),
                evidence.get("condition", ""),
                confidence,
                uncertainty,
                evidence.get("reason", ""),
            )
            self.run.log(
                {
                    "planning/step": event.step,
                    "planning/event": event.event,
                    "planning/subgoal": event.subgoal_id,
                    "planning/event_confidence": confidence,
                    "planning/event_uncertainty": uncertainty,
                    "planning/cumulative_replans": sum(
                        previous.event == "replan" for previous in trace.events if previous.step <= event.step
                    ),
                }
            )
        subgoal_table = wandb.Table(
            columns=[
                "subgoal",
                "action",
                "target",
                "status",
                "attempts",
                "condition",
                "evidence_count",
                "failure_reason",
            ],
            data=[
                [
                    value.subgoal_id,
                    value.action,
                    value.target,
                    str(value.status),
                    value.attempts,
                    value.verification_condition,
                    len(value.evidence),
                    value.failure_reason,
                ]
                for value in trace.task_graph.subgoals
            ],
        )
        verified = sum(str(value.status) == "verified" for value in trace.task_graph.subgoals)
        payload: dict[str, Any] = {
            "planning/step": max((value.step for value in trace.events), default=0) + 1,
            "planning/task_success": int(trace.success),
            "planning/subgoal_progress": verified / max(len(trace.task_graph.subgoals), 1),
            "planning/replans": trace.replans,
            "planning/verification_actions": trace.verification_actions,
            "planning/failure_attribution_count": sum(value.event == "failure_attribution" for value in trace.events),
            "planning/recovery_success": int(trace.success and trace.replans > 0),
            "planning/event_count": len(trace.events),
            "planning/mean_verification_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
            "planning/max_verification_uncertainty": max(uncertainty_values, default=1.0),
            "planning/event_trace": event_table,
            "planning/subgoals": subgoal_table,
        }
        if mpc is not None:
            payload.update(
                {
                    "dynamics/mpc_score": mpc["score"],
                    "dynamics/predicted_uncertainty": mpc["predicted_uncertainty"],
                    "dynamics/horizon": mpc["horizon"],
                    "dynamics/action_dim": mpc["action_dim"],
                    "dynamics/action_abs_mean": mpc["action_abs_mean"],
                    "dynamics/token_count": mpc["token_count"],
                    "dynamics/token_dim": mpc["token_dim"],
                    "dynamics/real_vjepa_features": int(mpc.get("real_vjepa_features", False)),
                }
            )
        self.run.log(payload)

    def log_benchmark_suite(self, report: Any) -> None:
        """Log Phase-6 stage estimates, intervals, latency, and typed failures."""
        if not self.enabled:
            return
        import wandb

        metric_table = wandb.Table(columns=["stage", "benchmark", "metric", "mean", "lower", "upper", "samples"])
        stage_table = wandb.Table(columns=["stage", "benchmark", "repetitions", "successes", "failures", "latency_ms"])
        for index, stage in enumerate(report.stages):
            stage_table.add_data(
                stage.stage,
                stage.name,
                stage.repetitions,
                stage.successes,
                stage.failures,
                stage.latency_ms.mean,
            )
            payload: dict[str, Any] = {
                "benchmark/stage_index": index,
                "benchmark/stage": stage.stage,
                "benchmark/name": stage.name,
                "benchmark/successes": stage.successes,
                "benchmark/failures": stage.failures,
                "benchmark/latency_ms": stage.latency_ms.mean,
            }
            for name, estimate in stage.metrics.items():
                metric_table.add_data(
                    stage.stage,
                    stage.name,
                    name,
                    estimate.mean,
                    estimate.lower,
                    estimate.upper,
                    estimate.samples,
                )
                payload[f"benchmark/metrics/{name}"] = estimate.mean
            self.run.log(payload)
        failure_table = wandb.Table(
            columns=["benchmark", "stage", "sample", "category", "message", "contributing"],
            data=[
                [
                    value.benchmark,
                    value.stage,
                    value.sample_id,
                    str(value.category),
                    value.message,
                    ",".join(str(item) for item in value.contributing),
                ]
                for value in report.failures
            ],
        )
        self.run.log(
            {
                "benchmark/stage_index": len(report.stages),
                "benchmark/suite_id": report.suite_id,
                "benchmark/stage_table": stage_table,
                "benchmark/metric_estimates": metric_table,
                "benchmark/failure_table": failure_table,
                "benchmark/total_failures": len(report.failures),
                "benchmark/stages_completed": len(report.stages),
            }
        )

    def log_memory_snapshot(self, memory: FourDMemoryCore) -> None:
        if not self.enabled:
            return
        import wandb

        object_table = wandb.Table(
            columns=[
                "object_id",
                "category",
                "confidence",
                "first_seen",
                "last_seen",
                "observations",
                "history",
                "region",
            ]
        )
        for value in memory.scene_graph.objects.values():
            object_table.add_data(
                value.object_id,
                value.category,
                value.confidence,
                value.first_seen_time,
                value.last_seen_time,
                value.observation_count,
                len(value.history),
                value.region_id,
            )
        event_table = wandb.Table(
            columns=["event_id", "timestamp", "type", "description", "entities"],
            data=[
                [event.event_id, event.timestamp, event.event_type, event.description, ",".join(event.entity_ids)]
                for event in memory.episodic_memory.events
            ],
        )
        self.run.log(
            {
                "memory/revision": memory.revision,
                "memory/object_table": object_table,
                "memory/event_table": event_table,
                "memory/snapshot": memory.snapshot().to_serializable(),
            }
        )

    def log_identity_ablation(
        self,
        results: dict[str, dict[str, dict[str, float]]],
        sweeps: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        import wandb

        table = wandb.Table(
            columns=[
                "dataset",
                "variant",
                "label",
                "pairwise_f1",
                "precision",
                "recall",
                "id_switches",
                "false_merges",
                "fragments",
                "track_survival",
                "predicted_tracks",
            ]
        )
        index = 0
        for dataset, variants in results.items():
            for variant, metrics in variants.items():
                table.add_data(
                    dataset,
                    variant,
                    f"{dataset}/{variant}",
                    metrics["pairwise_f1"],
                    metrics["pairwise_precision"],
                    metrics["pairwise_recall"],
                    metrics["id_switches"],
                    metrics["false_merges"],
                    metrics["fragments"],
                    metrics["track_survival"],
                    metrics["predicted_tracks"],
                )
                self.run.log(
                    {
                        "identity/variant_index": index,
                        "identity/dataset": dataset,
                        "identity/variant": variant,
                        "identity/pairwise_f1": metrics["pairwise_f1"],
                        "identity/id_switches": metrics["id_switches"],
                        "identity/false_merges": metrics["false_merges"],
                        "identity/fragments": metrics["fragments"],
                    }
                )
                index += 1
        payload: dict[str, Any] = {
            "identity/results": table,
            "identity/f1_comparison": wandb.plot.bar(
                table, "label", "pairwise_f1", title="Pairwise identity F1 by dataset and variant"
            ),
            "identity/switch_comparison": wandb.plot.bar(
                table, "label", "id_switches", title="Identity switches by dataset and variant"
            ),
        }
        if sweeps:
            sweep_table = wandb.Table(
                columns=["dataset", "appearance_weight", "iou_weight", "threshold", "f1", "switches", "merges"],
                data=[
                    [
                        value["dataset"],
                        value["appearance_weight"],
                        value["iou_weight"],
                        value["threshold"],
                        value["pairwise_f1"],
                        value["id_switches"],
                        value["false_merges"],
                    ]
                    for value in sweeps
                ],
            )
            payload["identity/operating_point_sweep"] = sweep_table
            payload["identity/sweep_f1"] = wandb.plot.scatter(
                sweep_table,
                "threshold",
                "f1",
                title="Exploratory identity F1 across thresholds and appearance weights",
            )
        self.run.log(payload)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if self.run is not None:
            if summary:
                self.run.summary.update(summary)
            self.run.finish()
