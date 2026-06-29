"""Ground queried objects and persist an initial scene-graph memory."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from jepa4d.data.rgb_input import load_rgb_input
from jepa4d.memory.memory_update import FourDMemoryCore
from jepa4d.memory.persistence import MemoryPersistence
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.object_slot_grounder import ObjectSlotGrounder
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.object_report import build_object_report
from jepa4d.visualization.observability import ExperimentLogger

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    images: Annotated[list[Path], typer.Option("--images", "-i")],
    query: Annotated[list[str], typer.Option("--query", "-q")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/object_memory"),
    detector_backend: Annotated[str, typer.Option("--detector-backend")] = "mock",
    mask_backend: Annotated[str, typer.Option("--mask-backend")] = "box",
    detector_model_id: Annotated[str, typer.Option("--detector-model-id")] = "IDEA-Research/grounding-dino-tiny",
    sam2_model_id: Annotated[str, typer.Option("--sam2-model-id")] = "facebook/sam2-hiera-tiny",
    geometry_backend: Annotated[str, typer.Option("--geometry-backend")] = "mock",
    geometry_model_id: Annotated[str, typer.Option("--geometry-model-id")] = "facebook/VGGT-1B",
    jepa_checkpoint: Annotated[Path | None, typer.Option("--jepa-checkpoint")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    max_frames: Annotated[int, typer.Option("--max-frames")] = 16,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_mode: Annotated[str, typer.Option("--wandb-mode")] = "online",
    run_name: Annotated[str | None, typer.Option("--run-name")] = None,
) -> None:
    pipeline_started = time.perf_counter()
    batch = load_rgb_input(images, max_frames=max_frames)
    input_seconds = time.perf_counter() - pipeline_started
    name = run_name or f"objects-{detector_backend}-{mask_backend}-{batch.mode}"
    logger = ExperimentLogger(
        enabled=wandb,
        project=wandb_project,
        name=name,
        mode=wandb_mode,
        tags=["phase-3", "object-grounding", detector_backend, mask_backend, batch.mode, device],
        config={
            "queries": query,
            "detector_backend": detector_backend,
            "mask_backend": mask_backend,
            "detector_model_id": detector_model_id,
            "sam2_model_id": sam2_model_id,
            "geometry_backend": geometry_backend,
            "geometry_model_id": geometry_model_id,
            "jepa_checkpoint": None if jepa_checkpoint is None else str(jepa_checkpoint),
            "jepa_backend": "mock" if jepa_checkpoint is None else "checkpoint",
            "device_requested": device,
            "input": batch.to_serializable(),
        },
    )
    feature_started = time.perf_counter()
    feature_extractor = VJEPA21FeatureExtractor(
        mock=jepa_checkpoint is None,
        checkpoint=jepa_checkpoint,
        model_name="vjepa2_1_vit_base_384",
        device=device,
    )
    tokens = feature_extractor(batch)
    feature_seconds = time.perf_counter() - feature_started
    logger.log_feature_bundle(
        batch,
        tokens,
        {"total_s": feature_seconds, "model_s": tokens.metadata.get("forward_seconds", feature_seconds)},
        step=0,
    )
    geometry_started = time.perf_counter()
    geometry_head = GeometryBeliefHead(
        backend=geometry_backend,
        model_id=geometry_model_id,
        device=device,
        output_size=112,
    )
    geometry = geometry_head(batch)
    geometry_seconds = time.perf_counter() - geometry_started
    logger.log_geometry_belief(batch, geometry, step=1)
    grounder_load_started = time.perf_counter()
    grounder = ObjectSlotGrounder(
        detector_backend=detector_backend,
        mask_backend=mask_backend,
        detector_model_id=detector_model_id,
        sam2_model_id=sam2_model_id,
        device=device,
    )
    grounder_load_seconds = time.perf_counter() - grounder_load_started
    grounding_started = time.perf_counter()
    result = grounder(batch, query, tokens=tokens, geometry=geometry)
    grounding_seconds = time.perf_counter() - grounding_started
    logger.log_object_grounding(batch, result, step=2)
    persistence_started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    result_path = result.save_json(output / "objects.json")
    masks_path = result.save_masks(output / "masks.npz")
    memory = FourDMemoryCore()
    persistence = MemoryPersistence(output / "memory.db")
    timestamp = float(batch.timestamps.max())
    memory_update = memory.update(
        geometry,
        result.slots,
        batch.robot_state,
        timestamp=timestamp,
        persistence=persistence,
    )
    scene_path = output / "scene_graph.json"
    scene_path.write_text(json.dumps(memory.scene_graph.to_serializable(), indent=2) + "\n")
    report_path = build_object_report(batch, result, output / "report.html", wandb_url=logger.url)
    persistence_seconds = time.perf_counter() - persistence_started
    timings = {
        "input_load": input_seconds,
        "vjepa_features": feature_seconds,
        "geometry": geometry_seconds,
        "grounder_load": grounder_load_seconds,
        "grounding": grounding_seconds,
        "persistence_report": persistence_seconds,
    }
    logger.log_pipeline_summary(timings, step=3)
    experiment_path = output / "EXPERIMENT.md"
    experiment_path.write_text(
        f"# Object grounding experiment: {name}\n\n"
        f"- Timestamp: {datetime.now(UTC).isoformat()}\n"
        f"- Input mode: `{batch.mode}`\n- Queries: `{result.queries}`\n"
        f"- Detector/mask backend: `{detector_backend}/{mask_backend}`\n"
        f"- Observations/slots: `{len(result.observations)}/{len(result.slots)}`\n"
        f"- Geometry-attached slots: `{sum(slot.pose_map is not None for slot in result.slots)}`\n"
        f"- Memory revision: `{memory_update.revision}`\n"
        f"- Memory update: `{memory_update.to_serializable()}`\n"
        f"- Grounding runtime: `{result.metadata['runtime_seconds']:.6f} s`\n"
        f"- End-to-end runtime: `{sum(timings.values()):.6f} s`\n"
        f"- Stage timings: `{timings}`\n"
        f"- W&B: {logger.url or 'disabled'}\n\n"
        "## Interpretation\n\n"
        "Slots are associated observations, not verified physical truth. Mock detections validate contracts only; teacher "
        "outputs require benchmark calibration and multi-view verification.\n\n"
        f"## Artifacts\n\n- `{result_path}`\n- `{masks_path}`\n- `{output / 'memory.db'}`\n"
        f"- `{scene_path}`\n- `{report_path}`\n"
    )
    for path, artifact_type in (
        (result_path, "object-slots"),
        (masks_path, "object-masks"),
        (output / "memory.db", "world-memory"),
        (scene_path, "scene-graph"),
        (report_path, "interactive-report"),
        (experiment_path, "experiment-record"),
    ):
        logger.log_artifact(path, artifact_type)
    logger.finish(
        {
            "result": "success",
            "slot_count": len(result.slots),
            "detection_count": len(result.observations),
            "geometry_attached": sum(slot.pose_map is not None for slot in result.slots),
            "end_to_end_seconds": sum(timings.values()),
            "device_requested": device,
        }
    )
    typer.echo(
        json.dumps(
            {
                "objects": str(result_path),
                "masks": str(masks_path),
                "memory": str(output / "memory.db"),
                "scene_graph": str(scene_path),
                "report": str(report_path),
                "experiment": str(experiment_path),
                "wandb_url": logger.url,
                "slot_ids": [slot.object_id for slot in result.slots],
                "timings": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
