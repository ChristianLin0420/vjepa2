"""Build and export a geometry belief from RGB images or a short video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from jepa4d.data.rgb_input import load_rgb_input
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.geometry_export import export_colmap_text, export_geometry_npz, export_pointcloud_ply
from jepa4d.visualization.experiment_record import ArtifactRecord, ExperimentRecord, PanelRecord, StageRecord
from jepa4d.visualization.geometry_report import build_geometry_report
from jepa4d.visualization.observability import ExperimentLogger

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    images: Annotated[
        list[Path], typer.Option("--images", "-i", help="Repeat for a multi-view set or pass one video")
    ],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/reconstruction"),
    backend: Annotated[str, typer.Option("--backend")] = "mock",
    model_id: Annotated[str, typer.Option("--model-id")] = "facebook/VGGT-1B",
    device: Annotated[str, typer.Option("--device")] = "cpu",
    max_frames: Annotated[int, typer.Option("--max-frames")] = 16,
    output_size: Annotated[int, typer.Option("--output-size")] = 112,
    precision: Annotated[str, typer.Option("--precision")] = "float32",
    known_scale_prior: Annotated[bool, typer.Option("--known-scale-prior")] = False,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_mode: Annotated[str, typer.Option("--wandb-mode")] = "online",
    run_name: Annotated[str | None, typer.Option("--run-name")] = None,
) -> None:
    batch = load_rgb_input(images, max_frames=max_frames)
    head = GeometryBeliefHead(
        backend=backend,
        device=device,
        model_id=model_id,
        output_size=output_size,
        known_scale_prior=known_scale_prior,
        precision=precision,
    )
    name = run_name or f"geometry-{backend}-{batch.mode}-{batch.images.shape[1]}v{batch.images.shape[2]}t"
    logger = ExperimentLogger(
        enabled=wandb,
        project=wandb_project,
        name=name,
        mode=wandb_mode,
        tags=["phase-2", "geometry", backend, batch.mode],
        config={
            "backend": backend,
            "model_id": model_id,
            "device": device,
            "output_size": output_size,
            "known_scale_prior": known_scale_prior,
            "precision": precision,
            "input": batch.to_serializable(),
        },
    )
    belief = head(batch)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = export_geometry_npz(belief, output / "geometry_belief.npz")
    ply_path = export_pointcloud_ply(belief, batch, output / "pointcloud.ply")
    colmap_path = export_colmap_text(belief, batch, output / "colmap")
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(belief.to_serializable(), indent=2) + "\n")
    logger.log_geometry_belief(batch, belief)
    report_path = build_geometry_report(batch, belief, output / "report.html", wandb_url=logger.url)
    experiment_path = ExperimentRecord(
        title=f"Geometry belief: {name}",
        experiment_id=name,
        stage="geometry",
        status="complete",
        evidence_level="contract-only" if backend == "mock" else "integration",
        objective="Produce explicit camera, depth, point-map, track, and uncertainty beliefs from RGB.",
        hypothesis="Additional views improve geometric constraint while unknown monocular scale remains uncertain.",
        decision="Expose the belief downstream without treating adapter confidence as calibrated accuracy.",
        wandb_url=logger.url,
        config={
            "backend": backend,
            "model_id": model_id,
            "device": device,
            "mode": batch.mode,
            "views": batch.images.shape[1],
            "timesteps": batch.images.shape[2],
            "output_size": output_size,
            "precision": precision,
        },
        stages=[
            StageRecord(
                "geometry",
                backend,
                "pass",
                batch.mode,
                "camera/depth/point-map/tracks",
                "Outputs are beliefs; metric accuracy remains unmeasured in this run.",
            )
        ],
        panels=[
            PanelRecord("geometry/depth_map", "image", "Inspect spatial depth structure."),
            PanelRecord("geometry/depth_histogram", "histogram", "Inspect numerical depth support."),
            PanelRecord("geometry/depth_uncertainty", "image", "Locate uncertain regions."),
            PanelRecord("geometry/point_extent_xyz", "chart", "Sanity-check point-map coordinate extent."),
            PanelRecord("geometry/*_confidence", "scalar", "Separate scale, pose, and reconstruction belief."),
        ],
        metrics={
            "scale_confidence": belief.scale_confidence.tolist(),
            "pose_confidence": belief.pose_confidence.tolist(),
            "reconstruction_confidence": belief.reconstruction_confidence.tolist(),
            "runtime_s": belief.metadata["runtime_seconds"],
            "cuda_peak_memory_gb": (
                None
                if belief.metadata["cuda_peak_memory_bytes"] is None
                else belief.metadata["cuda_peak_memory_bytes"] / 1024**3
            ),
        },
        artifacts=[
            ArtifactRecord(p, p.suffix.lstrip("."), purpose)
            for p, purpose in (
                (npz_path, "Serialized geometry belief"),
                (ply_path, "Inspectable point cloud"),
                (colmap_path / "cameras.txt", "COLMAP camera model"),
                (colmap_path / "images.txt", "COLMAP camera poses"),
                (metadata_path, "JSON metadata"),
                (report_path, "Interactive diagnostics"),
            )
        ],
        limitations=[
            "Adapter confidence is not calibrated error.",
            "Single-image metric scale is ambiguous without a prior.",
        ],
        next_actions=["Evaluate depth, pose, tracks, and calibration on a versioned held-out dataset."],
    ).write(output / "EXPERIMENT.md")
    logger.log_artifact(npz_path, "geometry-belief")
    logger.log_artifact(ply_path, "point-cloud")
    logger.log_artifact(report_path, "interactive-report")
    logger.finish(
        {
            "result": "success",
            "scale_confidence": belief.scale_confidence.tolist(),
            "pose_confidence": belief.pose_confidence.tolist(),
            "reconstruction_confidence": belief.reconstruction_confidence.tolist(),
        }
    )
    typer.echo(
        json.dumps(
            {
                "belief": str(npz_path),
                "pointcloud": str(ply_path),
                "colmap": str(colmap_path),
                "metadata": str(metadata_path),
                "report": str(report_path),
                "experiment": str(experiment_path),
                "wandb_url": logger.url,
                "depth_shape": None if belief.depth_mean is None else list(belief.depth_mean.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
