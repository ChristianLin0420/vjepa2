"""Build and export a geometry belief from RGB images or a short video."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from jepa4d.data.rgb_input import load_rgb_input
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.geometry_export import export_geometry_npz, export_pointcloud_ply
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
            "input": batch.to_serializable(),
        },
    )
    belief = head(batch)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = export_geometry_npz(belief, output / "geometry_belief.npz")
    ply_path = export_pointcloud_ply(belief, batch, output / "pointcloud.ply")
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(belief.to_serializable(), indent=2) + "\n")
    logger.log_geometry_belief(batch, belief)
    report_path = build_geometry_report(batch, belief, output / "report.html", wandb_url=logger.url)
    experiment_path = output / "EXPERIMENT.md"
    experiment_path.write_text(
        f"# Geometry experiment: {name}\n\n"
        f"- Timestamp: {datetime.now(UTC).isoformat()}\n"
        f"- Backend: `{backend}`\n- Input mode: `{batch.mode}`\n"
        f"- Views/timesteps: `{batch.images.shape[1]}/{batch.images.shape[2]}`\n"
        f"- Scale confidence: `{belief.scale_confidence.tolist()}`\n"
        f"- Pose confidence: `{belief.pose_confidence.tolist()}`\n"
        f"- Reconstruction confidence: `{belief.reconstruction_confidence.tolist()}`\n"
        f"- Runtime: `{belief.metadata['runtime_seconds']:.6f} s`\n"
        f"- W&B: {logger.url or 'disabled'}\n\n"
        "## Interpretation\n\n"
        "Confidence values describe the adapter's belief and are not accuracy claims until calibrated on held-out geometry. "
        "Uncalibrated single-image scale is deliberately low.\n\n"
        f"## Artifacts\n\n- `{npz_path}`\n- `{ply_path}`\n- `{metadata_path}`\n- `{report_path}`\n"
    )
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
