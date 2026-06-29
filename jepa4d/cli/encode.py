"""Feature extraction CLI for images, view sets, and short videos."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import torch
import typer

from jepa4d.data.rgb_input import load_rgb_input
from jepa4d.data.schemas import JEPATokenBundle
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.experiment_record import ArtifactRecord, ExperimentRecord, PanelRecord, StageRecord
from jepa4d.visualization.html_report import build_feature_report
from jepa4d.visualization.observability import ExperimentLogger

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _save_bundle(bundle: JEPATokenBundle, output: Path) -> Path:
    if output.suffix == ".zarr":
        import zarr

        group = zarr.open_group(str(output), mode="w")
        group.create_array("dense_tokens", data=bundle.dense_tokens.detach().cpu().numpy())
        group.create_array("global_tokens", data=bundle.global_tokens.detach().cpu().numpy())
        group.create_array("valid_mask", data=bundle.valid_mask.detach().cpu().numpy())
        layers = group.create_group("layer_tokens")
        for layer, value in bundle.layer_tokens.items():
            layers.create_array(str(layer), data=value.detach().cpu().numpy())
        group.attrs.update(
            {
                "patch_grid": bundle.patch_grid,
                "feature_scale": bundle.feature_scale,
                "modality": bundle.modality,
                "metadata": bundle.metadata,
            }
        )
        return output
    return bundle.save(output)


def _write_experiment_markdown(
    path: Path,
    *,
    name: str,
    inputs: list[Path],
    bundle: JEPATokenBundle,
    runtime: dict[str, float],
    wandb_url: str | None,
    artifacts: list[Path],
) -> Path:
    model = bundle.metadata["model"]
    return ExperimentRecord(
        title=f"Feature extraction: {name}",
        experiment_id=name,
        stage="representation",
        status="complete",
        evidence_level="contract-only" if model["backend"] == "mock" else "integration",
        objective="Extract inspectable dense, global, and intermediate V-JEPA tokens from a normalized RGB input.",
        hypothesis="The selected backend produces finite, non-degenerate tokens with preserved view/time identity.",
        decision="Use this run as representation substrate evidence; downstream task quality requires separate evaluation.",
        wandb_url=wandb_url,
        config={"inputs": [str(value) for value in inputs], "model": model, "runtime": runtime},
        stages=[
            StageRecord("input", "RGB loader", "pass", str([str(v) for v in inputs]), "normalized batch"),
            StageRecord(
                "features",
                str(model["backend"]),
                "pass",
                "normalized RGB",
                str(list(bundle.dense_tokens.shape)),
                "Tokens are finite; task usefulness is not measured by this integration run.",
            ),
        ],
        panels=[
            PanelRecord("visualizations/input", "image", "Verify the exact input and preprocessing."),
            PanelRecord("visualizations/pca_rgb", "image", "Inspect dense spatial feature structure."),
            PanelRecord("features/value_histogram", "histogram", "Detect collapse and outliers."),
            PanelRecord("features/norm_histogram", "histogram", "Inspect token magnitude distribution."),
            PanelRecord("visualizations/temporal_consistency", "line", "Inspect adjacent-frame latent similarity."),
            PanelRecord("features/layer_summary", "table", "Compare intermediate representation statistics."),
            PanelRecord("inference/*", "scalar", "Measure latency and throughput."),
        ],
        metrics={
            "dense_token_shape": list(bundle.dense_tokens.shape),
            "all_finite": bool(torch.isfinite(bundle.dense_tokens).all()),
            "feature_mean": bundle.dense_tokens.float().mean().item(),
            "feature_std": bundle.dense_tokens.float().std().item(),
            "runtime_s": runtime["total_s"],
        },
        artifacts=[
            ArtifactRecord(value, value.suffix.lstrip(".") or "directory", "Reproducible run output")
            for value in artifacts
        ],
        limitations=["PCA and temporal similarity are diagnostics, not semantic or tracking accuracy metrics."],
        next_actions=["Evaluate the representation on a named downstream task or attach the next structured adapter."],
    ).write(path)


@app.command()
def main(
    input: Annotated[list[Path], typer.Option("--input", "-i", help="Image paths or one video path")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/features.pt"),
    model: Annotated[str, typer.Option("--model")] = "vjepa2_1_vit_base_384",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    mock: Annotated[bool, typer.Option("--mock")] = False,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    max_frames: Annotated[int, typer.Option("--max-frames")] = 16,
    report: Annotated[Path | None, typer.Option("--report")] = None,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_mode: Annotated[str, typer.Option("--wandb-mode")] = "online",
    run_name: Annotated[str | None, typer.Option("--run-name")] = None,
) -> None:
    """Extract and persist dense JEPA features."""
    started = time.perf_counter()
    batch = load_rgb_input(input, max_frames=max_frames)
    extractor = VJEPA21FeatureExtractor(model_name=model, mock=mock, checkpoint=checkpoint, device=device, frozen=True)
    name = (
        run_name
        or f"{'mock' if mock else 'real'}-{model}-{batch.mode}-{batch.images.shape[1]}v{batch.images.shape[2]}t"
    )
    logger = ExperimentLogger(
        enabled=wandb,
        project=wandb_project,
        name=name,
        mode=wandb_mode,
        tags=["phase-1", "feature-extraction", batch.mode, "mock" if mock else "real"],
        config={"model": extractor.model_config, "input": batch.to_serializable(), "output": str(output)},
    )
    bundle = extractor(batch)
    runtime = {"total_s": time.perf_counter() - started, "model_s": bundle.metadata["forward_seconds"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_path = _save_bundle(bundle, output)
    metadata_path = output.parent / "metadata.json"
    bundle.write_metadata(metadata_path)
    logger.log_feature_bundle(batch, bundle, runtime)
    report_path = build_feature_report(
        batch, bundle, report or output.parent / "report.html", runtime=runtime, wandb_url=logger.url
    )
    experiment_path = _write_experiment_markdown(
        output.parent / "EXPERIMENT.md",
        name=name,
        inputs=input,
        bundle=bundle,
        runtime=runtime,
        wandb_url=logger.url,
        artifacts=[feature_path, metadata_path, report_path],
    )
    logger.log_artifact(feature_path)
    logger.log_artifact(report_path, artifact_type="interactive-report")
    logger.finish({"result": "success", "dense_shape": list(bundle.dense_tokens.shape), "report": str(report_path)})
    typer.echo(
        json.dumps(
            {
                "features": str(feature_path),
                "metadata": str(metadata_path),
                "report": str(report_path),
                "experiment": str(experiment_path),
                "wandb_url": logger.url,
                "shape": list(bundle.dense_tokens.shape),
                "runtime": runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
