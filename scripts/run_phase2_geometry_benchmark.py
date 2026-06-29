"""Run the official TUM RGB-D Phase 2 quality/calibration benchmark on CUDA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer

from jepa4d.benchmarks.geometry.tum_rgbd import (
    calibration_metrics,
    depth_metrics,
    load_depth,
    load_tum_subset,
    point_metrics,
    pose_metrics,
    validate_archive,
)
from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.geometry_export import export_colmap_text, export_geometry_npz, export_pointcloud_ply
from jepa4d.visualization.observability import ExperimentLogger

app = typer.Typer(add_completion=False)


def _ground_truth_intrinsics(manifest: dict[str, Any], output_size: tuple[int, int]) -> torch.Tensor:
    camera = manifest["camera"]
    height, width = output_size
    return torch.tensor(
        [
            [camera["fx"] * width / camera["width"], 0.0, camera["cx"] * width / camera["width"]],
            [0.0, camera["fy"] * height / camera["height"], camera["cy"] * height / camera["height"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _mean_metrics(per_sample: list[dict[str, Any]], indices: list[int]) -> dict[str, float]:
    keys = sorted({key for index in indices for key in per_sample[index]["metrics"]})
    return {key: float(np.mean([per_sample[index]["metrics"][key] for index in indices])) for key in keys}


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")],
    archive: Annotated[Path, typer.Option("--archive")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/jepa4d_phase2/tum_rgbd_a100"),
    manifest_path: Annotated[Path, typer.Option("--manifest")] = Path(
        "jepa4d/config/benchmarks/manifests/tum_rgbd_freiburg1_xyz_mini.yaml"
    ),
    model_id: Annotated[str, typer.Option("--model-id")] = "checkpoints/VGGT-1B",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    run_name: Annotated[str, typer.Option("--run-name")] = "phase2-tum-rgbd-vggt-a100",
) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise typer.BadParameter("Phase 2 quality benchmark requires an available CUDA device")
    manifest = validate_archive(archive, manifest_path)
    selection = manifest["selection"]
    samples = load_tum_subset(
        dataset_root,
        frame_count=int(selection["frame_count"]),
        max_delta=float(selection["timestamp_max_delta_seconds"]),
    )
    timestamps = torch.tensor([[sample.timestamp for sample in samples]], dtype=torch.float64)
    batch = from_view_sequences([[sample.rgb_path for sample in samples]], timestamps=timestamps)
    camera = manifest["camera"]
    batch.intrinsics = torch.tensor(
        [[[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    logger = ExperimentLogger(
        enabled=wandb,
        project=wandb_project,
        name=run_name,
        tags=["phase-2", "geometry", "TUM-RGBD", "official-subset", "cuda", "calibration"],
        config={
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "model_id": model_id,
            "device": device,
            "frame_count": len(samples),
            "calibration_indices": selection["calibration_indices"],
            "test_indices": selection["test_indices"],
        },
        mode="online",
    )
    head = GeometryBeliefHead(backend="vggt", device=device, model_id=model_id, precision="float32")
    belief = head(batch)
    logger.log_geometry_belief(batch, belief)
    assert belief.depth_mean is not None and belief.depth_logvar is not None
    assert belief.camera_extrinsics is not None and belief.camera_intrinsics is not None
    output_resolution = tuple(belief.depth_mean.shape[-2:])
    targets = torch.stack([load_depth(sample.depth_path, output_resolution) for sample in samples])
    predictions = belief.depth_mean[0, 0].detach().cpu()
    logvars = belief.depth_logvar[0, 0].detach().cpu()
    gt_intrinsics = _ground_truth_intrinsics(manifest, output_resolution)
    per_sample: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sample in enumerate(samples):
        try:
            metrics, alignment_scale, _ = depth_metrics(predictions[index], targets[index])
            metrics.update(point_metrics(predictions[index], targets[index], gt_intrinsics))
            per_sample.append(
                {
                    "sample_id": sample.sample_id,
                    "timestamp": sample.timestamp,
                    "split": "calibration" if index in selection["calibration_indices"] else "test",
                    "alignment_scale": alignment_scale,
                    "metrics": metrics,
                }
            )
        except Exception as error:
            failures.append({"sample_id": sample.sample_id, "error": f"{type(error).__name__}: {error}"})
            per_sample.append({"sample_id": sample.sample_id, "timestamp": sample.timestamp, "metrics": {}})
    if failures:
        raise RuntimeError(f"per-sample metric failures: {failures}")
    calibration_indices = [int(value) for value in selection["calibration_indices"]]
    test_indices = [int(value) for value in selection["test_indices"]]
    aggregate = _mean_metrics(per_sample, test_indices)
    aggregate.update(
        calibration_metrics(predictions, targets, logvars, calibration_indices, test_indices)
    )
    aggregate.update(pose_metrics(belief.camera_extrinsics[0, 0], samples))
    aggregate.update(
        {
            "runtime_seconds_8_frames_float32": float(belief.metadata["runtime_seconds"]),
            "cuda_peak_memory_gb_8_frames_float32": float(belief.metadata["cuda_peak_memory_bytes"]) / 1024**3,
            "finite_fraction": float(torch.isfinite(predictions).float().mean()),
            "test_frames": float(len(test_indices)),
        }
    )

    profiles: list[dict[str, float | int | str]] = []
    for precision in ("float32", "bfloat16"):
        head.precision = precision
        for frame_count in (1, 2, 4, 8):
            profile_batch = from_view_sequences(
                [[sample.rgb_path for sample in samples[:frame_count]]],
                timestamps=timestamps[:, :frame_count],
            )
            profile_belief = head(profile_batch)
            profiles.append(
                {
                    "precision": precision,
                    "frames": frame_count,
                    "runtime_seconds": float(profile_belief.metadata["runtime_seconds"]),
                    "peak_memory_gb": float(profile_belief.metadata["cuda_peak_memory_bytes"]) / 1024**3,
                    "finite_fraction": float(torch.isfinite(profile_belief.depth_mean).float().mean()),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    npz_path = export_geometry_npz(belief, output / "geometry_belief.npz")
    ply_path = export_pointcloud_ply(belief, batch, output / "pointcloud.ply")
    colmap_path = export_colmap_text(belief, batch, output / "colmap")
    report = {
        "experiment_id": run_name,
        "evidence_level": "official-mini-subset",
        "dataset": manifest,
        "model_id": model_id,
        "device": torch.cuda.get_device_name(torch.device(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "metrics": aggregate,
        "profiles": profiles,
        "per_sample": per_sample,
        "failures": failures,
        "claim_boundary": (
            "Eight deterministic frames from one TUM RGB-D sequence; depth/point metrics are per-frame median-scale "
            "aligned and do not establish metric scale or cross-scene generalization."
        ),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    failures_path = output / "failures.json"
    failures_path.write_text(json.dumps(failures, indent=2) + "\n")
    experiment_path = output / "EXPERIMENT.md"
    experiment_path.write_text(
        "\n".join(
            [
                "# Phase 2 TUM RGB-D A100 benchmark",
                "",
                f"- W&B: {logger.url}",
                f"- Dataset: `{manifest['dataset_id']}` / `{manifest['revision']}`",
                f"- GPU: `{report['device']}`",
                f"- Test AbsRel (median aligned): `{aggregate['abs_rel']:.6f}`",
                f"- Test RMSE (median aligned): `{aggregate['rmse_m']:.6f} m`",
                f"- Sim(3) ATE RMSE: `{aggregate['pose_ate_rmse_m_sim3']:.6f} m`",
                f"- Calibrated NLL: `{aggregate['uncertainty_calibrated_nll']:.6f}`",
                f"- Eight-frame FP32 runtime: `{aggregate['runtime_seconds_8_frames_float32']:.6f} s`",
                f"- Eight-frame FP32 peak memory: `{aggregate['cuda_peak_memory_gb_8_frames_float32']:.6f} GiB`",
                "",
                "## Claim boundary",
                "",
                report["claim_boundary"],
                "",
            ]
        )
    )
    if logger.run is not None:
        import wandb as wandb_module

        sample_table = wandb_module.Table(columns=["sample", "split", *sorted(per_sample[0]["metrics"])])
        for value in per_sample:
            sample_table.add_data(
                value["sample_id"], value["split"], *[value["metrics"][key] for key in sorted(value["metrics"])]
            )
        profile_table = wandb_module.Table(
            columns=["precision", "frames", "runtime_seconds", "peak_memory_gb", "finite_fraction"]
        )
        for value in profiles:
            profile_table.add_data(*value.values())
        logger.run.log(
            {
                **{f"phase2_quality/{key}": value for key, value in aggregate.items()},
                "phase2_quality/per_sample": sample_table,
                "phase2_quality/cuda_profile": profile_table,
                "phase2_quality/failures": len(failures),
            }
        )
    for path, artifact_type in (
        (report_path, "benchmark-report"),
        (failures_path, "benchmark-failures"),
        (experiment_path, "experiment-record"),
        (npz_path, "geometry-belief"),
        (ply_path, "point-cloud"),
        (colmap_path / "cameras.txt", "colmap-model"),
        (colmap_path / "images.txt", "colmap-model"),
    ):
        logger.log_artifact(path, artifact_type)
    logger.finish({"result": "success", **aggregate, "failures": len(failures)})
    typer.echo(json.dumps({"report": str(report_path), "wandb_url": logger.url, "metrics": aggregate}, indent=2))


if __name__ == "__main__":
    app()
