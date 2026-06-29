"""Build immutable Phase-2e SUN RGB-D V-JEPA/VGGT feature caches on CUDA."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn.functional as F
import typer

from jepa4d.benchmarks.geometry.sun_rgbd import load_sensor_blocked_manifest, sha256_file
from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.phase2e_feature_cache import (
    CACHE_SCHEMA,
    FROZEN_MANIFEST_SHA256,
    FROZEN_MODEL_IDENTITIES,
    FROZEN_SPLIT_SHA256,
    RECEIPT_SCHEMA,
    VIEW_POLICY,
    PreparedSplit,
    SplitName,
    build_separate_cache_payloads,
    centered_log_depth_teacher,
    normalize_final_features,
    prepare_sunrgbd_split,
    write_feature_cache,
)
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.phase2e_cache_report import build_phase2e_cache_report

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _path_identity(path: Path) -> dict[str, Any]:
    root = path.resolve(strict=True)
    files = (
        [root]
        if root.is_file()
        else sorted(
            value
            for value in root.rglob("*")
            if value.is_file() and "__pycache__" not in value.parts and value.suffix != ".pyc"
        )
    )
    if not files:
        raise ValueError(f"model asset has no files: {root}")
    combined = hashlib.sha256()
    total_bytes = 0
    for value in files:
        relative = value.name if root.is_file() else value.relative_to(root).as_posix()
        digest = sha256_file(value)
        size = value.stat().st_size
        combined.update(relative.encode())
        combined.update(str(size).encode())
        combined.update(digest.encode())
        total_bytes += size
    return {
        "path": str(root),
        "kind": "file" if root.is_file() else "directory",
        "files": len(files),
        "bytes": total_bytes,
        "content_manifest_sha256": combined.hexdigest(),
    }


def _single_image_batch(images: torch.Tensor) -> Any:
    return collate_rgb_inputs([from_view_sequences([[image]]) for image in images])


def _extract_vjepa_final(
    extractor: VJEPA21FeatureExtractor,
    prepared: PreparedSplit,
    *,
    chunk_size: int,
    progress: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if extractor.embed_dim != 768:
        raise ValueError(f"Phase-2e cache requires V-JEPA ViT-B width 768, found {extractor.embed_dim}")
    flattened = prepared.images_384.flatten(0, 1)
    grids = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(torch.device(extractor.device_name))
    with torch.inference_mode():
        for offset in range(0, len(flattened), chunk_size):
            bundle = extractor(_single_image_batch(flattened[offset : offset + chunk_size]))
            grid = bundle.dense_tokens[:, 0, 0].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2)
            grids.append(grid.detach().cpu().contiguous().half())
            progress("vjepa", prepared.name, min(offset + chunk_size, len(flattened)), len(flattened))
    torch.cuda.synchronize(torch.device(extractor.device_name))
    values = torch.cat(grids).reshape(prepared.count, prepared.views, 768, 24, 24)
    return values, {
        "frames": len(flattened),
        "seconds": time.perf_counter() - started,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated(torch.device(extractor.device_name)) / 1024**3,
    }


def _extract_vggt_teacher(
    checkpoint: Path,
    prepared_train: PreparedSplit,
    *,
    device: str,
    chunk_size: int,
    progress: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if prepared_train.name != "train" or prepared_train.views != 2:
        raise ValueError("VGGT teacher extraction accepts only the paired training split")
    head = GeometryBeliefHead(
        backend="vggt",
        device=device,
        model_id=str(checkpoint),
        precision="bfloat16",
    )
    flattened = prepared_train.images_384.flatten(0, 1)
    depths = []
    started = time.perf_counter()
    peak_memory_gb = 0.0
    with torch.inference_mode():
        for offset in range(0, len(flattened), chunk_size):
            belief = head(_single_image_batch(flattened[offset : offset + chunk_size]))
            if belief.depth_mean is None:
                raise RuntimeError("VGGT returned no depth teacher")
            depth = belief.depth_mean[:, 0, 0].float()
            depth_24 = F.interpolate(depth.unsqueeze(1), size=(24, 24), mode="bilinear", align_corners=False)[:, 0]
            if not torch.isfinite(depth_24).all() or not (depth_24 > 0).all():
                raise ValueError("VGGT teacher depth is non-finite or non-positive")
            depths.append(depth_24.cpu())
            peak = belief.metadata.get("cuda_peak_memory_bytes")
            peak_memory_gb = max(peak_memory_gb, 0.0 if peak is None else float(peak) / 1024**3)
            progress("vggt_teacher", "train", min(offset + chunk_size, len(flattened)), len(flattened))
    values = torch.cat(depths).reshape(prepared_train.count, 2, 24, 24)
    centered = centered_log_depth_teacher(values)
    del head
    torch.cuda.empty_cache()
    return centered, {
        "frames": len(flattened),
        "seconds": time.perf_counter() - started,
        "peak_cuda_memory_gb": peak_memory_gb,
        "precision": "bfloat16",
        "split": "train-only",
        "output": "centered-log-depth-shape-no-metric-scale",
    }


def _normalizer_identity(normalizer: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(normalizer):
        value = normalizer[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _split_summary(
    prepared: PreparedSplit,
    raw_features: torch.Tensor,
    normalized_features: torch.Tensor,
) -> dict[str, Any]:
    cached_prefix = [prepared.count, 2] if prepared.name == "train" else [prepared.count]
    summary: dict[str, Any] = {
        "samples": prepared.count,
        "views": prepared.views,
        "sensor_counts": dict(sorted(Counter(prepared.sensor_ids).items())),
        "shapes": {
            "features": [*cached_prefix, 768, 24, 24],
            "rgb": [*cached_prefix, 3, 96, 96],
            "intrinsics_384": [*cached_prefix, 3, 3],
            "targets": [*cached_prefix, 24, 24],
        },
    }
    if prepared.name == "test":
        summary["input_tensors"] = {
            "access": "opaque_until_final_evaluation",
            "schema_validation": "pass",
            "statistics_computed": False,
            "preview_generated": False,
        }
        summary["target_tensor"] = {
            "access": "opaque_until_final_evaluation",
            "schema_validation": "pass",
            "shape": [*cached_prefix, 24, 24],
            "dtype": "float32",
            "statistics_computed": False,
            "preview_generated": False,
        }
        return summary

    intrinsics = prepared.intrinsics_384.float()
    summary["intrinsics_384"] = {
        "fx_mean": float(intrinsics[..., 0, 0].mean()),
        "fy_mean": float(intrinsics[..., 1, 1].mean()),
        "cx_mean": float(intrinsics[..., 0, 2].mean()),
        "cy_mean": float(intrinsics[..., 1, 2].mean()),
        "fx_min": float(intrinsics[..., 0, 0].min()),
        "fx_max": float(intrinsics[..., 0, 0].max()),
    }
    summary["raw_features"] = {
        "mean": float(raw_features.float().mean()),
        "std": float(raw_features.float().std(unbiased=False)),
        "min": float(raw_features.float().min()),
        "max": float(raw_features.float().max()),
        "finite_fraction": float(torch.isfinite(raw_features).float().mean()),
    }
    summary["normalized_features"] = {
        "mean": float(normalized_features.float().mean()),
        "std": float(normalized_features.float().std(unbiased=False)),
        "min": float(normalized_features.float().min()),
        "max": float(normalized_features.float().max()),
        "finite_fraction": float(torch.isfinite(normalized_features).float().mean()),
    }
    valid = torch.isfinite(prepared.targets_24) & (prepared.targets_24 > 0.1) & (prepared.targets_24 < 10.0)
    depth = prepared.targets_24[valid].float()
    summary["target_depth_m"] = {
        "valid_fraction": float(valid.float().mean()),
        "valid_min_m": float(depth.min()),
        "valid_max_m": float(depth.max()),
        "valid_mean_m": float(depth.mean()),
        "valid_std_m": float(depth.std(unbiased=False)),
    }
    return summary


def _previews(prepared: dict[str, PreparedSplit], limit: int = 8) -> list[dict[str, Any]]:
    output = []
    # Test targets remain opaque until the one final evaluator. Cache-stage
    # visualizations are deliberately limited to train and validation.
    for split in ("train", "validation"):
        value = prepared[split]
        selected_sensors: set[str] = set()
        for index, sensor in enumerate(value.sensor_ids):
            if sensor in selected_sensors:
                continue
            selected_sensors.add(sensor)
            output.append(
                {
                    "label": f"{split}/{sensor}/center_square",
                    "rgb": value.rgb_96[index, 0].permute(1, 2, 0).float().numpy(),
                    "depth": value.targets_24[index, 0].float().numpy(),
                    "sample_id": value.sample_ids[index],
                }
            )
            if split == "train" and value.views == 2 and len(output) < limit:
                output.append(
                    {
                        "label": f"{split}/{sensor}/center_crop_0.85",
                        "rgb": value.rgb_96[index, 1].permute(1, 2, 0).float().numpy(),
                        "depth": value.targets_24[index, 1].float().numpy(),
                        "sample_id": value.sample_ids[index],
                    }
                )
            if len(output) >= limit:
                return output
    return output


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    vjepa_checkpoint: Annotated[Path, typer.Option("--vjepa-checkpoint")],
    vjepa_implementation: Annotated[Path, typer.Option("--vjepa-implementation")],
    vggt_checkpoint: Annotated[Path, typer.Option("--vggt-checkpoint")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/jepa4d_phase2e/sunrgbd_cache"),
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    vjepa_chunk_size: Annotated[int, typer.Option("--vjepa-chunk-size")] = 8,
    vggt_chunk_size: Annotated[int, typer.Option("--vggt-chunk-size")] = 2,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_entity: Annotated[str | None, typer.Option("--wandb-entity")] = None,
    run_name: Annotated[str, typer.Option("--run-name")] = "phase2e-sunrgbd-feature-cache",
) -> None:
    """Build hash-bound train/validation and isolated-test caches with online W&B."""
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"output directory must be new or empty: {output}")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise typer.BadParameter("Phase-2e real-model cache building requires an allocated CUDA device")
    if vjepa_chunk_size <= 0 or vggt_chunk_size <= 0:
        raise typer.BadParameter("model chunk sizes must be positive")
    paths = {
        "dataset_root": dataset_root.resolve(strict=True),
        "manifest": manifest.resolve(strict=True),
        "vjepa_checkpoint": vjepa_checkpoint.resolve(strict=True),
        "vjepa_implementation": vjepa_implementation.resolve(strict=True),
        "vggt_checkpoint": vggt_checkpoint.resolve(strict=True),
    }
    if sha256_file(paths["manifest"]) != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("SUN RGB-D manifest differs from the frozen Phase-2e protocol")
    asset_identities = {
        "vjepa_checkpoint": _path_identity(paths["vjepa_checkpoint"]),
        "vjepa_implementation": _path_identity(paths["vjepa_implementation"]),
        "vggt_checkpoint": _path_identity(paths["vggt_checkpoint"]),
    }
    for name, expected in FROZEN_MODEL_IDENTITIES.items():
        observed = {key: asset_identities[name][key] for key in expected}
        if observed != expected:
            raise RuntimeError(f"frozen Phase-2e model identity changed for {name}: {observed}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bundle = load_sensor_blocked_manifest(
        paths["dataset_root"], paths["manifest"], verify_file_hashes=True, validate_depth=True
    )
    if bundle.split_hash != FROZEN_SPLIT_SHA256:
        raise RuntimeError("SUN RGB-D split differs from the frozen Phase-2e protocol")
    clamp = bundle.manifest["protocol"]["depth_decode"]["clamp_max_depth_m"]

    import wandb

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        mode="online",
        job_type="phase2e-feature-cache",
        tags=["phase-2e", "SUN-RGBD", "feature-cache", "sensor-blocked", "cuda"],
        config={
            "dataset_root": str(paths["dataset_root"]),
            "manifest": str(paths["manifest"]),
            "split_hash": bundle.split_hash,
            "view_policy": VIEW_POLICY,
            "device": device,
            "vjepa_chunk_size": vjepa_chunk_size,
            "vggt_chunk_size": vggt_chunk_size,
            "model_metrics_computed": False,
        },
    )
    if run.offline:
        raise RuntimeError("Phase-2e cache build requires online W&B")
    run.define_metric("cache/progress_step")
    run.define_metric("cache/progress_fraction", step_metric="cache/progress_step")
    progress_step = 0

    def progress(stage: str, split: str, completed: int, total: int) -> None:
        nonlocal progress_step
        progress_step += 1
        event = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "split": split,
            "completed": completed,
            "total": total,
        }
        typer.echo(json.dumps(event, sort_keys=True))
        if completed == total or progress_step % 8 == 0:
            run.log(
                {
                    "cache/progress_step": progress_step,
                    "cache/progress_fraction": completed / max(total, 1),
                    "cache/stage": stage,
                    "cache/split": split,
                }
            )

    try:
        prepared: dict[str, PreparedSplit] = {}

        def preprocessing_progress(name: SplitName) -> Callable[[int, int], None]:
            def report(completed: int, total: int) -> None:
                progress("preprocess", name, completed, total)

            return report

        split_names: tuple[SplitName, ...] = ("train", "validation", "test")
        for split in split_names:
            prepared[split] = prepare_sunrgbd_split(
                split,
                bundle.splits[split],
                clamp_max_depth_m=clamp,
                progress=preprocessing_progress(split),
            )

        extractor = VJEPA21FeatureExtractor(
            model_name="vjepa2_1_vit_base_384",
            checkpoint=paths["vjepa_checkpoint"],
            implementation_path=paths["vjepa_implementation"],
            backend="hf_compat",
            device=device,
            frozen=True,
            capture_layers=(),
        )
        raw_features: dict[str, torch.Tensor] = {}
        vjepa_profiles = {}
        for split in ("train", "validation", "test"):
            raw_features[split], vjepa_profiles[split] = _extract_vjepa_final(
                extractor, prepared[split], chunk_size=vjepa_chunk_size, progress=progress
            )
        del extractor
        torch.cuda.empty_cache()
        train_features, validation_features, test_features, normalizer = normalize_final_features(
            raw_features["train"], raw_features["validation"], raw_features["test"]
        )
        normalized_features = {
            "train": train_features,
            "validation": validation_features,
            "test": test_features,
        }

        teacher, vggt_profile = _extract_vggt_teacher(
            paths["vggt_checkpoint"],
            prepared["train"],
            device=device,
            chunk_size=vggt_chunk_size,
            progress=progress,
        )
        train_validation_payload, test_payload = build_separate_cache_payloads(prepared, normalized_features, teacher)
        train_validation_path = write_feature_cache(output / "train_validation_cache.pt", train_validation_payload)
        test_path = write_feature_cache(output / "test_cache.pt", test_payload)
        normalizer_path = output / "feature_normalization.pt"
        torch.save(normalizer, normalizer_path)

        split_summaries = {
            split: _split_summary(prepared[split], raw_features[split], normalized_features[split])
            for split in ("train", "validation", "test")
        }
        preview_rows = _previews(prepared)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "pass",
            "evidence_level": "feature-cache-build",
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "dataset_id": bundle.manifest["dataset_id"],
                "version": bundle.manifest["version"],
                "manifest": str(paths["manifest"]),
                "manifest_sha256": sha256_file(paths["manifest"]),
                "split_hash": bundle.split_hash,
                "split_policy": bundle.manifest["protocol"]["split_policy"],
                "depth_decode": bundle.manifest["protocol"]["depth_decode"],
            },
            "models": asset_identities,
            "view_policy": {
                "name": VIEW_POLICY,
                "train": ["center_square", "center_crop_0.85"],
                "validation": ["center_square"],
                "test": ["center_square"],
                "resize": {
                    "vjepa_rgb": [384, 384],
                    "cached_rgb": [96, 96],
                    "target_depth": [24, 24],
                    "rgb_mode": "bilinear-align_corners_false",
                    "depth_mode": "nearest",
                },
                "intrinsics": "update_intrinsics_for_crop_resize with half-pixel centres",
            },
            "feature_normalization": {
                "policy": "channel mean/std over training samples, both train views, and 24x24 spatial grid only",
                "artifact": str(normalizer_path.resolve()),
                "sha256": sha256_file(normalizer_path),
                "content_sha256": _normalizer_identity(normalizer),
                "mean_min": float(normalizer["mean"].min()),
                "mean_max": float(normalizer["mean"].max()),
                "std_min": float(normalizer["std"].min()),
                "std_max": float(normalizer["std"].max()),
            },
            "teacher_policy": {
                "backend": "official VGGT-1B",
                "precision": "bfloat16",
                "split": "train-only",
                "views": ["center_square", "center_crop_0.85"],
                "target": "spatially centered log-depth at 24x24",
                "metric_scale_fitted": False,
                "validation_teacher_computed": False,
                "test_teacher_computed": False,
            },
            "caches": {
                "train_validation": {
                    "path": str(train_validation_path.resolve()),
                    "bytes": train_validation_path.stat().st_size,
                    "sha256": sha256_file(train_validation_path),
                    "schema_version": CACHE_SCHEMA,
                    "splits": ["train", "validation"],
                },
                "test": {
                    "path": str(test_path.resolve()),
                    "bytes": test_path.stat().st_size,
                    "sha256": sha256_file(test_path),
                    "schema_version": CACHE_SCHEMA,
                    "splits": ["test"],
                },
            },
            "split_summaries": split_summaries,
            "sample_metadata": {split: prepared[split].metadata_rows() for split in prepared},
            "profiles": {"vjepa": vjepa_profiles, "vggt": vggt_profile},
            "runtime": {
                "seconds": time.perf_counter() - started,
                "device": device,
                "gpu": torch.cuda.get_device_name(torch.device(device)),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "slurm": {
                    key: os.environ[key]
                    for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")
                    if key in os.environ
                },
            },
            "wandb_url": run.url,
            "model_metrics_computed": False,
            "large_caches_uploaded_to_wandb": False,
            "test_target_access": {
                "policy": "opaque cache payload only until final evaluation",
                "integrity_and_schema_validated": True,
                "statistics_computed": False,
                "preview_generated": False,
                "logged_to_wandb": False,
            },
        }
        report_path = build_phase2e_cache_report(receipt, preview_rows, output / "feature_cache_report.html")
        receipt["report"] = {
            "path": str(report_path.resolve()),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
            "self_contained": True,
        }
        receipt_path = output / "feature_cache_receipt.json"
        _write_json(receipt_path, receipt)

        sensor_table = wandb.Table(columns=["split", "sensor", "samples"])
        for split_name, summary in split_summaries.items():
            for sensor, count in summary["sensor_counts"].items():
                sensor_table.add_data(split_name, sensor, count)
        stats_table = wandb.Table(
            columns=[
                "split",
                "fx_mean",
                "fy_mean",
                "valid_depth_fraction",
                "depth_mean_m",
                "feature_mean",
                "feature_std",
            ]
        )
        for split_name in ("train", "validation"):
            summary = split_summaries[split_name]
            stats_table.add_data(
                split_name,
                summary["intrinsics_384"]["fx_mean"],
                summary["intrinsics_384"]["fy_mean"],
                summary["target_depth_m"]["valid_fraction"],
                summary["target_depth_m"]["valid_mean_m"],
                summary["normalized_features"]["mean"],
                summary["normalized_features"]["std"],
            )
        media = {}
        for index, preview in enumerate(preview_rows):
            media[f"cache/examples/{index}_rgb"] = wandb.Image(preview["rgb"], caption=preview["label"])
            media[f"cache/examples/{index}_depth"] = wandb.Image(preview["depth"], caption=preview["label"])
        run.log(
            {
                "cache/sensor_counts": sensor_table,
                "cache/statistics": stats_table,
                "cache/train_validation_sha256": receipt["caches"]["train_validation"]["sha256"],
                "cache/test_sha256": receipt["caches"]["test"]["sha256"],
                "cache/report": wandb.Html(str(report_path), inject=False),
                **media,
            }
        )
        artifact = wandb.Artifact(f"{run_name}-receipt", type="phase2e-feature-cache-receipt")
        artifact.add_file(str(receipt_path), name=receipt_path.name)
        artifact.add_file(str(report_path), name=report_path.name)
        uploaded = run.log_artifact(artifact).wait(timeout=900)
        wandb_receipt = {
            "schema_version": "jepa4d-phase2e-cache-wandb-receipt-v1",
            "status": "uploaded",
            "mode": "online",
            "run_id": str(run.id),
            "run_url": str(run.url),
            "run_path": str(run.path),
            "artifact_id": str(uploaded.id),
            "artifact_name": str(uploaded.name),
            "artifact_qualified_name": str(uploaded.qualified_name),
            "artifact_version": str(uploaded.version),
            "artifact_digest": str(uploaded.digest),
            "receipt_sha256": sha256_file(receipt_path),
            "report_sha256": sha256_file(report_path),
        }
        _write_json(output / "wandb_receipt.json", wandb_receipt)
        run.summary.update(
            {
                "result": "success",
                "split_hash": bundle.split_hash,
                "train_validation_cache_sha256": receipt["caches"]["train_validation"]["sha256"],
                "test_cache_sha256": receipt["caches"]["test"]["sha256"],
                "large_caches_uploaded": False,
                "model_metrics_computed": False,
            }
        )
        run.finish(exit_code=0)
        typer.echo(
            json.dumps(
                {
                    "train_validation_cache": str(train_validation_path),
                    "test_cache": str(test_path),
                    "normalization": str(normalizer_path),
                    "receipt": str(receipt_path),
                    "report": str(report_path),
                    "wandb_url": receipt["wandb_url"],
                },
                indent=2,
            )
        )
    except Exception:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    app()
