"""Run post-hoc same-checkpoint attribution for the completed Phase-2c fusion gate.

The script performs no optimization.  It reloads the immutable Phase-2c
comparison, checkpoints, and training-only normalization, recomputes frozen
validation/test V-JEPA features, and evaluates deterministic gate interventions.
Run real-model execution inside a Slurm GPU allocation.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from PIL import Image

from jepa4d.benchmarks.geometry.tum_rgbd_bundle import load_cross_sequence_bundle
from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.fusion_attribution import (
    GateIntervention,
    build_attribution_record,
    evaluate_checkpoint_attribution,
    load_phase2c_artifacts,
    normalize_phase2c_feature_grids,
    sha256,
    write_full_predictions_npz,
    write_qualitative_examples_npz,
)
from jepa4d.evaluation.phase2c_source import validate_phase2c_source
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.fusion_attribution_report import build_fusion_attribution_report
from slurm.validate_phase2d_test_receipt import validate_receipt

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _center_crop_square(value: torch.Tensor) -> torch.Tensor:
    height, width = value.shape[-2:]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return value[..., top : top + size, left : left + size]


def _images(samples: Sequence[Any], size: int = 384) -> torch.Tensor:
    values = [
        torch.from_numpy(np.asarray(Image.open(sample.rgb_path).convert("RGB"), dtype=np.uint8).copy()).permute(
            2, 0, 1
        )
        for sample in samples
    ]
    cropped = torch.stack([_center_crop_square(value) for value in values]).float() / 255.0
    return F.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)


def _targets(samples: Sequence[Any], size: tuple[int, int]) -> torch.Tensor:
    values = []
    for sample in samples:
        depth_scale = float(sample.depth_scale)
        if depth_scale <= 0:
            raise ValueError(f"invalid depth scale for {sample.sample_id}: {depth_scale}")
        raw = np.asarray(Image.open(sample.depth_path), dtype=np.uint16).copy()
        values.append(_center_crop_square(torch.from_numpy(raw.astype(np.float32) / depth_scale)))
    return F.interpolate(torch.stack(values).unsqueeze(1), size=size, mode="nearest")[:, 0]


def _single_image_batch(samples: Sequence[Any]) -> Any:
    return collate_rgb_inputs([from_view_sequences([[image]]) for image in _images(samples)])


def _extract_feature_grids(
    extractor: VJEPA21FeatureExtractor,
    samples: Sequence[Any],
    *,
    chunk_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks: dict[str, list[torch.Tensor]] = {
        "vjepa_final": [],
        "vjepa_layer_2": [],
        "vjepa_layer_5": [],
        "vjepa_layer_8": [],
    }
    started = time.perf_counter()
    if torch.cuda.is_available() and str(extractor.device_name).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for offset in range(0, len(samples), chunk_size):
            bundle = extractor(_single_image_batch(samples[offset : offset + chunk_size]))
            final = bundle.dense_tokens[:, 0, 0].reshape(-1, 24, 24, extractor.embed_dim).permute(0, 3, 1, 2)
            chunks["vjepa_final"].append(final.detach().cpu().contiguous().half())
            for layer in (2, 5, 8):
                value = bundle.layer_tokens[layer][:, 0, 0]
                grid = value.reshape(-1, 24, 24, extractor.embed_dim).permute(0, 3, 1, 2)
                chunks[f"vjepa_layer_{layer}"].append(grid.detach().cpu().contiguous().half())
            typer.echo(
                json.dumps(
                    {
                        "stage": "feature_extraction",
                        "completed": min(offset + chunk_size, len(samples)),
                        "total": len(samples),
                    },
                    sort_keys=True,
                )
            )
    elapsed = time.perf_counter() - started
    peak_memory = None
    if torch.cuda.is_available() and str(extractor.device_name).startswith("cuda"):
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    grids = {key: torch.cat(values) for key, values in chunks.items()}
    return grids, {
        "frames": len(samples),
        "seconds": elapsed,
        "ms_per_frame": elapsed * 1000.0 / max(len(samples), 1),
        "peak_cuda_memory_gb": peak_memory,
        "shapes": {key: list(value.shape) for key, value in grids.items()},
    }


def _resolve_path(value: Path | None, fallback: str | None, label: str) -> Path:
    path = value if value is not None else None if fallback is None else Path(fallback)
    if path is None or not path.exists():
        raise typer.BadParameter(f"{label} does not exist: {path}")
    return path.resolve()


def _select_qualitative_indices(sequence_ids: Sequence[str], maximum: int = 4) -> list[int]:
    """Choose a deterministic bounded set while balancing the test sequences."""
    if maximum <= 0:
        raise ValueError("qualitative sample maximum must be positive")
    grouped: dict[str, list[int]] = {}
    for index, sequence_id in enumerate(sequence_ids):
        grouped.setdefault(str(sequence_id), []).append(index)
    if not grouped:
        raise ValueError("cannot select qualitative examples from an empty test split")
    candidates: dict[str, list[int]] = {}
    for sequence_id, indices in sorted(grouped.items()):
        positions = np.linspace(0, len(indices) - 1, num=min(maximum, len(indices)), dtype=np.int64)
        candidates[sequence_id] = list(dict.fromkeys(indices[position] for position in positions))
    selected: list[int] = []
    offset = 0
    while len(selected) < min(maximum, len(sequence_ids)):
        added = False
        for sequence_id in sorted(candidates):
            values = candidates[sequence_id]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == min(maximum, len(sequence_ids)):
                    break
        if not added:
            break
        offset += 1
    if len(selected) != min(maximum, len(sequence_ids)):
        raise RuntimeError("qualitative selection did not produce the requested bounded sample count")
    return selected


def _execution_provenance(repo_root: Path, test_receipt: Path) -> dict[str, Any]:
    """Bind this GPU inference job to its clean tested commit and Slurm allocation."""
    resolved = test_receipt.resolve(strict=True)
    receipt = validate_receipt(repo_root, resolved)
    environment_names = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION",
        "nodelist": "SLURM_JOB_NODELIST",
    }
    slurm = {name: str(os.getenv(variable, "")).strip() for name, variable in environment_names.items()}
    if any(not value for value in slurm.values()):
        raise RuntimeError("Phase 2d attribution requires a complete Slurm allocation identity")
    receipt_slurm = receipt.get("slurm")
    if not isinstance(receipt_slurm, dict):
        raise RuntimeError("passing test receipt has no Slurm allocation")
    return {
        "git_commit": str(receipt["git_commit"]),
        "test_receipt": {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
            "test_job_id": str(receipt_slurm["SLURM_JOB_ID"]),
        },
        "slurm": slurm,
    }


@app.command()
def main(
    phase2c_output: Annotated[Path, typer.Option("--phase2c-output", help="Completed formal Phase-2c output")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/jepa4d_phase2d/fusion_attribution"),
    test_receipt: Annotated[Path, typer.Option("--test-receipt", exists=True, dir_okay=False)] = Path(
        "outputs/phase2d-gates/tests.json"
    ),
    dataset_parent: Annotated[Path | None, typer.Option("--dataset-parent")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    vjepa_checkpoint: Annotated[Path | None, typer.Option("--vjepa-checkpoint")] = None,
    vjepa_implementation: Annotated[Path | None, typer.Option("--vjepa-implementation")] = None,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    chunk_size: Annotated[int, typer.Option("--chunk-size")] = 8,
    probe_batch_size: Annotated[int, typer.Option("--probe-batch-size")] = 8,
) -> None:
    """Evaluate gate causality using each trained Phase-2c learned-fusion probe."""
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"output must be new or empty: {output}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise typer.BadParameter(f"requested {device}, but CUDA is unavailable")
    if chunk_size <= 0 or probe_batch_size <= 0:
        raise typer.BadParameter("chunk sizes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    execution_provenance = _execution_provenance(repo_root, test_receipt)
    artifacts = load_phase2c_artifacts(phase2c_output)
    resolved_config_path = artifacts.root / "resolved_config.json"
    if not resolved_config_path.is_file():
        raise typer.BadParameter("Phase-2c output has no resolved_config.json")
    resolved_config = json.loads(resolved_config_path.read_text())
    dataset_parent_path = _resolve_path(dataset_parent, resolved_config.get("dataset_root"), "dataset parent")
    manifest_path = _resolve_path(manifest, resolved_config.get("manifest"), "bundle manifest")
    checkpoint_path = _resolve_path(vjepa_checkpoint, resolved_config.get("vjepa_checkpoint"), "V-JEPA checkpoint")
    implementation_path = _resolve_path(
        vjepa_implementation, resolved_config.get("vjepa_implementation"), "V-JEPA implementation"
    )
    source_identity = validate_phase2c_source(
        artifacts.root,
        dataset_manifest=manifest_path,
        vjepa_checkpoint=checkpoint_path,
        vjepa_implementation=implementation_path,
    )
    source_identity_path = output / "source_identity.json"
    _write_json(source_identity_path, source_identity)
    bundle = load_cross_sequence_bundle(dataset_parent_path, manifest_path)
    if bundle.split_hash != artifacts.comparison.get("split_hash"):
        raise RuntimeError("dataset bundle split hash does not match the completed Phase-2c comparison")
    validation_samples = bundle.splits["validation"]
    test_samples = bundle.splits["test"]

    typer.echo(json.dumps({"stage": "model_loading", "device": device}, sort_keys=True))
    extractor = VJEPA21FeatureExtractor(
        checkpoint=checkpoint_path,
        implementation_path=implementation_path,
        backend="hf_compat",
        device=device,
        frozen=True,
        capture_layers=(2, 5, 8),
    )
    validation_grids, validation_profile = _extract_feature_grids(extractor, validation_samples, chunk_size=chunk_size)
    test_grids, test_profile = _extract_feature_grids(extractor, test_samples, chunk_size=chunk_size)
    validation_features = normalize_phase2c_feature_grids(validation_grids, artifacts.normalization)
    test_features = normalize_phase2c_feature_grids(test_grids, artifacts.normalization)
    del validation_grids, test_grids, extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    typer.echo(json.dumps({"stage": "target_loading"}, sort_keys=True))
    validation_targets_24 = _targets(validation_samples, (24, 24))
    test_targets_24 = _targets(test_samples, (24, 24))
    test_targets_full = _targets(test_samples, (518, 518))
    sequence_ids = [sample.sequence_id for sample in test_samples]
    qualitative_indices = _select_qualitative_indices(sequence_ids)
    qualitative_sample_ids = [test_samples[index].sample_id for index in qualitative_indices]
    qualitative_sequence_ids = [sequence_ids[index] for index in qualitative_indices]
    seed_results = []
    full_prediction_tensors: list[torch.Tensor] = []
    full_prediction_variant_ids: list[str] = []
    full_prediction_seeds: list[int] = []
    qualitative_predictions: list[torch.Tensor] = []
    qualitative_log_variances: list[torch.Tensor] = []
    qualitative_sigmas: list[torch.Tensor] = []
    qualitative_variant_ids: list[str] = []
    qualitative_seeds: list[int] = []
    persisted_interventions = {"original", "zero", "fixed_average"}

    def capture_prediction(
        seed: int,
        intervention: GateIntervention,
        prediction_full: torch.Tensor,
        prediction_24: torch.Tensor,
        log_variance_24: torch.Tensor,
        variance_multiplier: float,
    ) -> None:
        if intervention.intervention_id not in persisted_interventions:
            return
        variant_id = f"seed{seed}:{intervention.intervention_id}"
        full_prediction_tensors.append(prediction_full.detach().cpu().float().clone())
        full_prediction_variant_ids.append(variant_id)
        full_prediction_seeds.append(seed)
        selected = torch.as_tensor(qualitative_indices, dtype=torch.long)
        qualitative_predictions.append(prediction_24[selected].detach().cpu().float().clone())
        qualitative_log_variances.append(log_variance_24[selected].detach().cpu().float().clone())
        sigma = (0.5 * log_variance_24[selected]).exp() * math.sqrt(variance_multiplier)
        qualitative_sigmas.append(sigma.detach().cpu().float().clone())
        qualitative_variant_ids.append(variant_id)
        qualitative_seeds.append(seed)

    for seed in sorted(artifacts.checkpoints):
        typer.echo(json.dumps({"stage": "seed_attribution", "seed": seed}, sort_keys=True))
        seed_results.append(
            evaluate_checkpoint_attribution(
                artifacts.checkpoints[seed],
                artifacts.learned_rows[seed],
                validation_features,
                test_features,
                validation_targets_24,
                test_targets_24,
                test_targets_full,
                sequence_ids,
                device=device,
                batch_size=probe_batch_size,
                prediction_callback=capture_prediction,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record = build_attribution_record(
        artifacts=artifacts,
        seed_results=seed_results,
        dataset_manifest=manifest_path,
        dataset_split_hash=bundle.split_hash,
        sample_ids=[sample.sample_id for sample in test_samples],
        output_directory=output,
    )
    record["source_identity"] = source_identity
    record["execution_provenance"] = execution_provenance
    record["runtime"] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "total_seconds": time.perf_counter() - started,
        "feature_extraction": {"validation": validation_profile, "test": test_profile},
        "device_requested": device,
        "slurm": {
            key: os.environ[key]
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")
            if key in os.environ
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
    }
    prediction_path = write_full_predictions_npz(
        output / "full_predictions.npz",
        predictions=full_prediction_tensors,
        target=test_targets_full,
        sample_ids=[sample.sample_id for sample in test_samples],
        sequence_ids=sequence_ids,
        variant_ids=full_prediction_variant_ids,
        seeds=full_prediction_seeds,
    )
    qualitative_path = write_qualitative_examples_npz(
        output / "qualitative_examples.npz",
        predictions=qualitative_predictions,
        log_variances=qualitative_log_variances,
        calibrated_log_depth_sigmas=qualitative_sigmas,
        target=test_targets_24[qualitative_indices],
        sample_ids=qualitative_sample_ids,
        sequence_ids=qualitative_sequence_ids,
        variant_ids=qualitative_variant_ids,
        seeds=qualitative_seeds,
    )
    record["prediction_handoff"] = {
        "schema_version": "jepa4d-phase2d-depth-predictions-v1",
        "audit_scope": "full_phase2c_test",
        "selection": "original, zero, and fixed-average-equivalent for every checkpoint seed",
        "variant_ids": full_prediction_variant_ids,
        "shape": [len(full_prediction_tensors), len(test_samples), *test_targets_full.shape[-2:]],
        "path": str(prediction_path.resolve()),
        "sha256": sha256(prediction_path),
    }
    record["qualitative_handoff"] = {
        "schema_version": "jepa4d-phase2d-qualitative-v1",
        "selection_policy": "deterministic sequence-balanced fixed samples; maximum four",
        "path": str(qualitative_path.resolve()),
        "sha256": sha256(qualitative_path),
        "sample_count": len(qualitative_indices),
        "variant_count": len(qualitative_variant_ids),
        "sample_ids": qualitative_sample_ids,
        "sequence_ids": qualitative_sequence_ids,
        "variant_ids": qualitative_variant_ids,
        "fields": ["prediction_m", "target_m", "log_variance", "calibrated_log_depth_sigma"],
    }
    result_path = output / "fusion_attribution.json"
    _write_json(result_path, record)
    report_path = build_fusion_attribution_report(
        record,
        output / "fusion_attribution_report.html",
        qualitative_npz=qualitative_path,
    )
    receipt = {
        "schema_version": "jepa4d-phase2d-output-receipt-v1",
        "status": "pass",
        "source_comparison_sha256": artifacts.comparison_sha256,
        "source_identity": {
            "path": str(source_identity_path.resolve()),
            "sha256": sha256(source_identity_path),
            "schema_version": source_identity["schema_version"],
        },
        "fusion_attribution_json": {"path": str(result_path.resolve()), "sha256": sha256(result_path)},
        "fusion_attribution_html": {"path": str(report_path.resolve()), "sha256": sha256(report_path)},
        "full_predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256(prediction_path),
            "schema_version": "jepa4d-phase2d-depth-predictions-v1",
            "variants": len(full_prediction_tensors),
            "frames": len(test_samples),
            "audit_scope": "full_phase2c_test",
        },
        "qualitative_examples": {
            "path": str(qualitative_path.resolve()),
            "bytes": qualitative_path.stat().st_size,
            "sha256": sha256(qualitative_path),
            "schema_version": "jepa4d-phase2d-qualitative-v1",
            "samples": len(qualitative_indices),
            "variants": len(qualitative_variant_ids),
        },
        "execution_provenance": execution_provenance,
    }
    _write_json(output / "receipt.json", receipt)
    typer.echo(
        json.dumps(
            {
                "result": str(result_path),
                "report": str(report_path),
                "full_predictions": str(prediction_path),
                "qualitative_examples": str(qualitative_path),
                "receipt": str(output / "receipt.json"),
                "seeds": len(seed_results),
                "controls_per_seed": len(seed_results[0]["interventions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
