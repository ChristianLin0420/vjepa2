"""Train and compare Phase 2b RGB/V-JEPA geometry probes on an immutable split."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn.functional as F
import typer
from PIL import Image

from jepa4d.benchmarks.geometry.tum_rgbd import depth_metrics, load_depth, load_tum_indices, validate_archive
from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.evaluation.comparison import ComparisonRecord, VariantResult
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.geometry_student import DenseGeometryProbe, geometry_probe_loss, rgb_grid_features
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor

app = typer.Typer(add_completion=False)


def _images(samples: list[Any]) -> torch.Tensor:
    values = [torch.from_numpy(np.asarray(Image.open(item.rgb_path).convert("RGB"), dtype=np.uint8).copy()) for item in samples]
    return torch.stack(values).permute(0, 3, 1, 2).float() / 255.0


def _targets(samples: list[Any], size: tuple[int, int]) -> torch.Tensor:
    return torch.stack([load_depth(item.depth_path, size) for item in samples])


def _valid(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target) & (target > 0.1) & (target < 10.0)


def _extract_vjepa(
    extractor: VJEPA21FeatureExtractor, samples: list[Any], chunk_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    layers: dict[int, list[torch.Tensor]] = defaultdict(list)
    final: list[torch.Tensor] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(samples), chunk_size):
        chunk = samples[offset : offset + chunk_size]
        batch = from_view_sequences([[item.rgb_path] for item in chunk])
        bundle = extractor(batch)
        final.append(bundle.dense_tokens[0, :, 0].detach().cpu())
        for layer, value in bundle.layer_tokens.items():
            layers[layer].append(value[0, :, 0].detach().cpu())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    def grid(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.reshape(len(samples), 24, 24, -1).permute(0, 3, 1, 2).contiguous().half()

    final_grid = grid(torch.cat(final))
    layer_grids = [grid(torch.cat(layers[layer])) for layer in sorted(layers)]
    return (
        {"vjepa_final": final_grid, "vjepa_multilayer": torch.cat(layer_grids, dim=1)},
        {
            "total_seconds": elapsed,
            "per_frame_ms": elapsed * 1000.0 / len(samples),
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        },
    )


def _teacher_depth(
    head: GeometryBeliefHead, samples: list[Any], targets: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, dict[str, float]]:
    predictions = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(samples), chunk_size):
        chunk = samples[offset : offset + chunk_size]
        belief = head(from_view_sequences([[item.rgb_path] for item in chunk]))
        assert belief.depth_mean is not None
        depth = F.interpolate(
            belief.depth_mean[0, :, 0].unsqueeze(1).float(), size=targets.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0].cpu()
        predictions.append(depth)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = torch.cat(predictions)
    aligned = []
    for value, target in zip(prediction, targets, strict=True):
        mask = _valid(target) & torch.isfinite(value) & (value > 0)
        scale = target[mask].median() / value[mask].median().clamp_min(1e-8)
        aligned.append(value * scale)
    return torch.stack(aligned), {
        "total_seconds": elapsed,
        "per_frame_ms": elapsed * 1000.0 / len(samples),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def _normalize(
    train: torch.Tensor, validation: torch.Tensor, test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mean = train.float().mean(dim=(0, 2, 3), keepdim=True)
    std = train.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    return (
        ((train.float() - mean) / std).half(),
        ((validation.float() - mean) / std).half(),
        ((test.float() - mean) / std).half(),
        {"mean": mean.cpu(), "std": std.cpu()},
    )


def _raw_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = _valid(target) & torch.isfinite(predicted) & (predicted > 0)
    prediction, truth = predicted[mask], target[mask]
    error = prediction - truth
    ratio = torch.maximum(prediction / truth, truth / prediction.clamp_min(1e-8))
    return {
        "metric_abs_rel": float((error.abs() / truth).mean()),
        "metric_rmse_m": float(torch.sqrt(error.square().mean())),
        "metric_delta_1": float((ratio < 1.25).float().mean()),
    }


def _evaluate_depths(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    rows = []
    for predicted, target in zip(predictions, targets, strict=True):
        raw = _raw_metrics(predicted, target)
        aligned, _, _ = depth_metrics(predicted, target)
        rows.append({**raw, **{f"aligned_{key}": value for key, value in aligned.items()}})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _calibrate_log_variance(
    model: DenseGeometryProbe,
    validation_features: torch.Tensor,
    validation_target: torch.Tensor,
    test_features: torch.Tensor,
    test_target: torch.Tensor,
    device: str,
) -> tuple[float, float, float]:
    model.eval()
    with torch.inference_mode():
        val_log_depth, val_logvar = model(validation_features.to(device))
        test_log_depth, test_logvar = model(test_features.to(device))
    val_truth = validation_target.to(device).clamp_min(1e-4).log()
    test_truth = test_target.to(device).clamp_min(1e-4).log()
    val_mask, test_mask = _valid(validation_target).to(device), _valid(test_target).to(device)
    multiplier = float(
        (((val_log_depth - val_truth).square() / val_logvar.exp().clamp_min(1e-8))[val_mask]).mean().clamp(1e-4, 1e4)
    )
    residual = (test_log_depth - test_truth)[test_mask]
    raw_variance = test_logvar.exp()[test_mask].clamp_min(1e-8)
    calibrated_variance = raw_variance * multiplier
    raw_nll = 0.5 * (raw_variance.log() + residual.square() / raw_variance)
    calibrated_nll = 0.5 * (calibrated_variance.log() + residual.square() / calibrated_variance)
    return multiplier, float(raw_nll.mean()), float(calibrated_nll.mean())


def _head_latency(model: DenseGeometryProbe, features: torch.Tensor, device: str) -> float:
    value = features[:1].to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(value)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            model(value)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 10.0


def _train_variant(
    variant: str,
    seed: int,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    test_features: torch.Tensor,
    train_target: torch.Tensor,
    validation_target: torch.Tensor,
    test_target_518: torch.Tensor,
    teacher_target: torch.Tensor,
    output: Path,
    device: str,
    epochs: int,
    run: Any,
    encoder_runtime: dict[str, float],
) -> VariantResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DenseGeometryProbe(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    batch_size = 8
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_features), generator=generator)
        losses = []
        for offset in range(0, len(order), batch_size):
            index = order[offset : offset + batch_size]
            features = train_features[index].to(device)
            target = train_target[index].to(device)
            teacher = teacher_target[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            log_depth, logvar = model(features)
            loss, parts = geometry_probe_loss(log_depth, logvar, target, _valid(target), teacher_depth=teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_features.to(device))[0].exp().cpu()
        validation_absrel = _evaluate_depths(validation_prediction, validation_target)["metric_abs_rel"]
        if validation_absrel < best_score:
            best_score = validation_absrel
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if run is not None:
            run.log(
                {
                    "training/global_step": epoch,
                    f"training/{variant}/seed_{seed}/loss": float(np.mean(losses)),
                    f"training/{variant}/seed_{seed}/validation_abs_rel": validation_absrel,
                    f"training/{variant}/seed_{seed}/nll": float(parts["nll"]),
                    f"training/{variant}/seed_{seed}/distillation": float(parts["distillation"]),
                }
            )
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / f"{variant}-seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"variant": variant, "seed": seed, "state_dict": best_state, "validation_abs_rel": best_score}, checkpoint
    )
    model.eval()
    with torch.inference_mode():
        test_log_depth, _ = model(test_features.to(device))
        test_prediction = F.interpolate(
            test_log_depth.exp().unsqueeze(1), size=test_target_518.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0].cpu()
    metrics = _evaluate_depths(test_prediction, test_target_518)
    multiplier, raw_nll, calibrated_nll = _calibrate_log_variance(
        model, validation_features, validation_target, test_features, F.interpolate(
            test_target_518.unsqueeze(1), size=test_features.shape[-2:], mode="nearest"
        )[:, 0], device
    )
    metrics.update(
        {
            "validation_metric_abs_rel": best_score,
            "variance_multiplier": multiplier,
            "raw_log_depth_nll": raw_nll,
            "calibrated_log_depth_nll": calibrated_nll,
        }
    )
    head_ms = _head_latency(model, test_features, device)
    parameters = sum(value.numel() for value in model.parameters())
    family = "rgb" if variant == "rgb_probe" else "vjepa"
    role = "non_jepa_baseline" if variant == "rgb_probe" else ("ablation" if variant == "vjepa_final" else "ours")
    return VariantResult(
        variant_id=variant,
        family=family,
        role=role,
        seed=seed,
        metrics=metrics,
        runtime={
            "encoder_ms_per_frame": encoder_runtime["per_frame_ms"],
            "head_ms_per_frame": head_ms,
            "total_ms_per_frame": encoder_runtime["per_frame_ms"] + head_ms,
            "peak_encoder_memory_gb": encoder_runtime["peak_memory_gb"],
        },
        parameters=parameters,
        checkpoint=str(checkpoint),
        notes=["Best checkpoint selected only on validation metric AbsRel.", "VGGT-aligned auxiliary loss weight=0.25."],
    )


def _aggregate(results: list[VariantResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[VariantResult]] = defaultdict(list)
    for value in results:
        grouped[value.variant_id].append(value)
    output: dict[str, dict[str, float]] = {}
    for variant, values in grouped.items():
        metrics: dict[str, float] = {}
        for key in values[0].metrics:
            numbers = np.asarray([value.metrics[key] for value in values])
            metrics[f"{key}_mean"] = float(numbers.mean())
            metrics[f"{key}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        metrics["total_ms_per_frame_mean"] = float(np.mean([value.runtime["total_ms_per_frame"] for value in values]))
        metrics["parameters"] = float(values[0].parameters)
        output[variant] = metrics
    return output


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Option("--dataset-root")],
    archive: Annotated[Path, typer.Option("--archive")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/jepa4d_phase2b/tum_rgbd_v1"),
    manifest_path: Annotated[Path, typer.Option("--manifest")] = Path(
        "jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml"
    ),
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    epochs: Annotated[int, typer.Option("--epochs")] = 60,
    wandb_enabled: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    run_name: Annotated[str, typer.Option("--run-name")] = "phase2b-jepa-geometry-distillation-v1",
) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise typer.BadParameter("Phase 2b training requires CUDA")
    manifest = validate_archive(archive, manifest_path)
    split_hash = hashlib.sha256(
        json.dumps({key: manifest[key] for key in ("train_indices", "validation_indices", "test_indices")}, sort_keys=True).encode()
    ).hexdigest()
    splits = {
        name: load_tum_indices(dataset_root, [int(value) for value in manifest[f"{name}_indices"]])
        for name in ("train", "validation", "test")
    }
    targets_24 = {name: _targets(samples, (24, 24)) for name, samples in splits.items()}
    test_target_518 = _targets(splits["test"], (518, 518))
    output.mkdir(parents=True, exist_ok=True)
    run = None
    if wandb_enabled:
        import wandb

        run = wandb.init(
            project=wandb_project,
            name=run_name,
            tags=["phase-2b", "geometry-distillation", "TUM-RGBD", "vjepa", "baselines", "cuda"],
            config={
                "manifest": manifest,
                "split_hash": split_hash,
                "epochs": epochs,
                "seeds": [0, 1, 2],
                "device": device,
                "vjepa_checkpoint": "checkpoints/vjepa2.1-vitb-fpc64-384",
                "vggt_checkpoint": "checkpoints/VGGT-1B",
            },
        )
    extractor = VJEPA21FeatureExtractor(
        checkpoint="checkpoints/vjepa2.1-vitb-fpc64-384",
        implementation_path="checkpoints/vjepa21_hf_impl",
        backend="hf_compat",
        device=device,
    )
    vjepa_features: dict[str, dict[str, torch.Tensor]] = {}
    vjepa_runtime: dict[str, dict[str, float]] = {}
    for name, samples in splits.items():
        vjepa_features[name], vjepa_runtime[name] = _extract_vjepa(extractor, samples, chunk_size=8)
    teacher = GeometryBeliefHead(
        backend="vggt", device=device, model_id="checkpoints/VGGT-1B", precision="bfloat16"
    )
    teacher_targets: dict[str, torch.Tensor] = {}
    teacher_runtime: dict[str, dict[str, float]] = {}
    for name, samples in splits.items():
        teacher_targets[name], teacher_runtime[name] = _teacher_depth(teacher, samples, targets_24[name], chunk_size=8)
    teacher_test_518 = F.interpolate(
        teacher_targets["test"].unsqueeze(1), size=(518, 518), mode="bilinear", align_corners=False
    )[:, 0]
    teacher_metrics = _evaluate_depths(teacher_test_518, test_target_518)
    teacher_result = VariantResult(
        "vggt_teacher",
        "vggt",
        "teacher_baseline",
        None,
        teacher_metrics,
        {
            "encoder_ms_per_frame": teacher_runtime["test"]["per_frame_ms"],
            "head_ms_per_frame": 0.0,
            "total_ms_per_frame": teacher_runtime["test"]["per_frame_ms"],
            "peak_encoder_memory_gb": teacher_runtime["test"]["peak_memory_gb"],
        },
        sum(value.numel() for value in teacher.model.parameters()) if teacher.model is not None else 0,
        notes=["BF16 official VGGT-1B teacher; per-frame median aligned because teacher scale is relative."],
    )

    rgb = {name: rgb_grid_features(_images(samples), 24).half() for name, samples in splits.items()}
    rgb_runtime = {"per_frame_ms": 0.0, "peak_memory_gb": 0.0}
    variant_features: dict[str, dict[str, torch.Tensor]] = {"rgb_probe": rgb}
    for variant in ("vjepa_final", "vjepa_multilayer"):
        normalized = _normalize(
            vjepa_features["train"][variant],
            vjepa_features["validation"][variant],
            vjepa_features["test"][variant],
        )
        variant_features[variant] = {"train": normalized[0], "validation": normalized[1], "test": normalized[2]}
        torch.save(normalized[3], output / f"{variant}-normalization.pt")

    results = [teacher_result]
    failures: list[dict[str, str]] = []
    for variant, features in variant_features.items():
        for seed in (0, 1, 2):
            try:
                runtime = rgb_runtime if variant == "rgb_probe" else vjepa_runtime["test"]
                results.append(
                    _train_variant(
                        variant,
                        seed,
                        features["train"],
                        features["validation"],
                        features["test"],
                        targets_24["train"],
                        targets_24["validation"],
                        test_target_518,
                        teacher_targets["train"],
                        output,
                        device,
                        epochs,
                        run,
                        runtime,
                    )
                )
            except Exception as error:
                failures.append({"variant": variant, "seed": str(seed), "error": f"{type(error).__name__}: {error}"})
    aggregates = _aggregate(results)
    record = ComparisonRecord(
        experiment_id=run_name,
        schema_version="jepa4d-phase2b-comparison-v1",
        dataset_manifest=str(manifest_path),
        split_hash=split_hash,
        metric_policy={
            "primary": "metric_abs_rel on chronological held-out test frames",
            "secondary": "median-aligned AbsRel/RMSE/delta, validation-fitted log-depth variance NLL",
            "checkpoint_selection": "minimum validation metric_abs_rel",
            "seeds": [0, 1, 2],
            "teacher_auxiliary_weight": 0.25,
        },
        variants=results,
        failures=failures,
        aggregates=aggregates,
        wandb_url=None if run is None else run.url,
    )
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(record.to_serializable(), indent=2) + "\n")
    failures_path = output / "failures.json"
    failures_path.write_text(json.dumps(failures, indent=2) + "\n")
    if run is not None:
        import wandb

        table = wandb.Table(
            columns=[
                "variant", "role", "seed", "metric_abs_rel", "aligned_abs_rel", "aligned_rmse_m",
                "calibrated_nll", "total_ms_per_frame", "parameters",
            ]
        )
        for value in results:
            table.add_data(
                value.variant_id,
                value.role,
                value.seed,
                value.metrics.get("metric_abs_rel"),
                value.metrics.get("aligned_abs_rel"),
                value.metrics.get("aligned_rmse_m"),
                value.metrics.get("calibrated_log_depth_nll"),
                value.runtime["total_ms_per_frame"],
                value.parameters,
            )
        run.log({"comparison/results": table, "comparison/failures": len(failures)})
        for variant, values in aggregates.items():
            run.summary.update({f"comparison/{variant}/{key}": value for key, value in values.items()})
        artifact = wandb.Artifact(f"{run.id}-phase2b-comparison", type="geometry-comparison")
        artifact.add_file(str(report_path))
        artifact.add_file(str(failures_path))
        for checkpoint in sorted((output / "checkpoints").glob("*.pt")):
            artifact.add_file(str(checkpoint), name=f"checkpoints/{checkpoint.name}")
        run.log_artifact(artifact)
        run.summary.update({"result": "success" if not failures else "partial", "variants": len(results)})
        run.finish()
    typer.echo(json.dumps({"comparison": str(report_path), "wandb_url": record.wandb_url, "aggregates": aggregates, "failures": failures}, indent=2))


if __name__ == "__main__":
    app()
