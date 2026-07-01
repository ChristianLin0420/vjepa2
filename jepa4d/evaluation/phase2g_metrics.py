"""Formal Phase 2g-A metrics, aggregation, controls, and descriptive intervals."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from jepa4d.evaluation.phase2f_metrics import evaluate_depth_predictions, require_finite_tree
from jepa4d.training.phase2g_protocol import (
    ADDITIONAL_METRICS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FAMILIES,
    FORMAL_SEEDS,
    METRICS_SCHEMA,
    PRIMARY_METRICS,
    SAMPLES_PER_FAMILY,
)


def opaque_frame_id(sample_id: str, *, profile: str | None = None) -> str:
    """Return a stable non-path identifier safe for aggregate/W&B artifacts."""

    suffix = "" if profile is None else f"::{profile}"
    return hashlib.sha256(f"phase2g-frame-v1::{sample_id}{suffix}".encode()).hexdigest()


def _validate_inputs(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(not isinstance(value, torch.Tensor) for value in (log_depth, log_variance, target_depth, valid_mask)):
        raise TypeError("Phase 2g metric inputs must be tensors")
    if (
        log_depth.ndim != 3
        or log_depth.shape != log_variance.shape
        or log_depth.shape != target_depth.shape
        or log_depth.shape != valid_mask.shape
        or valid_mask.dtype != torch.bool
    ):
        raise ValueError("Phase 2g metrics require matching [N,H,W] tensors and a boolean mask")
    if not bool(torch.isfinite(log_depth).all()) or not bool(torch.isfinite(log_variance).all()):
        raise ValueError("Phase 2g predictions must be finite")
    if bool((valid_mask.flatten(1).sum(1) == 0).any()) or not bool(torch.isfinite(target_depth[valid_mask]).all()):
        raise ValueError("every Phase 2g frame needs finite valid target pixels")
    return (
        log_depth.detach().cpu().double(),
        log_variance.detach().cpu().double(),
        target_depth.detach().cpu().double(),
        valid_mask.detach().cpu(),
    )


def evaluate_phase2g_predictions(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    variance_multiplier: float,
    frame_ids: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, Any]:
    """Compute the frozen point, uncertainty, RMSE, Delta-1, and reliability metrics."""

    prediction, variance_log, target, valid = _validate_inputs(log_depth, log_variance, target_depth, valid_mask)
    if len(frame_ids) != len(prediction) or len(family_ids) != len(prediction):
        raise ValueError("frame/family identities must match prediction rows")
    base = evaluate_depth_predictions(
        prediction.float(),
        variance_log.float(),
        target.float(),
        valid_mask=valid,
        variance_multiplier=variance_multiplier,
        frame_ids=frame_ids,
        group_ids=family_ids,
    )
    coverage_levels = (0.5, 0.8, 0.9, 0.95)
    per_frame: list[dict[str, Any]] = []
    for index, base_row in enumerate(base["per_frame"]):
        mask = valid[index]
        predicted_log = prediction[index][mask]
        target_values = target[index][mask]
        target_log = target_values.log()
        signed_scale = float((predicted_log - target_log).median())
        predicted_depth = predicted_log.exp()
        aligned_depth = (predicted_log - signed_scale).exp()
        ratio = torch.maximum(predicted_depth / target_values, target_values / predicted_depth)
        calibrated_std = (variance_log[index][mask] + math.log(variance_multiplier)).mul(0.5).exp()
        coverage = {}
        # Frozen two-sided Gaussian z values for 50/80/90/95% intervals.
        z_values = (0.6744897501960817, 1.2815515655446004, 1.6448536269514722, 1.959963984540054)
        for level, z_value in zip(coverage_levels, z_values, strict=True):
            coverage[f"coverage_{int(level * 100)}"] = float(
                ((predicted_log - target_log).abs() <= z_value * calibrated_std).double().mean()
            )
        reliability_error = float(
            np.mean([abs(coverage[f"coverage_{int(level * 100)}"] - level) for level in coverage_levels])
        )
        row = {
            **base_row,
            "raw_rmse": float((predicted_depth - target_values).square().mean().sqrt()),
            "aligned_rmse": float((aligned_depth - target_values).square().mean().sqrt()),
            "delta1": float((ratio < 1.25).double().mean()),
            "reliability_error": reliability_error,
            **coverage,
        }
        per_frame.append(row)
    names = (*PRIMARY_METRICS, *ADDITIONAL_METRICS)
    per_family: dict[str, dict[str, float]] = {}
    for family in sorted(set(family_ids)):
        rows = [row for row in per_frame if row["group_id"] == family]
        per_family[family] = {name: float(np.mean([float(row[name]) for row in rows])) for name in names}
    frame_macro = {name: float(np.mean([float(row[name]) for row in per_frame])) for name in names}
    equal_family = {name: float(np.mean([values[name] for values in per_family.values()])) for name in names}
    result = {
        "schema_version": METRICS_SCHEMA,
        "variance_multiplier": float(variance_multiplier),
        "frames": len(per_frame),
        "valid_frames": len(per_frame),
        "failure_count": 0,
        "typed_failures": [],
        "valid_pixels": int(valid.sum()),
        "per_frame": per_frame,
        "per_family": per_family,
        "frame_macro": frame_macro,
        "equal_family_macro": equal_family,
        "coverage": base["coverage"],
        "risk_coverage": base["risk_coverage"],
        "aggregation": "pixels_within_frame_then_frames_within_family_then_equal_family",
    }
    require_finite_tree(result, "phase2g_metrics")
    return result


def scale_mechanism_diagnostics(
    predicted_log_scale: torch.Tensor,
    optimal_log_scale: torch.Tensor,
    *,
    frame_ids: Sequence[str],
) -> dict[str, Any]:
    """Report per-frame scale predictions, optimal targets, correlation, and residuals."""

    predicted = predicted_log_scale.detach().cpu().double().reshape(-1)
    optimal = optimal_log_scale.detach().cpu().double().reshape(-1)
    if len(predicted) == 0 or predicted.shape != optimal.shape or len(frame_ids) != len(predicted):
        raise ValueError("scale mechanism arrays must be equal non-empty vectors")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(optimal).all()):
        raise ValueError("scale mechanism arrays must be finite")
    pred_np, optimal_np = predicted.numpy(), optimal.numpy()
    if float(np.std(pred_np)) == 0.0 or float(np.std(optimal_np)) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(pred_np, optimal_np)[0, 1])
    residual = pred_np - optimal_np
    result = {
        "correlation": correlation,
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "per_frame": [
            {
                "frame_id": frame_id,
                "predicted_log_scale": float(pred),
                "optimal_log_scale": float(optimum),
                "residual": float(pred - optimum),
            }
            for frame_id, pred, optimum in zip(frame_ids, pred_np, optimal_np, strict=True)
        ],
    }
    require_finite_tree(result, "scale_mechanism")
    return result


def aggregate_evaluation_rows(
    receipts: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = (*PRIMARY_METRICS, *ADDITIONAL_METRICS),
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Aggregate 48 eval receipts as frames -> seeds -> equal held-out families."""

    frame_rows: list[dict[str, Any]] = []
    cells: set[tuple[str, str, int]] = set()
    fixed_family_ids: dict[str, tuple[str, ...]] = {}
    for receipt in receipts:
        arm = str(receipt["arm"])
        seed = int(receipt["seed"])
        family = str(receipt["heldout_family"])
        cell = (arm, family, seed)
        if cell in cells:
            raise ValueError(f"duplicate evaluation aggregation cell: {cell}")
        cells.add(cell)
        metrics = receipt.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"evaluation cell lacks metrics: {cell}")
        rows = metrics.get("per_frame")
        if (
            not isinstance(rows, list)
            or len(rows) != SAMPLES_PER_FAMILY
            or metrics.get("frames") != SAMPLES_PER_FAMILY
            or metrics.get("valid_frames") != SAMPLES_PER_FAMILY
            or metrics.get("failure_count") != 0
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            raise ValueError(f"evaluation cell is not a complete {SAMPLES_PER_FAMILY}-frame result: {cell}")
        typed_rows = [dict(row) for row in rows]
        frame_ids = tuple(str(row.get("frame_id", "")) for row in typed_rows)
        if (
            len(frame_ids) != len(set(frame_ids))
            or any(
                len(frame_id) != 64 or any(character not in "0123456789abcdef" for character in frame_id)
                for frame_id in frame_ids
            )
            or any(row.get("group_id") != family for row in typed_rows)
        ):
            raise ValueError(f"evaluation frame identity/family contract failed: {cell}")
        expected_ids = fixed_family_ids.setdefault(family, frame_ids)
        if frame_ids != expected_ids:
            raise ValueError(f"held-out membership differs across cells for family {family}")
        for row in typed_rows:
            frame_rows.append({"arm": arm, "seed": seed, "family": family, **dict(row)})
    family_seed: dict[tuple[str, str, int], dict[str, float]] = {}
    grouped: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(row["arm"], row["family"], row["seed"])].append(row)
    for key, rows in grouped.items():
        family_seed[key] = {name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names}
    arms = sorted({str(receipt["arm"]) for receipt in receipts})
    aggregates: dict[str, dict[str, float]] = {}
    for arm in arms:
        missing = [
            (family, seed) for family in FAMILIES for seed in FORMAL_SEEDS if (arm, family, seed) not in family_seed
        ]
        if missing:
            raise ValueError(f"evaluation aggregation is incomplete for {arm}: {missing}")
        per_family = {
            family: {
                name: float(np.mean([family_seed[(arm, family, seed)][name] for seed in FORMAL_SEEDS]))
                for name in metric_names
            }
            for family in FAMILIES
        }
        aggregates[arm] = {
            name: float(np.mean([per_family[family][name] for family in FAMILIES])) for name in metric_names
        }
        aggregates[arm]["_per_family"] = per_family  # type: ignore[assignment]
    return aggregates, frame_rows


def paired_hierarchical_bootstrap(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    reference: str = "M0",
    metric: str = "raw_abs_rel",
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired four-family bootstrap after optimizer-seed averaging.

    Pairing is by ``(family, frame_id, seed)``.  Seeds are averaged within each
    frame before families and frames are resampled.  The interval is explicitly
    descriptive because the protocol has only four family clusters.
    """

    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("bootstrap resamples must be a positive integer")
    table: dict[tuple[str, str, int, str], float] = {}
    for row in frame_rows:
        arm = str(row["arm"])
        if arm not in {candidate, reference}:
            continue
        key = (str(row["family"]), str(row["frame_id"]), int(row["seed"]), arm)
        if key in table:
            raise ValueError(f"duplicate paired bootstrap row: {key}")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError("bootstrap metric must be finite")
        table[key] = value
    effects: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        frame_ids = sorted({frame_id for fam, frame_id, _, arm in table if fam == family and arm == candidate})
        family_effects: list[float] = []
        for frame_id in frame_ids:
            differences = []
            for optimizer_seed in FORMAL_SEEDS:
                candidate_key = (family, frame_id, optimizer_seed, candidate)
                reference_key = (family, frame_id, optimizer_seed, reference)
                if candidate_key not in table or reference_key not in table:
                    raise ValueError(f"bootstrap pairing is incomplete: {(family, frame_id, optimizer_seed)}")
                differences.append(table[candidate_key] - table[reference_key])
            family_effects.append(float(np.mean(differences)))
        if not family_effects:
            raise ValueError(f"bootstrap family has no paired frames: {family}")
        effects[family] = np.asarray(family_effects, dtype=np.float64)

    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    family_count = len(FAMILIES)
    # Chunking bounds temporary frame-index arrays for the formal 100k x 1024 case.
    chunk_size = min(1_000, resamples)
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        size = stop - start
        selected_families = rng.integers(0, family_count, size=(size, family_count))
        totals = np.zeros(size, dtype=np.float64)
        for position in range(family_count):
            choices = selected_families[:, position]
            position_effect = np.empty(size, dtype=np.float64)
            for family_index, family in enumerate(FAMILIES):
                mask = choices == family_index
                count = int(mask.sum())
                if count:
                    values = effects[family]
                    indices = rng.integers(0, len(values), size=(count, len(values)))
                    position_effect[mask] = values[indices].mean(axis=1)
            totals += position_effect
        distribution[start:stop] = totals / family_count
    observed = float(np.mean([values.mean() for values in effects.values()]))
    result = {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "effect": "candidate_minus_reference",
        "observed": observed,
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "resamples": resamples,
        "seed": seed,
        "family_clusters": len(FAMILIES),
        "paired_optimizer_seeds": list(FORMAL_SEEDS),
        "descriptive_only": True,
    }
    require_finite_tree(result, "hierarchical_bootstrap")
    return result
