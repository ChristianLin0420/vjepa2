"""Versioned quality-first geometry metrics for post-Wave-A experiments.

This module deliberately leaves the historical Phase 2f metric implementation
unchanged.  The new schema adds the metrics promised by the systematic Phase 2
plan, preserves frame/group weighting explicitly, and emits observations that
can be checked against a registered split manifest before population inference.
Resource measurements do not appear in this quality contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import torch

from jepa4d.evaluation.statistics import ClusteredMetricObservation
from jepa4d.validation._content import sha256_value

GEOMETRY_QUALITY_SCHEMA = "jepa4d-geometry-quality-metrics-v1"
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
_PSEUDONYMOUS_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CREDENTIAL_IDENTIFIER_PATTERNS = (
    re.compile(r"(?i)^wandb_v1_[A-Za-z0-9_-]+$"),
    re.compile(r"(?i)^hf_[A-Za-z0-9]{20,}$"),
)
_CREDENTIAL_IDENTIFIER_TOKENS = frozenset(
    {"api", "apikey", "authorization", "cookie", "credential", "key", "netrc", "password", "secret", "token"}
)
METRIC_SPECS: Mapping[str, tuple[str, str]] = {
    "raw_abs_rel": ("lower", "ratio"),
    "raw_rmse_m": ("lower", "metres"),
    "log_rmse": ("lower", "log-depth"),
    "aligned_abs_rel": ("lower", "ratio"),
    "aligned_rmse_m": ("lower", "metres"),
    "signed_log_scale_error": ("zero", "log-scale"),
    "absolute_log_scale_error": ("lower", "log-scale"),
    "delta_1": ("higher", "fraction"),
    "delta_2": ("higher", "fraction"),
    "delta_3": ("higher", "fraction"),
    "nll": ("lower", "log-depth Gaussian NLL without constant"),
    "ause": ("lower", "AbsRel risk-coverage area"),
    "reliability_error": ("lower", "mean absolute coverage gap"),
}


def _require_finite_tree(value: Any, label: str = "value") -> None:
    if value is None or isinstance(value, bool | str | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")
        return
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def _validate_identifiers(values: Sequence[str], count: int, label: str, *, unique: bool) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or len(values) != count:
        raise ValueError(f"{label} must contain exactly one identifier per frame")
    normalized = tuple(values)
    if any(not isinstance(value, str) or _PSEUDONYMOUS_IDENTIFIER.fullmatch(value) is None for value in normalized):
        raise ValueError(f"{label} must contain bounded path-safe pseudonymous identifiers")
    for value in normalized:
        tokens = {token for token in re.split(r"[_.-]+", value.casefold()) if token}
        if tokens & _CREDENTIAL_IDENTIFIER_TOKENS or any(
            pattern.fullmatch(value) for pattern in _CREDENTIAL_IDENTIFIER_PATTERNS
        ):
            raise ValueError(f"{label} must not contain credential-like identifiers")
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must be unique")
    return normalized


def geometry_grouping_assignment_sha256(
    unit_ids: Sequence[str],
    cluster_ids: Sequence[str],
    group_ids: Sequence[str],
) -> str:
    """Hash the exact ordered unit-to-cluster-to-group assignment."""

    if not (len(unit_ids) == len(cluster_ids) == len(group_ids)):
        raise ValueError("grouping identity sequences must have equal lengths")
    return sha256_value(
        [
            {"unit_id": unit_id, "cluster_id": cluster_id, "group_id": group_id}
            for unit_id, cluster_id, group_id in zip(unit_ids, cluster_ids, group_ids, strict=True)
        ]
    )


def _validate_inputs(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (log_depth, log_variance, target_depth)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("log_depth, log_variance, and target_depth must be torch tensors")
    if log_depth.ndim != 3 or log_depth.shape != log_variance.shape or log_depth.shape != target_depth.shape:
        raise ValueError("metric tensors must share non-empty shape [N,H,W]")
    if log_depth.shape[0] < 1 or any(not torch.is_floating_point(value) for value in tensors):
        raise ValueError("metric tensors must be non-empty floating point values")
    if not bool(torch.isfinite(log_depth).all()) or not bool(torch.isfinite(log_variance).all()):
        raise ValueError("predicted log-depth and log-variance must be finite")
    target_valid = torch.isfinite(target_depth) & (target_depth > 0)
    if valid_mask is None:
        valid = target_valid
    else:
        if valid_mask.shape != target_depth.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean and match target_depth")
        valid = valid_mask & target_valid
    if bool((valid.flatten(1).sum(1) == 0).any()):
        raise ValueError("every evaluated frame must contain at least one valid target pixel")
    return tuple(value.detach().cpu().double() for value in tensors) + (valid.detach().cpu(),)  # type: ignore[return-value]


def _risk_coverage(error: np.ndarray, uncertainty: np.ndarray) -> tuple[list[float], list[float], float]:
    if error.ndim != 1 or uncertainty.shape != error.shape or len(error) < 1:
        raise ValueError("risk-coverage arrays must be non-empty, paired, and one-dimensional")
    if not np.isfinite(error).all() or not np.isfinite(uncertainty).all():
        raise ValueError("risk-coverage values must be finite")
    count = len(error)
    coverage = np.arange(1, count + 1, dtype=np.float64) / count
    predicted_order = np.argsort(uncertainty, kind="stable")
    oracle_order = np.argsort(error, kind="stable")
    denominator = np.arange(1, count + 1, dtype=np.float64)
    ordered_error = error[predicted_order]
    ordered_uncertainty = uncertainty[predicted_order]
    predicted_cumulative = np.empty(count, dtype=np.float64)
    prior_count = 0
    prior_sum = 0.0
    while prior_count < count:
        stop = prior_count + 1
        while stop < count and ordered_uncertainty[stop] == ordered_uncertainty[prior_count]:
            stop += 1
        tie_mean = float(np.mean(ordered_error[prior_count:stop]))
        take = np.arange(1, stop - prior_count + 1, dtype=np.float64)
        predicted_cumulative[prior_count:stop] = prior_sum + take * tie_mean
        prior_sum += float(np.sum(ordered_error[prior_count:stop]))
        prior_count = stop
    predicted_risk = predicted_cumulative / denominator
    oracle_risk = np.cumsum(error[oracle_order]) / denominator
    ause = max(0.0, float(np.trapz(predicted_risk - oracle_risk, coverage)))  # noqa: NPY201
    indices = np.unique(np.linspace(0, count - 1, num=min(count, 256), dtype=np.int64))
    return coverage[indices].tolist(), predicted_risk[indices].tolist(), ause


def _mean_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in METRIC_SPECS}


def evaluate_geometry_quality(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target_depth: torch.Tensor,
    *,
    unit_ids: Sequence[str],
    cluster_ids: Sequence[str],
    group_ids: Sequence[str],
    target_depth_unit: Literal["metres"],
    grouping_receipt_sha256: str,
    grouping_assignment_sha256: str,
    validity_policy_sha256: str,
    valid_mask: torch.Tensor | None = None,
    variance_multiplier: float = 1.0,
    calibration_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate quality/calibration with explicit unit, cluster, and group IDs.

    Pixels are averaged within frames, frames within groups, and groups equally
    for ``group_macro``.  This prevents a large image or sensor family from
    silently dominating the primary endpoint.  The Gaussian log-depth NLL uses
    the historical Phase 2f convention and omits ``0.5*log(2*pi)``. Receipt
    digests are caller-declared provenance: a governed runner must load and
    verify their typed contents before using this pure metric function.
    """

    if target_depth_unit != "metres":
        raise ValueError("geometry quality targets must be metric depth in metres")
    for name, value in (
        ("grouping_receipt_sha256", grouping_receipt_sha256),
        ("grouping_assignment_sha256", grouping_assignment_sha256),
        ("validity_policy_sha256", validity_policy_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if calibration_receipt_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", calibration_receipt_sha256):
        raise ValueError("calibration_receipt_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(variance_multiplier, int | float) or isinstance(variance_multiplier, bool):
        raise TypeError("variance_multiplier must be numeric")
    if not math.isfinite(float(variance_multiplier)) or not 1e-3 <= float(variance_multiplier) <= 1e3:
        raise ValueError("variance_multiplier must be finite and in [1e-3,1e3]")
    if float(variance_multiplier) != 1.0 and calibration_receipt_sha256 is None:
        raise ValueError("a non-default variance_multiplier requires a validation calibration receipt")
    prediction_log, variance_log, target, valid = _validate_inputs(log_depth, log_variance, target_depth, valid_mask)
    frame_count = prediction_log.shape[0]
    units = _validate_identifiers(unit_ids, frame_count, "unit_ids", unique=True)
    clusters = _validate_identifiers(cluster_ids, frame_count, "cluster_ids", unique=False)
    groups = _validate_identifiers(group_ids, frame_count, "group_ids", unique=False)
    observed_grouping_sha256 = geometry_grouping_assignment_sha256(units, clusters, groups)
    if grouping_assignment_sha256 != observed_grouping_sha256:
        raise ValueError("grouping_assignment_sha256 does not match the supplied unit/cluster/group mapping")
    calibrated_log_variance = variance_log + math.log(float(variance_multiplier))

    rows: list[dict[str, Any]] = []
    pooled_error: list[np.ndarray] = []
    pooled_uncertainty: list[np.ndarray] = []
    for index in range(frame_count):
        mask = valid[index]
        predicted_log = prediction_log[index][mask]
        expected_depth = target[index][mask]
        expected_log = expected_depth.log()
        predicted_depth = predicted_log.exp()
        calibrated_variance = calibrated_log_variance[index][mask].exp()
        if not bool(torch.isfinite(predicted_depth).all()) or not bool(torch.isfinite(calibrated_variance).all()):
            raise ValueError("decoded depth and calibrated variance must be finite")
        if not bool((predicted_depth > 0).all()) or not bool((calibrated_variance > 0).all()):
            raise ValueError("decoded depth and calibrated variance must be positive")

        residual = predicted_log - expected_log
        signed_scale = float(residual.median())
        aligned_depth = (predicted_log - signed_scale).exp()
        raw_error = predicted_depth - expected_depth
        aligned_error = aligned_depth - expected_depth
        raw_abs_rel = raw_error.abs() / expected_depth
        aligned_abs_rel = aligned_error.abs() / expected_depth
        ratio = torch.maximum(predicted_depth / expected_depth, expected_depth / predicted_depth)
        nll = 0.5 * (residual.square() / calibrated_variance + calibrated_variance.log())
        uncertainty = calibrated_variance.sqrt()
        error_array = raw_abs_rel.numpy()
        uncertainty_array = uncertainty.numpy()
        pooled_error.append(error_array)
        pooled_uncertainty.append(uncertainty_array)
        _, _, ause = _risk_coverage(error_array, uncertainty_array)
        empirical_coverage = {
            f"{int(level * 100)}": float(
                (residual.abs() <= NormalDist().inv_cdf((1 + level) / 2) * uncertainty).double().mean()
            )
            for level in COVERAGE_LEVELS
        }
        reliability_error = float(
            np.mean([abs(empirical_coverage[str(int(level * 100))] - level) for level in COVERAGE_LEVELS])
        )
        rows.append(
            {
                "unit_id": units[index],
                "cluster_id": clusters[index],
                "group_id": groups[index],
                "valid_pixels": int(mask.sum()),
                "raw_abs_rel": float(raw_abs_rel.mean()),
                "raw_rmse_m": float(raw_error.square().mean().sqrt()),
                "log_rmse": float(residual.square().mean().sqrt()),
                "aligned_abs_rel": float(aligned_abs_rel.mean()),
                "aligned_rmse_m": float(aligned_error.square().mean().sqrt()),
                "signed_log_scale_error": signed_scale,
                "absolute_log_scale_error": abs(signed_scale),
                "delta_1": float((ratio < 1.25).double().mean()),
                "delta_2": float((ratio < 1.25**2).double().mean()),
                "delta_3": float((ratio < 1.25**3).double().mean()),
                "nll": float(nll.mean()),
                "ause": ause,
                "reliability_error": reliability_error,
                "coverage": empirical_coverage,
            }
        )

    per_group: dict[str, dict[str, Any]] = {}
    for group_id in sorted(set(groups)):
        selected = [row for row in rows if row["group_id"] == group_id]
        group_coverage = {
            str(int(level * 100)): float(np.mean([float(row["coverage"][str(int(level * 100))]) for row in selected]))
            for level in COVERAGE_LEVELS
        }
        per_group[group_id] = {
            "frames": len(selected),
            "metrics": _mean_rows(selected),
            "coverage": group_coverage,
        }
    frame_macro = _mean_rows(rows)
    group_macro = {
        name: float(np.mean([float(value["metrics"][name]) for value in per_group.values()])) for name in METRIC_SPECS
    }
    group_macro_coverage = {
        str(int(level * 100)): float(
            np.mean([float(value["coverage"][str(int(level * 100))]) for value in per_group.values()])
        )
        for level in COVERAGE_LEVELS
    }
    curve_coverage, curve_risk, pooled_ause = _risk_coverage(
        np.concatenate(pooled_error), np.concatenate(pooled_uncertainty)
    )
    result: dict[str, Any] = {
        "schema_version": GEOMETRY_QUALITY_SCHEMA,
        "nll_convention": "0.5*(squared_log_residual/variance+log_variance); constant omitted",
        "variance_multiplier": float(variance_multiplier),
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "target_depth_unit": target_depth_unit,
        "grouping_receipt_sha256": grouping_receipt_sha256,
        "grouping_assignment_sha256": grouping_assignment_sha256,
        "validity_policy_sha256": validity_policy_sha256,
        "validity_mask_sha256": hashlib.sha256(
            str(tuple(valid.shape)).encode("ascii") + valid.numpy().astype(np.uint8, copy=False).tobytes()
        ).hexdigest(),
        "provenance_verification": (
            "caller-declared digests; formal use requires an external governed wrapper that verifies typed receipts"
        ),
        "metric_specs": {
            name: {"direction": direction, "unit": unit} for name, (direction, unit) in METRIC_SPECS.items()
        },
        "aggregation": {
            "pixel_to_unit": "mean",
            "unit_to_group": "equal-unit mean",
            "group_macro": "equal-group mean",
            "independent_population_interval": "requires registered-manifest cluster bootstrap",
            "group_macro_interval": "requires a separately preregistered hierarchical/stratified group-cluster method",
        },
        "frames": frame_count,
        "groups": len(per_group),
        "clusters": len(set(clusters)),
        "valid_pixels": sum(int(row["valid_pixels"]) for row in rows),
        "per_unit": rows,
        "per_group": per_group,
        "frame_macro": frame_macro,
        "group_macro": group_macro,
        "group_macro_coverage": group_macro_coverage,
        "risk_coverage": {
            "coverage": curve_coverage,
            "risk": curve_risk,
            "pooled_pixel_ause": pooled_ause,
            "claim": (
                "descriptive pooled-pixel discrete curve integrated from coverage 1/n; promotion uses "
                "per-unit/group metrics and AUSE is not compared across differing valid-pixel supports"
            ),
        },
    }
    _require_finite_tree(result, "geometry_quality")
    return result


def metric_observations(
    report: Mapping[str, Any],
    metric_name: str,
    *,
    estimand: Literal["equal_cluster"] = "equal_cluster",
) -> tuple[ClusteredMetricObservation, ...]:
    """Emit observations for an equal-cluster estimand, never a group-macro interval."""

    if report.get("schema_version") != GEOMETRY_QUALITY_SCHEMA:
        raise ValueError("unexpected geometry quality schema")
    if metric_name not in METRIC_SPECS:
        raise ValueError(f"unknown geometry metric {metric_name!r}")
    if estimand != "equal_cluster":
        raise ValueError("metric observations support only the equal_cluster estimand")
    rows = report.get("per_unit")
    if not isinstance(rows, list) or not rows:
        raise ValueError("geometry quality report has no per-unit rows")
    unit_ids: list[str] = []
    cluster_ids: list[str] = []
    metric_values: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("per-unit geometry rows must be mappings")
        unit_id = row.get("unit_id")
        cluster_id = row.get("cluster_id")
        value = row.get(metric_name)
        if (
            not isinstance(unit_id, str)
            or not isinstance(cluster_id, str)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            raise ValueError("per-unit geometry row is missing its metric or manifest identity")
        unit_ids.append(unit_id)
        cluster_ids.append(cluster_id)
        metric_values.append(float(value))
    units = _validate_identifiers(unit_ids, len(rows), "per-unit unit IDs", unique=True)
    clusters = _validate_identifiers(cluster_ids, len(rows), "per-unit cluster IDs", unique=False)
    observations: list[ClusteredMetricObservation] = []
    for unit_id, cluster_id, value in zip(units, clusters, metric_values, strict=True):
        observations.append(
            ClusteredMetricObservation(
                pair_id=unit_id,
                unit_id=unit_id,
                cluster_id=cluster_id,
                value=value,
            )
        )
    return tuple(observations)
