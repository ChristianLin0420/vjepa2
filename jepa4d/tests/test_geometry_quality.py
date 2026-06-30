from __future__ import annotations

import math

import pytest
import torch

from jepa4d.evaluation.geometry_quality import (
    GEOMETRY_QUALITY_SCHEMA,
    evaluate_geometry_quality,
    geometry_grouping_assignment_sha256,
    metric_observations,
)

GROUPING_RECEIPT = "b" * 64
CALIBRATION_RECEIPT = "c" * 64
VALIDITY_POLICY = "d" * 64
SINGLE_GROUPING = geometry_grouping_assignment_sha256(("unit-0",), ("scene-0",), ("sensor-a",))


def _evaluate(predicted_depth: torch.Tensor, target_depth: torch.Tensor, *, groups: tuple[str, ...] | None = None):
    predicted_depth = predicted_depth.double()
    target_depth = target_depth.double()
    count = predicted_depth.shape[0]
    units = tuple(f"unit-{index}" for index in range(count))
    clusters = tuple(f"scene-{index}" for index in range(count))
    group_ids = groups or tuple("sensor-a" for _ in range(count))
    return evaluate_geometry_quality(
        predicted_depth.log(),
        torch.zeros_like(predicted_depth),
        target_depth,
        unit_ids=units,
        cluster_ids=clusters,
        group_ids=group_ids,
        target_depth_unit="metres",
        grouping_receipt_sha256=GROUPING_RECEIPT,
        grouping_assignment_sha256=geometry_grouping_assignment_sha256(units, clusters, group_ids),
        validity_policy_sha256=VALIDITY_POLICY,
    )


def test_perfect_prediction_has_exact_quality_and_delta_metrics() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = _evaluate(target.clone(), target)
    assert result["schema_version"] == GEOMETRY_QUALITY_SCHEMA
    metrics = result["group_macro"]
    for name in (
        "raw_abs_rel",
        "raw_rmse_m",
        "log_rmse",
        "aligned_abs_rel",
        "aligned_rmse_m",
        "absolute_log_scale_error",
    ):
        assert metrics[name] == pytest.approx(0.0, abs=1e-12)
    assert metrics["delta_1"] == pytest.approx(1.0)
    assert metrics["delta_2"] == pytest.approx(1.0)
    assert metrics["delta_3"] == pytest.approx(1.0)


def test_constant_scale_error_is_removed_only_from_aligned_metrics() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = _evaluate(2 * target, target)
    metrics = result["group_macro"]
    assert metrics["raw_abs_rel"] == pytest.approx(1.0)
    assert metrics["signed_log_scale_error"] == pytest.approx(math.log(2), rel=1e-6)
    assert metrics["absolute_log_scale_error"] == pytest.approx(math.log(2), rel=1e-6)
    assert metrics["aligned_abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["aligned_rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["delta_3"] == pytest.approx(0.0)


def test_rmse_absrel_and_delta_have_analytic_values() -> None:
    target = torch.tensor([[[1.0, 2.0]]])
    prediction = torch.tensor([[[2.0, 2.0]]])
    metrics = _evaluate(prediction, target)["group_macro"]
    assert metrics["raw_abs_rel"] == pytest.approx(0.5)
    assert metrics["raw_rmse_m"] == pytest.approx(math.sqrt(0.5))
    assert metrics["delta_1"] == pytest.approx(0.5)
    assert metrics["delta_2"] == pytest.approx(0.5)
    assert metrics["delta_3"] == pytest.approx(0.5)


def test_group_macro_weights_sensor_families_equally() -> None:
    target = torch.ones((3, 1, 1))
    prediction = torch.tensor([[[1.0]], [[1.0]], [[3.0]]])
    result = _evaluate(prediction, target, groups=("large", "large", "small"))
    assert result["frame_macro"]["raw_abs_rel"] == pytest.approx(2 / 3)
    assert result["group_macro"]["raw_abs_rel"] == pytest.approx(1.0)
    assert result["per_group"]["large"]["frames"] == 2
    assert result["per_group"]["small"]["frames"] == 1


def test_coverage_reliability_and_risk_outputs_are_finite() -> None:
    target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]])
    prediction = target * torch.tensor([1.05, 1.20]).view(2, 1, 1)
    result = _evaluate(prediction, target, groups=("kv1", "kv2"))
    assert set(result["group_macro_coverage"]) == {"50", "80", "90", "95"}
    assert 0 <= result["group_macro"]["reliability_error"] <= 1
    assert result["risk_coverage"]["pooled_pixel_ause"] >= 0
    assert all(math.isfinite(value) for value in result["risk_coverage"]["risk"])


def test_oracle_uncertainty_ranking_has_zero_ause_and_reversal_is_worse() -> None:
    target = torch.ones((1, 1, 3), dtype=torch.float64)
    prediction = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    oracle = evaluate_geometry_quality(
        prediction.log(),
        torch.tensor([[[-4.0, 0.0, 2.0]]], dtype=torch.float64),
        target,
        unit_ids=("unit-0",),
        cluster_ids=("scene-0",),
        group_ids=("sensor-a",),
        target_depth_unit="metres",
        grouping_receipt_sha256=GROUPING_RECEIPT,
        grouping_assignment_sha256=SINGLE_GROUPING,
        validity_policy_sha256=VALIDITY_POLICY,
    )
    reversed_ranking = evaluate_geometry_quality(
        prediction.log(),
        torch.tensor([[[2.0, 0.0, -4.0]]], dtype=torch.float64),
        target,
        unit_ids=("unit-0",),
        cluster_ids=("scene-0",),
        group_ids=("sensor-a",),
        target_depth_unit="metres",
        grouping_receipt_sha256=GROUPING_RECEIPT,
        grouping_assignment_sha256=SINGLE_GROUPING,
        validity_policy_sha256=VALIDITY_POLICY,
    )
    assert oracle["group_macro"]["ause"] == pytest.approx(0.0)
    assert reversed_ranking["group_macro"]["ause"] > oracle["group_macro"]["ause"]


def test_variance_multiplier_changes_nll_without_changing_point_metrics() -> None:
    target = torch.ones((1, 1, 1), dtype=torch.float64)
    prediction_log = torch.full_like(target, math.log(2))
    unscaled = evaluate_geometry_quality(
        prediction_log,
        torch.zeros_like(target),
        target,
        unit_ids=("unit-0",),
        cluster_ids=("scene-0",),
        group_ids=("sensor-a",),
        target_depth_unit="metres",
        grouping_receipt_sha256=GROUPING_RECEIPT,
        grouping_assignment_sha256=SINGLE_GROUPING,
        validity_policy_sha256=VALIDITY_POLICY,
    )
    scaled = evaluate_geometry_quality(
        prediction_log,
        torch.zeros_like(target),
        target,
        unit_ids=("unit-0",),
        cluster_ids=("scene-0",),
        group_ids=("sensor-a",),
        target_depth_unit="metres",
        grouping_receipt_sha256=GROUPING_RECEIPT,
        grouping_assignment_sha256=SINGLE_GROUPING,
        validity_policy_sha256=VALIDITY_POLICY,
        variance_multiplier=4.0,
        calibration_receipt_sha256=CALIBRATION_RECEIPT,
    )
    residual = math.log(2)
    assert unscaled["group_macro"]["nll"] == pytest.approx(0.5 * residual**2)
    assert scaled["group_macro"]["nll"] == pytest.approx(0.5 * (residual**2 / 4 + math.log(4)))
    assert scaled["group_macro"]["raw_abs_rel"] == unscaled["group_macro"]["raw_abs_rel"]


def test_invalid_target_values_outside_explicit_mask_are_not_evaluated() -> None:
    target = torch.tensor([[[1.0, float("nan")]]], dtype=torch.float64)
    prediction = torch.ones_like(target)
    prediction[..., 1] = 2.0
    report = evaluate_geometry_quality(
        prediction.log(),
        torch.zeros_like(prediction),
        target,
        unit_ids=("unit-0",),
        cluster_ids=("scene-0",),
        group_ids=("sensor-a",),
        target_depth_unit="metres",
        grouping_receipt_sha256=GROUPING_RECEIPT,
        grouping_assignment_sha256=SINGLE_GROUPING,
        validity_policy_sha256=VALIDITY_POLICY,
        valid_mask=torch.tensor([[[True, False]]]),
    )
    assert report["valid_pixels"] == 1
    assert report["group_macro"]["raw_abs_rel"] == pytest.approx(0.0)


def test_metric_observations_preserve_manifest_unit_and_cluster() -> None:
    target = torch.ones((2, 1, 1))
    report = _evaluate(target, target, groups=("kv1", "kv2"))
    observations = metric_observations(report, "raw_abs_rel")
    assert [value.unit_id for value in observations] == ["unit-0", "unit-1"]
    assert [value.cluster_id for value in observations] == ["scene-0", "scene-1"]
    assert all(value.value == 0 for value in observations)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate-unit", "unit_ids must be unique"),
        ("empty-frame", "at least one valid target pixel"),
        ("nonfinite-prediction", "predicted log-depth"),
    ],
)
def test_invalid_geometry_inputs_fail_closed(mutation: str, message: str) -> None:
    target = torch.ones((2, 1, 1))
    prediction = target.clone()
    units = ("unit-0", "unit-1")
    mask = torch.ones_like(target, dtype=torch.bool)
    if mutation == "duplicate-unit":
        units = ("unit-0", "unit-0")
    elif mutation == "empty-frame":
        mask[1] = False
    else:
        prediction[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match=message):
        evaluate_geometry_quality(
            prediction.log(),
            torch.zeros_like(prediction),
            target,
            unit_ids=units,
            cluster_ids=("scene-0", "scene-1"),
            group_ids=("kv1", "kv2"),
            target_depth_unit="metres",
            grouping_receipt_sha256=GROUPING_RECEIPT,
            grouping_assignment_sha256=geometry_grouping_assignment_sha256(
                units, ("scene-0", "scene-1"), ("kv1", "kv2")
            ),
            validity_policy_sha256=VALIDITY_POLICY,
            valid_mask=mask,
        )


@pytest.mark.parametrize("label", ("unit_ids", "cluster_ids", "group_ids"))
@pytest.mark.parametrize(
    ("identifier", "message"),
    [
        ("/lustre/restricted/frame.png", "path-safe pseudonymous"),
        ("../relative-frame", "path-safe pseudonymous"),
        (r"C:\\restricted\\frame.png", "path-safe pseudonymous"),
        ("unit\nforged", "path-safe pseudonymous"),
        ("x" * 129, "path-safe pseudonymous"),
        ("wandb_v1_" + "A" * 40, "credential-like"),
        ("hf_" + "a" * 32, "credential-like"),
        ("api-key-secret", "credential-like"),
    ],
)
def test_identifiers_reject_paths_controls_credentials_and_oversize(
    label: str,
    identifier: str,
    message: str,
) -> None:
    target = torch.ones((1, 1, 1))
    identities = {
        "unit_ids": ("unit-0",),
        "cluster_ids": ("scene-0",),
        "group_ids": ("sensor-a",),
    }
    identities[label] = (identifier,)
    with pytest.raises(ValueError, match=message):
        evaluate_geometry_quality(
            target.log(),
            torch.zeros_like(target),
            target,
            **identities,  # type: ignore[arg-type]
            target_depth_unit="metres",
            grouping_receipt_sha256=GROUPING_RECEIPT,
            grouping_assignment_sha256=geometry_grouping_assignment_sha256(
                identities["unit_ids"], identities["cluster_ids"], identities["group_ids"]
            ),
            validity_policy_sha256=VALIDITY_POLICY,
        )


def test_unknown_metric_and_schema_are_rejected() -> None:
    target = torch.ones((1, 1, 1))
    report = _evaluate(target, target)
    with pytest.raises(ValueError, match="unknown geometry metric"):
        metric_observations(report, "latency_ms")
    with pytest.raises(ValueError, match="unexpected geometry quality schema"):
        metric_observations({**report, "schema_version": "wrong"}, "raw_abs_rel")
    with pytest.raises(ValueError, match="equal_cluster"):
        metric_observations(report, "raw_abs_rel", estimand="group_macro")  # type: ignore[arg-type]
    report["per_unit"][0]["raw_abs_rel"] = True
    with pytest.raises(ValueError, match="missing its metric"):
        metric_observations(report, "raw_abs_rel")


def test_tied_uncertainty_is_permutation_invariant_and_curve_keeps_full_coverage() -> None:
    target = torch.ones((1, 1, 4))
    first = _evaluate(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), target)
    second = _evaluate(torch.tensor([[[4.0, 3.0, 2.0, 1.0]]]), target)
    assert first["group_macro"]["ause"] == pytest.approx(second["group_macro"]["ause"])
    assert first["group_macro"]["ause"] > 0
    assert first["risk_coverage"]["coverage"][-1] == pytest.approx(1.0)


def test_nondefault_calibration_and_group_assignment_require_bound_provenance() -> None:
    target = torch.ones((1, 1, 1))
    common = {
        "unit_ids": ("unit-0",),
        "cluster_ids": ("scene-0",),
        "group_ids": ("sensor-a",),
        "target_depth_unit": "metres",
        "grouping_receipt_sha256": GROUPING_RECEIPT,
        "grouping_assignment_sha256": SINGLE_GROUPING,
        "validity_policy_sha256": VALIDITY_POLICY,
    }
    with pytest.raises(ValueError, match="calibration receipt"):
        evaluate_geometry_quality(
            target.log(),
            torch.zeros_like(target),
            target,
            variance_multiplier=2.0,
            **common,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_geometry_quality(
            target.log(),
            torch.zeros_like(target),
            target,
            **{**common, "grouping_assignment_sha256": "e" * 64},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="metric depth in metres"):
        evaluate_geometry_quality(
            target.log(),
            torch.zeros_like(target),
            target,
            **{**common, "target_depth_unit": "feet"},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("variance_multiplier", [0.0, 1e-4, 1e4, float("nan"), True])
def test_invalid_variance_multiplier_is_rejected(variance_multiplier: float) -> None:
    target = torch.ones((1, 1, 1))
    with pytest.raises((TypeError, ValueError), match="variance_multiplier"):
        evaluate_geometry_quality(
            target.log(),
            torch.zeros_like(target),
            target,
            unit_ids=("unit-0",),
            cluster_ids=("scene-0",),
            group_ids=("sensor-a",),
            target_depth_unit="metres",
            grouping_receipt_sha256=GROUPING_RECEIPT,
            grouping_assignment_sha256=SINGLE_GROUPING,
            validity_policy_sha256=VALIDITY_POLICY,
            variance_multiplier=variance_multiplier,
        )
