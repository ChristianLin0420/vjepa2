#!/usr/bin/env python3
"""Apply frozen Phase 2g quality/mechanism gates and select a development survivor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jepa4d.evaluation.phase2g_data import canonical_record_sha256, file_identity
from jepa4d.evaluation.phase2g_metrics import (
    aggregate_evaluation_rows,
    paired_hierarchical_bootstrap,
)
from jepa4d.evaluation.phase2g_visualization import write_aggregate_visualizations
from jepa4d.training.phase2g_protocol import (
    ARMS,
    BOOTSTRAP_RESAMPLES,
    CANDIDATES,
    EVALUATION_RECEIPT_SCHEMA,
    FAMILIES,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    MECHANISM_THRESHOLDS,
    PRIMARY_METRICS,
    QUALITY_THRESHOLDS,
    RAW_ABS_REL_TIE_RELATIVE,
    ROTATIONS,
    SAMPLES_PER_FAMILY,
    SCALE_ERROR_TIE_RELATIVE,
    SELECTOR_SCHEMA,
    SIMPLICITY_ORDER,
    expected_matrix_size,
)
from jepa4d.training.phase2g_runtime import (
    assert_same_execution,
    atomic_json,
    complete_output,
    finish_wandb_run,
    load_execution_provenance,
    load_json,
    prepare_output,
    start_wandb_run,
)


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"invalid ratio inputs for {label}")
    return numerator / denominator


def _validate_matrix(receipts: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> None:
    if len(receipts) != expected_matrix_size("evaluation"):
        raise ValueError("Phase 2g selector requires all 48 evaluation receipts")
    assert_same_execution(receipts, provenance)
    cells: set[tuple[str, str, int]] = set()
    for receipt in receipts:
        key = (str(receipt.get("arm")), str(receipt.get("rotation")), int(receipt.get("seed", -1)))
        if (
            receipt.get("schema_version") != EVALUATION_RECEIPT_SCHEMA
            or receipt.get("status") != "success"
            or key[0] not in ARMS
            or key[1] not in ROTATIONS
            or key[2] not in FORMAL_SEEDS
            or receipt.get("heldout_family") != ROTATIONS[key[1]]["heldout"]
            or receipt.get("all_values_finite") is not True
            or receipt.get("checkpoint_frozen_before_heldout_access") is not True
            or receipt.get("external_final_authorized") is not False
            or receipt.get("external_final_accessed") is not False
            or receipt.get("wandb", {}).get("mode") != "online"
            or receipt.get("wandb", {}).get("status") != "success"
            or receipt.get("qualitative", {}).get("count") != 16
            or len(receipt.get("training_diagnostics", {}).get("epoch_diagnostics", [])) != FORMAL_EPOCHS
        ):
            raise ValueError(f"invalid/incomplete Phase 2g evaluation receipt: {key}")
        if key[0] != "M0" and len(receipt.get("scale_mechanism", {}).get("per_frame", [])) != SAMPLES_PER_FAMILY:
            raise ValueError(f"scale-mechanism frame diagnostics are incomplete: {key}")
        if key[0] == "M3" and any(
            len(receipt.get("zero_field_intervention", {}).get(name, [])) != SAMPLES_PER_FAMILY
            for name in ("per_frame_field_mean", "per_frame_field_sd")
        ):
            raise ValueError(f"M3 field-distribution diagnostics are incomplete: {key}")
        if key in cells:
            raise ValueError(f"duplicate Phase 2g evaluation cell: {key}")
        cells.add(key)
    expected = {(arm, rotation, seed) for arm in ARMS for rotation in ROTATIONS for seed in FORMAL_SEEDS}
    if cells != expected:
        raise ValueError("Phase 2g evaluation matrix is incomplete")


def _family_values(aggregates: Mapping[str, Any], arm: str, metric: str) -> dict[str, float]:
    return {family: float(aggregates[arm]["_per_family"][family][metric]) for family in FAMILIES}


def _camera_gate(receipts: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    arm_receipts = [receipt for receipt in receipts if receipt["arm"] == arm]
    controls = ("stale", "wrong", "permuted")
    family_condition: dict[str, dict[str, float]] = {}
    structural_checks: list[bool] = []
    for family in FAMILIES:
        family_receipts = [receipt for receipt in arm_receipts if receipt["heldout_family"] == family]
        values: dict[str, list[float]] = {condition: [] for condition in ("updated", *controls)}
        for receipt in family_receipts:
            camera = receipt["camera_controls"]
            if camera.get("status") != "evaluated" or camera.get("consumes_intrinsics") is not True:
                raise ValueError(f"{arm} lacks evaluated camera controls")
            for condition in values:
                values[condition].append(float(camera["raw_abs_rel"][condition]))
            structural_checks.append(
                camera.get("distinct_analytic_intrinsics_per_source_min")
                == MECHANISM_THRESHOLDS["distinct_analytic_intrinsics_per_source"]
                and camera.get("distinct_analytic_intrinsics_per_source_max")
                == MECHANISM_THRESHOLDS["distinct_analytic_intrinsics_per_source"]
                and camera.get("permutation_assignment_change_fraction")
                == MECHANISM_THRESHOLDS["permutation_assignment_change_fraction"]
                and camera.get("permutation_matrix_change_fraction")
                == MECHANISM_THRESHOLDS["permutation_matrix_change_fraction"]
                and all(
                    float(delta) > MECHANISM_THRESHOLDS["minimum_mean_absolute_prediction_delta_m_exclusive"]
                    for delta in camera["mean_absolute_prediction_delta_m"].values()
                )
            )
        family_condition[family] = {condition: float(np.mean(numbers)) for condition, numbers in values.items()}
    equal_family = {
        condition: float(np.mean([family_condition[family][condition] for family in FAMILIES]))
        for condition in ("updated", *controls)
    }
    ratios = {
        condition: _ratio(equal_family["updated"], equal_family[condition], f"{arm}/updated/{condition}")
        for condition in controls
    }
    family_wins = {
        condition: sum(
            family_condition[family]["updated"] < family_condition[family][condition] for family in FAMILIES
        )
        for condition in controls
    }
    checks = {
        **{
            f"updated_to_{condition}_ratio": ratio <= MECHANISM_THRESHOLDS["updated_to_control_raw_abs_rel_ratio_max"]
            for condition, ratio in ratios.items()
        },
        **{
            f"updated_beats_{condition}_families": count
            >= MECHANISM_THRESHOLDS["updated_control_improving_families_min"]
            for condition, count in family_wins.items()
        },
        "analytic_and_sensitivity_audit": all(structural_checks),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ratios": ratios,
        "family_wins": family_wins,
        "per_family": family_condition,
        "equal_family": equal_family,
    }


def _negative_camera_gate(receipts: Sequence[Mapping[str, Any]], arm: str) -> bool:
    return all(
        receipt["camera_controls"].get("status") == "not_applicable_nonconsumer"
        and receipt["camera_controls"].get("consumes_intrinsics") is False
        and receipt["camera_controls"].get("camera_parameters") == 0
        and receipt["camera_controls"].get("evaluator_intrinsics_call_rejected") is True
        for receipt in receipts
        if receipt["arm"] == arm
    )


def _zero_field_gate(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        family_receipts = [
            receipt for receipt in receipts if receipt["arm"] == "M3" and receipt["heldout_family"] == family
        ]
        full = float(np.mean([receipt["metrics"]["equal_family_macro"]["raw_abs_rel"] for receipt in family_receipts]))
        zero = float(
            np.mean(
                [
                    receipt["zero_field_intervention"]["metrics"]["equal_family_macro"]["raw_abs_rel"]
                    for receipt in family_receipts
                ]
            )
        )
        if not all(receipt["zero_field_intervention"].get("same_checkpoint") is True for receipt in family_receipts):
            raise ValueError("M3 zero-field intervention did not reuse its selected checkpoint")
        per_family[family] = {"full": full, "zero": zero}
    equal_full = float(np.mean([value["full"] for value in per_family.values()]))
    equal_zero = float(np.mean([value["zero"] for value in per_family.values()]))
    ratio = _ratio(equal_full, equal_zero, "M3 full/zero field")
    wins = sum(value["full"] < value["zero"] for value in per_family.values())
    checks = {
        "full_to_zero_ratio": ratio <= MECHANISM_THRESHOLDS["m3_full_to_zero_field_raw_abs_rel_ratio_max"],
        "improving_families": wins >= MECHANISM_THRESHOLDS["m3_full_improving_families_min"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ratio": ratio,
        "family_wins": wins,
        "per_family": per_family,
    }


def _variation(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for family in FAMILIES:
            rows = [receipt for receipt in receipts if receipt["arm"] == arm and receipt["heldout_family"] == family]
            result[arm][family] = {
                metric: {
                    "seed_means": [float(row["metrics"]["equal_family_macro"][metric]) for row in rows],
                    "sample_sd": float(
                        np.std([float(row["metrics"]["equal_family_macro"][metric]) for row in rows], ddof=1)
                    ),
                }
                for metric in PRIMARY_METRICS
            }
    return result


def select_survivor(
    receipts: list[dict[str, Any]],
    provenance: dict[str, Any],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    _validate_matrix(receipts, provenance)
    aggregates, frame_rows = aggregate_evaluation_rows(receipts)
    reference = aggregates["M0"]
    if not _negative_camera_gate(receipts, "M0") or not _negative_camera_gate(receipts, "M1"):
        raise ValueError("M0/M1 structural camera-negative controls failed")
    camera = {arm: _camera_gate(receipts, arm) for arm in ("M2", "M3")}
    zero_field = _zero_field_gate(receipts)
    eligibility: dict[str, Any] = {}
    for arm in CANDIDATES:
        values = aggregates[arm]
        arm_families = _family_values(aggregates, arm, "raw_abs_rel")
        reference_families = _family_values(aggregates, "M0", "raw_abs_rel")
        improving = sum(arm_families[family] < reference_families[family] for family in FAMILIES)
        worst_ratios = {
            family: _ratio(arm_families[family], reference_families[family], f"{arm}/{family}/M0")
            for family in FAMILIES
        }
        checks = {
            "raw_abs_rel": _ratio(values["raw_abs_rel"], reference["raw_abs_rel"], f"{arm}/M0 raw")
            <= QUALITY_THRESHOLDS["raw_abs_rel_ratio_to_m0_max"],
            "absolute_log_scale_error": _ratio(
                values["absolute_log_scale_error"],
                reference["absolute_log_scale_error"],
                f"{arm}/M0 scale",
            )
            <= QUALITY_THRESHOLDS["absolute_log_scale_error_ratio_to_m0_max"],
            "aligned_abs_rel": _ratio(values["aligned_abs_rel"], reference["aligned_abs_rel"], f"{arm}/M0 aligned")
            <= QUALITY_THRESHOLDS["aligned_abs_rel_ratio_to_m0_max"],
            "nll": values["nll"] - reference["nll"] <= QUALITY_THRESHOLDS["nll_difference_to_m0_max"],
            "ause": _ratio(values["ause"], reference["ause"], f"{arm}/M0 ause")
            <= QUALITY_THRESHOLDS["ause_ratio_to_m0_max"],
            "family_consistency": improving >= QUALITY_THRESHOLDS["raw_abs_rel_improving_families_min"],
            "worst_family_protection": all(
                ratio <= QUALITY_THRESHOLDS["raw_abs_rel_worst_family_ratio_to_m0_max"]
                for ratio in worst_ratios.values()
            ),
            "completeness": len([receipt for receipt in receipts if receipt["arm"] == arm]) == 12,
        }
        mechanism: dict[str, bool] = {}
        if arm in {"M2", "M3"}:
            mechanism["camera_controls"] = camera[arm]["passed"]
        if arm == "M2":
            mechanism["m2_beats_m1"] = (
                _ratio(aggregates["M2"]["raw_abs_rel"], aggregates["M1"]["raw_abs_rel"], "M2/M1")
                <= MECHANISM_THRESHOLDS["m2_raw_abs_rel_ratio_to_m1_max"]
            )
        if arm == "M3":
            mechanism["m3_beats_m2"] = (
                _ratio(aggregates["M3"]["raw_abs_rel"], aggregates["M2"]["raw_abs_rel"], "M3/M2")
                <= MECHANISM_THRESHOLDS["m3_raw_abs_rel_ratio_to_m2_max"]
            )
            mechanism["zero_field"] = zero_field["passed"]
        eligibility[arm] = {
            "eligible": all(checks.values()) and all(mechanism.values()),
            "quality_checks": checks,
            "mechanism_checks": mechanism,
            "raw_abs_rel_improving_families": improving,
            "worst_family_ratios": worst_ratios,
        }
    eligible = [arm for arm in CANDIDATES if eligibility[arm]["eligible"]]
    survivor: str | None = None
    if eligible:
        best_raw = min(float(aggregates[arm]["raw_abs_rel"]) for arm in eligible)
        raw_tied = [
            arm
            for arm in eligible
            if float(aggregates[arm]["raw_abs_rel"]) <= best_raw * (1 + RAW_ABS_REL_TIE_RELATIVE)
        ]
        best_scale = min(float(aggregates[arm]["absolute_log_scale_error"]) for arm in raw_tied)
        scale_tied = [
            arm
            for arm in raw_tied
            if float(aggregates[arm]["absolute_log_scale_error"]) <= best_scale * (1 + SCALE_ERROR_TIE_RELATIVE)
        ]
        survivor = min(
            scale_tied,
            key=lambda arm: (
                float(aggregates[arm]["nll"]),
                SIMPLICITY_ORDER.index(arm),
            ),
        )
    bootstrap = {
        arm: {
            metric: paired_hierarchical_bootstrap(
                frame_rows,
                candidate=arm,
                metric=metric,
                resamples=bootstrap_resamples,
            )
            for metric in PRIMARY_METRICS
        }
        for arm in CANDIDATES
    }
    paired_effects = {
        arm: {metric: float(aggregates[arm][metric] - aggregates["M0"][metric]) for metric in PRIMARY_METRICS}
        for arm in CANDIDATES
    }
    result: dict[str, Any] = {
        "schema_version": SELECTOR_SCHEMA,
        "status": "success",
        "formal_matrix": {"arms": list(ARMS), "rotations": list(ROTATIONS), "seeds": list(FORMAL_SEEDS)},
        "aggregation": "frames_within_heldout_family_then_seeds_within_rotation_then_equal_four_families",
        "aggregates": aggregates,
        "variation": _variation(receipts),
        "paired_candidate_minus_m0": paired_effects,
        "hierarchical_bootstrap": bootstrap,
        "eligibility": eligibility,
        "camera_mechanisms": camera,
        "m3_zero_field_mechanism": zero_field,
        "eligible_arms": eligible,
        "survivor": survivor,
        "retained_arm": survivor or "M0",
        "no_survivor": survivor is None,
        "external_final_authorized": False,
        "external_final_accessed": False,
        "execution_provenance": provenance,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-receipt", type=Path, action="append", default=[])
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2g survivor selection may run only inside Slurm")
    output = prepare_output(args.output)
    provenance = load_execution_provenance(args.provenance)
    receipts = [load_json(path) for path in args.evaluation_receipt]
    result = select_survivor(receipts, provenance)
    result["source_receipts"] = [
        file_identity(path, schema=EVALUATION_RECEIPT_SCHEMA) for path in args.evaluation_receipt
    ]
    csv_path = output / "selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "eligible", *PRIMARY_METRICS))
        for arm in ARMS:
            writer.writerow(
                (
                    arm,
                    arm == "M0" or result["eligibility"].get(arm, {}).get("eligible", False),
                    *(result["aggregates"][arm][metric] for metric in PRIMARY_METRICS),
                )
            )
    report_path = atomic_json(
        output / "selection_report.json",
        {
            "schema_version": "jepa4d-phase2g-selection-report-v1",
            "status": "success",
            "aggregates": result["aggregates"],
            "eligibility": result["eligibility"],
            "camera_mechanisms": result["camera_mechanisms"],
            "m3_zero_field_mechanism": result["m3_zero_field_mechanism"],
            "survivor": result["survivor"],
            "retained_arm": result["retained_arm"],
            "external_final_authorized": False,
            "protected_paths_uploaded": False,
        },
    )
    aggregate_visuals = write_aggregate_visualizations(output, result=result, receipts=receipts)
    result["visualizations"] = {
        "categories": [
            "per-family-per-seed forest",
            "raw versus aligned AbsRel and scale error",
            "scale residuals and predicted-versus-optimal correlation",
            "reliability and risk-coverage/AUSE",
            "P1-P7 camera controls",
            "M3 full versus zero field",
            "fixed qualitative completeness (protected panels remain local)",
            "loss and allowed/forbidden-gradient diagnostics",
            "resource diagnostics",
            "provenance/failure/retry/completeness",
        ],
        "files": [file_identity(path) for path in aggregate_visuals],
        "protected_qualitative_pixels_uploaded": False,
    }
    result["selection_sha256_scope"] = "complete-record-excluding-selection_sha256-and-post-upload-wandb"
    result["selection_sha256"] = canonical_record_sha256(
        result,
        excluded_fields=("selection_sha256", "wandb"),
    )
    atomic_json(output / "selector.json", result)
    run = start_wandb_run(
        provenance=provenance,
        job_type="selection",
        semantic_name="selection",
        config={
            "cells": len(receipts),
            "quality_thresholds": QUALITY_THRESHOLDS,
            "mechanism_thresholds": MECHANISM_THRESHOLDS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-selection-{provenance['execution_id']}",
        job_type="selection",
        files=(csv_path, report_path, *aggregate_visuals),
        summary={
            "survivor": result["survivor"] or "none",
            "retained_arm": result["retained_arm"],
            "eligible_candidates": len(result["eligible_arms"]),
            "external_final_authorized": False,
        },
    )
    complete_output(output, receipt_name="selector.json", receipt=result, wandb_receipt=wandb_receipt)
    print(json.dumps({"status": "success", "survivor": result["survivor"]}, sort_keys=True))


if __name__ == "__main__":
    main()
