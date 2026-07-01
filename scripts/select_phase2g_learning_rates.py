#!/usr/bin/env python3
"""Select one healthy learning rate for every Phase 2g arm/rotation pair."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2g_data import canonical_record_sha256, file_identity
from jepa4d.training.phase2g_protocol import (
    ARMS,
    LEARNING_RATES,
    LR_SELECTION_SCHEMA,
    ROTATIONS,
    TUNING_RECEIPT_SCHEMA,
    TUNING_SEED,
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


def select_learning_rates(receipts: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    if len(receipts) != expected_matrix_size("tuning"):
        raise ValueError("LR selector requires all 48 tuning receipts")
    assert_same_execution(receipts, provenance)
    cells: dict[tuple[str, str, float], dict[str, Any]] = {}
    for receipt in receipts:
        if receipt.get("schema_version") != TUNING_RECEIPT_SCHEMA or receipt.get("status") != "success":
            raise ValueError("LR selector received an invalid tuning receipt")
        arm, rotation = str(receipt.get("arm")), str(receipt.get("rotation"))
        seed, learning_rate = int(receipt.get("seed", -1)), float(receipt.get("learning_rate", math.nan))
        key = (arm, rotation, learning_rate)
        if arm not in ARMS or rotation not in ROTATIONS or seed != TUNING_SEED or learning_rate not in LEARNING_RATES:
            raise ValueError(f"invalid tuning cell identity: {key}")
        if key in cells:
            raise ValueError(f"duplicate tuning cell: {key}")
        health = receipt.get("health", {})
        if not (
            health.get("all_finite") is True
            and float(health.get("maximum_forbidden_gradient_norm", math.inf)) == 0.0
            and health.get("all_expected_allowed_gradients_seen") is True
            and health.get("model_changed_from_initialization") is True
            and health.get("objective_decreased") is True
            and receipt.get("exact_reload") is True
            and receipt.get("wandb", {}).get("mode") == "online"
            and receipt.get("wandb", {}).get("status") == "success"
        ):
            raise ValueError(f"tuning cell is unhealthy; partial selection is forbidden: {key}")
        cells[key] = receipt
    expected = {
        (arm, rotation, learning_rate) for arm in ARMS for rotation in ROTATIONS for learning_rate in LEARNING_RATES
    }
    if set(cells) != expected:
        raise ValueError("tuning matrix is incomplete")
    selected: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for rotation in ROTATIONS:
            ranked = sorted(
                (cells[(arm, rotation, learning_rate)] for learning_rate in LEARNING_RATES),
                key=lambda receipt: (
                    float(receipt["validation_metrics"]["equal_family_macro"]["raw_abs_rel"]),
                    float(receipt["validation_metrics"]["equal_family_macro"]["absolute_log_scale_error"]),
                    float(receipt["learning_rate"]),
                ),
            )
            winner = ranked[0]
            selected[f"{arm}:{rotation}"] = {
                "arm": arm,
                "rotation": rotation,
                "learning_rate": float(winner["learning_rate"]),
                "validation_raw_abs_rel": float(winner["validation_metrics"]["equal_family_macro"]["raw_abs_rel"]),
                "validation_absolute_log_scale_error": float(
                    winner["validation_metrics"]["equal_family_macro"]["absolute_log_scale_error"]
                ),
                "checkpoint_sha256": winner["checkpoint"]["sha256"],
            }
    result: dict[str, Any] = {
        "schema_version": LR_SELECTION_SCHEMA,
        "status": "pass",
        "policy": "healthy_only_then_validation_raw_abs_rel_then_absolute_log_scale_error_then_lower_lr",
        "learning_rates": list(LEARNING_RATES),
        "tuning_seed": TUNING_SEED,
        "selected": selected,
        "external_final_authorized": False,
        "execution_provenance": provenance,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2g LR selection may run only inside Slurm")
    output = prepare_output(args.output)
    provenance = load_execution_provenance(args.provenance)
    receipts = [load_json(path) for path in args.receipt]
    result = select_learning_rates(receipts, provenance)
    result["source_receipts"] = [file_identity(path, schema=TUNING_RECEIPT_SCHEMA) for path in args.receipt]
    result["selection_sha256_scope"] = "complete-record-excluding-selection_sha256-and-post-upload-wandb"
    result["selection_sha256"] = canonical_record_sha256(
        result,
        excluded_fields=("selection_sha256", "wandb"),
    )
    atomic_json(output / "lr_selection.json", result)
    report = atomic_json(
        output / "lr_selection_report.json",
        {
            "schema_version": "jepa4d-phase2g-lr-selection-report-v1",
            "status": "pass",
            "policy": result["policy"],
            "selected": result["selected"],
            "protected_paths_uploaded": False,
        },
    )
    run = start_wandb_run(
        provenance=provenance,
        job_type="lr-selection",
        semantic_name="lr-selection",
        config={"cells": len(receipts), "policy": result["policy"]},
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-lr-selection-{provenance['execution_id']}",
        job_type="lr-selection",
        files=(report,),
        summary={"status": "pass", "selected_cells": len(result["selected"])},
    )
    complete_output(output, receipt_name="lr_selection.json", receipt=result, wandb_receipt=wandb_receipt)
    print(json.dumps({"status": "pass", "selected_cells": len(result["selected"])}, sort_keys=True))


if __name__ == "__main__":
    main()
