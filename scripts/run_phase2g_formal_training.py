#!/usr/bin/env python3
"""Train one of the 48 frozen Phase 2g-A formal cells without held-out targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from jepa4d.evaluation.phase2g_data import require_canonical_record_sha256
from jepa4d.training.phase2g_protocol import ARMS, FORMAL_SEEDS, LR_SELECTION_SCHEMA, ROTATIONS
from jepa4d.training.phase2g_runtime import (
    assert_same_execution,
    complete_output,
    finish_wandb_run,
    load_execution_provenance,
    load_json,
    prepare_output,
    start_wandb_run,
)
from jepa4d.training.phase2g_training import run_training_cell


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--rotation", choices=tuple(ROTATIONS), required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--cache-root", type=Path, required=True, help="Rotation-scoped train+validation cache view")
    parser.add_argument("--lr-selection", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2g formal training may run only inside Slurm")
    output = prepare_output(args.output)
    provenance = load_execution_provenance(args.provenance)
    selection = load_json(args.lr_selection)
    if selection.get("schema_version") != LR_SELECTION_SCHEMA or selection.get("status") != "pass":
        raise ValueError("formal training requires a passing Phase 2g LR selection")
    if selection.get("selection_sha256_scope") != "complete-record-excluding-selection_sha256-and-post-upload-wandb":
        raise ValueError("formal training received an unknown LR-selection hash scope")
    require_canonical_record_sha256(
        selection,
        field="selection_sha256",
        excluded_fields=("selection_sha256", "wandb"),
    )
    assert_same_execution((selection,), provenance)
    key = f"{args.arm}:{args.rotation}"
    selected = selection.get("selected", {}).get(key)
    if not isinstance(selected, dict) or "learning_rate" not in selected:
        raise ValueError(f"LR selection lacks {key}")
    learning_rate = float(selected["learning_rate"])
    semantic = f"formal-{args.arm.lower()}-{args.rotation.lower()}-s{args.seed}"
    run = start_wandb_run(
        provenance=provenance,
        job_type="formal-training",
        semantic_name=semantic,
        config={
            "arm": args.arm,
            "rotation": args.rotation,
            "seed": args.seed,
            "learning_rate": learning_rate,
            "lr_selection_sha256": selection.get("selection_sha256"),
        },
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    receipt, files = run_training_cell(
        stage="formal",
        arm=args.arm,
        rotation=args.rotation,
        seed=args.seed,
        learning_rate=learning_rate,
        cache_root=args.cache_root,
        output=output,
        provenance=provenance,
        device=torch.device(args.device),
        wandb_run=run,
    )
    receipt["lr_selection_sha256"] = selection.get("selection_sha256")
    macro = receipt["validation_metrics"]["equal_family_macro"]
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-{semantic}-{provenance['execution_id']}",
        job_type="formal-training",
        files=files,
        summary={
            "validation_raw_abs_rel": macro["raw_abs_rel"],
            "validation_absolute_log_scale_error": macro["absolute_log_scale_error"],
            "best_epoch": receipt["best_epoch"],
        },
    )
    complete_output(output, receipt_name="training_receipt.json", receipt=receipt, wandb_receipt=wandb_receipt)
    print(
        json.dumps(
            {"status": "success", "arm": args.arm, "rotation": args.rotation, "seed": args.seed}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
