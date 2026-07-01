#!/usr/bin/env python3
"""Run one of the 48 frozen Phase 2g-A learning-rate tuning cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from jepa4d.training.phase2g_protocol import ARMS, LEARNING_RATES, ROTATIONS, TUNING_SEED
from jepa4d.training.phase2g_runtime import (
    complete_output,
    finish_wandb_run,
    load_execution_provenance,
    prepare_output,
    start_wandb_run,
)
from jepa4d.training.phase2g_training import run_training_cell


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--rotation", choices=tuple(ROTATIONS), required=True)
    parser.add_argument("--learning-rate", type=float, choices=LEARNING_RATES, required=True)
    parser.add_argument("--cache-root", type=Path, required=True, help="Rotation-scoped train+validation cache view")
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2g tuning may run only inside Slurm")
    output = prepare_output(args.output)
    provenance = load_execution_provenance(args.provenance)
    semantic = f"tuning-{args.arm.lower()}-{args.rotation.lower()}-lr{args.learning_rate:g}"
    run = start_wandb_run(
        provenance=provenance,
        job_type="tuning",
        semantic_name=semantic,
        config={
            "arm": args.arm,
            "rotation": args.rotation,
            "seed": TUNING_SEED,
            "learning_rate": args.learning_rate,
        },
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    receipt, files = run_training_cell(
        stage="tuning",
        arm=args.arm,
        rotation=args.rotation,
        seed=TUNING_SEED,
        learning_rate=args.learning_rate,
        cache_root=args.cache_root,
        output=output,
        provenance=provenance,
        device=torch.device(args.device),
        wandb_run=run,
    )
    macro = receipt["validation_metrics"]["equal_family_macro"]
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-{semantic}-{provenance['execution_id']}",
        job_type="tuning",
        files=files,
        summary={
            "validation_raw_abs_rel": macro["raw_abs_rel"],
            "validation_absolute_log_scale_error": macro["absolute_log_scale_error"],
            "healthy": True,
        },
    )
    complete_output(output, receipt_name="training_receipt.json", receipt=receipt, wandb_receipt=wandb_receipt)
    print(json.dumps({"status": "success", "arm": args.arm, "rotation": args.rotation}, sort_keys=True))


if __name__ == "__main__":
    main()
