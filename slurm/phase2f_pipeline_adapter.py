#!/usr/bin/env python3
"""Bind Phase 2f aggregate/selector library results to the current Slurm provenance before W&B upload."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from jepa4d.evaluation.phase2f_metrics import atomic_json, publish_online_wandb, self_contained_html
from scripts.aggregate_phase2f_qualification import (
    ARMS,
    _write_outputs,
    aggregate_latency_receipts,
    aggregate_pilot_receipts,
)
from scripts.select_phase2f_survivor import METRICS, _plot, select_survivor
from slurm.phase2f_contract import load_json


def _publish(*, output: Path, result: dict, provenance: dict, job_type: str, artifact_files: tuple[Path, ...]) -> dict:
    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    return publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type=job_type,
        run_name=f"{execution_id}-{job_type}-{job_id}",
        config={
            "execution_id": execution_id,
            "git_commit": provenance["git_commit"],
            "job_label": provenance["job_label"],
        },
        summary={"status": result["status"]},
        artifact_name=f"phase2f-{job_type}-{execution_id}",
        artifact_files=artifact_files,
    )


def run_qualification(args: argparse.Namespace, provenance: dict) -> None:
    if args.mode == "latency":
        result = aggregate_latency_receipts(args.receipt, current_provenance=provenance)
        job_type = "latency-aggregate"
    else:
        result = aggregate_pilot_receipts(args.latency_gate, args.receipt, current_provenance=provenance)
        job_type = "pilot-gate"
    result["execution_provenance"] = provenance
    output = args.output.resolve()
    _write_outputs(output, result, args.mode)
    files = (
        output / "qualification.csv",
        output / "qualification.png",
        output / "qualification.npz",
        output / "report.html",
    )
    wandb = _publish(output=output, result=result, provenance=provenance, job_type=job_type, artifact_files=files)
    result["wandb"] = wandb
    atomic_json(output / "qualification.json", result)
    atomic_json(output / "wandb_receipt.json", wandb)
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")


def run_selection(args: argparse.Namespace, provenance: dict) -> None:
    result = select_survivor(args.latency_gate, args.pilot_gate, args.formal_receipt, current_provenance=provenance)
    result["execution_provenance"] = provenance
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "eligible", *METRICS))
        for arm in ARMS:
            values = result["development_aggregates"].get(arm, {})
            writer.writerow(
                (
                    arm,
                    arm == "M0" or result["eligibility"].get(arm, {}).get("eligible", False),
                    *(values.get(name, "") for name in METRICS),
                )
            )
    figure = output / "selection.png"
    _plot(figure, result)
    np.savez_compressed(
        output / "selection.npz",
        arms=np.asarray(ARMS),
        raw_abs_rel=np.asarray(
            [result["development_aggregates"].get(arm, {}).get("raw_abs_rel", np.nan) for arm in ARMS]
        ),
    )
    report = output / "report.html"
    report.write_text(
        self_contained_html(
            "Phase 2f development selection",
            {
                "final_authorized": result["final_authorized"],
                "survivor": result["survivor"] or "none",
                "eligible_arms": ", ".join(result["eligible_arms"]) or "none",
            },
            images=(("Development raw AbsRel", figure),),
            claim_boundary=result["claim_boundary"],
        ),
        encoding="utf-8",
    )
    files = (output / "selection.csv", figure, output / "selection.npz", report)
    wandb = _publish(output=output, result=result, provenance=provenance, job_type="selection", artifact_files=files)
    result["wandb"] = wandb
    atomic_json(output / "selector.json", result)
    atomic_json(output / "wandb_receipt.json", wandb)
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("latency", "pilot", "selection"), required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--formal-receipt", type=Path, action="append", default=[])
    parser.add_argument("--latency-gate", type=Path)
    parser.add_argument("--pilot-gate", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = load_json(args.provenance)
    if args.mode == "latency":
        if len(args.receipt) != 12 or args.latency_gate or args.formal_receipt:
            raise ValueError("latency adapter requires exactly 12 --receipt inputs")
        run_qualification(args, provenance)
    elif args.mode == "pilot":
        if args.latency_gate is None or not args.receipt:
            raise ValueError("pilot adapter requires --latency-gate and qualified --receipt inputs")
        run_qualification(args, provenance)
    else:
        if args.latency_gate is None or args.pilot_gate is None or len(args.formal_receipt) != 48:
            raise ValueError("selection adapter requires both gates and exactly 48 formal receipts")
        run_selection(args, provenance)
    print(json.dumps({"status": "success", "mode": args.mode, "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
