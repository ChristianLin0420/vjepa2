#!/usr/bin/env python3
"""Aggregate Phase 2f latency replicas or pilot receipts into hard gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from jepa4d.evaluation.phase2f_metrics import (
    atomic_json,
    canonical_sha256,
    file_identity,
    publish_online_wandb,
    require_finite_tree,
    self_contained_html,
)

ARMS = ("M0", "M1", "M2", "M3")
LATENCY_REPLICA_SCHEMA = "jepa4d-phase2f-latency-replica-v1"
LATENCY_GATE_SCHEMA = "jepa4d-phase2f-latency-qualification-v1"
PILOT_GATE_SCHEMA = "jepa4d-phase2f-pilot-qualification-v1"
TRAINING_RECEIPT_SCHEMA = "jepa4d-phase2f-training-run-v1"
GPU_NAME = "NVIDIA A100-SXM4-80GB"
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 260629
PARAMETER_LIMIT = 95_042
PARAMETER_TARGET = 90_722
LATENCY_RATIO_LIMIT = 1.10


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    require_finite_tree(value, str(path))
    return value


def _common_identity(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not receipts:
        raise ValueError("at least one receipt is required")
    keys = (
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "test_receipt_sha256",
        "dependency_graph_sha256",
    )
    common: dict[str, Any] = {}
    for key in keys:
        values = {receipt.get("execution_provenance", {}).get(key) for receipt in receipts}
        if len(values) != 1 or None in values or "" in values:
            raise ValueError(f"qualification receipts disagree on execution_provenance.{key}")
        common[key] = values.pop()
    return common


def _current_provenance(path: Path, parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = _read_json(path)
    provenance = value.get("execution_provenance", value)
    if not isinstance(provenance, dict):
        raise ValueError("current-job provenance must be a JSON object")
    common = _common_identity(parents)
    for key, expected in common.items():
        if provenance.get(key) != expected:
            raise ValueError(f"current-job provenance {key} differs from parent receipts")
    if not isinstance(provenance.get("slurm"), Mapping) or not provenance["slurm"].get("job_id"):
        raise ValueError("current-job provenance lacks its Slurm identity")
    return provenance


def _paired_cluster_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap paired independent allocation clusters and return ratio/CI."""

    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(baseline) != 12:
        raise ValueError("paired cluster bootstrap requires exactly 12 paired allocation means")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all() or np.any(baseline <= 0):
        raise ValueError("latency cluster means must be finite and baseline-positive")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    chunk = 10_000
    for offset in range(0, resamples, chunk):
        size = min(chunk, resamples - offset)
        indices = rng.integers(0, len(baseline), size=(size, len(baseline)))
        values[offset : offset + size] = candidate[indices].mean(axis=1) / baseline[indices].mean(axis=1)
    ratio = float(candidate.mean() / baseline.mean())
    low, high = np.quantile(values, (0.025, 0.975))
    return ratio, float(low), float(high)


def aggregate_latency_receipts(
    paths: Sequence[Path],
    *,
    current_provenance: Mapping[str, Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Validate 12 receipts and apply the preregistered parameter/latency gates."""

    if len(paths) != 12:
        raise ValueError("latency aggregation requires exactly 12 independent replica receipts")
    values = [_read_json(path) for path in paths]
    common = _common_identity(values)
    if any(current_provenance.get(key) != expected for key, expected in common.items()):
        raise ValueError("latency aggregate current-job provenance differs from parent receipts")
    replicas: dict[int, Mapping[str, Any]] = {}
    for path, value in zip(paths, values, strict=True):
        if value.get("schema_version") != LATENCY_REPLICA_SCHEMA or value.get("status") != "success":
            raise ValueError(f"invalid latency replica schema/status: {path}")
        replica = value.get("replica")
        if not isinstance(replica, int) or not 0 <= replica < 12 or replica in replicas:
            raise ValueError(f"invalid or duplicate latency replica index: {replica}")
        if value.get("hardware", {}).get("gpu_name") != GPU_NAME:
            raise ValueError(f"replica {replica} did not run on {GPU_NAME}")
        if value.get("config") != {
            "initialization_seed": 260629,
            "warmups_per_path": 30,
            "blocks": 30,
            "iterations_per_block": 100,
            "batch_size": 1,
        }:
            raise ValueError(f"replica {replica} latency config drifted")
        arms = value.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise ValueError(f"replica {replica} must report exactly M0-M3")
        for arm in ARMS:
            arm_value = arms[arm]
            samples = arm_value.get("complete_head_wall_ms")
            if not isinstance(samples, list) or len(samples) != 30:
                raise ValueError(f"replica {replica} {arm} must contain 30 complete-head blocks")
            if any(not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0 for item in samples):
                raise ValueError(f"replica {replica} {arm} has invalid complete-head samples")
            if not isinstance(arm_value.get("peak_allocation_bytes"), int):
                raise ValueError(f"replica {replica} {arm} lacks peak allocation")
        replicas[replica] = value
    if set(replicas) != set(range(12)):
        raise ValueError("latency replica indices must be exactly 0..11")

    baseline = np.asarray([np.mean(replicas[index]["arms"]["M0"]["complete_head_wall_ms"]) for index in range(12)])
    arms_result: dict[str, Any] = {}
    for arm in ARMS:
        candidate = np.asarray([np.mean(replicas[index]["arms"][arm]["complete_head_wall_ms"]) for index in range(12)])
        ratio, low, high = _paired_cluster_bootstrap(baseline, candidate, resamples=resamples)
        parameter_values = {int(replicas[index]["arms"][arm]["parameter_count"]) for index in range(12)}
        if len(parameter_values) != 1:
            raise ValueError(f"{arm} parameter count differs among replicas")
        parameters = parameter_values.pop()
        parameter_pass = parameters <= PARAMETER_LIMIT
        latency_pass = high <= LATENCY_RATIO_LIMIT
        qualified = parameter_pass and latency_pass
        arms_result[arm] = {
            "parameter_count": parameters,
            "parameter_gate_limit": PARAMETER_LIMIT,
            "parameter_target": PARAMETER_TARGET,
            "parameter_pass": parameter_pass,
            "complete_head_wall_ms_mean": float(candidate.mean()),
            "ratio_to_m0": ratio,
            "ratio_ci95": [low, high],
            "latency_ratio_upper_limit": LATENCY_RATIO_LIMIT,
            "latency_pass": latency_pass,
            "qualified": qualified,
        }
    if not arms_result["M0"]["qualified"]:
        raise RuntimeError("M0 failed its static/latency reference gate; formal DAG must stop")
    result = {
        "schema_version": LATENCY_GATE_SCHEMA,
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "replica_count": 12,
        "bootstrap": {"resamples": resamples, "seed": BOOTSTRAP_SEED, "resampling_unit": "slurm_allocation"},
        "arms": arms_result,
        "qualified_arms": [arm for arm in ARMS if arms_result[arm]["qualified"]],
        "source_receipts": [file_identity(path, schema=LATENCY_REPLICA_SCHEMA) for path in paths],
        "execution_provenance": dict(current_provenance),
    }
    require_finite_tree(result, "latency_gate")
    return result


def _strict_control_pass(camera: Mapping[str, Any]) -> bool:
    required = ("updated", "stale", "wrong", "permuted")
    metrics = camera.get("raw_abs_rel")
    if not isinstance(metrics, Mapping) or any(name not in metrics for name in required):
        return False
    try:
        values = {name: float(metrics[name]) for name in required}
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in values.values())
        and values["updated"] < values["stale"]
        and values["updated"] < values["wrong"]
        and values["updated"] < values["permuted"]
        and camera.get("permutation_bijective") is True
        and float(camera.get("permutation_change_fraction", 0.0)) == 1.0
        and float(camera.get("minimum_output_delta_m", 0.0)) > 1e-6
    )


def aggregate_pilot_receipts(
    latency_gate_path: Path,
    paths: Sequence[Path],
    *,
    current_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply finite/reload/firewall/camera gates to exactly the latency-qualified pilots."""

    latency_gate = _read_json(latency_gate_path)
    if latency_gate.get("schema_version") != LATENCY_GATE_SCHEMA or latency_gate.get("status") != "pass":
        raise ValueError("pilot gate requires a passing latency qualification receipt")
    qualified = set(latency_gate.get("qualified_arms", []))
    if len(paths) != 4:
        raise ValueError("pilot aggregation requires exactly four predeclared pilot receipts")
    values = [_read_json(path) for path in paths]
    common = _common_identity([latency_gate, *values])
    if any(current_provenance.get(key) != expected for key, expected in common.items()):
        raise ValueError("pilot aggregate current-job provenance differs from parent receipts")
    by_arm: dict[str, Mapping[str, Any]] = {}
    for path, value in zip(paths, values, strict=True):
        if value.get("schema_version") != TRAINING_RECEIPT_SCHEMA or value.get("stage") != "pilot":
            raise ValueError(f"invalid pilot receipt: {path}")
        arm = value.get("arm")
        if arm not in ARMS or arm in by_arm:
            raise ValueError(f"unexpected or duplicate pilot arm {arm}")
        if value.get("rotation") != "R0" or value.get("seed") != 0:
            raise ValueError("pilots must use R0/seed 0")
        by_arm[str(arm)] = value
    if set(by_arm) != set(ARMS):
        raise ValueError("pilot receipts must cover all predeclared M0-M3 jobs")

    arms: dict[str, Any] = {}
    for arm in ARMS:
        if arm not in qualified:
            skipped_value = by_arm[arm]
            if skipped_value.get("status") != "skipped_not_qualified" or skipped_value.get("optimizer_steps") != 0:
                raise ValueError(f"latency-disqualified pilot {arm} did not write a zero-step skip receipt")
            arms[arm] = {"qualified": False, "reason": "latency_or_parameter_gate", "checks": {}}
            continue
        qualified_value = by_arm[arm]
        checks = {
            "status_success": qualified_value.get("status") == "success",
            "finite": qualified_value.get("finite") is True,
            "exact_reload": qualified_value.get("exact_reload") is True,
            "zero_forbidden_gradient": float(qualified_value.get("maximum_forbidden_gradient_norm", math.inf)) == 0.0,
            "camera_controls": arm not in {"M2", "M3"}
            or _strict_control_pass(qualified_value.get("camera_controls", {})),
        }
        arms[arm] = {"qualified": all(checks.values()), "checks": checks}
    if not arms["M0"]["qualified"]:
        raise RuntimeError("M0 failed the pilot reference gate; formal DAG must stop")
    result = {
        "schema_version": PILOT_GATE_SCHEMA,
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "arms": arms,
        "formal_allowlist": [arm for arm in ARMS if arms[arm]["qualified"]],
        "latency_gate": file_identity(latency_gate_path, schema=LATENCY_GATE_SCHEMA),
        "source_receipts": [file_identity(path, schema=TRAINING_RECEIPT_SCHEMA) for path in paths],
        "execution_provenance": dict(current_provenance),
    }
    require_finite_tree(result, "pilot_gate")
    return result


def _bar_plot(path: Path, result: Mapping[str, Any], mode: str) -> None:
    width, height = 920, 480
    image = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), f"Phase 2f {mode} qualification", fill="#17202e")
    values = result["arms"]
    for index, arm in enumerate(ARMS):
        top = 80 + index * 88
        item = values[arm]
        if mode == "latency":
            number = float(item["ratio_ci95"][1])
            label = f"upper CI {number:.3f}; params {item['parameter_count']:,}"
            fraction = min(1.0, number / 1.25)
        else:
            fraction = 1.0 if item["qualified"] else 0.12
            label = "qualified" if item["qualified"] else str(item.get("reason", "failed gate"))
        color = "#238636" if item["qualified"] else "#cf4a4a"
        draw.text((30, top), arm, fill="#17202e")
        draw.rectangle((100, top, 100 + int(650 * fraction), top + 28), fill=color)
        draw.text((110, top + 7), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_outputs(output: Path, result: dict[str, Any], mode: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "qualification.json"
    atomic_json(receipt, result)
    with (output / "qualification.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "qualified", "parameter_count", "latency_ratio", "latency_ci_high"))
        for arm in ARMS:
            row = result["arms"][arm]
            writer.writerow(
                (
                    arm,
                    row["qualified"],
                    row.get("parameter_count", ""),
                    row.get("ratio_to_m0", ""),
                    row.get("ratio_ci95", ["", ""])[1],
                )
            )
    figure = output / "qualification.png"
    _bar_plot(figure, result, mode)
    np.savez_compressed(
        output / "qualification.npz",
        qualified=np.asarray([bool(result["arms"][arm]["qualified"]) for arm in ARMS]),
        arms=np.asarray(ARMS),
    )
    summary = {
        "status": result["status"],
        "qualified": ", ".join(arm for arm in ARMS if result["arms"][arm]["qualified"]),
        "receipt_sha256": canonical_sha256(result),
    }
    (output / "report.html").write_text(
        self_contained_html(
            f"Phase 2f {mode} qualification",
            summary,
            images=(("Hard qualification gates", figure),),
            claim_boundary="Development-only qualification; this report contains no DIODE target result.",
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("latency", "pilot"), required=True)
    parser.add_argument("--receipt", action="append", type=Path, default=[])
    parser.add_argument("--latency-gate", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2f aggregate CLIs may run only inside a Slurm job")
    if args.mode == "latency":
        if args.latency_gate is not None:
            raise ValueError("--latency-gate is valid only in pilot mode")
        parents = [_read_json(path) for path in args.receipt]
        current = _current_provenance(args.provenance, parents)
        result = aggregate_latency_receipts(args.receipt, current_provenance=current)
    else:
        if args.latency_gate is None:
            raise ValueError("pilot mode requires --latency-gate")
        parents = [_read_json(args.latency_gate), *[_read_json(path) for path in args.receipt]]
        current = _current_provenance(args.provenance, parents)
        result = aggregate_pilot_receipts(args.latency_gate, args.receipt, current_provenance=current)
    output = args.output.resolve()
    _write_outputs(output, result, args.mode)
    provenance = result["execution_provenance"]
    execution_id = str(provenance["execution_id"])
    slurm_id = str(provenance.get("slurm", {}).get("job_id", "unknown"))
    job_type = "latency-aggregate" if args.mode == "latency" else "pilot-gate"
    wandb_receipt = publish_online_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"phase2f-{execution_id}",
        job_type=job_type,
        run_name=f"{execution_id}-{job_type}-{slurm_id}",
        config={"execution_id": execution_id, "git_commit": provenance["git_commit"], "mode": args.mode},
        summary={
            "status": result["status"],
            "qualified_count": sum(bool(result["arms"][arm]["qualified"]) for arm in ARMS),
        },
        artifact_name=f"phase2f-{job_type}-{execution_id}",
        artifact_files=(
            output / "qualification.csv",
            output / "qualification.png",
            output / "qualification.npz",
            output / "report.html",
        ),
    )
    atomic_json(output / "wandb_receipt.json", wandb_receipt)
    result["wandb"] = wandb_receipt
    atomic_json(output / "qualification.json", result)
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
