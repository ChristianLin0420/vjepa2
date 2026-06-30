#!/usr/bin/env python3
"""Select at most one Phase 2f survivor from complete development evidence."""

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
    file_identity,
    publish_online_wandb,
    require_finite_tree,
    self_contained_html,
)

ARMS = ("M0", "M1", "M2", "M3")
CANDIDATES = ("M1", "M2", "M3")
ROTATIONS = ("R0", "R1", "R2", "R3")
SEEDS = (0, 1, 2)
TRAINING_RECEIPT_SCHEMA = "jepa4d-phase2f-training-run-v1"
LATENCY_GATE_SCHEMA = "jepa4d-phase2f-latency-qualification-v1"
PILOT_GATE_SCHEMA = "jepa4d-phase2f-pilot-qualification-v1"
SELECTOR_SCHEMA = "jepa4d-phase2f-development-selector-v1"
METRICS = ("raw_abs_rel", "absolute_log_scale_error", "aligned_abs_rel", "nll", "ause")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    require_finite_tree(value, str(path))
    return value


def _identity(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    provenance = receipt.get("execution_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("receipt lacks embedded execution_provenance")
    keys = (
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "test_receipt_sha256",
        "dependency_graph_sha256",
    )
    values = tuple(provenance.get(key) for key in keys)
    if any(value is None or value == "" for value in values):
        raise ValueError("receipt has incomplete execution provenance")
    return values


def _load_current_provenance(path: Path, parent_identity: tuple[Any, ...]) -> dict[str, Any]:
    value = _load(path)
    provenance = value.get("execution_provenance", value)
    if not isinstance(provenance, dict):
        raise ValueError("selector current-job provenance must be an object")
    probe = {"execution_provenance": provenance}
    if _identity(probe) != parent_identity:
        raise ValueError("selector current-job provenance differs from parent execution identity")
    if not isinstance(provenance.get("slurm"), Mapping) or not provenance["slurm"].get("job_id"):
        raise ValueError("selector current-job provenance lacks its Slurm identity")
    return provenance


def _camera_pass(receipts: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, float]]:
    controls = ("updated", "stale", "wrong", "permuted")
    values: dict[str, list[float]] = {name: [] for name in controls}
    for receipt in receipts:
        camera = receipt.get("camera_controls")
        if not isinstance(camera, Mapping):
            return False, {}
        metrics = camera.get("raw_abs_rel")
        if not isinstance(metrics, Mapping):
            return False, {}
        if camera.get("permutation_bijective") is not True:
            return False, {}
        if float(camera.get("permutation_change_fraction", 0.0)) != 1.0:
            return False, {}
        if float(camera.get("minimum_output_delta_m", 0.0)) <= 1e-6:
            return False, {}
        for name in controls:
            number = float(metrics.get(name, math.nan))
            if not math.isfinite(number):
                return False, {}
            values[name].append(number)
    aggregate = {name: float(np.mean(numbers)) for name, numbers in values.items()}
    passed = all(aggregate["updated"] < aggregate[name] for name in ("stale", "wrong", "permuted"))
    return passed, aggregate


def select_survivor(
    latency_gate_path: Path,
    pilot_gate_path: Path,
    formal_paths: Sequence[Path],
    *,
    current_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the fixed 48-job matrix and apply the frozen deterministic rule."""

    latency = _load(latency_gate_path)
    pilot = _load(pilot_gate_path)
    if latency.get("schema_version") != LATENCY_GATE_SCHEMA or latency.get("status") != "pass":
        raise ValueError("selector requires a passing latency gate")
    if pilot.get("schema_version") != PILOT_GATE_SCHEMA or pilot.get("status") != "pass":
        raise ValueError("selector requires a passing pilot gate")
    allowlist = set(pilot.get("formal_allowlist", []))
    if "M0" not in allowlist or not allowlist <= set(ARMS):
        raise ValueError("formal allowlist is invalid")
    if len(formal_paths) != 48:
        raise ValueError("selector requires all 4 arms x 4 rotations x 3 seeds receipts")
    formal = [_load(path) for path in formal_paths]
    identities = {_identity(value) for value in [latency, pilot, *formal]}
    if len(identities) != 1:
        raise ValueError("selector inputs do not share one execution identity")
    if _identity({"execution_provenance": current_provenance}) != next(iter(identities)):
        raise ValueError("selector current-job provenance differs from parent receipts")

    runs: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    source_by_key: dict[tuple[str, str, int], Path] = {}
    for path, value in zip(formal_paths, formal, strict=True):
        if value.get("schema_version") != TRAINING_RECEIPT_SCHEMA or value.get("stage") != "formal":
            raise ValueError(f"invalid formal receipt: {path}")
        arm, rotation, seed = value.get("arm"), value.get("rotation"), value.get("seed")
        if arm not in ARMS or rotation not in ROTATIONS or seed not in SEEDS:
            raise ValueError(f"invalid formal arm/rotation/seed: {(arm, rotation, seed)}")
        key = (str(arm), str(rotation), int(seed))
        if key in runs:
            raise ValueError(f"duplicate formal receipt: {key}")
        expected_status = "success" if arm in allowlist else "skipped_not_qualified"
        if value.get("status") != expected_status:
            raise ValueError(f"{key} status {value.get('status')} != {expected_status}")
        if expected_status == "success":
            run_checks = (
                value.get("finite") is True,
                value.get("exact_reload") is True,
                float(value.get("maximum_forbidden_gradient_norm", math.inf)) == 0.0,
                value.get("wandb", {}).get("mode") == "online",
                value.get("wandb", {}).get("status") == "success",
            )
            if not all(run_checks):
                raise ValueError(f"formal receipt failed completeness checks: {key}")
            metrics = value.get("metrics", {}).get("development_test", {}).get("group_macro")
            if not isinstance(metrics, Mapping) or any(name not in metrics for name in METRICS):
                raise ValueError(f"formal receipt lacks development metrics: {key}")
            checkpoint = value.get("checkpoint")
            normalization = value.get("feature_normalization")
            if not isinstance(checkpoint, Mapping) or not isinstance(normalization, Mapping):
                raise ValueError(f"formal receipt lacks checkpoint/normalization identity: {key}")
            for item in (checkpoint, normalization):
                if not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                    raise ValueError(f"formal artifact identity is incomplete: {key}")
        else:
            if value.get("optimizer_steps") != 0:
                raise ValueError(f"disqualified formal job performed optimization: {key}")
        runs[key] = value
        source_by_key[key] = path
    expected = {(arm, rotation, seed) for arm in ARMS for rotation in ROTATIONS for seed in SEEDS}
    if set(runs) != expected:
        raise ValueError("formal receipt matrix is incomplete")

    aggregates: dict[str, dict[str, float]] = {}
    for arm in allowlist:
        arm_runs = [runs[(arm, rotation, seed)] for rotation in ROTATIONS for seed in SEEDS]
        aggregates[arm] = {
            name: float(
                np.mean([float(value["metrics"]["development_test"]["group_macro"][name]) for value in arm_runs])
            )
            for name in METRICS
        }
    reference = aggregates["M0"]
    eligibility: dict[str, Any] = {}
    for arm in CANDIDATES:
        if arm not in allowlist:
            eligibility[arm] = {"eligible": False, "reason": "not_pilot_qualified", "checks": {}}
            continue
        values = aggregates[arm]
        arm_runs = [runs[(arm, rotation, seed)] for rotation in ROTATIONS for seed in SEEDS]
        camera_pass, camera_values = _camera_pass(arm_runs) if arm in {"M2", "M3"} else (True, {})
        latency_arm = latency.get("arms", {}).get(arm, {})
        eligibility_checks = {
            "raw_abs_rel_lower": values["raw_abs_rel"] < reference["raw_abs_rel"],
            "scale_error_lower": values["absolute_log_scale_error"] < reference["absolute_log_scale_error"],
            "aligned_abs_rel_noninferior": values["aligned_abs_rel"] <= 1.02 * reference["aligned_abs_rel"],
            "nll_lower": values["nll"] < reference["nll"],
            "ause_no_worse": values["ause"] <= reference["ause"],
            "latency_frozen": latency_arm.get("qualified") is True
            and float(latency_arm.get("ratio_ci95", [math.inf, math.inf])[1]) <= 1.10,
            "parameters_frozen": int(latency_arm.get("parameter_count", 10**18)) <= 95_042,
            "camera_controls": camera_pass,
        }
        eligibility[arm] = {
            "eligible": all(eligibility_checks.values()),
            "checks": eligibility_checks,
            "development_metrics": values,
            "camera_controls": camera_values,
        }
    eligible = [arm for arm in CANDIDATES if eligibility[arm]["eligible"]]

    if eligible:
        lowest_raw = min(aggregates[arm]["raw_abs_rel"] for arm in eligible)
        tied = [arm for arm in eligible if abs(aggregates[arm]["raw_abs_rel"] - lowest_raw) <= 1e-12]
        tied.sort(
            key=lambda arm: (
                aggregates[arm]["absolute_log_scale_error"],
                aggregates[arm]["aligned_abs_rel"],
                CANDIDATES.index(arm),
            )
        )
        remaining = [arm for arm in eligible if arm not in tied]
        remaining.sort(key=lambda arm: (aggregates[arm]["raw_abs_rel"], CANDIDATES.index(arm)))
        eligible = [*tied, *remaining]
    survivor = eligible[0] if eligible else None
    checkpoints: dict[str, list[dict[str, Any]]] = {}
    if survivor is not None:
        for arm in ("M0", survivor):
            checkpoints[arm] = []
            for rotation in ROTATIONS:
                for seed in SEEDS:
                    run = runs[(arm, rotation, seed)]
                    checkpoints[arm].append(
                        {
                            "rotation": rotation,
                            "seed": seed,
                            "checkpoint": run["checkpoint"],
                            "feature_normalization": run["feature_normalization"],
                            "validation_variance_calibration": run["validation_variance_calibration"],
                            "formal_receipt": file_identity(
                                source_by_key[(arm, rotation, seed)], schema=TRAINING_RECEIPT_SCHEMA
                            ),
                        }
                    )
    result = {
        "schema_version": SELECTOR_SCHEMA,
        "status": "success",
        "created_utc": datetime.now(UTC).isoformat(),
        "formal_matrix": {"arms": list(ARMS), "rotations": list(ROTATIONS), "seeds": list(SEEDS)},
        "development_aggregates": aggregates,
        "eligibility": eligibility,
        "eligible_arms": eligible,
        "survivor": survivor,
        "final_authorized": survivor is not None,
        "checkpoint_set": checkpoints,
        "latency_gate": file_identity(latency_gate_path, schema=LATENCY_GATE_SCHEMA),
        "pilot_gate": file_identity(pilot_gate_path, schema=PILOT_GATE_SCHEMA),
        "source_receipts": [file_identity(path, schema=TRAINING_RECEIPT_SCHEMA) for path in formal_paths],
        "claim_boundary": "Development selection only; no DIODE archive or target was accessed.",
        "execution_provenance": dict(current_provenance),
    }
    require_finite_tree(result, "selector")
    return result


def _plot(path: Path, result: Mapping[str, Any]) -> None:
    image = Image.new("RGB", (1040, 520), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), "Phase 2f development survivor selection", fill="#17202e")
    aggregates = result["development_aggregates"]
    reference = float(aggregates["M0"]["raw_abs_rel"])
    maximum = max(float(value["raw_abs_rel"]) for value in aggregates.values())
    for index, arm in enumerate(ARMS):
        top = 90 + 90 * index
        if arm not in aggregates:
            draw.text((30, top), f"{arm}: not qualified", fill="#6b7280")
            continue
        value = float(aggregates[arm]["raw_abs_rel"])
        width = int(720 * value / max(maximum, 1e-12))
        eligible = arm == "M0" or result["eligibility"].get(arm, {}).get("eligible", False)
        color = "#238636" if eligible else "#cf4a4a"
        draw.text((30, top), arm, fill="#17202e")
        draw.rectangle((100, top, 100 + width, top + 30), fill=color)
        draw.text((110, top + 8), f"raw AbsRel {value:.5f} ({value / reference:.3f}x M0)", fill="white")
    image.save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-gate", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--formal-receipt", type=Path, action="append", default=[])
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2f selector may run only inside a Slurm job")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = _load(args.latency_gate)
    current = _load_current_provenance(args.provenance, _identity(parent))
    result = select_survivor(
        args.latency_gate,
        args.pilot_gate,
        args.formal_receipt,
        current_provenance=current,
    )
    receipt = atomic_json(output / "selector.json", result)
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
    summary = {
        "final_authorized": result["final_authorized"],
        "survivor": result["survivor"] or "none",
        "eligible_arms": ", ".join(result["eligible_arms"]) or "none",
    }
    report = output / "report.html"
    report.write_text(
        self_contained_html(
            "Phase 2f development selection",
            summary,
            images=(("Development raw AbsRel", figure),),
            claim_boundary=result["claim_boundary"],
        ),
        encoding="utf-8",
    )
    provenance = result["execution_provenance"]
    execution_id = str(provenance["execution_id"])
    slurm_id = str(provenance.get("slurm", {}).get("job_id", "unknown"))
    wandb_receipt = publish_online_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"phase2f-{execution_id}",
        job_type="selection",
        run_name=f"{execution_id}-selection-{slurm_id}",
        config={"execution_id": execution_id, "git_commit": provenance["git_commit"]},
        summary=summary,
        artifact_name=f"phase2f-selection-{execution_id}",
        artifact_files=(output / "selection.csv", figure, output / "selection.npz", report),
    )
    atomic_json(output / "wandb_receipt.json", wandb_receipt)
    result["wandb"] = wandb_receipt
    atomic_json(receipt, result)
    (output / "SUCCESS").write_text("success\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
