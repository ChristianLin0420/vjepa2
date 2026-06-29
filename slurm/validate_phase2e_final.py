"""Fail-closed postflight for the formal Phase-2e held-out evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

EVALUATION_SCHEMA = "jepa4d-phase2e-final-evaluation-v1"
MANIFEST_SCHEMA = "jepa4d-phase2e-final-artifact-manifest-v1"
WANDB_SCHEMA = "jepa4d-phase2e-final-wandb-receipt-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _nonempty(value: Any, label: str) -> str:
    rendered = str(value or "").strip()
    if not rendered or rendered == "None":
        raise RuntimeError(f"formal Phase-2e result lacks {label}")
    return rendered


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def validate_phase2e_final(output_path: Path) -> dict[str, Any]:
    root = output_path.resolve(strict=True)
    if (root / "run_failure.json").exists():
        raise RuntimeError("formal Phase-2e output contains a run_failure.json")
    paths = {
        "evaluation": root / "phase2e_final_evaluation.json",
        "predictions": root / "phase2e_final_predictions.npz",
        "per_sample": root / "phase2e_final_per_sample.csv",
        "report": root / "phase2e_final_report.html",
        "manifest": root / "artifact_manifest.json",
        "wandb": root / "wandb_receipt.json",
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"formal Phase-2e {label} artifact is absent or empty: {path}")

    evaluation = _json(paths["evaluation"])
    if evaluation.get("schema_version") != EVALUATION_SCHEMA or evaluation.get("status") != "success":
        raise RuntimeError("formal Phase-2e evaluation does not report success")
    counts = evaluation.get("counts")
    expected_counts = {
        "test_samples": 128,
        "formal_checkpoints": 24,
        "per_seed_rows": 42,
        "per_sample_rows": 5_376,
        "failures": 0,
    }
    if counts != expected_counts:
        raise RuntimeError(f"formal Phase-2e result counts differ from the frozen protocol: {counts}")
    if len(evaluation.get("aggregates", [])) != 14 or len(evaluation.get("per_seed", [])) != 42:
        raise RuntimeError("formal Phase-2e aggregate/seed coverage is incomplete")
    if len(evaluation.get("per_sample", [])) != 5_376:
        raise RuntimeError("formal Phase-2e per-sample coverage is incomplete")
    provenance = evaluation.get("provenance")
    if not isinstance(provenance, dict) or len(str(provenance.get("git_commit", ""))) != 40:
        raise RuntimeError("formal Phase-2e result lacks an execution commit")
    dependency_graph = provenance.get("slurm_dependency_graph")
    if (
        provenance.get("git_status") != ""
        or not isinstance(dependency_graph, dict)
        or len(str(dependency_graph.get("sha256", ""))) != 64
        or dependency_graph.get("graph", {}).get("schema_version") != "jepa4d-phase2e-dependency-graph-v1"
    ):
        raise RuntimeError("formal Phase-2e provenance lacks a clean tree or dependency graph")
    test_receipt = provenance.get("test_receipt")
    if not isinstance(test_receipt, dict) or len(str(test_receipt.get("sha256", ""))) != 64:
        raise RuntimeError("formal Phase-2e provenance lacks the passing test receipt")
    runtime = provenance.get("runtime")
    slurm = provenance.get("slurm")
    if (
        not isinstance(runtime, dict)
        or not runtime.get("gpu")
        or not runtime.get("cuda_build")
        or float(runtime.get("seconds", 0)) <= 0
        or not isinstance(slurm, dict)
        or not slurm.get("SLURM_JOB_ID")
        or not slurm.get("SLURM_JOB_PARTITION")
    ):
        raise RuntimeError("formal Phase-2e runtime/Slurm provenance is incomplete")
    identities = {(row["variant"], int(row["seed"]), row["intrinsics_control"]) for row in evaluation["per_seed"]}
    if len(identities) != 42:
        raise RuntimeError("formal Phase-2e seed/control identities are duplicated")
    if {str(row["sensor_id"]) for row in evaluation["per_sample"]} != {"kv2"}:
        raise RuntimeError("formal Phase-2e result includes a non-kv2 held-out sample")
    gate = evaluation.get("gate")
    if not isinstance(gate, dict) or gate.get("population_significance_claimed") is not False:
        raise RuntimeError("formal Phase-2e gate is missing its interpretation boundary")
    conditions = gate.get("conditions")
    if not isinstance(conditions, dict) or set(conditions.values()) - {True, False}:
        raise RuntimeError("formal Phase-2e operational gate conditions are incomplete")
    if gate.get("passed") is not all(conditions.values()):
        raise RuntimeError("formal Phase-2e gate decision is inconsistent with its conditions")
    if not _all_finite(evaluation):
        raise RuntimeError("formal Phase-2e evaluation contains a non-finite scalar")

    report_text = paths["report"].read_text()
    if re.search(r"<script[^>]+src\s*=", report_text, flags=re.IGNORECASE):
        raise RuntimeError("formal Phase-2e HTML report has an external script dependency")
    for marker in ("Operational gate", "Same-checkpoint camera controls", "Predicted vs true global log scale"):
        if marker not in report_text:
            raise RuntimeError(f"formal Phase-2e visual report lacks required panel: {marker}")

    with np.load(paths["predictions"], allow_pickle=False) as predictions:
        if predictions["prediction_m"].shape != (42, 128, 24, 24):
            raise RuntimeError("formal Phase-2e prediction tensor coverage is incorrect")
        if predictions["log_variance"].shape != (42, 128, 24, 24):
            raise RuntimeError("formal Phase-2e uncertainty tensor coverage is incorrect")
        if predictions["target_m"].shape != (128, 24, 24):
            raise RuntimeError("formal Phase-2e held-out target tensor coverage is incorrect")
        for name in ("prediction_m", "log_variance", "target_m"):
            if not np.isfinite(predictions[name]).all():
                raise RuntimeError(f"formal Phase-2e prediction bundle has non-finite {name}")
    with paths["per_sample"].open(newline="") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != 5_376:
            raise RuntimeError("formal Phase-2e CSV row count is incomplete")

    manifest = _json(paths["manifest"])
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("formal Phase-2e artifact manifest schema is unexpected")
    expected_roles = {
        "canonical_evaluation": paths["evaluation"],
        "full_predictions": paths["predictions"],
        "per_sample_metrics": paths["per_sample"],
        "visual_report": paths["report"],
    }
    files = manifest.get("files")
    if not isinstance(files, list) or {row.get("role") for row in files} != set(expected_roles):
        raise RuntimeError("formal Phase-2e artifact manifest roles are incomplete")
    for row in files:
        role = str(row["role"])
        path = (root / str(row["path"])).resolve(strict=True)
        if (
            path != expected_roles[role].resolve()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != _sha256(path)
        ):
            raise RuntimeError(f"formal Phase-2e artifact identity changed for {role}")

    wandb = _json(paths["wandb"])
    if (
        wandb.get("schema_version") != WANDB_SCHEMA
        or wandb.get("status") != "uploaded"
        or wandb.get("mode") != "online"
    ):
        raise RuntimeError("formal Phase-2e W&B upload receipt does not pass")
    for key in (
        "run_id",
        "run_url",
        "run_path",
        "artifact_id",
        "artifact_name",
        "artifact_qualified_name",
        "artifact_version",
        "artifact_digest",
    ):
        _nonempty(wandb.get(key), f"W&B {key}")
    if (
        wandb.get("artifact_manifest_sha256") != _sha256(paths["manifest"])
        or wandb.get("evaluation_sha256") != _sha256(paths["evaluation"])
        or wandb.get("report_sha256") != _sha256(paths["report"])
    ):
        raise RuntimeError("formal Phase-2e W&B receipt is not bound to the local result")
    return {
        "schema_version": "jepa4d-phase2e-final-postflight-v1",
        "status": "pass",
        "gate_passed": gate["passed"],
        "evaluation_sha256": _sha256(paths["evaluation"]),
        "report_sha256": _sha256(paths["report"]),
        "wandb_run_id": wandb["run_id"],
        "wandb_artifact_id": wandb["artifact_id"],
        "wandb_artifact_digest": wandb["artifact_digest"],
    }


def main(output: Annotated[Path, typer.Option("--output")]) -> None:
    typer.echo(json.dumps(validate_phase2e_final(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
