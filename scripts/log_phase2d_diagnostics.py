"""Upload durable Phase 2d diagnostic records to an online W&B run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import wandb

from jepa4d.evaluation.fusion_attribution import QUALITATIVE_SCHEMA


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribution_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in record["seeds"]:
        for value in seed["interventions"]:
            intervention = value["intervention"]
            macro = value["macro"]
            rows.append(
                {
                    "seed": seed["seed"],
                    "intervention": intervention["intervention_id"],
                    "final_coefficient": intervention["final_coefficient"],
                    "layer_2": intervention["effective_coefficients"]["2"],
                    "layer_5": intervention["effective_coefficients"]["5"],
                    "layer_8": intervention["effective_coefficients"]["8"],
                    "metric_abs_rel": macro["metric_abs_rel"],
                    "aligned_abs_rel": macro["aligned_abs_rel"],
                    "abs_log_scale_error": macro["metric_abs_log_scale_error"],
                    "calibrated_nll": macro["calibrated_log_depth_nll"],
                    "prediction_delta_relative": macro["prediction_delta_relative"],
                    "total_residual_ratio": macro["residual_total_norm_ratio"],
                }
            )
    return rows


def _attribution_sequence_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in record["seeds"]:
        for value in seed["interventions"]:
            intervention = value["intervention"]
            for sequence in value["per_sequence"]:
                metrics = sequence["metrics"]
                rows.append(
                    {
                        "seed": seed["seed"],
                        "intervention": intervention["intervention_id"],
                        "sequence_id": sequence["sequence_id"],
                        "frames": sequence["frames"],
                        "metric_abs_rel": metrics["metric_abs_rel"],
                        "aligned_abs_rel": metrics["aligned_abs_rel"],
                        "abs_log_scale_error": metrics["metric_abs_log_scale_error"],
                        "calibrated_nll": metrics["calibrated_log_depth_nll"],
                        "prediction_delta_relative": metrics["prediction_delta_relative"],
                        "residual_total_norm_ratio": metrics["residual_total_norm_ratio"],
                    }
                )
    return rows


def _wandb_qualitative_table(path: Path) -> Any:
    with np.load(path.resolve(strict=True), allow_pickle=False) as payload:
        if str(payload["schema_version"]) != QUALITATIVE_SCHEMA:
            raise ValueError("unexpected Phase 2d qualitative schema")
        predictions = np.asarray(payload["prediction_m"], dtype=np.float32)
        targets = np.asarray(payload["target_m"], dtype=np.float32)
        log_variances = np.asarray(payload["log_variance"], dtype=np.float32)
        sigmas = np.asarray(payload["calibrated_log_depth_sigma"], dtype=np.float32)
        sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
        sequence_ids = [str(value) for value in payload["sequence_ids"].tolist()]
        variant_ids = [str(value) for value in payload["variant_ids"].tolist()]
    if not (
        predictions.shape == log_variances.shape == sigmas.shape
        and predictions.ndim == 4
        and targets.shape == predictions.shape[1:]
        and predictions.shape[1] <= 8
    ):
        raise ValueError("malformed or unbounded Phase 2d qualitative bundle")
    table = wandb.Table(
        columns=[
            "variant_id",
            "sample_id",
            "sequence_id",
            "target",
            "prediction",
            "relative_error",
            "calibrated_log_depth_sigma",
        ]
    )
    for variant_index, variant_id in enumerate(variant_ids):
        for sample_index, sample_id in enumerate(sample_ids):
            target = targets[sample_index]
            prediction = predictions[variant_index, sample_index]
            relative_error = np.abs(prediction - target) / np.maximum(target, 1e-4)
            caption = f"{variant_id} · {sequence_ids[sample_index]} · {sample_id}"
            table.add_data(
                variant_id,
                sample_id,
                sequence_ids[sample_index],
                wandb.Image(target, caption=f"target · {sample_id}"),
                wandb.Image(prediction, caption=f"prediction · {caption}"),
                wandb.Image(relative_error, caption=f"relative error · {caption}"),
                wandb.Image(sigmas[variant_index, sample_index], caption=f"calibrated sigma · {caption}"),
            )
    return table


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _attribution_summary(record: dict[str, Any]) -> dict[str, float]:
    """Flatten aggregate control means into stable W&B summary scalars."""
    output: dict[str, float] = {}
    for control in record["aggregate"]["controls"]:
        intervention = str(control["intervention"]["intervention_id"])
        for metric, statistics in control["metrics"].items():
            if isinstance(statistics, dict) and isinstance(statistics.get("mean"), (int, float)):
                output[f"{intervention}/{metric}_mean"] = float(statistics["mean"])
    if not output:
        raise ValueError("attribution aggregate contains no scalar control means")
    return output


def main(
    kind: Annotated[str, typer.Option("--kind")],
    input_directory: Annotated[Path, typer.Option("--input")],
    run_name: Annotated[str, typer.Option("--run-name")],
    project: Annotated[str, typer.Option("--project")] = "jepa4d-worldmodel",
    entity: Annotated[str | None, typer.Option("--entity")] = None,
) -> None:
    if kind not in {"attribution", "calibration"}:
        raise typer.BadParameter("kind must be attribution or calibration")
    input_directory = input_directory.resolve(strict=True)
    sequence_rows: list[dict[str, Any]] = []
    if kind == "attribution":
        record_path = input_directory / "fusion_attribution.json"
        record = json.loads(record_path.read_text())
        rows = _attribution_rows(record)
        sequence_rows = _attribution_sequence_rows(record)
        summary = _attribution_summary(record)
    else:
        record_path = input_directory / "phase2d_calibration_scale_audit.json"
        record = json.loads(record_path.read_text())
        rows = _csv_rows(input_directory / "phase2d_oracle_summary.csv")
        summary = {
            "prediction_sets": len(record["scale_oracle_audits"]),
            "full_scope": float("full_phase2c_test" in record["audit_scopes"]),
            "sequence_count": len(record["calibration_audit"]["sequences"]),
        }
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type=f"phase2d-{kind}",
        mode="online",
        config={
            "kind": kind,
            "source": str(input_directory),
            "schema_version": record["schema_version"],
            "execution_provenance": record.get("execution_provenance"),
        },
        tags=["phase-2d", kind, "diagnostic", "slurm"],
    )
    try:
        if run.offline:
            raise RuntimeError("Phase 2d diagnostics require online W&B")
        table = wandb.Table(columns=list(rows[0]))
        for row in rows:
            table.add_data(*row.values())
        logged: dict[str, Any] = {f"{kind}/results": table}
        report_name = (
            "fusion_attribution_report.html" if kind == "attribution" else "phase2d_calibration_scale_audit.html"
        )
        logged[f"{kind}/report"] = wandb.Html(str(input_directory / report_name), inject=False)
        if kind == "attribution":
            sequence_table = wandb.Table(columns=list(sequence_rows[0]))
            for row in sequence_rows:
                sequence_table.add_data(*row.values())
            logged["attribution/seed_sequence_results"] = sequence_table
            logged["attribution/qualitative_examples"] = _wandb_qualitative_table(
                input_directory / "qualitative_examples.npz"
            )
        run.log(logged)
        for key, value in summary.items():
            if isinstance(value, (int, float, str, bool)):
                run.summary[f"{kind}/{key}"] = value
        artifact = wandb.Artifact(f"{run.id}-phase2d-{kind}", type=f"phase2d-{kind}")
        uploaded_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(input_directory.iterdir())
            if path.is_file() and path.name != "wandb_receipt.json"
        }
        for name in uploaded_files:
            artifact.add_file(str(input_directory / name))
        uploaded = run.log_artifact(artifact).wait(timeout=900)
        run.summary["result"] = "success"
        receipt = {
            "schema_version": "jepa4d-phase2d-wandb-receipt-v1",
            "status": "uploaded",
            "mode": "online",
            "kind": kind,
            "run_id": str(run.id),
            "run_url": str(run.url),
            "run_path": str(run.path),
            "artifact_id": str(uploaded.id),
            "artifact_name": str(uploaded.name),
            "artifact_qualified_name": str(uploaded.qualified_name),
            "artifact_version": str(uploaded.version),
            "artifact_digest": str(uploaded.digest),
            "source_record": record_path.name,
            "source_record_sha256": _sha256(record_path),
            "uploaded_files": uploaded_files,
        }
        (input_directory / "wandb_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    finally:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
