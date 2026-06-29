"""Strict Phase-2d latency and cross-diagnostic postflight validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from scripts.aggregate_phase2d_diagnostics import _camera_provenance, _execution_provenance
from scripts.aggregate_phase2d_latency import (
    EXPECTED_BLOCKS,
    EXPECTED_ITERATIONS,
    EXPECTED_REPLICATES,
    EXPECTED_WARMUPS,
    validate_latency_inputs,
)

WANDB_SCHEMA = "jepa4d-phase2d-wandb-receipt-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required Phase-2d artifact is absent or empty: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _check_recorded_identity(path: Path, identity: Any, label: str) -> None:
    actual = _file_identity(path)
    if not isinstance(identity, dict):
        raise RuntimeError(f"{label} identity is missing")
    if Path(str(identity.get("path", ""))).resolve() != path.resolve():
        raise RuntimeError(f"{label} path differs from its receipt")
    if int(identity.get("bytes", actual["bytes"])) != actual["bytes"] or identity.get("sha256") != actual["sha256"]:
        raise RuntimeError(f"{label} byte identity differs from its receipt")


def _validate_wandb(directory: Path, required_names: set[str], *, kind: str | None = None) -> dict[str, Any]:
    receipt_path = directory / "wandb_receipt.json"
    receipt = _json(receipt_path)
    if (
        receipt.get("schema_version") != WANDB_SCHEMA
        or receipt.get("status") != "uploaded"
        or receipt.get("mode") != "online"
    ):
        raise RuntimeError(f"Phase-2d W&B receipt is not an online upload: {receipt_path}")
    if kind is not None and receipt.get("kind") != kind:
        raise RuntimeError(f"Phase-2d W&B receipt kind differs from {kind}: {receipt_path}")
    backend = (
        "run_id",
        "run_url",
        "run_path",
        "artifact_id",
        "artifact_name",
        "artifact_qualified_name",
        "artifact_version",
        "artifact_digest",
    )
    if any(not str(receipt.get(key, "")).strip() or receipt.get(key) == "None" for key in backend):
        raise RuntimeError(f"Phase-2d W&B receipt lacks backend-confirmed identity: {receipt_path}")
    uploaded = receipt.get("uploaded_files")
    if not isinstance(uploaded, dict) or not required_names.issubset(uploaded):
        raise RuntimeError(f"Phase-2d W&B receipt lacks uploaded-file hashes: {receipt_path}")
    for name, identity in uploaded.items():
        path = directory / str(name)
        if not path.is_file() or not isinstance(identity, dict):
            raise RuntimeError(f"W&B-uploaded Phase-2d file is absent: {path}")
        if int(identity.get("bytes", -1)) != path.stat().st_size or identity.get("sha256") != _sha256(path):
            raise RuntimeError(f"W&B-uploaded Phase-2d file changed after upload: {path}")
    return receipt


def _require_finite(value: Any, label: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value in Phase-2d output at {label}")
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite(nested, f"{label}[{index}]")


def _self_contained(path: Path) -> None:
    text = path.read_text()
    if re.search(r"<script[^>]+src\s*=", text, flags=re.IGNORECASE):
        raise RuntimeError(f"Phase-2d HTML contains an external script dependency: {path}")


def validate_attribution(directory: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    root = directory.resolve(strict=True)
    result_path = root / "fusion_attribution.json"
    report_path = root / "fusion_attribution_report.html"
    prediction_path = root / "full_predictions.npz"
    qualitative_path = root / "qualitative_examples.npz"
    source_path = root / "source_identity.json"
    local_receipt_path = root / "receipt.json"
    result = _json(result_path)
    local_receipt = _json(local_receipt_path)
    source = _json(source_path)
    if (
        result.get("schema_version") != "jepa4d-phase2d-same-checkpoint-fusion-attribution-v1"
        or result.get("status") != "complete"
        or local_receipt.get("schema_version") != "jepa4d-phase2d-output-receipt-v1"
        or local_receipt.get("status") != "pass"
    ):
        raise RuntimeError("Phase-2d attribution result/receipt does not pass")
    if len(result.get("seeds", [])) != 3 or any(len(seed.get("interventions", [])) != 15 for seed in result["seeds"]):
        raise RuntimeError("Phase-2d attribution does not contain 3 seeds x 15 interventions")
    if int(result.get("test_samples", {}).get("count", -1)) != 128:
        raise RuntimeError("Phase-2d attribution is not the full 128-frame consumed test scope")
    if result.get("source_identity") != source:
        raise RuntimeError("Phase-2d attribution result differs from source_identity.json")
    if result.get("execution_provenance") != local_receipt.get("execution_provenance"):
        raise RuntimeError("Phase-2d attribution execution provenance differs from its local receipt")
    _check_recorded_identity(result_path, local_receipt.get("fusion_attribution_json"), "attribution JSON")
    _check_recorded_identity(report_path, local_receipt.get("fusion_attribution_html"), "attribution HTML")
    _check_recorded_identity(prediction_path, local_receipt.get("full_predictions"), "attribution predictions")
    _check_recorded_identity(
        qualitative_path, local_receipt.get("qualitative_examples"), "attribution qualitative examples"
    )
    _check_recorded_identity(source_path, local_receipt.get("source_identity"), "attribution source identity")
    if (
        local_receipt["full_predictions"].get("frames") != 128
        or local_receipt["full_predictions"].get("audit_scope") != "full_phase2c_test"
    ):
        raise RuntimeError("Phase-2d prediction handoff scope is incomplete")
    qualitative_handoff = result.get("qualitative_handoff")
    qualitative_receipt = local_receipt.get("qualitative_examples")
    if (
        not isinstance(qualitative_handoff, dict)
        or not isinstance(qualitative_receipt, dict)
        or qualitative_handoff.get("schema_version") != "jepa4d-phase2d-qualitative-v1"
        or qualitative_receipt.get("schema_version") != "jepa4d-phase2d-qualitative-v1"
        or qualitative_handoff.get("sha256") != _sha256(qualitative_path)
        or qualitative_receipt.get("sha256") != _sha256(qualitative_path)
    ):
        raise RuntimeError("Phase-2d qualitative handoff identity is incomplete")
    with np.load(qualitative_path, allow_pickle=False) as qualitative:
        schema = str(np.asarray(qualitative["schema_version"]).item())
        predictions = np.asarray(qualitative["prediction_m"])
        targets = np.asarray(qualitative["target_m"])
        log_variance = np.asarray(qualitative["log_variance"])
        sigma = np.asarray(qualitative["calibrated_log_depth_sigma"])
        sample_ids = [str(value) for value in np.asarray(qualitative["sample_ids"]).tolist()]
        variant_ids = [str(value) for value in np.asarray(qualitative["variant_ids"]).tolist()]
        seeds = np.asarray(qualitative["seeds"])
    expected_variants = {f"seed{seed}:{name}" for seed in range(3) for name in ("original", "zero", "fixed_average")}
    if (
        schema != "jepa4d-phase2d-qualitative-v1"
        or predictions.ndim != 4
        or predictions.shape != log_variance.shape
        or predictions.shape != sigma.shape
        or targets.shape != predictions.shape[1:]
        or not 1 <= predictions.shape[1] <= 8
        or predictions.shape[0] != 9
        or len(sample_ids) != predictions.shape[1]
        or len(set(sample_ids)) != len(sample_ids)
        or set(variant_ids) != expected_variants
        or tuple(seeds.shape) != (9,)
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(log_variance).all()
        or not np.isfinite(sigma).all()
        or not (predictions > 0).all()
        or not (sigma > 0).all()
    ):
        raise RuntimeError("Phase-2d qualitative bundle schema, scope, or tensors are invalid")
    if (
        qualitative_handoff.get("sample_count") != predictions.shape[1]
        or qualitative_handoff.get("variant_count") != predictions.shape[0]
        or qualitative_handoff.get("sample_ids") != sample_ids
        or qualitative_handoff.get("variant_ids") != variant_ids
        or qualitative_receipt.get("samples") != predictions.shape[1]
        or qualitative_receipt.get("variants") != predictions.shape[0]
    ):
        raise RuntimeError("Phase-2d qualitative metadata differs from its NPZ")
    _validate_wandb(
        root,
        {
            result_path.name,
            report_path.name,
            prediction_path.name,
            qualitative_path.name,
            source_path.name,
            local_receipt_path.name,
        },
        kind="attribution",
    )
    _self_contained(report_path)
    _require_finite(result)
    return (
        source,
        _sha256(prediction_path),
        {
            "result": _file_identity(result_path),
            "local_receipt": _file_identity(local_receipt_path),
            "qualitative_examples": _file_identity(qualitative_path),
            "wandb_receipt": _file_identity(root / "wandb_receipt.json"),
        },
    )


def validate_calibration(
    directory: Path, *, expected_source: dict[str, Any], prediction_sha256: str
) -> dict[str, Any]:
    root = directory.resolve(strict=True)
    result_path = root / "phase2d_calibration_scale_audit.json"
    report_path = root / "phase2d_calibration_scale_audit.html"
    oracle_csv = root / "phase2d_oracle_summary.csv"
    camera_csv = root / "phase2d_calibration_table.csv"
    result = _json(result_path)
    if (
        result.get("schema_version") != "jepa4d-phase2d-calibration-scale-audit-v1"
        or result.get("diagnostic_only") is not True
    ):
        raise RuntimeError("Phase-2d calibration result is not a diagnostic-only completed schema")
    if result.get("audit_scopes") != ["full_phase2c_test"] or len(result.get("scale_oracle_audits", [])) != 9:
        raise RuntimeError("Phase-2d calibration does not cover all 9 full-test prediction sets")
    prediction_sources = result.get("prediction_sources")
    if (
        not isinstance(prediction_sources, list)
        or len(prediction_sources) != 1
        or prediction_sources[0].get("sha256") != prediction_sha256
    ):
        raise RuntimeError("Phase-2d calibration is not bound to the attribution prediction handoff")
    if result.get("manifest_sha256") != expected_source.get("dataset_manifest", {}).get("sha256"):
        raise RuntimeError("Phase-2d calibration and attribution use different dataset manifests")
    _validate_wandb(root, {result_path.name, report_path.name, oracle_csv.name, camera_csv.name}, kind="calibration")
    _self_contained(report_path)
    _require_finite(result)
    return {"result": _file_identity(result_path), "wandb_receipt": _file_identity(root / "wandb_receipt.json")}


def validate_latency(latency_root: Path, latency_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    records, inputs, source = validate_latency_inputs(latency_root)
    root = latency_output.resolve(strict=True)
    result_path = root / "latency_aggregate.json"
    report_path = root / "latency_aggregate_report.html"
    result = _json(result_path)
    protocol = result.get("protocol")
    if (
        result.get("schema_version") != "jepa4d-phase2d-latency-aggregate-v1"
        or result.get("status") != "complete"
        or not isinstance(protocol, dict)
        or protocol.get("replicate_ids") != list(EXPECTED_REPLICATES)
        or protocol.get("warmups_per_path") != EXPECTED_WARMUPS
        or protocol.get("blocks") != EXPECTED_BLOCKS
        or protocol.get("iterations_per_block") != EXPECTED_ITERATIONS
    ):
        raise RuntimeError("Phase-2d latency aggregate differs from the frozen protocol")
    if (
        result.get("source_identity") != source
        or result.get("inputs") != inputs
        or int(result.get("replicate_count", -1)) != 12
        or int(result.get("e2e_block_count", -1)) != 12 * 30 * 4
        or int(result.get("head_only_block_count", -1)) != 12 * 30 * 5
        or len(result.get("rows", [])) != 12 * 270
    ):
        raise RuntimeError("Phase-2d latency aggregate is not a complete reconstruction of its 12 inputs")
    systems = result.get("systems")
    expected_job_ids = [str(record["slurm"]["job_id"]) for record in records]
    if (
        not isinstance(systems, dict)
        or systems.get("slurm_job_ids") != expected_job_ids
        or systems.get("test_receipt") != inputs[0]["test_receipt"]
    ):
        raise RuntimeError("Phase-2d latency aggregate systems provenance is incomplete")
    latency_wandb = _validate_wandb(root, {result_path.name, report_path.name})
    if (
        latency_wandb.get("test_receipt_sha256") != systems["test_receipt"]["sha256"]
        or latency_wandb.get("slurm_job_ids") != systems["slurm_job_ids"]
    ):
        raise RuntimeError("Phase-2d latency aggregate W&B receipt lacks systems provenance")
    _self_contained(report_path)
    _require_finite(result)
    return source, {
        "result": _file_identity(result_path),
        "wandb_receipt": _file_identity(root / "wandb_receipt.json"),
    }


def validate_diagnostics(
    attribution: Path,
    calibration: Path,
    latency_root: Path,
    latency_output: Path,
    aggregate_output: Path,
    repo_root: Path,
    test_receipt: Path,
    dependency_graph: Path,
) -> dict[str, Any]:
    source, prediction_sha, attribution_identity = validate_attribution(attribution)
    calibration_identity = validate_calibration(calibration, expected_source=source, prediction_sha256=prediction_sha)
    latency_source, latency_identity = validate_latency(latency_root, latency_output)
    if latency_source != source:
        raise RuntimeError("Phase-2d attribution and latency use different Phase-2c source identities")

    root = aggregate_output.resolve(strict=True)
    result_path = root / "phase2d_diagnostics.json"
    report_path = root / "phase2d_diagnostics_report.html"
    result = _json(result_path)
    if (
        result.get("schema_version") != "jepa4d-phase2d-diagnostics-aggregate-v1"
        or result.get("status") != "complete"
        or result.get("failures") != []
        or result.get("source_identity") != source
    ):
        raise RuntimeError("Phase-2d diagnostic aggregate is incomplete or source-inconsistent")
    expected_execution = _execution_provenance(repo_root, test_receipt, dependency_graph)
    if result.get("execution") != expected_execution:
        raise RuntimeError("Phase-2d diagnostic aggregate execution provenance is stale")
    graph_latency_jobs = [str(value) for value in expected_execution["dependency_graph"]["graph"]["latency_job_ids"]]
    validated_latency = _json(latency_output / "latency_aggregate.json")
    if set(graph_latency_jobs) != set(validated_latency["systems"]["slurm_job_ids"]):
        raise RuntimeError("Phase-2d dependency graph latency jobs differ from validated replicate allocations")
    attribution_record = _json(attribution / "fusion_attribution.json")
    attribution_execution = attribution_record.get("execution_provenance", {})
    attribution_job_id = str(attribution_execution.get("slurm", {}).get("job_id", ""))
    if attribution_job_id != str(expected_execution["dependency_graph"]["graph"]["attribution_job_id"]):
        raise RuntimeError("Phase-2d dependency graph attribution job differs from its execution provenance")
    expected_test = expected_execution["test_receipt"]
    attribution_test = attribution_execution.get("test_receipt", {})
    if (
        attribution_execution.get("git_commit") != expected_execution["git_commit"]
        or attribution_test.get("sha256") != expected_test["sha256"]
        or attribution_test.get("test_job_id") != expected_test["test_job_id"]
    ):
        raise RuntimeError("Phase-2d attribution uses a different tested commit/receipt")
    calibration_record = _json(calibration / "phase2d_calibration_scale_audit.json")
    if result.get("camera_provenance") != _camera_provenance(calibration_record):
        raise RuntimeError("Phase-2d diagnostic aggregate camera provenance is stale or over-claimed")
    expected_inputs = {
        "attribution": attribution_identity,
        "calibration": calibration_identity,
        "latency": latency_identity,
    }
    if result.get("inputs") != expected_inputs:
        raise RuntimeError("Phase-2d diagnostic aggregate input identity map is stale")
    aggregate_wandb = _validate_wandb(root, {result_path.name, report_path.name})
    if aggregate_wandb.get("execution") != expected_execution:
        raise RuntimeError("Phase-2d aggregate W&B receipt is not bound to execution provenance")
    _self_contained(report_path)
    _require_finite(result)
    return {
        "schema_version": "jepa4d-phase2d-postflight-v1",
        "status": "pass",
        "source_identity_sha256": _sha256(attribution / "source_identity.json"),
        "attribution": attribution_identity,
        "calibration": calibration_identity,
        "latency": latency_identity,
        "aggregate": {
            "result": _file_identity(result_path),
            "wandb_receipt": _file_identity(root / "wandb_receipt.json"),
        },
        "failures": [],
    }


def main(
    latency_root: Annotated[Path, typer.Option("--latency-root")],
    latency_output: Annotated[Path, typer.Option("--latency-output")],
    attribution: Annotated[Path | None, typer.Option("--attribution")] = None,
    calibration: Annotated[Path | None, typer.Option("--calibration")] = None,
    aggregate_output: Annotated[Path | None, typer.Option("--aggregate-output")] = None,
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
    test_receipt: Annotated[Path | None, typer.Option("--test-receipt")] = None,
    dependency_graph: Annotated[Path | None, typer.Option("--dependency-graph")] = None,
    receipt_output: Annotated[Path | None, typer.Option("--receipt-output")] = None,
) -> None:
    supplied = (attribution, calibration, aggregate_output)
    if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
        raise typer.BadParameter("attribution, calibration, and aggregate-output must be supplied together")
    if attribution is None or calibration is None or aggregate_output is None:
        source, identity = validate_latency(latency_root, latency_output)
        payload = {
            "schema_version": "jepa4d-phase2d-latency-postflight-v1",
            "status": "pass",
            "source_identity": source,
            "latency": identity,
            "failures": [],
        }
    else:
        provenance = (repo_root, test_receipt, dependency_graph)
        if not all(value is not None for value in provenance):
            raise typer.BadParameter("repo-root, test-receipt, and dependency-graph are required for final postflight")
        assert repo_root is not None and test_receipt is not None and dependency_graph is not None
        payload = validate_diagnostics(
            attribution,
            calibration,
            latency_root,
            latency_output,
            aggregate_output,
            repo_root,
            test_receipt,
            dependency_graph,
        )
    if receipt_output is not None:
        receipt_output.parent.mkdir(parents=True, exist_ok=True)
        receipt_output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    typer.run(main)
