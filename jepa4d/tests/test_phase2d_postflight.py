from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.aggregate_phase2d_diagnostics import _camera_provenance, _execution_provenance
from scripts.aggregate_phase2d_latency import (
    E2E_VARIANTS,
    HEAD_VARIANTS,
    _summarize_values,
    validate_latency_inputs,
)
from scripts.aggregate_phase2d_latency import (
    main as aggregate_latency,
)
from slurm.validate_phase2d_diagnostics import _validate_wandb, validate_diagnostics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source() -> dict[str, Any]:
    return {
        "schema_version": "jepa4d-phase2c-source-identity-v1",
        "phase2c_git_commit": "a" * 40,
        "split_hash": "b" * 64,
        "dataset_manifest": {"path": "/dataset/manifest.yaml", "sha256": "c" * 64},
        "source_files": {},
        "assets": {},
        "wandb": {},
    }


def _rows(replicate: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, variants, base in (("e2e", E2E_VARIANTS, 10.0), ("head_only", HEAD_VARIANTS, 0.1)):
        for block in range(30):
            for order, variant in enumerate(variants):
                value = base + replicate * 0.01 + block * 0.001 + order * 0.0001
                rows.append(
                    {
                        "scope": scope,
                        "block": block,
                        "order": order,
                        "variant": variant,
                        "iterations": 100,
                        "wall_ms_per_frame": value,
                        "cuda_ms_per_frame": value * 0.9,
                        "sample_offset": block % 16 if scope == "e2e" else 0,
                    }
                )
    return rows


def _refresh_receipt(directory: Path) -> None:
    names = ("latency.json", "latency_rows.csv", "latency_report.html", "source_identity.json", "gpu_telemetry.csv")
    latency = json.loads((directory / "latency.json").read_text())
    receipt = {
        "schema_version": "jepa4d-phase2d-wandb-receipt-v1",
        "status": "uploaded",
        "mode": "online",
        "run_id": "run-id",
        "run_url": "https://wandb.invalid/run-id",
        "run_path": "entity/project/run-id",
        "artifact_id": "artifact-id",
        "artifact_name": "artifact:v0",
        "artifact_qualified_name": "entity/project/artifact:v0",
        "artifact_version": "v0",
        "artifact_digest": "digest",
        "slurm_job_id": latency["slurm"]["job_id"],
        "test_receipt_sha256": latency["test_receipt"]["sha256"],
        "uploaded_files": {
            name: {"bytes": (directory / name).stat().st_size, "sha256": _sha256(directory / name)} for name in names
        },
    }
    _write_json(directory / "wandb_receipt.json", receipt)


def _generic_wandb_receipt(directory: Path, names: tuple[str, ...], kind: str | None = None) -> None:
    receipt = {
        "schema_version": "jepa4d-phase2d-wandb-receipt-v1",
        "status": "uploaded",
        "mode": "online",
        "run_id": "run-id",
        "run_url": "https://wandb.invalid/run-id",
        "run_path": "entity/project/run-id",
        "artifact_id": "artifact-id",
        "artifact_name": "artifact:v0",
        "artifact_qualified_name": "entity/project/artifact:v0",
        "artifact_version": "v0",
        "artifact_digest": "digest",
        "uploaded_files": {
            name: {"bytes": (directory / name).stat().st_size, "sha256": _sha256(directory / name)} for name in names
        },
    }
    if kind is not None:
        receipt["kind"] = kind
    _write_json(directory / "wandb_receipt.json", receipt)


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _replicate(root: Path, replicate: int) -> Path:
    directory = root / f"replicate-{replicate:02d}"
    directory.mkdir(parents=True)
    source = _source()
    _write_json(directory / "source_identity.json", source)
    test_receipt_path = root / "test-receipt.json"
    if not test_receipt_path.exists():
        _write_json(test_receipt_path, {"status": "pass", "git_commit": "d" * 40, "test_job_id": "100"})
    telemetry_path = directory / "gpu_telemetry.csv"
    telemetry_path.write_text(
        "timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm\nnow,50,1024,60,200,1200\n"
    )
    telemetry_statistics = {
        name: {"mean": value, "min": value, "max": value, "p50": value, "p95": value}
        for name, value in {
            "utilization_gpu_pct": 50.0,
            "memory_used_mib": 1024.0,
            "temperature_c": 60.0,
            "power_w": 200.0,
            "clocks_sm_mhz": 1200.0,
        }.items()
    }
    rows = _rows(replicate)
    summary = {}
    for scope, variants in (("e2e", E2E_VARIANTS), ("head_only", HEAD_VARIANTS)):
        for variant in variants:
            summary[f"{scope}/{variant}"] = _summarize_values(rows, scope, variant)
    record = {
        "schema_version": "jepa4d-phase2d-latency-replicate-v1",
        "replicate": replicate,
        "seed": 20260629 + replicate * 1009,
        "split_hash": source["split_hash"],
        "source_identity": source,
        "source_identity_sha256": _sha256(directory / "source_identity.json"),
        "slurm": {
            "job_id": str(200 + replicate),
            "job_name": f"latency-{replicate:02d}",
            "partition": "test",
            "nodelist": f"node{replicate:02d}",
        },
        "test_receipt": {
            "path": str(test_receipt_path.resolve()),
            "bytes": test_receipt_path.stat().st_size,
            "sha256": _sha256(test_receipt_path),
            "git_commit": "d" * 40,
            "test_job_id": "100",
        },
        "gpu_name": "test-gpu",
        "gpu_uuid": f"GPU-{replicate:02d}",
        "torch": "test",
        "python": "test",
        "cuda": "test",
        "warmups_per_path": 30,
        "blocks": 30,
        "iterations_per_block": 100,
        "peak_cuda_memory_gb": 2.0 + replicate * 0.01,
        "peak_cuda_reserved_memory_gb": 2.5 + replicate * 0.01,
        "gpu_telemetry": {
            "path": str(telemetry_path.resolve()),
            "bytes": telemetry_path.stat().st_size,
            "sha256": _sha256(telemetry_path),
            "sample_count": 1,
            "statistics": telemetry_statistics,
        },
        "decision_scope": "profiling-only",
        "summary": summary,
        "rows": rows,
    }
    _write_json(directory / "latency.json", record)
    with (directory / "latency_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "latency_report.html").write_text("<html><body>self contained</body></html>")
    _refresh_receipt(directory)
    return directory


def _latency_root(tmp_path: Path) -> Path:
    root = tmp_path / "latency"
    for replicate in range(12):
        _replicate(root, replicate)
    return root


def test_strict_latency_inputs_accept_exact_frozen_protocol(tmp_path: Path) -> None:
    records, inputs, source = validate_latency_inputs(_latency_root(tmp_path))
    assert [record["replicate"] for record in records] == list(range(12))
    assert [value["replicate"] for value in inputs] == list(range(12))
    assert source["split_hash"] == "b" * 64


def test_latency_aggregate_persists_input_map_and_head_only_summary(tmp_path: Path) -> None:
    root = _latency_root(tmp_path)
    output = tmp_path / "aggregate"
    aggregate_latency(root, output, expected_replicates=12, wandb_enabled=False)
    result = json.loads((output / "latency_aggregate.json").read_text())
    assert result["protocol"]["replicate_ids"] == list(range(12))
    assert len(result["inputs"]) == 12
    assert set(result["head_only_aggregate"]) == set(HEAD_VARIANTS)
    assert result["head_only_block_count"] == 12 * 30 * 5
    assert result["systems"]["slurm_job_ids"] == [str(value) for value in range(200, 212)]
    assert result["systems"]["peak_cuda_memory_gb"]["max"] > 2.0
    assert "Arithmetic-only paths" in (output / "latency_aggregate_report.html").read_text()


def test_strict_latency_inputs_reject_wrong_protocol_even_when_receipted(tmp_path: Path) -> None:
    root = _latency_root(tmp_path)
    directory = root / "replicate-07"
    record = json.loads((directory / "latency.json").read_text())
    record["warmups_per_path"] = 29
    _write_json(directory / "latency.json", record)
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="30 warmups"):
        validate_latency_inputs(root)


def test_strict_latency_inputs_reject_post_upload_byte_change(tmp_path: Path) -> None:
    root = _latency_root(tmp_path)
    path = root / "replicate-03" / "latency_rows.csv"
    path.write_text(path.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="identity changed"):
        validate_latency_inputs(root)


def test_strict_latency_inputs_require_unique_slurm_jobs_and_valid_telemetry(tmp_path: Path) -> None:
    root = _latency_root(tmp_path)
    directory = root / "replicate-11"
    record = json.loads((directory / "latency.json").read_text())
    record["slurm"]["job_id"] = "210"
    _write_json(directory / "latency.json", record)
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="reuse Slurm job ID"):
        validate_latency_inputs(root)

    record["slurm"]["job_id"] = "211"
    record["gpu_telemetry"]["statistics"]["power_w"]["min"] = 300.0
    _write_json(directory / "latency.json", record)
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="summary ordering is invalid"):
        validate_latency_inputs(root)


def test_strict_latency_inputs_require_exact_ids_zero_through_eleven(tmp_path: Path) -> None:
    root = _latency_root(tmp_path)
    (root / "replicate-11").rename(root / "replicate-12")
    with pytest.raises(ValueError, match="exactly replicate-00"):
        validate_latency_inputs(root)


def test_diagnostic_postflight_binds_backend_receipt_to_local_bytes(tmp_path: Path) -> None:
    result = tmp_path / "phase2d_diagnostics.json"
    report = tmp_path / "phase2d_diagnostics_report.html"
    result.write_text("{}\n")
    report.write_text("<html>ok</html>")
    receipt = {
        "schema_version": "jepa4d-phase2d-wandb-receipt-v1",
        "status": "uploaded",
        "mode": "online",
        "run_id": "run-id",
        "run_url": "https://wandb.invalid/run-id",
        "run_path": "entity/project/run-id",
        "artifact_id": "artifact-id",
        "artifact_name": "artifact:v0",
        "artifact_qualified_name": "entity/project/artifact:v0",
        "artifact_version": "v0",
        "artifact_digest": "digest",
        "uploaded_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in (result, report)
        },
    }
    _write_json(tmp_path / "wandb_receipt.json", receipt)
    _validate_wandb(tmp_path, {result.name, report.name})
    report.write_text("<html>changed</html>")
    with pytest.raises(RuntimeError, match="changed after upload"):
        _validate_wandb(tmp_path, {result.name, report.name})


def test_full_diagnostic_postflight_closes_cross_artifact_identity_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latency_root = _latency_root(tmp_path)
    latency_output = tmp_path / "latency-aggregate"
    aggregate_latency(latency_root, latency_output, expected_replicates=12, wandb_enabled=False)
    _generic_wandb_receipt(
        latency_output,
        ("latency_aggregate.json", "latency_aggregate_report.html"),
    )
    latency_payload = json.loads((latency_output / "latency_aggregate.json").read_text())
    latency_wandb_path = latency_output / "wandb_receipt.json"
    latency_wandb = json.loads(latency_wandb_path.read_text())
    latency_wandb["test_receipt_sha256"] = latency_payload["systems"]["test_receipt"]["sha256"]
    latency_wandb["slurm_job_ids"] = latency_payload["systems"]["slurm_job_ids"]
    _write_json(latency_wandb_path, latency_wandb)

    attribution = tmp_path / "attribution"
    attribution.mkdir()
    source_path = attribution / "source_identity.json"
    _write_json(source_path, _source())
    prediction_path = attribution / "full_predictions.npz"
    prediction_path.write_bytes(b"full-predictions")
    qualitative_path = attribution / "qualitative_examples.npz"
    qualitative_variants = [
        f"seed{seed}:{name}" for seed in range(3) for name in ("original", "zero", "fixed_average")
    ]
    with qualitative_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray("jepa4d-phase2d-qualitative-v1"),
            prediction_m=np.ones((9, 2, 2, 2), dtype=np.float32),
            target_m=np.ones((2, 2, 2), dtype=np.float32),
            log_variance=np.zeros((9, 2, 2, 2), dtype=np.float32),
            calibrated_log_depth_sigma=np.ones((9, 2, 2, 2), dtype=np.float32),
            sample_ids=np.asarray(["sample-a", "sample-b"]),
            sequence_ids=np.asarray(["sequence-a", "sequence-b"]),
            variant_ids=np.asarray(qualitative_variants),
            seeds=np.repeat(np.arange(3), 3),
        )
    attribution_execution = {
        "git_commit": "d" * 40,
        "test_receipt": {"path": "/test", "bytes": 1, "sha256": "e" * 64, "test_job_id": "100"},
        "slurm": {"job_id": "101", "job_name": "attribution", "partition": "test", "nodelist": "node"},
    }
    attribution_result = attribution / "fusion_attribution.json"
    _write_json(
        attribution_result,
        {
            "schema_version": "jepa4d-phase2d-same-checkpoint-fusion-attribution-v1",
            "status": "complete",
            "source_identity": _source(),
            "execution_provenance": attribution_execution,
            "qualitative_handoff": {
                "schema_version": "jepa4d-phase2d-qualitative-v1",
                "path": str(qualitative_path.resolve()),
                "sha256": _sha256(qualitative_path),
                "sample_count": 2,
                "variant_count": 9,
                "sample_ids": ["sample-a", "sample-b"],
                "variant_ids": qualitative_variants,
            },
            "test_samples": {"count": 128},
            "seeds": [{"seed": seed, "interventions": [{} for _ in range(15)]} for seed in range(3)],
        },
    )
    attribution_report = attribution / "fusion_attribution_report.html"
    attribution_report.write_text("<html>attribution</html>")
    attribution_receipt = attribution / "receipt.json"
    _write_json(
        attribution_receipt,
        {
            "schema_version": "jepa4d-phase2d-output-receipt-v1",
            "status": "pass",
            "fusion_attribution_json": {
                "path": str(attribution_result.resolve()),
                "sha256": _sha256(attribution_result),
            },
            "fusion_attribution_html": {
                "path": str(attribution_report.resolve()),
                "sha256": _sha256(attribution_report),
            },
            "full_predictions": {
                "path": str(prediction_path.resolve()),
                "sha256": _sha256(prediction_path),
                "frames": 128,
                "audit_scope": "full_phase2c_test",
            },
            "source_identity": {"path": str(source_path.resolve()), "sha256": _sha256(source_path)},
            "qualitative_examples": {
                "path": str(qualitative_path.resolve()),
                "bytes": qualitative_path.stat().st_size,
                "sha256": _sha256(qualitative_path),
                "schema_version": "jepa4d-phase2d-qualitative-v1",
                "samples": 2,
                "variants": 9,
            },
            "execution_provenance": attribution_execution,
        },
    )
    _generic_wandb_receipt(
        attribution,
        (
            attribution_result.name,
            attribution_report.name,
            prediction_path.name,
            qualitative_path.name,
            source_path.name,
            attribution_receipt.name,
        ),
        kind="attribution",
    )

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    calibration_result = calibration / "phase2d_calibration_scale_audit.json"
    calibration_payload = {
        "schema_version": "jepa4d-phase2d-calibration-scale-audit-v1",
        "diagnostic_only": True,
        "audit_scopes": ["full_phase2c_test"],
        "manifest_sha256": _source()["dataset_manifest"]["sha256"],
        "prediction_sources": [{"sha256": _sha256(prediction_path)}],
        "scale_oracle_audits": [{"audit_scope": "full_phase2c_test"} for _ in range(9)],
        "calibration_audit": {
            "sequences": [
                {
                    "distortion": {"status": "unknown_not_declared"},
                    "rgb_depth_registration_status": "unknown_not_declared",
                    "depth": {
                        "provenance_status": "unknown_not_declared",
                        "duplicate_correction_status": "unknown_not_declared",
                    },
                }
            ]
        },
    }
    _write_json(calibration_result, calibration_payload)
    calibration_report = calibration / "phase2d_calibration_scale_audit.html"
    calibration_report.write_text("<html>calibration</html>")
    oracle_csv = calibration / "phase2d_oracle_summary.csv"
    oracle_csv.write_text("metric\n1\n")
    camera_csv = calibration / "phase2d_calibration_table.csv"
    camera_csv.write_text("fx\n1\n")
    _generic_wandb_receipt(
        calibration,
        (calibration_result.name, calibration_report.name, oracle_csv.name, camera_csv.name),
        kind="calibration",
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "phase2d@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Phase2d Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    test_receipt = tmp_path / "test-receipt.json"
    _write_json(
        test_receipt,
        {
            "schema_version": "jepa4d-phase2d-test-receipt-v1",
            "status": "pass",
            "git_commit": commit,
            "slurm": {"SLURM_JOB_ID": "100"},
        },
    )
    dependency_graph = tmp_path / "dependency-graph.json"
    _write_json(
        dependency_graph,
        {
            "schema_version": "jepa4d-phase2d-dependency-graph-v1",
            "test_job_id": "100",
            "attribution_job_id": "101",
            "calibration_job_id": "102",
            "latency_job_ids": [str(value) for value in range(200, 212)],
            "latency_aggregate_job_id": "300",
            "aggregate_job_id": "400",
        },
    )
    monkeypatch.setenv("SLURM_JOB_ID", "400")
    monkeypatch.setenv("SLURM_JOB_NAME", "phase2d-aggregate")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "test")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node0")
    execution = _execution_provenance(repo, test_receipt, dependency_graph)
    attribution_execution = {
        "git_commit": commit,
        "test_receipt": {
            "path": str(test_receipt.resolve()),
            "bytes": test_receipt.stat().st_size,
            "sha256": _sha256(test_receipt),
            "test_job_id": "100",
        },
        "slurm": {"job_id": "101", "job_name": "attribution", "partition": "test", "nodelist": "node"},
    }
    attribution_payload = json.loads(attribution_result.read_text())
    attribution_payload["execution_provenance"] = attribution_execution
    _write_json(attribution_result, attribution_payload)
    attribution_receipt_payload = json.loads(attribution_receipt.read_text())
    attribution_receipt_payload["execution_provenance"] = attribution_execution
    attribution_receipt_payload["fusion_attribution_json"]["sha256"] = _sha256(attribution_result)
    _write_json(attribution_receipt, attribution_receipt_payload)
    _generic_wandb_receipt(
        attribution,
        (
            attribution_result.name,
            attribution_report.name,
            prediction_path.name,
            qualitative_path.name,
            source_path.name,
            attribution_receipt.name,
        ),
        kind="attribution",
    )

    aggregate = tmp_path / "diagnostics"
    aggregate.mkdir()
    aggregate_result = aggregate / "phase2d_diagnostics.json"
    _write_json(
        aggregate_result,
        {
            "schema_version": "jepa4d-phase2d-diagnostics-aggregate-v1",
            "status": "complete",
            "failures": [],
            "source_identity": _source(),
            "execution": execution,
            "camera_provenance": _camera_provenance(calibration_payload),
            "latency": {"replicates": latency_payload["replicates"]},
            "inputs": {
                "attribution": {
                    "result": _identity(attribution_result),
                    "local_receipt": _identity(attribution_receipt),
                    "qualitative_examples": _identity(qualitative_path),
                    "wandb_receipt": _identity(attribution / "wandb_receipt.json"),
                },
                "calibration": {
                    "result": _identity(calibration_result),
                    "wandb_receipt": _identity(calibration / "wandb_receipt.json"),
                },
                "latency": {
                    "result": _identity(latency_output / "latency_aggregate.json"),
                    "wandb_receipt": _identity(latency_output / "wandb_receipt.json"),
                },
            },
        },
    )
    aggregate_report = aggregate / "phase2d_diagnostics_report.html"
    aggregate_report.write_text("<html>diagnostics</html>")
    _generic_wandb_receipt(aggregate, (aggregate_result.name, aggregate_report.name))
    aggregate_wandb_path = aggregate / "wandb_receipt.json"
    aggregate_wandb = json.loads(aggregate_wandb_path.read_text())
    aggregate_wandb["execution"] = execution
    _write_json(aggregate_wandb_path, aggregate_wandb)

    postflight = validate_diagnostics(
        attribution,
        calibration,
        latency_root,
        latency_output,
        aggregate,
        repo,
        test_receipt,
        dependency_graph,
    )
    assert postflight["status"] == "pass"
    assert postflight["failures"] == []
