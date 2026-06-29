"""Create the single visual decision surface for all Phase 2d diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Annotated, Any

import plotly.graph_objects as go
import typer
import wandb
from plotly.subplots import make_subplots


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _execution_provenance(repo_root: Path, test_receipt_path: Path, dependency_graph_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    test_receipt_path = test_receipt_path.resolve(strict=True)
    dependency_graph_path = dependency_graph_path.resolve(strict=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip()
    if status:
        raise RuntimeError("Phase-2d diagnostic aggregation requires a clean Git worktree")
    test_receipt = json.loads(test_receipt_path.read_text())
    if (
        test_receipt.get("schema_version") != "jepa4d-phase2d-test-receipt-v1"
        or test_receipt.get("status") != "pass"
        or test_receipt.get("git_commit") != commit
    ):
        raise RuntimeError("Phase-2d test receipt is not bound to the current clean commit")
    dependency_graph = json.loads(dependency_graph_path.read_text())
    required_graph_keys = {
        "schema_version",
        "test_job_id",
        "attribution_job_id",
        "calibration_job_id",
        "latency_job_ids",
        "latency_aggregate_job_id",
        "aggregate_job_id",
    }
    if not isinstance(dependency_graph, dict) or set(dependency_graph) != required_graph_keys:
        raise RuntimeError("Phase-2d dependency graph schema/keys are incomplete")
    latency_job_ids = [str(value) for value in dependency_graph["latency_job_ids"]]
    scalar_job_ids = [
        str(dependency_graph[key])
        for key in (
            "test_job_id",
            "attribution_job_id",
            "calibration_job_id",
            "latency_aggregate_job_id",
            "aggregate_job_id",
        )
    ]
    if (
        dependency_graph["schema_version"] != "jepa4d-phase2d-dependency-graph-v1"
        or len(latency_job_ids) != 12
        or len(set(latency_job_ids)) != 12
        or any(not value for value in [*scalar_job_ids, *latency_job_ids])
    ):
        raise RuntimeError("Phase-2d dependency graph job identities are invalid")
    test_job_id = str(test_receipt.get("slurm", {}).get("SLURM_JOB_ID", ""))
    if test_job_id != str(dependency_graph["test_job_id"]):
        raise RuntimeError("Phase-2d dependency graph test job differs from the passing test receipt")
    current_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not current_job_id or current_job_id != str(dependency_graph["aggregate_job_id"]):
        raise RuntimeError("current Slurm job differs from the Phase-2d dependency graph aggregate job")
    return {
        "repo_root": str(repo_root),
        "git_commit": commit,
        "git_status": status,
        "test_receipt": {**_identity(test_receipt_path), "test_job_id": test_job_id},
        "dependency_graph": {**_identity(dependency_graph_path), "graph": dependency_graph},
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST")
        },
    }


def _control(record: dict[str, Any], intervention_id: str) -> dict[str, Any]:
    matches = [
        row for row in record["aggregate"]["controls"] if row["intervention"]["intervention_id"] == intervention_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one aggregate control {intervention_id}, found {len(matches)}")
    return matches[0]


def _oracle_summary(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    selected = [
        audit
        for audit in record["scale_oracle_audits"]
        if audit["audit_scope"] == "full_phase2c_test" and audit["variant_id"].endswith(":original")
    ]
    if len(selected) != 3:
        raise ValueError(f"expected three full-test original prediction sets, found {len(selected)}")
    names = list(selected[0]["oracles"])
    summary = {}
    for name in names:
        macros = [audit["oracles"][name]["macro_equal_sequence_weight"] for audit in selected]
        summary[name] = {
            "metric_abs_rel": statistics.fmean(value["metric"]["abs_rel"] for value in macros),
            "aligned_abs_rel": statistics.fmean(value["aligned"]["abs_rel"] for value in macros),
            "abs_log_scale_error": statistics.fmean(value["metric"]["abs_log_scale_error"] for value in macros),
        }
    return summary


def _camera_provenance(record: dict[str, Any]) -> dict[str, Any]:
    sequences = record["calibration_audit"]["sequences"]
    distortion_statuses = sorted({str(sequence["distortion"]["status"]) for sequence in sequences})
    registration_statuses = sorted({str(sequence["rgb_depth_registration_status"]) for sequence in sequences})
    depth_provenance_statuses = sorted({str(sequence["depth"]["provenance_status"]) for sequence in sequences})
    duplicate_statuses = sorted({str(sequence["depth"]["duplicate_correction_status"]) for sequence in sequences})
    distortion = "declared" if distortion_statuses == ["declared"] else "incomplete"
    registration = "declared" if registration_statuses == ["declared"] else "incomplete"
    if "duplicate_detected" in duplicate_statuses:
        depth_correction = "duplicate_detected"
    elif depth_provenance_statuses == ["declared"] and duplicate_statuses == ["no_duplicate_detected"]:
        depth_correction = "declared_no_duplicate_detected"
    else:
        depth_correction = "incomplete"
    overall = (
        "complete"
        if distortion == "declared"
        and registration == "declared"
        and depth_correction == "declared_no_duplicate_detected"
        else "incomplete"
    )
    return {
        "overall": overall,
        "distortion": distortion,
        "rgb_depth_registration": registration,
        "depth_correction": depth_correction,
        "observed_statuses": {
            "distortion": distortion_statuses,
            "rgb_depth_registration": registration_statuses,
            "depth_provenance": depth_provenance_statuses,
            "duplicate_correction": duplicate_statuses,
        },
    }


def _render(path: Path, payload: dict[str, Any]) -> None:
    attribution = payload["attribution"]
    oracle = payload["scale_oracles"]
    latency = payload["latency"]
    controls = attribution["controls"]
    control_names = ["original", "zero", "fixed_average"]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Same-checkpoint gate interventions",
            "Scale-oracle diagnostic ladder",
            "Residual contribution",
            "Independent latency-job ratios",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.13,
    )
    figure.add_trace(
        go.Bar(
            x=control_names,
            y=[controls[name]["metric_abs_rel"] for name in control_names],
            marker_color=["#ef4444", "#2563eb", "#f59e0b"],
            text=[f"{controls[name]['metric_abs_rel']:.4f}" for name in control_names],
            textposition="outside",
            name="AbsRel",
        ),
        row=1,
        col=1,
    )
    oracle_names = list(oracle)
    figure.add_trace(
        go.Bar(
            x=oracle_names,
            y=[oracle[name]["metric_abs_rel"] for name in oracle_names],
            marker_color=["#64748b", "#0ea5e9", "#14b8a6", "#8b5cf6", "#f59e0b"],
            name="oracle AbsRel",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=control_names,
            y=[controls[name]["residual_total_norm_ratio"] for name in control_names],
            marker_color=["#ef4444", "#2563eb", "#f59e0b"],
            name="residual/final norm",
        ),
        row=2,
        col=1,
    )
    replicate_rows = latency["replicates"]
    figure.add_trace(
        go.Scatter(
            x=[row["replicate"] for row in replicate_rows],
            y=[row["learned_to_final_wall_ratio"] for row in replicate_rows],
            mode="lines+markers",
            marker={"size": 9, "color": "#ef4444"},
            line={"color": "#fecaca"},
            name="learned/final",
        ),
        row=2,
        col=2,
    )
    figure.add_hline(y=1.10, line_dash="dash", line_color="#b91c1c", row=2, col=2)
    figure.update_layout(
        template="plotly_white",
        height=900,
        showlegend=False,
        font={"family": "Inter,system-ui,sans-serif"},
        margin={"l": 55, "r": 30, "t": 85, "b": 75},
    )
    figure.update_yaxes(title_text="AbsRel", row=1, col=1)
    figure.update_yaxes(title_text="AbsRel", row=1, col=2)
    figure.update_yaxes(title_text="L2 ratio", row=2, col=1)
    figure.update_yaxes(title_text="latency ratio", row=2, col=2)
    plot = figure.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
    findings = payload["findings"]
    camera = payload["camera_provenance"]
    cards = "".join(
        f"<div class='card'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in (
            ("Gate causal ΔAbsRel", f"{findings['original_minus_zero_absrel']:+.5f}"),
            ("Per-image scalar gain", f"{findings['per_image_scalar_absrel_gain'] * 100:.1f}%"),
            ("Latency ratio 95% CI", findings["latency_ratio_ci"]),
            ("Camera provenance", camera["overall"]),
            ("Distortion", camera["distortion"]),
            ("RGB-depth registration", camera["rgb_depth_registration"]),
            ("Depth correction", camera["depth_correction"]),
        )
    )
    path.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>Phase 2d diagnostics</title>
<style>body{{margin:0;background:#f8fafc;color:#0f172a;font-family:Inter,system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:32px}}h1{{margin-bottom:5px}}.sub{{color:#475569}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}}.card{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px;box-shadow:0 4px 16px #0f172a0a}}.card span{{display:block;color:#64748b;font-size:13px;text-transform:uppercase}}.card strong{{display:block;font-size:24px;margin-top:7px}}
.panel{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:10px}}.boundary{{margin-top:18px;background:#fff7ed;border-left:4px solid #f59e0b;border-radius:8px;padding:14px 18px}}
</style></head><body><main><h1>Phase 2d · Causal and systems diagnostics</h1><div class='sub'>One reading surface for gate causality, camera/scale structure, and independent latency confirmation</div>
<div class='cards'>{cards}</div><div class='panel'>{plot}</div><div class='boundary'>Freiburg-3 is consumed. These results diagnose mechanism and timing; they do not create a new generalization claim or change the frozen Phase-2c decision.</div></main></body></html>"""
    )


def main(
    attribution_json: Annotated[Path, typer.Option("--attribution-json")],
    calibration_json: Annotated[Path, typer.Option("--calibration-json")],
    latency_json: Annotated[Path, typer.Option("--latency-json")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    repo_root: Annotated[Path, typer.Option("--repo-root")],
    test_receipt: Annotated[Path, typer.Option("--test-receipt")],
    dependency_graph: Annotated[Path, typer.Option("--dependency-graph")],
    project: Annotated[str, typer.Option("--project")] = "jepa4d-worldmodel",
    entity: Annotated[str | None, typer.Option("--entity")] = None,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    attribution_record = json.loads(attribution_json.resolve(strict=True).read_text())
    calibration_record = json.loads(calibration_json.resolve(strict=True).read_text())
    latency_record = json.loads(latency_json.resolve(strict=True).read_text())
    if attribution_record["schema_version"] != "jepa4d-phase2d-same-checkpoint-fusion-attribution-v1":
        raise ValueError("unexpected attribution schema")
    if calibration_record["schema_version"] != "jepa4d-phase2d-calibration-scale-audit-v1":
        raise ValueError("unexpected calibration schema")
    if latency_record["schema_version"] != "jepa4d-phase2d-latency-aggregate-v1":
        raise ValueError("unexpected latency schema")
    attribution_source = attribution_record.get("source_identity")
    latency_source = latency_record.get("source_identity")
    if not isinstance(attribution_source, dict) or attribution_source != latency_source:
        raise ValueError("attribution and latency do not share one immutable Phase-2c source identity")
    execution = _execution_provenance(repo_root, test_receipt, dependency_graph)
    controls = {}
    for name in ("original", "zero", "fixed_average"):
        value = _control(attribution_record, name)
        controls[name] = {
            "metric_abs_rel": value["metrics"]["metric_abs_rel"]["mean"],
            "aligned_abs_rel": value["metrics"]["aligned_abs_rel"]["mean"],
            "calibrated_nll": value["metrics"]["calibrated_log_depth_nll"]["mean"],
            "residual_total_norm_ratio": value["metrics"]["residual_total_norm_ratio"]["mean"],
        }
    oracle = _oracle_summary(calibration_record)
    latency_ratio = latency_record["ratio_aggregate"]
    camera_provenance = _camera_provenance(calibration_record)
    payload: dict[str, Any] = {
        "schema_version": "jepa4d-phase2d-diagnostics-aggregate-v1",
        "status": "complete",
        "source_identity": attribution_source,
        "execution": execution,
        "inputs": {
            "attribution": {
                "result": _identity(attribution_json),
                "local_receipt": _identity(attribution_json.parent / "receipt.json"),
                "qualitative_examples": _identity(attribution_json.parent / "qualitative_examples.npz"),
                "wandb_receipt": _identity(attribution_json.parent / "wandb_receipt.json"),
            },
            "calibration": {
                "result": _identity(calibration_json),
                "wandb_receipt": _identity(calibration_json.parent / "wandb_receipt.json"),
            },
            "latency": {
                "result": _identity(latency_json),
                "wandb_receipt": _identity(latency_json.parent / "wandb_receipt.json"),
            },
        },
        "attribution": {"controls": controls},
        "scale_oracles": oracle,
        "latency": {
            "ratio_aggregate": latency_ratio,
            "replicates": latency_record["replicates"],
        },
        "camera_provenance": camera_provenance,
        "findings": {
            "original_minus_zero_absrel": controls["original"]["metric_abs_rel"] - controls["zero"]["metric_abs_rel"],
            "per_image_scalar_absrel_gain": 1.0
            - oracle["per_image_scalar"]["metric_abs_rel"] / oracle["raw"]["metric_abs_rel"],
            "latency_ratio_ci": f"[{latency_ratio['ci95_low']:.3f}, {latency_ratio['ci95_high']:.3f}]×",
            "latency_confirmation": latency_ratio["confirmation_status"],
            "camera_provenance": camera_provenance["overall"],
        },
        "claim_boundary": "post-hoc diagnostic only; Phase 2c decision remains retain_final_layer",
        "failures": [],
    }
    result_path = output / "phase2d_diagnostics.json"
    report_path = output / "phase2d_diagnostics_report.html"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _render(report_path, payload)
    run = wandb.init(
        project=project,
        entity=entity,
        name="phase2d-diagnostics-aggregate",
        job_type="phase2d-diagnostics-aggregate",
        mode="online",
        config={
            "schema_version": payload["schema_version"],
            "phase2c_git_commit": attribution_source["phase2c_git_commit"],
            "split_hash": attribution_source["split_hash"],
            "claim_boundary": payload["claim_boundary"],
            "execution_git_commit": execution["git_commit"],
            "test_receipt_sha256": execution["test_receipt"]["sha256"],
            "dependency_graph_sha256": execution["dependency_graph"]["sha256"],
            "aggregate_slurm_job_id": execution["slurm"]["SLURM_JOB_ID"],
        },
        tags=["phase-2d", "diagnostics", "aggregate", "visual-report"],
    )
    try:
        run.log({"diagnostics/report": wandb.Html(str(report_path), inject=False)})
        for key, value in payload["findings"].items():
            run.summary[f"findings/{key}"] = value
        artifact = wandb.Artifact(f"{run.id}-phase2d-diagnostics", type="phase2d-report")
        artifact.add_file(str(result_path))
        artifact.add_file(str(report_path))
        uploaded = run.log_artifact(artifact).wait(timeout=900)
        run.summary["result"] = "success"
        uploaded_files = {
            name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in ("phase2d_diagnostics.json", "phase2d_diagnostics_report.html")
        }
        (output / "wandb_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": "jepa4d-phase2d-wandb-receipt-v1",
                    "status": "uploaded",
                    "mode": "online",
                    "run_id": str(run.id),
                    "run_url": str(run.url),
                    "run_path": str(run.path),
                    "artifact_id": str(uploaded.id),
                    "artifact_name": str(uploaded.name),
                    "artifact_qualified_name": str(uploaded.qualified_name),
                    "artifact_version": str(uploaded.version),
                    "artifact_digest": str(uploaded.digest),
                    "uploaded_files": uploaded_files,
                    "execution": execution,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
