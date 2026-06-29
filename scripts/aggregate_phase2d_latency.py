"""Aggregate independent Phase 2d latency jobs into a cluster-aware report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import plotly.graph_objects as go
import typer
import wandb
from plotly.subplots import make_subplots

EXPECTED_REPLICATES = tuple(range(12))
EXPECTED_WARMUPS = 30
EXPECTED_BLOCKS = 30
EXPECTED_ITERATIONS = 100
E2E_VARIANTS = ("final_deployment", "final_capture_all", "fixed_deployment", "learned_deployment")
HEAD_VARIANTS = ("final_head", "fixed_head", "learned_head", "zero_gate_same_head", "fixed_equivalent_same_head")
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2d-wandb-receipt-v1"
SOURCE_IDENTITY_SCHEMA = "jepa4d-phase2c-source-identity-v1"
TELEMETRY_FIELDS = ("utilization_gpu_pct", "memory_used_mib", "temperature_c", "power_w", "clocks_sm_mhz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_backend_receipt(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    receipt_path = directory / "wandb_receipt.json"
    receipt = _read_json(receipt_path)
    if (
        receipt.get("schema_version") != WANDB_RECEIPT_SCHEMA
        or receipt.get("status") != "uploaded"
        or receipt.get("mode") != "online"
    ):
        raise ValueError(f"latency W&B receipt is not an online upload: {receipt_path}")
    backend_keys = (
        "run_id",
        "run_url",
        "run_path",
        "artifact_id",
        "artifact_name",
        "artifact_qualified_name",
        "artifact_version",
        "artifact_digest",
    )
    missing = [key for key in backend_keys if not str(receipt.get(key, "")).strip() or receipt.get(key) == "None"]
    if missing:
        raise ValueError(f"latency W&B receipt lacks backend identities {missing}: {receipt_path}")
    uploaded = receipt.get("uploaded_files")
    expected_names = {
        "latency.json",
        "latency_rows.csv",
        "latency_report.html",
        "source_identity.json",
        "gpu_telemetry.csv",
    }
    if not isinstance(uploaded, dict) or set(uploaded) != expected_names:
        raise ValueError(f"latency W&B receipt has an incomplete uploaded_files map: {receipt_path}")
    identities: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        path = directory / name
        identity = uploaded.get(name)
        if not path.is_file() or not isinstance(identity, dict):
            raise ValueError(f"latency uploaded artifact is absent: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if int(identity.get("bytes", -1)) != actual["bytes"] or identity.get("sha256") != actual["sha256"]:
            raise ValueError(f"latency uploaded artifact identity changed: {path}")
        identities[name] = actual
    identities["wandb_receipt.json"] = {"bytes": receipt_path.stat().st_size, "sha256": _sha256(receipt_path)}
    return receipt, identities


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive, found {number}")
    return number


def _summarize_values(rows: list[dict[str, Any]], scope: str, variant: str) -> dict[str, float]:
    selected = [row for row in rows if row["scope"] == scope and row["variant"] == variant]
    wall = [float(row["wall_ms_per_frame"]) for row in selected]
    cuda = [float(row["cuda_ms_per_frame"]) for row in selected]
    return {
        "blocks": float(len(selected)),
        "iterations": float(sum(int(row["iterations"]) for row in selected)),
        "wall_mean_ms": float(statistics.fmean(wall)),
        "wall_median_ms": float(statistics.median(wall)),
        "wall_p90_ms": float(np.percentile(wall, 90)),
        "wall_p95_ms": float(np.percentile(wall, 95)),
        "cuda_mean_ms": float(statistics.fmean(cuda)),
        "cuda_median_ms": float(statistics.median(cuda)),
        "cuda_p90_ms": float(np.percentile(cuda, 90)),
        "cuda_p95_ms": float(np.percentile(cuda, 95)),
    }


def _validate_summary(record: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary = record.get("summary")
    if not isinstance(summary, dict) or set(summary) != {
        *(f"e2e/{name}" for name in E2E_VARIANTS),
        *(f"head_only/{name}" for name in HEAD_VARIANTS),
    }:
        raise ValueError("latency summary path coverage differs from the frozen protocol")
    for scope, variants in (("e2e", E2E_VARIANTS), ("head_only", HEAD_VARIANTS)):
        for variant in variants:
            observed = summary[f"{scope}/{variant}"]
            expected = _summarize_values(rows, scope, variant)
            for key, expected_value in expected.items():
                observed_value = float(observed.get(key, float("nan")))
                if not math.isfinite(observed_value) or not math.isclose(
                    observed_value, expected_value, rel_tol=1e-10, abs_tol=1e-10
                ):
                    raise ValueError(f"latency summary mismatch for {scope}/{variant}/{key}")


def _validate_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = record.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_BLOCKS * (len(E2E_VARIANTS) + len(HEAD_VARIANTS)):
        raise ValueError("latency replicate does not contain exactly 270 frozen-protocol block rows")
    expected_fields = {
        "scope",
        "block",
        "order",
        "variant",
        "iterations",
        "wall_ms_per_frame",
        "cuda_ms_per_frame",
        "sample_offset",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("latency row schema differs from the frozen protocol")
        if int(row["iterations"]) != EXPECTED_ITERATIONS:
            raise ValueError("latency row does not contain exactly 100 serial iterations")
        _finite_positive(row["wall_ms_per_frame"], "wall latency")
        _finite_positive(row["cuda_ms_per_frame"], "CUDA latency")
    for scope, variants in (("e2e", E2E_VARIANTS), ("head_only", HEAD_VARIANTS)):
        for block in range(EXPECTED_BLOCKS):
            selected = [row for row in rows if row["scope"] == scope and int(row["block"]) == block]
            if {str(row["variant"]) for row in selected} != set(variants):
                raise ValueError(f"latency {scope} block {block} has incomplete variant coverage")
            if {int(row["order"]) for row in selected} != set(range(len(variants))):
                raise ValueError(f"latency {scope} block {block} has invalid randomized schedule positions")
    _validate_summary(record, rows)
    return rows


def _validate_systems_metadata(directory: Path, record: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    slurm = record.get("slurm")
    if not isinstance(slurm, dict) or set(slurm) != {"job_id", "job_name", "partition", "nodelist"}:
        raise ValueError(f"latency Slurm identity schema is incomplete: {directory}")
    if any(not str(value).strip() for value in slurm.values()):
        raise ValueError(f"latency Slurm identity contains an empty value: {directory}")

    test_receipt = record.get("test_receipt")
    if not isinstance(test_receipt, dict) or set(test_receipt) != {
        "path",
        "bytes",
        "sha256",
        "git_commit",
        "test_job_id",
    }:
        raise ValueError(f"latency test-receipt identity schema is incomplete: {directory}")
    receipt_path = Path(str(test_receipt["path"])).resolve(strict=True)
    if (
        int(test_receipt["bytes"]) != receipt_path.stat().st_size
        or test_receipt["sha256"] != _sha256(receipt_path)
        or len(str(test_receipt["git_commit"])) != 40
        or not str(test_receipt["test_job_id"])
    ):
        raise ValueError(f"latency test receipt byte identity is invalid: {directory}")

    telemetry = record.get("gpu_telemetry")
    if not isinstance(telemetry, dict) or set(telemetry) != {"path", "bytes", "sha256", "sample_count", "statistics"}:
        raise ValueError(f"latency GPU telemetry identity schema is incomplete: {directory}")
    telemetry_path = Path(str(telemetry["path"])).resolve(strict=True)
    if telemetry_path != (directory / "gpu_telemetry.csv").resolve():
        raise ValueError(f"latency GPU telemetry path escapes its replicate output: {directory}")
    if int(telemetry["bytes"]) != telemetry_path.stat().st_size or telemetry["sha256"] != _sha256(telemetry_path):
        raise ValueError(f"latency GPU telemetry byte identity changed: {directory}")
    with telemetry_path.open(newline="") as stream:
        telemetry_rows = list(csv.DictReader(stream))
    if int(telemetry["sample_count"]) <= 0 or int(telemetry["sample_count"]) != len(telemetry_rows):
        raise ValueError(f"latency GPU telemetry sample count is invalid: {directory}")
    statistics_payload = telemetry.get("statistics")
    if not isinstance(statistics_payload, dict) or set(statistics_payload) != set(TELEMETRY_FIELDS):
        raise ValueError(f"latency GPU telemetry statistics are incomplete: {directory}")
    for metric in TELEMETRY_FIELDS:
        values = statistics_payload[metric]
        if not isinstance(values, dict) or set(values) != {"mean", "min", "max", "p50", "p95"}:
            raise ValueError(f"latency GPU telemetry summary is incomplete for {metric}: {directory}")
        numeric = {key: float(value) for key, value in values.items()}
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"latency GPU telemetry contains non-finite {metric} values: {directory}")
        if not (
            numeric["min"] <= numeric["p50"] <= numeric["p95"] <= numeric["max"]
            and numeric["min"] <= numeric["mean"] <= numeric["max"]
        ):
            raise ValueError(f"latency GPU telemetry summary ordering is invalid for {metric}: {directory}")
    peak = _finite_positive(record.get("peak_cuda_memory_gb"), "peak CUDA memory")
    reserved = _finite_positive(record.get("peak_cuda_reserved_memory_gb"), "peak reserved CUDA memory")
    if reserved < peak:
        raise ValueError(f"reserved CUDA memory is lower than allocated peak: {directory}")
    if str(record.get("gpu_uuid", "")) == "" or str(record.get("gpu_name", "")) == "":
        raise ValueError(f"latency GPU identity is incomplete: {directory}")
    return str(slurm["job_id"]), test_receipt, telemetry


def validate_latency_inputs(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate all 12 byte-bound independent latency jobs before aggregation."""

    root = input_root.resolve(strict=True)
    expected_directories = {f"replicate-{replicate:02d}" for replicate in EXPECTED_REPLICATES}
    observed_directories = {path.name for path in root.glob("replicate-*") if path.is_dir()}
    if observed_directories != expected_directories:
        raise ValueError(
            f"latency inputs must be exactly replicate-00..replicate-11; found {sorted(observed_directories)}"
        )
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    common_source: dict[str, Any] | None = None
    common_test_receipt: dict[str, Any] | None = None
    slurm_job_ids: set[str] = set()
    for replicate in EXPECTED_REPLICATES:
        directory = root / f"replicate-{replicate:02d}"
        latency_path = directory / "latency.json"
        record = _read_json(latency_path)
        if record.get("schema_version") != "jepa4d-phase2d-latency-replicate-v1":
            raise ValueError(f"unexpected latency schema for replicate {replicate}")
        if int(record.get("replicate", -1)) != replicate or int(record.get("seed", -1)) != 20260629 + replicate * 1009:
            raise ValueError(f"latency identity/seed mismatch for replicate {replicate}")
        if (
            int(record.get("warmups_per_path", -1)) != EXPECTED_WARMUPS
            or int(record.get("blocks", -1)) != EXPECTED_BLOCKS
            or int(record.get("iterations_per_block", -1)) != EXPECTED_ITERATIONS
        ):
            raise ValueError(
                f"latency protocol differs from 30 warmups/30 blocks/100 iterations for replicate {replicate}"
            )
        rows = _validate_rows(record)
        slurm_job_id, test_receipt, telemetry = _validate_systems_metadata(directory, record)
        if slurm_job_id in slurm_job_ids:
            raise ValueError(f"latency replicates reuse Slurm job ID {slurm_job_id}")
        slurm_job_ids.add(slurm_job_id)
        if common_test_receipt is None:
            common_test_receipt = test_receipt
        elif test_receipt != common_test_receipt:
            raise ValueError(f"latency replicate {replicate} uses a different passing test receipt")
        source_path = directory / "source_identity.json"
        source = _read_json(source_path)
        if source.get("schema_version") != SOURCE_IDENTITY_SCHEMA:
            raise ValueError(f"unexpected Phase-2c source identity for replicate {replicate}")
        if record.get("source_identity") != source or record.get("source_identity_sha256") != _sha256(source_path):
            raise ValueError(f"latency JSON is not bound to source_identity.json for replicate {replicate}")
        if record.get("split_hash") != source.get("split_hash"):
            raise ValueError(f"latency split differs from Phase-2c source for replicate {replicate}")
        if common_source is None:
            common_source = source
        elif source != common_source:
            raise ValueError(f"latency replicate {replicate} uses a different Phase-2c source")
        receipt, identities = _require_backend_receipt(directory)
        if receipt.get("slurm_job_id") != slurm_job_id or receipt.get("test_receipt_sha256") != test_receipt["sha256"]:
            raise ValueError(f"latency W&B receipt provenance differs from replicate {replicate}")
        csv_path = directory / "latency_rows.csv"
        with csv_path.open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != len(rows):
            raise ValueError(f"latency CSV row count differs from JSON for replicate {replicate}")
        records.append(record)
        inputs.append(
            {
                "replicate": replicate,
                "slurm": record["slurm"],
                "test_receipt": test_receipt,
                "gpu_telemetry": telemetry,
                "peak_cuda_memory_gb": float(record["peak_cuda_memory_gb"]),
                "peak_cuda_reserved_memory_gb": float(record["peak_cuda_reserved_memory_gb"]),
                "directory": str(directory.resolve()),
                "files": identities,
                "wandb": {
                    key: receipt[key]
                    for key in ("run_id", "run_url", "artifact_id", "artifact_version", "artifact_digest")
                },
            }
        )
    assert common_source is not None
    return records, inputs, common_source


def _bootstrap_interval(values: list[float], seed: int = 20260629) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("at least two independent replicates are required")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(10_000, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _render(path: Path, payload: dict[str, Any]) -> None:
    replicate_rows = payload["replicates"]
    raw_rows = payload["rows"]
    variants = list(E2E_VARIANTS)
    colors = ["#2563eb", "#60a5fa", "#f59e0b", "#ef4444"]
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "violin"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=(
            "Block latency distribution",
            "Independent replicate ratios",
            "Median and p90 across all blocks",
            "Schedule-position sensitivity",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )
    for variant, color in zip(variants, colors, strict=True):
        selected = [
            row["wall_ms_per_frame"] for row in raw_rows if row["scope"] == "e2e" and row["variant"] == variant
        ]
        figure.add_trace(
            go.Violin(
                x=[variant] * len(selected),
                y=selected,
                name=variant,
                box_visible=True,
                meanline_visible=True,
                points=False,
                line_color=color,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    ratios = [row["learned_to_final_wall_ratio"] for row in replicate_rows]
    figure.add_trace(
        go.Scatter(
            x=[row["replicate"] for row in replicate_rows],
            y=ratios,
            mode="lines+markers",
            marker={"size": 10, "color": "#ef4444"},
            line={"color": "#fecaca"},
            name="learned/final",
        ),
        row=1,
        col=2,
    )
    figure.add_hline(y=1.10, line_dash="dash", line_color="#b91c1c", row=1, col=2)
    figure.add_trace(
        go.Bar(
            x=variants,
            y=[payload["aggregate"][name]["wall_median_ms"] for name in variants],
            marker_color=colors,
            name="median",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=variants,
            y=[payload["aggregate"][name]["wall_p90_ms"] for name in variants],
            marker_color=colors,
            opacity=0.45,
            name="p90",
        ),
        row=2,
        col=1,
    )
    for variant, color in zip(variants, colors, strict=True):
        selected = [row for row in raw_rows if row["scope"] == "e2e" and row["variant"] == variant]
        by_position = {
            position: statistics.fmean(row["wall_ms_per_frame"] for row in selected if row["order"] == position)
            for position in sorted({row["order"] for row in selected})
        }
        figure.add_trace(
            go.Scatter(
                x=list(by_position),
                y=list(by_position.values()),
                mode="lines+markers",
                name=variant,
                line={"color": color},
            ),
            row=2,
            col=2,
        )
    figure.update_layout(
        template="plotly_white",
        height=920,
        barmode="group",
        font={"family": "Inter,system-ui,sans-serif"},
        margin={"l": 55, "r": 25, "t": 85, "b": 50},
        legend={"orientation": "h", "y": -0.12},
    )
    figure.update_yaxes(title_text="ms/frame", row=1, col=1)
    figure.update_yaxes(title_text="ratio", row=1, col=2)
    figure.update_yaxes(title_text="ms/frame", row=2, col=1)
    figure.update_yaxes(title_text="mean ms/frame", row=2, col=2)
    figure.update_xaxes(title_text="randomized position", row=2, col=2)
    plot = figure.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
    aggregate = payload["ratio_aggregate"]
    status = aggregate["confirmation_status"]
    color = "#15803d" if status == "within_1.10x" else "#b91c1c"
    cards = "".join(
        f"<div class='card'><span>{name}</span><strong>{value}</strong></div>"
        for name, value in (
            ("Independent jobs", str(payload["replicate_count"])),
            ("Paired ratio mean", f"{aggregate['mean']:.3f}×"),
            ("95% cluster CI", f"[{aggregate['ci95_low']:.3f}, {aggregate['ci95_high']:.3f}]"),
            ("Measured blocks", str(payload["e2e_block_count"])),
            ("Peak CUDA memory", f"{payload['systems']['peak_cuda_memory_gb']['max']:.2f} GiB"),
            ("Max GPU temperature", f"{payload['systems']['telemetry']['gpu_temperature_max_c']:.1f} °C"),
        )
    )
    head_rows = "".join(
        "<tr>"
        f"<td>{name}</td>"
        f"<td>{payload['head_only_aggregate'][name]['wall_median_ms']:.4f}</td>"
        f"<td>{payload['head_only_aggregate'][name]['wall_p90_ms']:.4f}</td>"
        f"<td>{payload['head_only_aggregate'][name]['wall_p95_ms']:.4f}</td>"
        f"<td>{payload['head_only_aggregate'][name]['cuda_median_ms']:.4f}</td>"
        "</tr>"
        for name in HEAD_VARIANTS
    )
    job_rows = "".join(
        "<tr>"
        f"<td>{row['replicate']}</td><td>{row['slurm_job_id']}</td><td>{row['slurm_partition']}</td>"
        f"<td>{row['gpu_name']}</td><td>{float(row['peak_cuda_memory_gb']):.3f}</td>"
        f"<td>{float(row['gpu_utilization_mean_pct']):.1f}</td><td>{float(row['gpu_temperature_max_c']):.1f}</td>"
        f"<td>{float(row['gpu_power_mean_w']):.1f}</td><td>{float(row['gpu_clocks_sm_mean_mhz']):.1f}</td>"
        "</tr>"
        for row in sorted(replicate_rows, key=lambda value: value["replicate"])
    )
    path.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>Phase 2d latency aggregate</title>
<style>body{{margin:0;background:#f8fafc;color:#0f172a;font-family:Inter,system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:32px}}
h1{{margin-bottom:5px}}.sub{{color:#475569}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:24px 0}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:18px;box-shadow:0 4px 16px #0f172a0a}}.card span{{display:block;color:#64748b;font-size:13px;text-transform:uppercase}}.card strong{{display:block;font-size:26px;margin-top:7px}}
.status{{display:inline-block;background:{color};color:white;border-radius:999px;padding:8px 14px;font-weight:700}}.panel{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:10px;margin-top:18px}}
.note{{background:#fff7ed;border-left:4px solid #f59e0b;border-radius:8px;padding:14px 18px;margin-top:18px}}
table{{border-collapse:collapse;width:100%;margin-top:18px}}th,td{{border-bottom:1px solid #e2e8f0;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head>
<body><main><h1>Phase 2d · Latency confirmation</h1><div class='sub'>Independent Slurm allocations · randomized/interleaved order · cluster-aware uncertainty</div>
<div class='cards'>{cards}</div><span class='status'>{status.replace("_", " ")}</span>
<div class='note'>This report confirms measurement behavior only. It does not alter the frozen Phase 2c retain-final decision.</div>
<div class='panel'>{plot}</div><div class='panel'><h2>Arithmetic-only paths on identical cached features</h2>
<table><thead><tr><th>Path</th><th>Wall p50</th><th>Wall p90</th><th>Wall p95</th><th>CUDA p50</th></tr></thead>
<tbody>{head_rows}</tbody></table></div><div class='panel'><h2>Independent Slurm allocations and telemetry</h2>
<table><thead><tr><th>Replicate</th><th>Job ID</th><th>Partition</th><th>GPU</th><th>Peak GiB</th>
<th>Util %</th><th>Max °C</th><th>Mean W</th><th>Mean SM MHz</th></tr></thead><tbody>{job_rows}</tbody></table>
</div></main></body></html>"""
    )


def main(
    input_root: Annotated[Path, typer.Option("--input-root")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    expected_replicates: Annotated[int, typer.Option("--expected-replicates")] = 12,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_entity: Annotated[str | None, typer.Option("--wandb-entity")] = None,
    wandb_enabled: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    if expected_replicates != len(EXPECTED_REPLICATES):
        raise typer.BadParameter("Phase 2d aggregation is frozen to exactly 12 independent replicates (IDs 0..11)")
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records, input_identities, common_source = validate_latency_inputs(input_root)
    rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for record in records:
        for row in record["rows"]:
            rows.append({**row, "replicate": int(record["replicate"]), "gpu_uuid": record["gpu_uuid"]})
        final = record["summary"]["e2e/final_deployment"]["wall_median_ms"]
        learned = record["summary"]["e2e/learned_deployment"]["wall_median_ms"]
        replicate_rows.append(
            {
                "replicate": int(record["replicate"]),
                "slurm_job_id": record["slurm"]["job_id"],
                "slurm_partition": record["slurm"]["partition"],
                "gpu_uuid": record["gpu_uuid"],
                "gpu_name": record["gpu_name"],
                "peak_cuda_memory_gb": record["peak_cuda_memory_gb"],
                "peak_cuda_reserved_memory_gb": record["peak_cuda_reserved_memory_gb"],
                "gpu_utilization_mean_pct": record["gpu_telemetry"]["statistics"]["utilization_gpu_pct"]["mean"],
                "gpu_memory_used_mean_mib": record["gpu_telemetry"]["statistics"]["memory_used_mib"]["mean"],
                "gpu_temperature_max_c": record["gpu_telemetry"]["statistics"]["temperature_c"]["max"],
                "gpu_power_mean_w": record["gpu_telemetry"]["statistics"]["power_w"]["mean"],
                "gpu_clocks_sm_mean_mhz": record["gpu_telemetry"]["statistics"]["clocks_sm_mhz"]["mean"],
                "final_wall_median_ms": final,
                "learned_wall_median_ms": learned,
                "learned_to_final_wall_ratio": learned / final,
            }
        )
    ratios = [row["learned_to_final_wall_ratio"] for row in replicate_rows]
    low, high = _bootstrap_interval(ratios)
    aggregate = {variant: _summarize_values(rows, "e2e", variant) for variant in E2E_VARIANTS}
    head_aggregate = {variant: _summarize_values(rows, "head_only", variant) for variant in HEAD_VARIANTS}
    systems = {
        "slurm_job_ids": [row["slurm_job_id"] for row in sorted(replicate_rows, key=lambda value: value["replicate"])],
        "test_receipt": input_identities[0]["test_receipt"],
        "peak_cuda_memory_gb": {
            "mean": float(statistics.fmean(float(row["peak_cuda_memory_gb"]) for row in replicate_rows)),
            "max": float(max(float(row["peak_cuda_memory_gb"]) for row in replicate_rows)),
        },
        "peak_cuda_reserved_memory_gb": {
            "mean": float(statistics.fmean(float(row["peak_cuda_reserved_memory_gb"]) for row in replicate_rows)),
            "max": float(max(float(row["peak_cuda_reserved_memory_gb"]) for row in replicate_rows)),
        },
        "telemetry": {
            "gpu_utilization_mean_pct": float(
                statistics.fmean(float(row["gpu_utilization_mean_pct"]) for row in replicate_rows)
            ),
            "gpu_memory_used_mean_mib": float(
                statistics.fmean(float(row["gpu_memory_used_mean_mib"]) for row in replicate_rows)
            ),
            "gpu_temperature_max_c": float(max(float(row["gpu_temperature_max_c"]) for row in replicate_rows)),
            "gpu_power_mean_w": float(statistics.fmean(float(row["gpu_power_mean_w"]) for row in replicate_rows)),
            "gpu_clocks_sm_mean_mhz": float(
                statistics.fmean(float(row["gpu_clocks_sm_mean_mhz"]) for row in replicate_rows)
            ),
        },
    }
    payload = {
        "schema_version": "jepa4d-phase2d-latency-aggregate-v1",
        "status": "complete",
        "protocol": {
            "replicate_ids": list(EXPECTED_REPLICATES),
            "warmups_per_path": EXPECTED_WARMUPS,
            "blocks": EXPECTED_BLOCKS,
            "iterations_per_block": EXPECTED_ITERATIONS,
            "e2e_variants": list(E2E_VARIANTS),
            "head_only_variants": list(HEAD_VARIANTS),
        },
        "replicate_count": len(records),
        "e2e_block_count": sum(row["scope"] == "e2e" for row in rows),
        "head_only_block_count": sum(row["scope"] == "head_only" for row in rows),
        "gpu_uuids": sorted({str(row["gpu_uuid"]) for row in replicate_rows}),
        "source_identity": common_source,
        "systems": systems,
        "inputs": input_identities,
        "replicates": sorted(replicate_rows, key=lambda row: row["replicate"]),
        "aggregate": aggregate,
        "head_only_aggregate": head_aggregate,
        "ratio_aggregate": {
            "mean": float(statistics.fmean(ratios)),
            "median": float(statistics.median(ratios)),
            "ci95_low": low,
            "ci95_high": high,
            "confirmation_status": "within_1.10x" if high <= 1.10 else "exceeds_or_uncertain_1.10x",
        },
        "rows": rows,
        "claim_boundary": "profiling confirmation only; independent job is the resampling unit",
    }
    (output / "latency_aggregate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _render(output / "latency_aggregate_report.html", payload)
    if wandb_enabled:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name="phase2d-latency-aggregate",
            job_type="profiling-aggregate",
            mode="online",
            config={
                "expected_replicates": expected_replicates,
                "bootstrap_resamples": 10_000,
                "phase2c_git_commit": common_source["phase2c_git_commit"],
                "split_hash": common_source["split_hash"],
                "test_receipt_sha256": systems["test_receipt"]["sha256"],
                "slurm_job_ids": systems["slurm_job_ids"],
            },
            tags=["phase-2d", "latency", "aggregate", "cluster-bootstrap"],
        )
        try:
            table = wandb.Table(columns=list(replicate_rows[0]))
            for row in replicate_rows:
                table.add_data(*row.values())
            head_table = wandb.Table(columns=["variant", "wall_p50_ms", "wall_p90_ms", "wall_p95_ms", "cuda_p50_ms"])
            for variant in HEAD_VARIANTS:
                value = head_aggregate[variant]
                head_table.add_data(
                    variant,
                    value["wall_median_ms"],
                    value["wall_p90_ms"],
                    value["wall_p95_ms"],
                    value["cuda_median_ms"],
                )
            run.log(
                {
                    "latency/replicate_summary": table,
                    "latency/head_only_summary": head_table,
                    "latency/report": wandb.Html(str(output / "latency_aggregate_report.html"), inject=False),
                }
            )
            run.summary.update({f"latency/{key}": value for key, value in payload["ratio_aggregate"].items()})
            run.summary["runtime/peak_cuda_memory_max_gb"] = systems["peak_cuda_memory_gb"]["max"]
            run.summary["runtime/peak_cuda_reserved_memory_max_gb"] = systems["peak_cuda_reserved_memory_gb"]["max"]
            for key, value in systems["telemetry"].items():
                run.summary[f"runtime/{key}"] = value
            artifact = wandb.Artifact(f"{run.id}-phase2d-latency-aggregate", type="phase2d-report")
            artifact.add_file(str(output / "latency_aggregate.json"))
            artifact.add_file(str(output / "latency_aggregate_report.html"))
            uploaded = run.log_artifact(artifact).wait(timeout=900)
            run.summary["result"] = "success"
            uploaded_files = {
                name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
                for name in ("latency_aggregate.json", "latency_aggregate_report.html")
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
                        "test_receipt_sha256": systems["test_receipt"]["sha256"],
                        "slurm_job_ids": systems["slurm_job_ids"],
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
