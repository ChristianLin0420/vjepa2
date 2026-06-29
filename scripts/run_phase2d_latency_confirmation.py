"""Randomized, paired latency confirmation for Phase 2c frozen checkpoints.

The script intentionally performs no training and never changes the registered
Phase 2c decision.  It separates deployment-path cost (encoder capture plus
probe) from arithmetic-only cost on an identical precomputed feature tensor.
"""

from __future__ import annotations

import copy
import csv
import html as html_module
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import plotly.graph_objects as go
import torch
import typer
import wandb
from plotly.subplots import make_subplots

from jepa4d.benchmarks.geometry.tum_rgbd_bundle import load_cross_sequence_bundle
from jepa4d.evaluation.phase2c_source import sha256_file, validate_phase2c_source
from jepa4d.models.geometry_student import DenseGeometryProbe, ResidualFusionGeometryProbe
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from scripts.run_phase2b_geometry_distillation import _single_image_batch
from slurm.validate_phase2d_test_receipt import validate_receipt

GPU_TELEMETRY_FIELDS = {
    "utilization_gpu_pct": ("utilization.gpu",),
    "memory_used_mib": ("memory.used",),
    "temperature_c": ("temperature.gpu",),
    "power_w": ("power.draw",),
    "clocks_sm_mhz": ("clocks.current.sm", "clocks.sm"),
}
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


@dataclass(frozen=True, slots=True)
class TimingRow:
    scope: str
    block: int
    order: int
    variant: str
    iterations: int
    wall_ms_per_frame: float
    cuda_ms_per_frame: float
    sample_offset: int


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _slurm_identity(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the allocation identity that makes one replicate an independent job."""
    values = os.environ if environment is None else environment
    mapping = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION",
        "nodelist": "SLURM_JOB_NODELIST",
    }
    identity = {name: str(values.get(variable, "")).strip() for name, variable in mapping.items()}
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise RuntimeError(f"Phase 2d latency requires a complete Slurm allocation identity; missing {missing}")
    return identity


def _test_receipt_identity(path: Path, repo_root: Path) -> dict[str, Any]:
    """Revalidate and reduce the passing receipt to a durable content identity."""
    resolved = path.resolve(strict=True)
    receipt = validate_receipt(repo_root, resolved)
    slurm = receipt.get("slurm")
    if not isinstance(slurm, dict):
        raise RuntimeError("passing test receipt has no Slurm identity")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "git_commit": str(receipt["git_commit"]),
        "test_job_id": str(slurm["SLURM_JOB_ID"]),
    }


def _numeric(value: Any) -> float | None:
    match = _NUMBER.search(str(value))
    return None if match is None else float(match.group())


def _telemetry_value(row: Mapping[str, Any], prefixes: tuple[str, ...]) -> float | None:
    for name, value in row.items():
        normalized = str(name).strip().lower()
        if any(normalized.startswith(prefix) for prefix in prefixes):
            return _numeric(value)
    return None


def _summarize_gpu_telemetry(path: Path) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Parse nvidia-smi CSV without depending on driver-specific unit suffixes."""
    with path.open(newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise RuntimeError("Phase 2d GPU telemetry has no monitor samples")
    rows: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {name: [] for name in GPU_TELEMETRY_FIELDS}
    for raw in raw_rows:
        row: dict[str, Any] = {
            "timestamp": str(raw.get("timestamp", "")).strip(),
            "gpu_index": _telemetry_value(raw, ("index",)),
            "gpu_uuid": str(raw.get(" uuid", raw.get("uuid", ""))).strip(),
            "gpu_name": str(raw.get(" name", raw.get("name", ""))).strip(),
            "pstate": str(raw.get(" pstate", raw.get("pstate", ""))).strip(),
        }
        for name, prefixes in GPU_TELEMETRY_FIELDS.items():
            value = _telemetry_value(raw, prefixes)
            row[name] = value
            if value is not None and math.isfinite(value):
                values[name].append(value)
        rows.append(row)
    missing = [name for name, observed in values.items() if not observed]
    if missing:
        raise RuntimeError(f"Phase 2d GPU telemetry lacks required numeric fields: {missing}")
    statistics_payload = {
        name: {
            "mean": float(statistics.fmean(observed)),
            "min": float(min(observed)),
            "max": float(max(observed)),
            "p50": _percentile(observed, 50),
            "p95": _percentile(observed, 95),
        }
        for name, observed in values.items()
    }
    return statistics_payload, rows


def _snapshot_gpu_telemetry(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log_directory = os.getenv("JEPA4D_JOB_LOG_DIR")
    if not log_directory:
        raise RuntimeError("JEPA4D_JOB_LOG_DIR is required for Phase 2d GPU telemetry")
    source = Path(log_directory) / "gpu-telemetry.csv"
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"Slurm GPU monitor output is absent or empty: {source}")
    destination = output / "gpu_telemetry.csv"
    shutil.copyfile(source, destination)
    statistics_payload, rows = _summarize_gpu_telemetry(destination)
    return (
        {
            "path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "sample_count": len(rows),
            "statistics": statistics_payload,
        },
        rows,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty latency list")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summary(rows: list[TimingRow]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for scope in sorted({row.scope for row in rows}):
        for variant in sorted({row.variant for row in rows if row.scope == scope}):
            selected = [row for row in rows if row.scope == scope and row.variant == variant]
            wall = [row.wall_ms_per_frame for row in selected]
            cuda = [row.cuda_ms_per_frame for row in selected]
            result[f"{scope}/{variant}"] = {
                "blocks": float(len(selected)),
                "iterations": float(sum(row.iterations for row in selected)),
                "wall_mean_ms": float(statistics.fmean(wall)),
                "wall_median_ms": float(statistics.median(wall)),
                "wall_p90_ms": _percentile(wall, 90),
                "wall_p95_ms": _percentile(wall, 95),
                "cuda_mean_ms": float(statistics.fmean(cuda)),
                "cuda_median_ms": float(statistics.median(cuda)),
                "cuda_p90_ms": _percentile(cuda, 90),
                "cuda_p95_ms": _percentile(cuda, 95),
            }
    final = result.get("e2e/final_deployment")
    if final is not None:
        for key, values in list(result.items()):
            if not key.startswith("e2e/") or key == "e2e/final_deployment":
                continue
            values["paired_wall_median_ratio_to_final"] = values["wall_median_ms"] / final["wall_median_ms"]
            values["paired_cuda_median_ratio_to_final"] = values["cuda_median_ms"] / final["cuda_median_ms"]
    return result


def _time_block(
    callback: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    iterations: int,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter_ns()
    start_event.record()
    output: tuple[torch.Tensor, torch.Tensor] | None = None
    for index in range(iterations):
        output = callback(index)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_started) / 1e6 / iterations
    cuda_ms = start_event.elapsed_time(end_event) / iterations
    if output is None or not all(torch.isfinite(value).all() for value in output):
        raise RuntimeError("latency callback returned missing or non-finite output")
    return wall_ms, cuda_ms


def _load_models(checkpoint_root: Path, device: str) -> dict[str, torch.nn.Module]:
    mapping = {
        "final": ("vjepa_final-seed0.pt", DenseGeometryProbe),
        "fixed": ("vjepa_multilayer-seed0.pt", DenseGeometryProbe),
        "learned": ("vjepa_learned_fusion-seed0.pt", ResidualFusionGeometryProbe),
    }
    models: dict[str, torch.nn.Module] = {}
    for name, (filename, model_type) in mapping.items():
        payload = torch.load(checkpoint_root / filename, map_location="cpu", weights_only=True)
        model = model_type(int(payload["input_dim"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().to(device)
        models[name] = model
    for name, gate in (("learned_zero", 0.0), ("learned_fixed_equivalent", math.atanh(0.75))):
        model = copy.deepcopy(models["learned"])
        assert isinstance(model, ResidualFusionGeometryProbe)
        with torch.no_grad():
            model.fusion.raw_gates.fill_(gate)
        models[name] = model
    return models


def _render_report(
    path: Path,
    rows: list[TimingRow],
    summary: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> None:
    e2e = {key.split("/", 1)[1]: value for key, value in summary.items() if key.startswith("e2e/")}
    variants = sorted(e2e)
    colors = {
        "final_deployment": "#2563eb",
        "final_capture_all": "#60a5fa",
        "fixed_deployment": "#f59e0b",
        "learned_deployment": "#ef4444",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"colspan": 2}, None]],
        subplot_titles=("Median end-to-end latency", "Tail latency (p90)", "Randomized block trace"),
        vertical_spacing=0.18,
    )
    figure.add_trace(
        go.Bar(
            x=variants,
            y=[e2e[name]["wall_median_ms"] for name in variants],
            marker_color=[colors.get(name, "#64748b") for name in variants],
            text=[f"{e2e[name]['wall_median_ms']:.2f} ms" for name in variants],
            textposition="outside",
            name="wall median",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=variants,
            y=[e2e[name]["wall_p90_ms"] for name in variants],
            marker_color=[colors.get(name, "#64748b") for name in variants],
            text=[f"{e2e[name]['wall_p90_ms']:.2f} ms" for name in variants],
            textposition="outside",
            name="wall p90",
        ),
        row=1,
        col=2,
    )
    for variant in variants:
        selected = [row for row in rows if row.scope == "e2e" and row.variant == variant]
        figure.add_trace(
            go.Scatter(
                x=[row.block for row in selected],
                y=[row.wall_ms_per_frame for row in selected],
                mode="lines+markers",
                marker={"size": 5},
                line={"color": colors.get(variant, "#64748b")},
                name=variant,
                hovertemplate="block=%{x}<br>wall=%{y:.3f} ms<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.update_layout(
        template="plotly_white",
        height=860,
        margin={"l": 55, "r": 25, "t": 90, "b": 50},
        legend={"orientation": "h", "y": -0.10},
        font={"family": "Inter, system-ui, sans-serif"},
    )
    figure.update_yaxes(title_text="ms / frame", rangemode="tozero")
    final = e2e["final_deployment"]["wall_median_ms"]
    learned = e2e["learned_deployment"]["wall_median_ms"]
    ratio = learned / final
    badge = "PASS" if ratio <= 1.10 else "FAIL"
    badge_color = "#15803d" if badge == "PASS" else "#b91c1c"
    telemetry = metadata.get("gpu_telemetry", {}).get("statistics", {})
    utilization = telemetry.get("utilization_gpu_pct", {})
    temperature = telemetry.get("temperature_c", {})
    power = telemetry.get("power_w", {})
    clocks = telemetry.get("clocks_sm_mhz", {})
    cards = "".join(
        f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div></div>"
        for label, value in (
            ("Replicate", str(metadata["replicate"])),
            ("GPU", str(metadata["gpu_name"])),
            ("Final median", f"{final:.2f} ms"),
            ("Learned / final", f"{ratio:.3f}×"),
            ("Peak CUDA memory", f"{float(metadata.get('peak_cuda_memory_gb', float('nan'))):.2f} GiB"),
            ("GPU utilization p95", f"{float(utilization.get('p95', float('nan'))):.1f}%"),
        )
    )
    telemetry_rows = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>"
        for label, value in (
            ("Slurm job", html_module.escape(str(metadata.get("slurm", {}).get("job_id", "unknown")))),
            (
                "GPU utilization mean / p95",
                f"{utilization.get('mean', float('nan')):.1f}% / {utilization.get('p95', float('nan')):.1f}%",
            ),
            (
                "Temperature mean / max",
                f"{temperature.get('mean', float('nan')):.1f} / {temperature.get('max', float('nan')):.1f} °C",
            ),
            ("Power mean / max", f"{power.get('mean', float('nan')):.1f} / {power.get('max', float('nan')):.1f} W"),
            (
                "SM clock mean / p95",
                f"{clocks.get('mean', float('nan')):.1f} / {clocks.get('p95', float('nan')):.1f} MHz",
            ),
            ("Monitor samples", str(metadata.get("gpu_telemetry", {}).get("sample_count", 0))),
        )
    )
    plot = figure.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>JEPA-4D Phase 2d latency</title>
<style>
body{{margin:0;background:#f8fafc;color:#0f172a;font-family:Inter,system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:32px}} h1{{margin-bottom:4px}} .sub{{color:#475569}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px;box-shadow:0 4px 16px #0f172a0a}}
.label{{font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}} .value{{font-size:27px;font-weight:700;margin-top:5px}}
.badge{{display:inline-block;background:{badge_color};color:white;padding:7px 12px;border-radius:999px;font-weight:700}}
	.panel{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:10px;margin-top:16px}}
	.note{{background:#eff6ff;border-left:4px solid #2563eb;padding:14px 18px;border-radius:8px;margin-top:18px}}
	table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e2e8f0;padding:9px;text-align:left}}
	</style></head><body><main><h1>Phase 2d · Paired latency confirmation</h1>
<div class='sub'>Randomized/interleaved blocks · synchronized wall clock + CUDA events · frozen checkpoints</div>
	<div class='cards'>{cards}</div><span class='badge'>{badge} ≤1.10× gate</span>
	<div class='note'>This profiling-only confirmation does not retroactively change the Phase 2c decision. Deployment and capture-all arithmetic paths are reported separately.</div>
	<div class='panel'>{plot}</div><div class='panel'><h2>Allocation and GPU telemetry</h2><table>{telemetry_rows}</table></div></main></body></html>"""
    path.write_text(html, encoding="utf-8")


def main(
    phase2c_output: Annotated[Path, typer.Option("--phase2c-output")],
    dataset_parent: Annotated[Path, typer.Option("--dataset-parent")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    vjepa_checkpoint: Annotated[Path, typer.Option("--vjepa-checkpoint")],
    vjepa_implementation: Annotated[Path, typer.Option("--vjepa-implementation")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    test_receipt: Annotated[Path, typer.Option("--test-receipt", exists=True, dir_okay=False)],
    replicate: Annotated[int, typer.Option("--replicate")] = 0,
    warmups: Annotated[int, typer.Option("--warmups")] = 30,
    blocks: Annotated[int, typer.Option("--blocks")] = 30,
    iterations: Annotated[int, typer.Option("--iterations")] = 100,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_entity: Annotated[str | None, typer.Option("--wandb-entity")] = None,
    wandb_enabled: Annotated[bool, typer.Option("--wandb/--no-wandb")] = True,
) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise typer.BadParameter("Phase 2d latency confirmation requires an allocated CUDA device")
    if min(warmups, blocks, iterations) <= 0:
        raise typer.BadParameter("warmups, blocks, and iterations must all be positive")
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter(f"output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    slurm_identity = _slurm_identity()
    test_receipt_identity = _test_receipt_identity(test_receipt, repo_root)

    source_identity = validate_phase2c_source(
        phase2c_output,
        dataset_manifest=manifest,
        vjepa_checkpoint=vjepa_checkpoint,
        vjepa_implementation=vjepa_implementation,
    )
    source_identity_path = output / "source_identity.json"
    _json(source_identity_path, source_identity)

    seed = 20260629 + replicate * 1009
    rng = random.Random(seed)
    torch.manual_seed(seed)
    bundle = load_cross_sequence_bundle(dataset_parent, manifest)
    test_samples = bundle.splits["test"]
    batches = [_single_image_batch([sample]) for sample in test_samples[:: max(1, len(test_samples) // 16)][:16]]
    if not batches:
        raise RuntimeError("no test batches were loaded")

    checkpoint_root = phase2c_output / "checkpoints"
    models = _load_models(checkpoint_root, device)
    statistics = torch.load(
        phase2c_output / "vjepa_learned_fusion-normalization.pt", map_location="cpu", weights_only=True
    )
    device_statistics = {
        key: {name: tensor.to(device) for name, tensor in values.items()} for key, values in statistics.items()
    }
    extractor = VJEPA21FeatureExtractor(
        checkpoint=vjepa_checkpoint,
        implementation_path=vjepa_implementation,
        backend="hf_compat",
        device=device,
        capture_layers=(),
    )

    def normalized_grid(bundle_value: Any, key: str, layer: int | None = None) -> torch.Tensor:
        tokens = bundle_value.dense_tokens if layer is None else bundle_value.layer_tokens[layer]
        grid = tokens[:, 0, 0].reshape(1, 24, 24, -1).permute(0, 3, 1, 2).contiguous()
        values = device_statistics[key]
        return ((grid.float() - values["mean"]) / values["std"]).half()

    def features_for(batch: Any, capture_all: bool) -> tuple[torch.Tensor, torch.Tensor]:
        extractor.capture_layers = (2, 5, 8) if capture_all else ()
        bundle_value = extractor(batch)
        final = normalized_grid(bundle_value, "vjepa_final")
        if not capture_all:
            empty = torch.empty(0, device=device, dtype=final.dtype)
            return final, empty
        layers = [normalized_grid(bundle_value, f"vjepa_layer_{layer}", layer) for layer in (2, 5, 8)]
        return final, torch.stack(layers, dim=1)

    def e2e_forward(variant: str, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = batches[batch_index % len(batches)]
        capture_all = variant != "final_deployment"
        final, layers = features_for(batch, capture_all)
        if variant in {"final_deployment", "final_capture_all"}:
            return models["final"](final)
        if variant == "fixed_deployment":
            return models["fixed"](torch.cat((final.unsqueeze(1), layers), dim=1).mean(dim=1))
        if variant == "learned_deployment":
            return models["learned"](torch.cat((final.unsqueeze(1), layers), dim=1))
        raise KeyError(variant)

    with torch.inference_mode():
        reference_final, reference_layers = features_for(batches[0], True)
        stacked_reference = torch.cat((reference_final.unsqueeze(1), reference_layers), dim=1)

    head_callbacks: dict[str, Callable[[int], tuple[torch.Tensor, torch.Tensor]]] = {
        "final_head": lambda _: models["final"](reference_final),
        "fixed_head": lambda _: models["fixed"](stacked_reference.mean(dim=1)),
        "learned_head": lambda _: models["learned"](stacked_reference),
        "zero_gate_same_head": lambda _: models["learned_zero"](stacked_reference),
        "fixed_equivalent_same_head": lambda _: models["learned_fixed_equivalent"](stacked_reference),
    }
    e2e_variants = ["final_deployment", "final_capture_all", "fixed_deployment", "learned_deployment"]

    with torch.inference_mode():
        for variant in e2e_variants:
            for index in range(warmups):
                e2e_forward(variant, index)
        for callback in head_callbacks.values():
            for index in range(warmups):
                callback(index)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    run = None
    if wandb_enabled:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"phase2d-latency-r{replicate:02d}",
            group="phase2d-latency-confirmation",
            job_type="profiling",
            mode="online",
            config={
                "replicate": replicate,
                "seed": seed,
                "warmups_per_path": warmups,
                "blocks": blocks,
                "iterations_per_block": iterations,
                "split_hash": bundle.split_hash,
                "phase2c_output": str(phase2c_output.resolve()),
                "phase2c_git_commit": source_identity["phase2c_git_commit"],
                "phase2c_source_identity_sha256": sha256_file(source_identity_path),
                "slurm_job_id": slurm_identity["job_id"],
                "test_receipt_sha256": test_receipt_identity["sha256"],
            },
            tags=["phase-2d", "latency", "paired", "slurm", "frozen-checkpoint"],
        )
        if run.offline:
            raise RuntimeError("Phase 2d latency confirmation requires online W&B")

    rows: list[TimingRow] = []
    try:
        with torch.inference_mode():
            for block in range(blocks):
                order = e2e_variants.copy()
                rng.shuffle(order)
                for position, variant in enumerate(order):
                    offset = rng.randrange(len(batches))

                    def timed_e2e(
                        index: int, name: str = variant, start: int = offset
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return e2e_forward(name, start + index)

                    wall_ms, cuda_ms = _time_block(timed_e2e, iterations)
                    row = TimingRow("e2e", block, position, variant, iterations, wall_ms, cuda_ms, offset)
                    rows.append(row)
                    if run is not None:
                        run.log(
                            {
                                f"latency/e2e/{variant}/wall_ms": wall_ms,
                                f"latency/e2e/{variant}/cuda_ms": cuda_ms,
                                "schedule/block": block,
                                "schedule/position": position,
                            }
                        )
            for block in range(blocks):
                order = list(head_callbacks)
                rng.shuffle(order)
                for position, variant in enumerate(order):
                    wall_ms, cuda_ms = _time_block(head_callbacks[variant], iterations)
                    row = TimingRow("head_only", block, position, variant, iterations, wall_ms, cuda_ms, 0)
                    rows.append(row)
                    if run is not None:
                        run.log(
                            {
                                f"latency/head_only/{variant}/wall_ms": wall_ms,
                                f"latency/head_only/{variant}/cuda_ms": cuda_ms,
                            }
                        )

        summary = _summary(rows)
        torch.cuda.synchronize()
        peak_cuda_memory_gb = float(torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3)
        peak_cuda_reserved_memory_gb = float(torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3)
        gpu_telemetry, gpu_telemetry_rows = _snapshot_gpu_telemetry(output)
        metadata = {
            "schema_version": "jepa4d-phase2d-latency-replicate-v1",
            "replicate": replicate,
            "seed": seed,
            "split_hash": bundle.split_hash,
            "source_identity": source_identity,
            "source_identity_sha256": sha256_file(source_identity_path),
            "slurm": slurm_identity,
            "test_receipt": test_receipt_identity,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "warmups_per_path": warmups,
            "blocks": blocks,
            "iterations_per_block": iterations,
            "peak_cuda_memory_gb": peak_cuda_memory_gb,
            "peak_cuda_reserved_memory_gb": peak_cuda_reserved_memory_gb,
            "gpu_telemetry": gpu_telemetry,
            "decision_scope": "profiling-only; Phase 2c retain_final_layer remains frozen",
        }
        payload = {**metadata, "summary": summary, "rows": [asdict(row) for row in rows]}
        _json(output / "latency.json", payload)
        with (output / "latency_rows.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        _render_report(output / "latency_report.html", rows, summary, metadata)
        if run is not None:
            table = wandb.Table(columns=list(asdict(rows[0])))
            for row in rows:
                table.add_data(*asdict(row).values())
            telemetry_table = wandb.Table(columns=list(gpu_telemetry_rows[0]))
            for telemetry_row in gpu_telemetry_rows:
                telemetry_table.add_data(*telemetry_row.values())
            run.log(
                {
                    "latency/raw_blocks": table,
                    "runtime/gpu_telemetry": telemetry_table,
                    "latency/report": wandb.Html(str(output / "latency_report.html"), inject=False),
                }
            )
            for key, values in summary.items():
                run.summary.update({f"summary/{key}/{name}": value for name, value in values.items()})
            run.summary["runtime/peak_cuda_memory_gb"] = peak_cuda_memory_gb
            run.summary["runtime/peak_cuda_reserved_memory_gb"] = peak_cuda_reserved_memory_gb
            for metric, values in gpu_telemetry["statistics"].items():
                for statistic, value in values.items():
                    run.summary[f"runtime/{metric}/{statistic}"] = value
            artifact = wandb.Artifact(f"{run.id}-phase2d-latency", type="phase2d-latency-replicate")
            artifact.add_file(str(output / "latency.json"))
            artifact.add_file(str(output / "latency_rows.csv"))
            artifact.add_file(str(output / "latency_report.html"))
            artifact.add_file(str(source_identity_path))
            artifact.add_file(str(output / "gpu_telemetry.csv"))
            uploaded = run.log_artifact(artifact).wait(timeout=900)
            run.summary["result"] = "success"
            uploaded_files = {
                name: {
                    "bytes": (output / name).stat().st_size,
                    "sha256": sha256_file(output / name),
                }
                for name in (
                    "latency.json",
                    "latency_rows.csv",
                    "latency_report.html",
                    "source_identity.json",
                    "gpu_telemetry.csv",
                )
            }
            _json(
                output / "wandb_receipt.json",
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
                    "slurm_job_id": slurm_identity["job_id"],
                    "test_receipt_sha256": test_receipt_identity["sha256"],
                    "uploaded_files": uploaded_files,
                },
            )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    typer.run(main)
