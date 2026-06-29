"""Offline diagnostics for Phase-2b geometry-student comparisons.

The report deliberately keeps accuracy, latency, memory, and each training
objective on separate axes.  It accepts the stable comparison record plus
optional locally persisted epoch/per-frame rows, so useful diagnostics do not
depend on W&B being reachable after a Slurm job finishes.
"""

from __future__ import annotations

import html
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jepa4d.evaluation.comparison import ComparisonRecord

_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
_DASHES = ("solid", "dash", "dot", "dashdot")
_HISTORY_METRICS: dict[str, tuple[str, ...]] = {
    "loss": ("loss", "train_loss", "objective"),
    "validation_abs_rel": ("validation_abs_rel", "validation_metric_abs_rel", "val_abs_rel"),
    "nll": ("nll", "train_nll"),
    "distillation": ("distillation", "distillation_loss"),
    "scale_invariant": ("scale_invariant", "scale_invariant_loss"),
    "gradient": ("gradient", "gradient_loss"),
}


@dataclass(frozen=True, slots=True)
class GeometryStudentReportArtifacts:
    """Files and non-fatal diagnostics produced by the report builder."""

    html_path: Path
    png_path: Path | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _VariantSummary:
    variant_id: str
    family: str
    role: str
    seed_count: int
    primary_values: tuple[float, ...]
    primary_mean: float | None
    primary_std: float | None
    runtime_ms: float | None
    memory_gib: float | None
    parameters: float | None
    raw_nll: float | None
    calibrated_nll: float | None
    variance_multiplier: float | None


@dataclass(frozen=True, slots=True)
class _DepthGrid:
    label: str
    frame_index: int
    abs_rel: float
    prediction: np.ndarray
    target: np.ndarray
    relative_error: np.ndarray


RecordInput = ComparisonRecord | Mapping[str, Any] | str | Path
RowsInput = Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | Path | None


def write_phase2b_report(
    output: Path,
    comparison: dict[str, Any],
    histories: list[dict[str, Any]],
    diagnostics: dict[str, str] | None = None,
) -> Path:
    """Runner-friendly wrapper that writes the canonical local report path.

    Diagnostic artifact paths are retained in the embedded provenance.  A JSON
    or JSONL path under ``per_frame_predictions``, ``per_frame_metrics``, or
    ``frame_metrics`` is also rendered as the optional frame-error panel.
    """

    record = dict(comparison)
    per_frame: str | Path | None = None
    if diagnostics:
        record["diagnostic_artifacts"] = dict(diagnostics)
        for key in ("per_frame_predictions", "per_frame_metrics", "frame_metrics"):
            candidate = diagnostics.get(key)
            if candidate is not None and Path(candidate).is_file():
                per_frame = candidate
                break
    artifacts = build_geometry_student_report(
        record,
        output / "geometry_student_report.html",
        training_history=histories,
        per_frame_predictions=per_frame,
        diagnostic_npz=diagnostics,
        static_png=output / "geometry_student_report.png",
    )
    return artifacts.html_path


def build_geometry_student_report(
    comparison: RecordInput,
    output: str | Path,
    *,
    training_history: RowsInput = None,
    per_frame_predictions: RowsInput = None,
    diagnostic_npz: Mapping[str, str | Path] | None = None,
    primary_metric: str | None = None,
    static_png: bool | str | Path = True,
) -> GeometryStudentReportArtifacts:
    """Build a self-contained Phase-2b HTML report and an optional summary PNG.

    ``comparison`` may be a :class:`ComparisonRecord`, its serialized mapping,
    or a JSON path.  Optional row inputs may be iterables, JSON/JSONL paths, or
    ``{"rows": [...]}`` mappings.

    Training rows use ``variant``, ``seed``, ``epoch`` and any of ``loss``,
    ``validation_abs_rel``, ``nll``, ``distillation``, ``scale_invariant`` or
    ``gradient``.  Per-frame rows use ``variant``, ``seed``, ``frame_id`` and a
    metric such as ``metric_abs_rel``.  They may instead contain
    ``predicted_depth`` and ``target_depth`` arrays; AbsRel is then derived for
    the visualization without embedding those dense arrays in the report.
    ``diagnostic_npz`` maps run labels to archives containing ``prediction_m``
    and ``target_m`` stacks.  These produce bounded-resolution depth/error
    grids and per-frame AbsRel traces without loading pickled data.

    Plotly JavaScript is embedded inline.  PNG export is attempted through
    Plotly's configured image backend (normally Kaleido); a missing backend is
    reported as a non-fatal warning and never prevents the HTML artifact.
    """

    record = _load_record(comparison)
    history_rows = _load_rows(training_history, "training_history")
    frame_rows = _load_rows(per_frame_predictions, "per_frame_predictions")
    html_path = _html_output_path(Path(output))
    html_path.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    variants = _variant_rows(record, warnings)
    selected_metric = _select_primary_metric(record, variants, primary_metric)
    summaries = _summaries(variants, selected_metric)
    colors = {value.variant_id: _COLORS[index % len(_COLORS)] for index, value in enumerate(summaries)}

    _comparison_diagnostics(record, variants, summaries, selected_metric, warnings)
    overview = _overview_figure(summaries, selected_metric, colors)
    training_figure = _training_figure(history_rows, colors, warnings)
    frame_values = _frame_values(frame_rows, selected_metric, warnings) if frame_rows else []
    npz_frame_values, depth_grids = _load_npz_diagnostics(diagnostic_npz, selected_metric, warnings)
    # Canonical per-frame rows carry dataset IDs/timestamps. NPZ-derived rows
    # are only a fallback for older runs; always retain NPZ depth grids.
    if not frame_values:
        frame_values.extend(npz_frame_values)
    if not frame_values:
        warnings.append(
            "No per-frame predictions or metrics were supplied; outlier-frame diagnostics are unavailable."
        )
    frame_figure = _frame_figure(frame_values, selected_metric, colors)
    depth_figure = _depth_grid_figure(depth_grids)

    png_path = _export_png(overview, html_path, static_png, warnings)
    figures: list[tuple[str, go.Figure]] = [("overview", overview)]
    if training_figure is not None:
        figures.append(("training", training_figure))
    if frame_figure is not None:
        figures.append(("frame", frame_figure))
    if depth_figure is not None:
        figures.append(("depth", depth_figure))
    figure_html = {
        name: figure.to_html(
            full_html=False,
            include_plotlyjs=index == 0,
            config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "png"}},
        )
        for index, (name, figure) in enumerate(figures)
    }

    html_path.write_text(
        _document(
            html_path=html_path,
            record=record,
            summaries=summaries,
            variants=variants,
            failures=_failure_rows(record),
            history_rows=history_rows,
            frame_values=frame_values,
            primary_metric=selected_metric,
            warnings=warnings,
            overview_html=figure_html["overview"],
            training_html=figure_html.get("training"),
            frame_html=figure_html.get("frame"),
            depth_grid_html=figure_html.get("depth"),
            depth_grid_count=len(depth_grids),
            has_training=training_figure is not None,
            png_path=png_path,
        ),
        encoding="utf-8",
    )
    return GeometryStudentReportArtifacts(html_path, png_path, tuple(warnings))


def _load_record(value: RecordInput) -> dict[str, Any]:
    if isinstance(value, ComparisonRecord):
        return value.to_serializable()
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"comparison JSON must contain an object: {path}")
    return loaded


def _load_rows(value: RowsInput, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    loaded: Any = value
    if isinstance(value, str | Path):
        path = Path(value)
        if path.suffix.lower() == ".jsonl":
            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, Mapping):
        if "rows" not in loaded:
            raise ValueError(f"{name} mapping must contain a 'rows' list")
        loaded = loaded["rows"]
    if isinstance(loaded, str | bytes) or not isinstance(loaded, Iterable):
        raise TypeError(f"{name} must be an iterable of row mappings")
    rows = []
    for index, row in enumerate(loaded):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name}[{index}] must be a mapping")
        rows.append(dict(row))
    return rows


def _html_output_path(value: Path) -> Path:
    if value.suffix.lower() in {".html", ".htm"}:
        return value
    return value / "geometry_student_report.html"


def _variant_rows(record: Mapping[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    raw = record.get("variants", [])
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("comparison variants must be a list")
    rows = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            warnings.append(f"Ignored malformed variant row {index}: expected an object.")
            continue
        row = dict(value)
        row.setdefault("variant_id", f"variant_{index}")
        if not isinstance(row.get("metrics", {}), Mapping):
            warnings.append(f"{row['variant_id']}: metrics were malformed and treated as missing.")
            row["metrics"] = {}
        if not isinstance(row.get("runtime", {}), Mapping):
            warnings.append(f"{row['variant_id']}: runtime was malformed and treated as missing.")
            row["runtime"] = {}
        rows.append(row)
    return rows


def _select_primary_metric(
    record: Mapping[str, Any], variants: Sequence[Mapping[str, Any]], explicit: str | None
) -> str:
    keys = {
        str(key)
        for value in variants
        for key in value.get("metrics", {})
        if isinstance(value.get("metrics", {}), Mapping)
    }
    if explicit is not None:
        if keys and explicit not in keys:
            raise ValueError(f"primary metric {explicit!r} is absent from all variant metrics")
        return explicit
    policy = record.get("metric_policy", {})
    policy_primary = str(policy.get("primary", "")) if isinstance(policy, Mapping) else ""
    mentioned = [key for key in keys if key in policy_primary]
    if mentioned:
        return max(mentioned, key=len)
    for candidate in ("metric_abs_rel", "abs_rel", "aligned_abs_rel", "metric_rmse_m", "aligned_rmse_m"):
        if candidate in keys:
            return candidate
    return sorted(keys)[0] if keys else "metric_abs_rel"


def _number(value: Any, *, positive: bool = False) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _sample_std(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _metric(row: Mapping[str, Any], key: str) -> float | None:
    metrics = row.get("metrics", {})
    return _number(metrics.get(key)) if isinstance(metrics, Mapping) else None


def _runtime(row: Mapping[str, Any], key: str, *, positive: bool = False) -> float | None:
    runtime = row.get("runtime", {})
    return _number(runtime.get(key), positive=positive) if isinstance(runtime, Mapping) else None


def _peak_memory(row: Mapping[str, Any]) -> float | None:
    values = [
        value
        for key in ("peak_encoder_memory_gb", "peak_head_memory_gb", "peak_memory_gb")
        if (value := _runtime(row, key, positive=True)) is not None
    ]
    return max(values) if values else None


def _summaries(variants: Sequence[Mapping[str, Any]], primary_metric: str) -> list[_VariantSummary]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in variants:
        grouped[str(row.get("variant_id", "unknown"))].append(row)
    output = []
    for variant_id, rows in grouped.items():
        primary = tuple(value for row in rows if (value := _metric(row, primary_metric)) is not None)
        seeds = {str(row.get("seed")) for row in rows if row.get("seed") is not None}
        parameters = _mean(_number(row.get("parameters")) for row in rows)
        output.append(
            _VariantSummary(
                variant_id=variant_id,
                family=str(rows[0].get("family", "unknown")),
                role=str(rows[0].get("role", "unknown")),
                seed_count=len(seeds),
                primary_values=primary,
                primary_mean=_mean(primary),
                primary_std=_sample_std(primary),
                runtime_ms=_mean(_runtime(row, "total_ms_per_frame", positive=True) for row in rows),
                memory_gib=_mean(_peak_memory(row) for row in rows),
                parameters=parameters,
                raw_nll=_mean(_metric(row, "raw_log_depth_nll") for row in rows),
                calibrated_nll=_mean(_metric(row, "calibrated_log_depth_nll") for row in rows),
                variance_multiplier=_mean(_metric(row, "variance_multiplier") for row in rows),
            )
        )
    return output


def _comparison_diagnostics(
    record: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    summaries: Sequence[_VariantSummary],
    primary_metric: str,
    warnings: list[str],
) -> None:
    failures = _failure_rows(record)
    if failures:
        warnings.append(f"Comparison contains {len(failures)} recorded failure(s); inspect the failure table.")
    if not variants:
        warnings.append("No completed variant rows were available.")
        return
    missing = [value.variant_id for value in summaries if value.primary_mean is None]
    if missing:
        warnings.append(f"Primary metric {primary_metric} is unavailable for: {', '.join(missing)}.")
    missing_runtime = [value.variant_id for value in summaries if value.runtime_ms is None]
    if missing_runtime:
        warnings.append(
            "Non-positive or missing latency was treated as unavailable for: " + ", ".join(missing_runtime) + "."
        )
    missing_memory = [value.variant_id for value in summaries if value.memory_gib is None]
    if missing_memory:
        warnings.append(
            "Non-positive or missing peak memory was treated as unavailable for: " + ", ".join(missing_memory) + "."
        )
    if any(
        _runtime(row, "encoder_ms_per_frame") is not None and _runtime(row, "head_ms_per_frame") is not None
        for row in variants
    ):
        warnings.append(
            "Reported total latency combines encoder per-frame throughput from chunked extraction with batch-1 head "
            "latency. It is not a measured batch-1 end-to-end latency; compare it only under this timing policy."
        )
    alignment_notes = []
    for row in variants:
        notes = row.get("notes", [])
        text = (
            " ".join(str(note) for note in notes)
            if isinstance(notes, Sequence) and not isinstance(notes, str)
            else str(notes)
        )
        if "median align" in text.lower() or "median-align" in text.lower():
            alignment_notes.append(str(row.get("variant_id")))
    if alignment_notes and len(set(alignment_notes)) < len({str(row.get("variant_id")) for row in variants}):
        warnings.append(
            f"Alignment notes differ across variants ({', '.join(sorted(set(alignment_notes)))} are aligned). "
            f"Confirm that {primary_metric} has identical scale semantics before ranking variants."
        )


def _overview_figure(
    summaries: Sequence[_VariantSummary], primary_metric: str, colors: Mapping[str, str]
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{_pretty(primary_metric)} seed variation",
            f"{_pretty(primary_metric)} vs end-to-end latency",
            f"{_pretty(primary_metric)} vs peak encoder memory",
            "Held-out log-depth calibration",
        ),
        horizontal_spacing=0.13,
        vertical_spacing=0.16,
    )
    for value in summaries:
        if value.primary_values:
            figure.add_trace(
                go.Scatter(
                    x=[value.variant_id] * len(value.primary_values),
                    y=list(value.primary_values),
                    mode="markers",
                    marker={"color": colors[value.variant_id], "size": 9, "line": {"color": "white", "width": 1}},
                    name=value.variant_id,
                    legendgroup=value.variant_id,
                    hovertemplate="variant=%{x}<br>value=%{y:.5g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=[value.variant_id],
                    y=[value.primary_mean],
                    mode="markers",
                    marker={"color": colors[value.variant_id], "size": 14, "symbol": "diamond"},
                    error_y={
                        "type": "data",
                        "array": [value.primary_std or 0.0],
                        "visible": value.primary_std is not None,
                    },
                    name=f"{value.variant_id} mean ± sample SD",
                    legendgroup=value.variant_id,
                    showlegend=False,
                    hovertemplate="mean=%{y:.5g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if value.primary_mean is not None and value.runtime_ms is not None:
            figure.add_trace(
                go.Scatter(
                    x=[value.runtime_ms],
                    y=[value.primary_mean],
                    mode="markers+text",
                    text=[value.variant_id],
                    textposition="top center",
                    marker={"color": colors[value.variant_id], "size": 13},
                    error_y={
                        "type": "data",
                        "array": [value.primary_std or 0.0],
                        "visible": value.primary_std is not None,
                    },
                    legendgroup=value.variant_id,
                    showlegend=False,
                    hovertemplate="latency=%{x:.4g} ms/frame<br>metric=%{y:.5g}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        if value.primary_mean is not None and value.memory_gib is not None:
            figure.add_trace(
                go.Scatter(
                    x=[value.memory_gib],
                    y=[value.primary_mean],
                    mode="markers+text",
                    text=[value.variant_id],
                    textposition="top center",
                    marker={"color": colors[value.variant_id], "size": 13},
                    error_y={
                        "type": "data",
                        "array": [value.primary_std or 0.0],
                        "visible": value.primary_std is not None,
                    },
                    legendgroup=value.variant_id,
                    showlegend=False,
                    hovertemplate="memory=%{x:.4g} GiB<br>metric=%{y:.5g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    calibrated = [value for value in summaries if value.raw_nll is not None or value.calibrated_nll is not None]
    if calibrated:
        figure.add_trace(
            go.Bar(
                x=[value.variant_id for value in calibrated],
                y=[value.raw_nll for value in calibrated],
                name="Raw NLL",
                marker_color="#999999",
                hovertemplate="raw NLL=%{y:.5g}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        figure.add_trace(
            go.Bar(
                x=[value.variant_id for value in calibrated],
                y=[value.calibrated_nll for value in calibrated],
                name="Validation-calibrated NLL",
                marker_color="#009E73",
                hovertemplate="calibrated NLL=%{y:.5g}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    figure.update_yaxes(title_text=f"{primary_metric} (lower is better)", row=1, col=1)
    figure.update_xaxes(
        title_text="Reported latency (ms/frame; chunked encoder + batch-1 head)", rangemode="tozero", row=1, col=2
    )
    figure.update_yaxes(title_text=f"{primary_metric} (lower is better)", row=1, col=2)
    figure.update_xaxes(title_text="Peak GPU allocation (GiB; max encoder/head)", rangemode="tozero", row=2, col=1)
    figure.update_yaxes(title_text=f"{primary_metric} (lower is better)", row=2, col=1)
    figure.update_yaxes(title_text="Gaussian NLL (lower is better)", row=2, col=2)
    figure.update_layout(
        barmode="group",
        height=920,
        template="plotly_white",
        title="Phase-2b geometry student: quality and resource trade-offs",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0},
        margin={"t": 135, "b": 70, "l": 80, "r": 40},
    )
    return figure


def _history_value(row: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    nested = row.get("metrics", {})
    for key in aliases:
        if key in row and (value := _number(row[key])) is not None:
            return value
        if isinstance(nested, Mapping) and key in nested and (value := _number(nested[key])) is not None:
            return value
    return None


def _training_figure(
    rows: Sequence[Mapping[str, Any]], colors: dict[str, str], warnings: list[str]
) -> go.Figure | None:
    if not rows:
        warnings.append("No local epoch history was supplied; training-curve diagnostics are unavailable.")
        return None
    present = [
        key
        for key, aliases in _HISTORY_METRICS.items()
        if any(_history_value(row, aliases) is not None for row in rows)
    ]
    if not present:
        warnings.append("Epoch history contained no recognized finite training metrics.")
        return None
    columns = 2
    row_count = math.ceil(len(present) / columns)
    figure = make_subplots(rows=row_count, cols=columns, subplot_titles=[_pretty(metric) for metric in present])
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant", row.get("variant_id", "unknown")))
        seed = str(row.get("seed", "none"))
        grouped[(variant, seed)].append(row)
        if variant not in colors:
            colors[variant] = _COLORS[len(colors) % len(_COLORS)]
    for panel, metric in enumerate(present):
        panel_row, panel_col = divmod(panel, columns)
        aliases = _HISTORY_METRICS[metric]
        for trace_index, ((variant, seed), trace_rows) in enumerate(grouped.items()):
            values = []
            for fallback_step, row in enumerate(trace_rows):
                value = _history_value(row, aliases)
                step = _number(row.get("epoch", row.get("global_step", row.get("step", fallback_step))))
                if value is not None and step is not None:
                    values.append((step, value))
            values.sort(key=lambda item: item[0])
            if not values:
                continue
            seed_number = int(seed) if seed.lstrip("-").isdigit() else trace_index
            figure.add_trace(
                go.Scatter(
                    x=[value[0] for value in values],
                    y=[value[1] for value in values],
                    mode="lines",
                    line={"color": colors[variant], "dash": _DASHES[seed_number % len(_DASHES)], "width": 2},
                    name=f"{variant} seed {seed}",
                    legendgroup=f"{variant}-{seed}",
                    showlegend=panel == 0,
                    hovertemplate="epoch=%{x}<br>value=%{y:.5g}<extra></extra>",
                ),
                row=panel_row + 1,
                col=panel_col + 1,
            )
        figure.update_xaxes(title_text="Epoch", row=panel_row + 1, col=panel_col + 1)
        figure.update_yaxes(title_text=_pretty(metric), row=panel_row + 1, col=panel_col + 1)
    figure.update_layout(
        height=max(450, 350 * row_count),
        template="plotly_white",
        title="Training diagnostics (one metric per axis)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0},
        margin={"t": 135, "b": 60},
    )
    return figure


def _frame_values(rows: Sequence[Mapping[str, Any]], primary_metric: str, warnings: list[str]) -> list[dict[str, Any]]:
    if not rows:
        warnings.append(
            "No per-frame predictions or metrics were supplied; outlier-frame diagnostics are unavailable."
        )
        return []
    output = []
    skipped = 0
    for index, row in enumerate(rows):
        metrics = row.get("metrics", {})
        value = _number(row.get(primary_metric))
        if value is None and isinstance(metrics, Mapping):
            value = _number(metrics.get(primary_metric))
        if value is None and primary_metric in {"metric_abs_rel", "abs_rel", "aligned_abs_rel"}:
            value = _depth_abs_rel(row, aligned=primary_metric == "aligned_abs_rel")
        if value is None:
            skipped += 1
            continue
        output.append(
            {
                "variant": str(row.get("variant", row.get("variant_id", "unknown"))),
                "seed": row.get("seed"),
                "frame_id": row.get("frame_id", row.get("index", index)),
                "value": value,
            }
        )
    if skipped:
        warnings.append(f"Ignored {skipped} per-frame row(s) without finite {primary_metric} values.")
    return output


def _depth_abs_rel(row: Mapping[str, Any], *, aligned: bool) -> float | None:
    predicted = row.get("predicted_depth", row.get("prediction"))
    target = row.get("target_depth", row.get("target"))
    if predicted is None or target is None:
        return None
    try:
        prediction = np.asarray(predicted, dtype=np.float64)
        truth = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if prediction.shape != truth.shape or not prediction.size:
        return None
    valid = np.isfinite(prediction) & np.isfinite(truth) & (prediction > 0) & (truth > 0.1) & (truth < 10.0)
    supplied_mask = row.get("valid_mask")
    if supplied_mask is not None:
        mask = np.asarray(supplied_mask, dtype=bool)
        if mask.shape != truth.shape:
            return None
        valid &= mask
    if not valid.any():
        return None
    if aligned:
        median = float(np.median(prediction[valid]))
        if not math.isfinite(median) or median <= 0:
            return None
        prediction = prediction * (float(np.median(truth[valid])) / median)
    return float(np.mean(np.abs(prediction[valid] - truth[valid]) / truth[valid]))


def _load_npz_diagnostics(
    paths: Mapping[str, str | Path] | None, primary_metric: str, warnings: list[str]
) -> tuple[list[dict[str, Any]], list[_DepthGrid]]:
    if not paths:
        return [], []
    frame_rows: list[dict[str, Any]] = []
    grids: list[_DepthGrid] = []
    supported_metric = primary_metric in {"metric_abs_rel", "abs_rel", "aligned_abs_rel"}
    for label, raw_path in paths.items():
        path = Path(raw_path)
        if path.suffix.lower() != ".npz":
            continue
        if not path.is_file():
            warnings.append(f"Diagnostic NPZ is missing for {label}: {path}.")
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                prediction = _depth_stack(payload["prediction_m"])
                target = _depth_stack(payload["target_m"])
        except Exception as error:  # A corrupt optional diagnostic must not hide the comparison report.
            warnings.append(f"Could not load diagnostic NPZ for {label} ({type(error).__name__}: {error}).")
            continue
        if prediction.shape != target.shape:
            warnings.append(
                f"Ignored diagnostic NPZ for {label}: prediction {prediction.shape} != target {target.shape}."
            )
            continue
        variant, seed = _diagnostic_identity(str(label))
        frame_metrics: list[float] = []
        frame_errors: list[np.ndarray] = []
        for frame_index, (predicted, truth) in enumerate(zip(prediction, target, strict=True)):
            valid = np.isfinite(predicted) & np.isfinite(truth) & (predicted > 0) & (truth > 0.1) & (truth < 10.0)
            if not valid.any():
                frame_metrics.append(float("nan"))
                frame_errors.append(np.full(truth.shape, np.nan, dtype=np.float32))
                continue
            evaluated = predicted
            if primary_metric == "aligned_abs_rel":
                median = float(np.median(predicted[valid]))
                if median > 0 and math.isfinite(median):
                    evaluated = predicted * (float(np.median(truth[valid])) / median)
            relative_error = np.full(truth.shape, np.nan, dtype=np.float32)
            relative_error[valid] = np.abs(evaluated[valid] - truth[valid]) / truth[valid]
            metric = float(np.mean(relative_error[valid]))
            frame_metrics.append(metric)
            frame_errors.append(relative_error)
            if supported_metric:
                frame_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "frame_id": frame_index,
                        "value": metric,
                    }
                )
        finite_indices = [index for index, value in enumerate(frame_metrics) if math.isfinite(value)]
        if not finite_indices:
            warnings.append(f"Diagnostic NPZ for {label} has no valid target/prediction pixels.")
            continue
        worst = max(finite_indices, key=lambda index: frame_metrics[index])
        grids.append(
            _DepthGrid(
                label=str(label),
                frame_index=worst,
                abs_rel=frame_metrics[worst],
                prediction=_downsample_grid(prediction[worst]),
                target=_downsample_grid(target[worst]),
                relative_error=_downsample_grid(frame_errors[worst]),
            )
        )
    if grids and not supported_metric:
        warnings.append(
            f"NPZ depth grids were rendered, but their per-frame AbsRel was not mixed with primary metric {primary_metric}."
        )
    return frame_rows, grids


def _depth_stack(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 4 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim == 2:
        result = result[None]
    if result.ndim != 3:
        raise ValueError(f"expected depth stack [N,H,W], got {result.shape}")
    return result


def _diagnostic_identity(label: str) -> tuple[str, int | None]:
    if "-seed" not in label:
        return label, None
    variant, seed = label.rsplit("-seed", 1)
    return (variant, int(seed)) if seed.isdigit() else (label, None)


def _downsample_grid(value: np.ndarray, limit: int = 96) -> np.ndarray:
    height, width = value.shape
    y = np.linspace(0, height - 1, min(height, limit), dtype=np.int64)
    x = np.linspace(0, width - 1, min(width, limit), dtype=np.int64)
    return value[np.ix_(y, x)]


def _depth_grid_figure(grids: Sequence[_DepthGrid]) -> go.Figure | None:
    if not grids:
        return None
    titles = [
        title
        for grid in grids
        for title in (
            f"{grid.label} · frame {grid.frame_index} · prediction",
            "Target depth",
            f"Relative error · AbsRel {grid.abs_rel:.4f}",
        )
    ]
    figure = make_subplots(
        rows=len(grids), cols=3, subplot_titles=titles, horizontal_spacing=0.035, vertical_spacing=0.06
    )
    depth_values = np.concatenate(
        [value[np.isfinite(value) & (value > 0)] for grid in grids for value in (grid.prediction, grid.target)]
    )
    error_values = np.concatenate([grid.relative_error[np.isfinite(grid.relative_error)] for grid in grids])
    depth_min, depth_max = np.percentile(depth_values, (1, 99)) if depth_values.size else (0.1, 10.0)
    error_max = float(np.percentile(error_values, 95)) if error_values.size else 1.0
    if not math.isfinite(error_max) or error_max <= 0:
        error_max = 1.0
    for row, grid in enumerate(grids, start=1):
        figure.add_trace(
            go.Heatmap(z=grid.prediction, coloraxis="coloraxis", hovertemplate="depth=%{z:.4g} m<extra></extra>"),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Heatmap(z=grid.target, coloraxis="coloraxis", hovertemplate="depth=%{z:.4g} m<extra></extra>"),
            row=row,
            col=2,
        )
        figure.add_trace(
            go.Heatmap(
                z=grid.relative_error,
                coloraxis="coloraxis2",
                hovertemplate="relative error=%{z:.4g}<extra></extra>",
            ),
            row=row,
            col=3,
        )
        for column in (1, 2, 3):
            figure.update_xaxes(showticklabels=False, row=row, col=column)
            figure.update_yaxes(showticklabels=False, autorange="reversed", row=row, col=column)
    figure.update_layout(
        height=max(520, 245 * len(grids)),
        template="plotly_white",
        title=(
            "Worst held-out depth/error grids · shared depth scale uses pooled 1st–99th percentiles; "
            "shared error scale saturates at pooled 95th percentile"
        ),
        coloraxis={
            "colorscale": "Viridis",
            "cmin": float(depth_min),
            "cmax": float(depth_max),
            "colorbar": {"title": "Depth (m)", "x": 1.01, "y": 0.78, "len": 0.3},
        },
        coloraxis2={
            "colorscale": "Magma",
            "cmin": 0.0,
            "cmax": error_max,
            "colorbar": {"title": "Abs. rel. error", "x": 1.01, "y": 0.22, "len": 0.3},
        },
        margin={"t": 145, "b": 40, "l": 30, "r": 115},
    )
    return figure


def _frame_figure(rows: Sequence[Mapping[str, Any]], primary_metric: str, colors: dict[str, str]) -> go.Figure | None:
    if not rows:
        return None
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row.get("seed", "none")))].append(row)
    figure = go.Figure()
    for trace_index, ((variant, seed), values) in enumerate(grouped.items()):
        if variant not in colors:
            colors[variant] = _COLORS[len(colors) % len(_COLORS)]
        seed_number = int(seed) if seed.lstrip("-").isdigit() else trace_index
        figure.add_trace(
            go.Scatter(
                x=[value["frame_id"] for value in values],
                y=[value["value"] for value in values],
                mode="lines+markers",
                line={"color": colors[variant], "dash": _DASHES[seed_number % len(_DASHES)]},
                name=f"{variant} seed {seed}",
                hovertemplate="frame=%{x}<br>value=%{y:.5g}<extra></extra>",
            )
        )
    figure.update_layout(
        height=480,
        template="plotly_white",
        title="Held-out per-frame error and outlier localization",
        xaxis_title="Chronological held-out frame",
        yaxis_title=f"{primary_metric} (lower is better)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0},
        margin={"t": 135, "b": 60},
    )
    return figure


def _export_png(figure: go.Figure, html_path: Path, static_png: bool | str | Path, warnings: list[str]) -> Path | None:
    if static_png is False:
        return None
    target = (
        Path(static_png)
        if isinstance(static_png, str | Path) and not isinstance(static_png, bool)
        else html_path.with_suffix(".png")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.write_image(str(target), width=1500, height=1000, scale=1)
    except Exception as error:  # Plotly image backends expose version-specific exception types.
        target.unlink(missing_ok=True)
        detail = f"{type(error).__name__}: {error}".replace("\n", " ")[:300]
        warnings.append(f"Static PNG export was unavailable; the self-contained HTML is complete ({detail}).")
        return None
    return target


def _failure_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = record.get("failures", [])
    if not isinstance(failures, Sequence) or isinstance(failures, str | bytes):
        return [{"error": "Malformed failures field in comparison record."}]
    return [dict(row) if isinstance(row, Mapping) else {"error": str(row)} for row in failures]


def _document(
    *,
    html_path: Path,
    record: Mapping[str, Any],
    summaries: Sequence[_VariantSummary],
    variants: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    frame_values: Sequence[Mapping[str, Any]],
    primary_metric: str,
    warnings: Sequence[str],
    overview_html: str,
    training_html: str | None,
    frame_html: str | None,
    depth_grid_html: str | None,
    depth_grid_count: int,
    has_training: bool,
    png_path: Path | None,
) -> str:
    experiment = html.escape(str(record.get("experiment_id", "unknown experiment")))
    schema = html.escape(str(record.get("schema_version", "unknown")))
    manifest = html.escape(str(record.get("dataset_manifest", "unknown")))
    split_hash = html.escape(str(record.get("split_hash", "unknown")))
    warnings_html = "".join(f"<li>{html.escape(value)}</li>" for value in warnings)
    if warnings_html:
        warnings_html = (
            f"<section class='warning'><h2>Diagnostics requiring attention</h2><ul>{warnings_html}</ul></section>"
        )
    else:
        warnings_html = (
            "<section class='success'><h2>Integrity summary</h2><p>No report-level warnings detected.</p></section>"
        )
    wandb = _external_link(record.get("wandb_url"), "Open W&B run")
    png = _artifact_link(png_path, html_path.parent, "Static overview PNG")
    training_section = (
        f"<p>{len(history_rows)} local epoch rows rendered. Solid/dashed line styles distinguish seeds.</p>{training_html}"
        if has_training and training_html is not None
        else "<p class='muted'>No local training history was available. Persist epoch rows in the formal runner to enable this section.</p>"
    )
    frame_section = (
        f"<p>{len(frame_values)} finite per-frame measurements rendered. The table highlights the worst held-out frames.</p>"
        f"{frame_html}{_worst_frame_table(frame_values, primary_metric)}"
        if frame_html is not None
        else "<p class='muted'>No usable per-frame measurements were available.</p>"
    )
    depth_section = (
        f"<p>{depth_grid_count} diagnostic NPZ artifact(s) rendered at a bounded 96×96 display resolution. Each row "
        "selects that run's worst finite held-out frame. Prediction and target share one global depth scale; relative "
        f"error uses a separate shared scale.</p>{depth_grid_html}"
        if depth_grid_html is not None
        else "<p class='muted'>No usable prediction/target NPZ diagnostics were available.</p>"
    )
    status = "PARTIAL / FAILURES RECORDED" if failures else "COMPLETE RECORD / NO RECORDED FAILURES"
    status_class = "bad" if failures else "good"
    provenance = html.escape(json.dumps(_json_safe(record), indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase-2b geometry student report — {experiment}</title>
<style>
:root{{--ink:#17212b;--muted:#5c6873;--line:#d8dee4;--panel:#f7f9fb;--warn:#fff4ce;--good:#e6f4ea;--bad:#fce8e6}}
body{{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);max-width:1500px;margin:0 auto;padding:2rem}}
h1,h2,h3{{line-height:1.2}} h2{{margin-top:2rem}} code{{overflow-wrap:anywhere}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.75rem;margin:1rem 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem}}
.status{{font-weight:700;letter-spacing:.03em}} .good{{background:var(--good)}} .bad{{background:var(--bad)}}
.warning{{background:var(--warn);border-left:5px solid #b7791f;padding:.7rem 1.1rem;border-radius:4px}}
.success{{background:var(--good);border-left:5px solid #16833f;padding:.7rem 1.1rem;border-radius:4px}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid var(--line);padding:.45rem .55rem;text-align:right;white-space:nowrap}} th{{background:#eef2f5}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} .muted{{color:var(--muted)}}
pre{{background:#0d1117;color:#e6edf3;padding:1rem;overflow:auto;border-radius:6px;max-height:34rem}}
</style></head><body>
<h1>Phase-2b geometry student diagnostics</h1>
<p class="status {status_class}">{status}</p>
<div class="meta"><div class="card"><b>Experiment</b><br>{experiment}</div><div class="card"><b>Schema</b><br>{schema}</div>
<div class="card"><b>Dataset manifest</b><br>{manifest}</div><div class="card"><b>Split hash</b><br><code>{split_hash}</code></div>
<div class="card"><b>Primary metric</b><br><code>{html.escape(primary_metric)}</code> (lower is better)</div>
<div class="card"><b>Linked artifacts</b><br>{wandb} · PNG: {png}</div></div>
{warnings_html}
<section><h2>Decision overview</h2><p>Diamonds and error bars show mean ± sample SD across completed seeds. Teacher or single-run rows have no inferred variance. Accuracy, latency, memory, and NLL use separate axes and retain their native units. Reported encoder timing is chunked per-frame throughput while head timing is batch-1 latency; their sum is not a measured batch-1 end-to-end latency.</p>
{overview_html}</section>
<section><h2>Variant summary</h2>{_summary_table(summaries, primary_metric)}</section>
<section><h2>Seed-level measurements</h2><p class="muted">Values are read directly from the comparison record; missing measurements remain explicit.</p>{_seed_table(variants, primary_metric)}</section>
<section><h2>Training curves</h2>{training_section}</section>
<section><h2>Per-frame diagnostics</h2>{frame_section}</section>
<section><h2>Depth and relative-error grids</h2>{depth_section}</section>
<section><h2>Failures</h2>{_failure_table(failures)}</section>
<section><h2>Metric policy and complete provenance</h2><p>Use the policy below to verify scale alignment, checkpoint selection, split identity, and artifact hashes before promotion.</p><pre>{provenance}</pre></section>
</body></html>"""


def _summary_table(values: Sequence[_VariantSummary], primary_metric: str) -> str:
    headers = (
        "Variant",
        "Role",
        "Seeds",
        f"{primary_metric} mean ± sample SD",
        "Latency ms/frame",
        "Peak GPU memory GiB (max component)",
        "Total parameters",
        "Raw NLL",
        "Calibrated NLL",
        "Variance multiplier",
    )
    rows = []
    for value in values:
        metric = _fmt(value.primary_mean)
        if value.primary_mean is not None and value.primary_std is not None:
            metric += f" ± {_fmt(value.primary_std)}"
        rows.append(
            (
                value.variant_id,
                value.role,
                str(value.seed_count) if value.seed_count else "fixed / unseeded",
                metric,
                _fmt(value.runtime_ms),
                _fmt(value.memory_gib),
                _fmt(value.parameters, integer=True),
                _fmt(value.raw_nll),
                _fmt(value.calibrated_nll),
                _fmt(value.variance_multiplier),
            )
        )
    return _table(headers, rows)


def _seed_table(variants: Sequence[Mapping[str, Any]], primary_metric: str) -> str:
    headers = (
        "Variant",
        "Seed",
        primary_metric,
        "Aligned AbsRel",
        "Metric RMSE (m)",
        "Delta &lt; 1.25",
        "Raw NLL",
        "Calibrated NLL",
        "Latency ms/frame",
        "Peak memory GiB",
        "Parameters",
    )
    rows = []
    for row in variants:
        memory = _peak_memory(row)
        rows.append(
            (
                str(row.get("variant_id", "unknown")),
                "—" if row.get("seed") is None else str(row.get("seed")),
                _fmt(_metric(row, primary_metric)),
                _fmt(_metric(row, "aligned_abs_rel")),
                _fmt(_metric(row, "metric_rmse_m")),
                _fmt(_metric(row, "metric_delta_1") or _metric(row, "aligned_delta_1")),
                _fmt(_metric(row, "raw_log_depth_nll")),
                _fmt(_metric(row, "calibrated_log_depth_nll")),
                _fmt(_runtime(row, "total_ms_per_frame", positive=True)),
                _fmt(memory),
                _fmt(_number(row.get("parameters")), integer=True),
            )
        )
    return _table(headers, rows)


def _worst_frame_table(rows: Sequence[Mapping[str, Any]], primary_metric: str) -> str:
    worst = sorted(rows, key=lambda row: float(row["value"]), reverse=True)[:20]
    return "<h3>Worst finite frame/variant rows</h3>" + _table(
        ("Variant", "Seed", "Frame", primary_metric),
        [
            (
                str(row["variant"]),
                "—" if row.get("seed") is None else str(row.get("seed")),
                str(row["frame_id"]),
                _fmt(float(row["value"])),
            )
            for row in worst
        ],
    )


def _failure_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p class='good card'>No failures were recorded.</p>"
    columns = []
    for preferred in ("variant", "seed", "stage", "error"):
        if any(preferred in row for row in rows):
            columns.append(preferred)
    columns.extend(sorted({str(key) for row in rows for key in row}.difference(columns)))
    return _table(
        tuple(_pretty(value) for value in columns), [tuple(str(row.get(key, "—")) for key in columns) for row in rows]
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{value}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    if not rows:
        body = f"<tr><td colspan='{len(headers)}'>No rows available.</td></tr>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _fmt(value: float | None, *, integer: bool = False) -> str:
    if value is None:
        return "—"
    if integer:
        return f"{value:,.0f}"
    return f"{value:.5g}"


def _pretty(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _external_link(value: Any, label: str) -> str:
    if value is None:
        return "W&B not linked"
    url = str(value)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return html.escape(url)
    return f"<a href='{html.escape(url, quote=True)}' rel='noopener noreferrer'>{html.escape(label)}</a>"


def _artifact_link(path: Path | None, report_directory: Path, label: str) -> str:
    if path is None:
        return "not exported"
    try:
        relative = path.relative_to(report_directory)
    except ValueError:
        return f"<code>{html.escape(str(path))}</code>"
    return f"<a href='{html.escape(relative.as_posix(), quote=True)}'>{html.escape(label)}</a>"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
