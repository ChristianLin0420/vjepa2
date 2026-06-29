"""Self-contained Plotly report for Phase-2d fusion attribution."""

from __future__ import annotations

import base64
import html
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from plotly.subplots import make_subplots

from jepa4d.evaluation.fusion_attribution import LAYER_ORDER, PHASE2D_SCHEMA, QUALITATIVE_SCHEMA

FAMILY_COLORS = {
    "original": "#2563eb",
    "zero": "#64748b",
    "fixed_average": "#16a34a",
    "layer_permutation": "#f59e0b",
    "sign_flip": "#dc2626",
}


def _label(intervention: Mapping[str, Any]) -> str:
    identifier = str(intervention["intervention_id"])
    replacements = {
        "original": "Original learned",
        "zero": "Zero gates",
        "fixed_average": "Fixed 4-layer avg",
        "permute_sources_": "Permute ",
        "sign_flip_": "Flip sign ",
    }
    for source, target in replacements.items():
        if identifier.startswith(source):
            return target + identifier.removeprefix(source).replace("_", "→" if source == "permute_sources_" else ",")
    return identifier


def _aggregate_rows(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = record.get("aggregate", {}).get("controls", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("attribution record has no aggregate control rows")
    return rows


def _metric(row: Mapping[str, Any], key: str, statistic: str = "mean") -> float:
    return float(row["metrics"][key][statistic])


def _bar_figure(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    labels = [_label(row["intervention"]) for row in rows]
    colors = [FAMILY_COLORS[str(row["intervention"]["family"])] for row in rows]
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Raw metric AbsRel ↓", "Median-aligned AbsRel ↓"))
    for column, metric in enumerate(("metric_abs_rel", "aligned_abs_rel"), start=1):
        figure.add_trace(
            go.Bar(
                x=labels,
                y=[_metric(row, metric) for row in rows],
                error_y={"type": "data", "array": [_metric(row, metric, "std") for row in rows]},
                marker_color=colors,
                customdata=[row["intervention"]["family"] for row in rows],
                hovertemplate="%{x}<br>%{y:.6f}<br>family=%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(tickangle=-55)
    figure.update_layout(height=540, margin={"l": 55, "r": 25, "t": 70, "b": 170}, template="plotly_white")
    return figure


def _causal_figure(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    original = _metric(rows[0], "metric_abs_rel")
    values = [100.0 * (_metric(row, "metric_abs_rel") / max(original, 1e-12) - 1.0) for row in rows]
    colors = ["#16a34a" if value < 0 else "#dc2626" if value > 0 else "#64748b" for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[_label(row["intervention"]) for row in rows],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>AbsRel change=%{x:.3f}%<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color="#0f172a", line_width=1)
    figure.update_layout(
        title="Causal raw-AbsRel change from the same checkpoint's original gates",
        xaxis_title="Relative change (%) — negative is better",
        height=560,
        margin={"l": 190, "r": 25, "t": 70, "b": 55},
        template="plotly_white",
    )
    return figure


def _sequence_figure(record: Mapping[str, Any]) -> go.Figure:
    seeds = record["seeds"]
    sequence_ids = sorted(
        {
            value["sequence_id"]
            for seed in seeds
            for intervention in seed["interventions"]
            for value in intervention["per_sequence"]
        }
    )
    intervention_ids = [row["intervention"]["intervention_id"] for row in seeds[0]["interventions"]]
    intervention_metadata = {
        row["intervention"]["intervention_id"]: row["intervention"] for row in seeds[0]["interventions"]
    }
    figure = make_subplots(
        rows=1,
        cols=len(sequence_ids),
        subplot_titles=tuple(value.replace("freiburg3_", "") for value in sequence_ids),
        shared_yaxes=True,
    )
    for column, sequence_id in enumerate(sequence_ids, start=1):
        means, standard_deviations = [], []
        for intervention_id in intervention_ids:
            values = []
            for seed in seeds:
                intervention = next(
                    value
                    for value in seed["interventions"]
                    if value["intervention"]["intervention_id"] == intervention_id
                )
                sequence = next(value for value in intervention["per_sequence"] if value["sequence_id"] == sequence_id)
                values.append(float(sequence["metrics"]["metric_abs_rel"]))
            means.append(sum(values) / len(values))
            variance = sum((value - means[-1]) ** 2 for value in values) / max(len(values) - 1, 1)
            standard_deviations.append(variance**0.5)
        figure.add_trace(
            go.Bar(
                x=[_label(intervention_metadata[value]) for value in intervention_ids],
                y=means,
                error_y={"type": "data", "array": standard_deviations},
                marker_color=[
                    FAMILY_COLORS[str(intervention_metadata[value]["family"])] for value in intervention_ids
                ],
                showlegend=False,
                hovertemplate="%{x}<br>AbsRel=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(tickangle=-55)
    figure.update_yaxes(title_text="Raw metric AbsRel ↓", row=1, col=1)
    figure.update_layout(height=560, margin={"l": 55, "r": 25, "t": 70, "b": 180}, template="plotly_white")
    return figure


def _nll_delta_figure(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Validation-calibrated test NLL ↓", "Prediction change from original ↓"),
    )
    labels = [_label(row["intervention"]) for row in rows]
    colors = [FAMILY_COLORS[str(row["intervention"]["family"])] for row in rows]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[_metric(row, "calibrated_log_depth_nll") for row in rows],
            error_y={"type": "data", "array": [_metric(row, "calibrated_log_depth_nll", "std") for row in rows]},
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>NLL=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[_metric(row, "prediction_delta_relative") for row in rows],
            error_y={"type": "data", "array": [_metric(row, "prediction_delta_relative", "std") for row in rows]},
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>mean relative prediction delta=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(tickangle=-55)
    figure.update_layout(height=530, margin={"l": 55, "r": 25, "t": 70, "b": 170}, template="plotly_white")
    return figure


def _coefficient_figure(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    labels = [_label(row["intervention"]) for row in rows]
    coefficients = [
        [float(row["intervention"]["effective_coefficients"][str(layer)]) for layer in LAYER_ORDER] for row in rows
    ]
    figure = go.Figure(
        go.Heatmap(
            z=coefficients,
            x=[f"layer {layer}" for layer in LAYER_ORDER],
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "effective<br>coefficient"},
            text=[[f"{value:+.5f}" for value in values] for values in coefficients],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:+.7f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Gate interventions (seed-0 values; fixed control is seed invariant)",
        height=590,
        margin={"l": 190, "r": 55, "t": 70, "b": 55},
        template="plotly_white",
    )
    return figure


def _contribution_figure(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    labels = [_label(row["intervention"]) for row in rows]
    metrics = [f"residual_layer_{layer}_norm_ratio" for layer in LAYER_ORDER]
    values = [[_metric(row, metric) for metric in metrics] for row in rows]
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[f"layer {layer}" for layer in LAYER_ORDER],
            y=labels,
            colorscale="Viridis",
            colorbar={"title": "||contribution||₂<br>/ ||final||₂"},
            text=[[f"{value:.5e}" for value in row] for row in values],
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.7e}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Mean residual contribution norm ratio across sequences and seeds",
        height=590,
        margin={"l": 190, "r": 70, "t": 70, "b": 55},
        template="plotly_white",
    )
    return figure


def _summary_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(_label(row['intervention']))}</td>"
            f"<td>{html.escape(str(row['intervention']['family']))}</td>"
            f"<td>{_metric(row, 'metric_abs_rel'):.6f} ± {_metric(row, 'metric_abs_rel', 'std'):.6f}</td>"
            f"<td>{_metric(row, 'aligned_abs_rel'):.6f} ± {_metric(row, 'aligned_abs_rel', 'std'):.6f}</td>"
            f"<td>{_metric(row, 'metric_abs_log_scale_error'):.6f}</td>"
            f"<td>{_metric(row, 'calibrated_log_depth_nll'):.6f}</td>"
            f"<td>{_metric(row, 'prediction_delta_relative'):.6e}</td>"
            f"<td>{_metric(row, 'residual_total_norm_ratio'):.6e}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Control</th><th>Family</th><th>Raw AbsRel ↓</th><th>Aligned AbsRel ↓</th>"
        "<th>|log scale| ↓</th><th>Calibrated NLL ↓</th><th>Prediction Δ rel</th><th>Residual / final</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _figure_html(figures: Sequence[go.Figure]) -> list[str]:
    parts = []
    for index, figure in enumerate(figures):
        parts.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs="inline" if index == 0 else False,
                config={"displaylogo": False, "responsive": True},
            )
        )
    return parts


def _image_data(value: np.ndarray, low: float, high: float, *, error: bool = False) -> str:
    normalized = np.clip((np.asarray(value, dtype=np.float32) - low) / max(high - low, 1e-8), 0.0, 1.0)
    if error:
        rgb = np.stack((normalized, 0.25 * (1.0 - normalized), 1.0 - normalized), axis=-1)
    else:
        rgb = np.stack((normalized, 1.0 - np.abs(2.0 * normalized - 1.0), 1.0 - normalized), axis=-1)
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").resize((192, 192), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _qualitative_gallery(path: Path, handoff: Mapping[str, Any]) -> str:
    with np.load(path.resolve(strict=True), allow_pickle=False) as payload:
        if str(payload["schema_version"]) != QUALITATIVE_SCHEMA:
            raise ValueError("unexpected Phase-2d qualitative schema")
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
        and len(sample_ids) == len(sequence_ids) == predictions.shape[1]
        and len(variant_ids) == predictions.shape[0]
    ):
        raise ValueError("Phase-2d qualitative tensor/identity coverage is inconsistent")
    if handoff.get("sample_ids") != sample_ids or handoff.get("variant_ids") != variant_ids:
        raise ValueError("Phase-2d qualitative handoff identities differ from the NPZ")
    if not all(np.isfinite(value).all() for value in (predictions, targets, log_variances, sigmas)):
        raise ValueError("Phase-2d qualitative bundle contains non-finite values")
    depth_low, depth_high = (float(value) for value in np.percentile(targets, (2, 98)))
    errors = np.abs(predictions - targets[None]) / np.maximum(targets[None], 1e-4)
    error_high = max(float(np.percentile(errors, 98)), 1e-6)
    sigma_low, sigma_high = (float(value) for value in np.percentile(sigmas, (2, 98)))
    rows = []
    for variant_index, variant_id in enumerate(variant_ids):
        for sample_index, sample_id in enumerate(sample_ids):
            panels = (
                ("Target depth", targets[sample_index], depth_low, depth_high, False),
                ("Prediction depth", predictions[variant_index, sample_index], depth_low, depth_high, False),
                ("|relative error|", errors[variant_index, sample_index], 0.0, error_high, True),
                (
                    "Calibrated log-depth σ",
                    sigmas[variant_index, sample_index],
                    sigma_low,
                    sigma_high,
                    True,
                ),
            )
            figures = "".join(
                "<figure>"
                f'<img alt="{html.escape(variant_id)} {html.escape(label)}" '
                f'src="data:image/png;base64,{_image_data(value, low, high, error=is_error)}">'
                f"<figcaption>{html.escape(label)}<br><small>[{float(np.min(value)):.3g}, "
                f"{float(np.max(value)):.3g}]</small></figcaption></figure>"
                for label, value, low, high, is_error in panels
            )
            rows.append(
                "<article class='qual-row'>"
                f"<h3>{html.escape(variant_id)} · {html.escape(sequence_ids[sample_index])} · "
                f"{html.escape(sample_id)}</h3><div class='qual-grid'>{figures}</div></article>"
            )
    return (
        "<p class='lead'>Fixed sequence-balanced samples only (maximum four); the exact same IDs are reused for every "
        "seed/control. Uncertainty is validation-calibrated log-depth standard deviation.</p>" + "".join(rows)
    )


def build_fusion_attribution_report(
    record: Mapping[str, Any], output: Path, *, qualitative_npz: Path | None = None
) -> Path:
    """Render a self-contained HTML report with no network dependencies."""
    if record.get("schema_version") != PHASE2D_SCHEMA:
        raise ValueError(f"unexpected attribution schema: {record.get('schema_version')!r}")
    rows = _aggregate_rows(record)
    source = record["source"]
    protocol = record["protocol"]
    figures = (
        _bar_figure(rows),
        _causal_figure(rows),
        _sequence_figure(record),
        _nll_delta_figure(rows),
        _coefficient_figure(rows),
        _contribution_figure(rows),
    )
    figure_html = _figure_html(figures)
    qualitative_html = ""
    if qualitative_npz is not None:
        handoff = record.get("qualitative_handoff")
        if not isinstance(handoff, Mapping):
            raise ValueError("attribution record lacks its qualitative handoff")
        qualitative_html = _qualitative_gallery(qualitative_npz, handoff)
    metadata_json = html.escape(json.dumps({"source": source, "protocol": protocol}, indent=2, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 2d same-checkpoint fusion attribution</title>
<style>
:root{{--ink:#0f172a;--muted:#475569;--line:#cbd5e1;--paper:#fff;--wash:#f1f5f9;--warn:#fff7ed;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 8px;font-size:30px}} h2{{margin-top:32px}}
.lead{{color:var(--muted);max-width:1050px}} .warning{{background:var(--warn);border-left:5px solid #f97316;padding:14px 18px;margin:20px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:20px 0}}
.card,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 2px 8px #0f172a0a}}
.card b{{display:block;font-size:22px}} .card span{{color:var(--muted)}} .panel{{margin:18px 0;overflow:hidden}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:right;white-space:nowrap}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
	pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:9px;overflow:auto}}
	.qual-row{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:14px;margin:14px 0}}
	.qual-row h3{{font-size:14px;margin:0 0 10px}} .qual-grid{{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px}}
	.qual-grid figure{{margin:0}} .qual-grid img{{width:100%;max-width:192px;image-rendering:pixelated;border-radius:7px}}
	.qual-grid figcaption{{font-size:12px;color:var(--muted)}} @media(max-width:760px){{.qual-grid{{grid-template-columns:repeat(2,1fr)}}}}
	</style></head><body><main>
<h1>Phase 2d · Same-checkpoint fusion attribution</h1>
<p class="lead">The dense probe and its checkpoint remain fixed. Only the three residual fusion gates are replaced, so
learned-vs-zero is the direct causal test of whether the selected checkpoint's gates change its predictions beneficially.</p>
<div class="warning"><strong>Claim boundary.</strong> {html.escape(str(protocol["claim_boundary"]))}</div>
<div class="cards">
  <div class="card"><b>{len(record["seeds"])}</b><span>checkpoint seeds</span></div>
  <div class="card"><b>{len(rows)}</b><span>gate interventions per seed</span></div>
  <div class="card"><b>{record["test_samples"]["count"]}</b><span>previously consumed test frames</span></div>
  <div class="card"><b>Probe frozen</b><span>only fusion.raw_gates changed</span></div>
</div>
<h2>Primary comparisons</h2><div class="panel">{figure_html[0]}{figure_html[1]}</div>
<h2>Sequence, uncertainty, and prediction sensitivity</h2><div class="panel">{figure_html[2]}{figure_html[3]}</div>
	<h2>Intervention mechanics</h2><div class="panel">{figure_html[4]}{figure_html[5]}</div>
	{f"<h2>Fixed qualitative prediction · target · error · uncertainty</h2>{qualitative_html}" if qualitative_html else ""}
	<h2>Aggregate table</h2><div class="panel">{_summary_table(rows)}</div>
<h2>Auditable source and protocol</h2><pre>{metadata_json}</pre>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
