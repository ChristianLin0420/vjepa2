"""Self-contained Phase-2e SUN RGB-D feature-cache report."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from jepa4d.evaluation.phase2e_feature_cache import RECEIPT_SCHEMA

SPLIT_COLORS = {"train": "#2563eb", "validation": "#f59e0b", "test": "#dc2626"}


def validate_test_target_boundary(receipt: Mapping[str, Any], previews: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed if cache-stage evidence discloses test target values."""
    summaries = receipt.get("split_summaries")
    if not isinstance(summaries, Mapping) or "test" not in summaries:
        raise ValueError("cache receipt must contain an opaque test split summary")
    test_summary = summaries["test"]
    if not isinstance(test_summary, Mapping):
        raise ValueError("test split summary must be a mapping")
    if "target_depth_m" in test_summary:
        raise ValueError("test target statistics are forbidden before final evaluation")
    forbidden_inputs = {"intrinsics_384", "raw_features", "normalized_features"} & set(test_summary)
    if forbidden_inputs:
        raise ValueError(f"test input statistics are forbidden before final evaluation: {sorted(forbidden_inputs)}")
    inputs = test_summary.get("input_tensors")
    if not isinstance(inputs, Mapping) or inputs.get("access") != "opaque_until_final_evaluation":
        raise ValueError("test input tensors must remain explicitly opaque until final evaluation")
    if inputs.get("statistics_computed") is not False or inputs.get("preview_generated") is not False:
        raise ValueError("test input receipt claims statistics or previews before final evaluation")
    target = test_summary.get("target_tensor")
    if not isinstance(target, Mapping) or target.get("access") != "opaque_until_final_evaluation":
        raise ValueError("test target tensor must remain explicitly opaque until final evaluation")
    if target.get("statistics_computed") is not False or target.get("preview_generated") is not False:
        raise ValueError("test target receipt claims statistics or previews before final evaluation")
    for preview in previews:
        split = str(preview.get("label", "")).split("/", maxsplit=1)[0]
        if split == "test":
            raise ValueError("test target previews are forbidden before final evaluation")


def _counts_figure(receipt: Mapping[str, Any]) -> go.Figure:
    summaries = receipt["split_summaries"]
    sensors = sorted({sensor for summary in summaries.values() for sensor in summary["sensor_counts"]})
    figure = go.Figure()
    for split, summary in summaries.items():
        figure.add_bar(
            name=split,
            x=sensors,
            y=[summary["sensor_counts"].get(sensor, 0) for sensor in sensors],
            marker_color=SPLIT_COLORS[split],
            hovertemplate=f"{split}<br>%{{x}}: %{{y}} samples<extra></extra>",
        )
    figure.update_layout(
        title="Frozen sensor-blocked sample counts",
        barmode="stack",
        xaxis_title="Sensor",
        yaxis_title="Source samples",
        template="plotly_white",
        height=430,
    )
    return figure


def _camera_depth_figure(receipt: Mapping[str, Any]) -> go.Figure:
    summaries = receipt["split_summaries"]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Intrinsics after crop + resize to 384",
            "Train/validation target depth statistics (test opaque)",
        ),
    )
    for split, summary in summaries.items():
        if "intrinsics_384" not in summary:
            continue
        camera = summary["intrinsics_384"]
        figure.add_trace(
            go.Scatter(
                x=[camera["fx_mean"]],
                y=[camera["fy_mean"]],
                mode="markers+text",
                text=[split],
                textposition="top center",
                marker={"size": 13, "color": SPLIT_COLORS[split]},
                name=split,
                hovertemplate=(
                    f"{split}<br>fx=%{{x:.2f}}<br>fy=%{{y:.2f}}<br>"
                    f"cx={camera['cx_mean']:.2f}<br>cy={camera['cy_mean']:.2f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        if "target_depth_m" in summary:
            depth = summary["target_depth_m"]
            figure.add_trace(
                go.Bar(
                    x=[split],
                    y=[depth["valid_mean_m"]],
                    error_y={"type": "data", "array": [depth["valid_std_m"]]},
                    marker_color=SPLIT_COLORS[split],
                    customdata=[[depth["valid_fraction"], depth["valid_min_m"], depth["valid_max_m"]]],
                    hovertemplate=(
                        "%{x}<br>mean=%{y:.3f}m<br>valid=%{customdata[0]:.3f}<br>"
                        "range=%{customdata[1]:.3f}–%{customdata[2]:.3f}m<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
    figure.update_xaxes(title_text="fx", row=1, col=1)
    figure.update_yaxes(title_text="fy", row=1, col=1)
    figure.update_yaxes(title_text="Mean valid depth (m)", row=1, col=2)
    figure.update_layout(template="plotly_white", height=450, margin={"t": 70})
    return figure


def _feature_figure(receipt: Mapping[str, Any]) -> go.Figure:
    summaries = receipt["split_summaries"]
    splits = [split for split, summary in summaries.items() if "normalized_features" in summary]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Train/validation V-JEPA feature moments (test opaque)", "Cached tensor inventory"),
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )
    figure.add_trace(
        go.Bar(
            x=splits,
            y=[summaries[split]["normalized_features"]["mean"] for split in splits],
            name="mean",
            marker_color="#0ea5e9",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=splits,
            y=[summaries[split]["normalized_features"]["std"] for split in splits],
            name="std",
            marker_color="#8b5cf6",
        ),
        row=1,
        col=1,
    )
    inventory = []
    for split, summary in summaries.items():
        inventory.append(
            [
                split,
                summary["views"],
                str(summary["shapes"]["features"]),
                str(summary["shapes"]["rgb"]),
                str(summary["shapes"]["targets"]),
            ]
        )
    figure.add_trace(
        go.Table(
            header={"values": ["split", "views", "features", "RGB", "depth"], "fill_color": "#e2e8f0"},
            cells={"values": list(map(list, zip(*inventory, strict=True))), "fill_color": "#f8fafc"},
        ),
        row=1,
        col=2,
    )
    figure.update_layout(template="plotly_white", barmode="group", height=450, margin={"t": 70})
    return figure


def _preview_figure(previews: Sequence[Mapping[str, Any]]) -> go.Figure:
    if not previews:
        return go.Figure().update_layout(title="No preview samples", template="plotly_white")
    figure = make_subplots(
        rows=len(previews),
        cols=2,
        subplot_titles=tuple(
            title
            for preview in previews
            for title in (f"{preview['label']} · RGB", f"{preview['label']} · target depth")
        ),
        vertical_spacing=min(0.08, 0.4 / len(previews)),
    )
    for row, preview in enumerate(previews, start=1):
        rgb = np.asarray(preview["rgb"])
        depth = np.asarray(preview["depth"])
        figure.add_trace(go.Image(z=np.clip(rgb * 255.0, 0, 255).astype(np.uint8)), row=row, col=1)
        figure.add_trace(
            go.Heatmap(z=depth, colorscale="Viridis", colorbar={"title": "m", "len": 0.15}, showscale=True),
            row=row,
            col=2,
        )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)
    figure.update_layout(
        title="Bounded preprocessing examples (diagnostics, not model predictions)",
        template="plotly_white",
        height=max(420, 300 * len(previews)),
        margin={"t": 90},
    )
    return figure


def build_phase2e_cache_report(
    receipt: Mapping[str, Any], previews: Sequence[Mapping[str, Any]], output: Path
) -> Path:
    """Write one portable report with Plotly embedded inline."""
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError(f"unexpected Phase-2e cache receipt schema: {receipt.get('schema_version')!r}")
    validate_test_target_boundary(receipt, previews)
    figures = (
        _counts_figure(receipt),
        _camera_depth_figure(receipt),
        _feature_figure(receipt),
        _preview_figure(previews),
    )
    fragments = [
        pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs="inline" if index == 0 else False,
            config={"displaylogo": False, "responsive": True},
        )
        for index, figure in enumerate(figures)
    ]
    audit = html.escape(
        json.dumps(
            {
                "dataset": receipt["dataset"],
                "view_policy": receipt["view_policy"],
                "feature_normalization": receipt["feature_normalization"],
                "teacher_policy": receipt["teacher_policy"],
                "caches": receipt["caches"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2e SUN RGB-D cache</title>
<style>
:root{{--ink:#0f172a;--muted:#475569;--line:#cbd5e1;--paper:#fff;--wash:#f1f5f9}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 8px;font-size:30px}} h2{{margin-top:32px}}
.lead{{color:var(--muted);max-width:1100px}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}
.card,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 2px 8px #0f172a0a}}
.card b{{display:block;font-size:21px}} .card span{{color:var(--muted)}} .panel{{margin:18px 0;overflow:hidden}}
.boundary{{border-left:5px solid #2563eb;background:#eff6ff;padding:14px 18px;margin:18px 0}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:9px;overflow:auto}}
</style></head><body><main>
<h1>Phase 2e · SUN RGB-D feature-cache audit</h1>
    <p class="lead">Frozen V-JEPA features, camera-aligned RGB/depth tensors, and a train-only centered VGGT shape teacher.
    This build stage computes no model-quality metric and performs no checkpoint selection. Test targets are cached as an
    opaque tensor for the final evaluator: no test-target statistic, preview, or W&amp;B media is produced here.</p>
<div class="boundary"><strong>Split boundary.</strong> Train/validation and untouched test tensors are stored in separate
hash-bound artifacts. VGGT is executed only for the two training views.</div>
<div class="cards">
 <div class="card"><b>{receipt["split_summaries"]["train"]["samples"]}</b><span>train sources × 2 views</span></div>
 <div class="card"><b>{receipt["split_summaries"]["validation"]["samples"]}</b><span>validation sources</span></div>
 <div class="card"><b>{receipt["split_summaries"]["test"]["samples"]}</b><span>isolated test sources</span></div>
</div>
<h2>Dataset and camera audit</h2><div class="panel">{fragments[0]}{fragments[1]}</div>
<h2>Feature and cache inventory</h2><div class="panel">{fragments[2]}</div>
<h2>Bounded examples</h2><div class="panel">{fragments[3]}</div>
<h2>Reproducibility record</h2><pre>{audit}</pre>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    return output
