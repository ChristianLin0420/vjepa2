"""Interactive comparison report for object-identity association ablations."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_identity_report(
    results: dict[str, dict[str, dict[str, float]]],
    output: str | Path,
    *,
    metadata: dict[str, Any],
    wandb_url: str | None = None,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(results)
    figure = make_subplots(
        rows=len(datasets),
        cols=2,
        subplot_titles=[title for dataset in datasets for title in (f"{dataset}: identity F1", f"{dataset}: errors")],
    )
    for row, dataset in enumerate(datasets, start=1):
        variants = list(results[dataset])
        figure.add_trace(
            go.Bar(
                x=variants,
                y=[results[dataset][variant]["pairwise_f1"] for variant in variants],
                name=f"{dataset} F1",
                text=[f"{results[dataset][variant]['pairwise_f1']:.3f}" for variant in variants],
            ),
            row=row,
            col=1,
        )
        for metric in ("id_switches", "false_merges", "fragments"):
            figure.add_trace(
                go.Bar(
                    x=variants,
                    y=[results[dataset][variant][metric] for variant in variants],
                    name=f"{dataset} {metric}",
                ),
                row=row,
                col=2,
            )
    figure.update_yaxes(range=[0, 1.05], col=1)
    figure.update_layout(
        height=max(650, 520 * len(datasets)),
        barmode="group",
        template="plotly_white",
        title="JEPA-4D identity association ablations",
    )
    payload = {"results": results, "metadata": metadata, "wandb_url": wandb_url}
    target.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Identity ablation report</title>"
        "<style>body{font-family:system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}"
        "pre{background:#f6f8fa;padding:1rem;overflow:auto}.warning{background:#fff3cd;padding:1rem}</style>"
        "</head><body><h1>Identity association ablations</h1>"
        "<p class='warning'>Synthetic results are controlled diagnostics. DAVIS operating-point sweeps are exploratory "
        "on the evaluated sequence and are not held-out performance.</p>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        f"<h2>Complete metrics and provenance</h2><pre>{html.escape(json.dumps(payload, indent=2))}</pre>"
        "</body></html>"
    )
    return target
