"""Interactive trajectory, confidence, and event report for Phase 4 memory."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jepa4d.memory.memory_update import FourDMemoryCore


def build_memory_report(
    memory: FourDMemoryCore,
    output: str | Path,
    *,
    persistence_stats: dict[str, int] | None = None,
    wandb_url: str | None = None,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "table", "colspan": 2}, None]],
        subplot_titles=("Object trajectories (map XY)", "Confidence history", "Current global memory"),
        row_heights=[0.62, 0.38],
    )
    for value in memory.scene_graph.objects.values():
        positioned = [entry for entry in value.history if entry.pose_map is not None]
        if positioned:
            figure.add_trace(
                go.Scatter(
                    x=[cast(list[float], entry.pose_map)[0] for entry in positioned],
                    y=[cast(list[float], entry.pose_map)[1] for entry in positioned],
                    mode="lines+markers",
                    name=f"{value.category}:{value.object_id[:8]}",
                    text=[f"t={entry.timestamp:.2f}" for entry in positioned],
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=[entry.timestamp for entry in value.history],
                y=[entry.confidence for entry in value.history],
                mode="lines+markers",
                name=f"confidence:{value.object_id[:8]}",
            ),
            row=1,
            col=2,
        )
    values = list(memory.scene_graph.objects.values())
    figure.add_trace(
        go.Table(
            header={"values": ["ID", "Category", "Confidence", "Observations", "First", "Last"]},
            cells={
                "values": [
                    [value.object_id[:12] for value in values],
                    [value.category for value in values],
                    [f"{value.confidence:.3f}" for value in values],
                    [value.observation_count for value in values],
                    [value.first_seen_time for value in values],
                    [value.last_seen_time for value in values],
                ]
            },
        ),
        row=2,
        col=1,
    )
    figure.update_layout(height=900, template="plotly_white", title="JEPA-4D Persistent Memory")
    metadata = {
        "snapshot": memory.snapshot().to_serializable(),
        "persistence": persistence_stats or {},
        "wandb_url": wandb_url,
    }
    target.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>JEPA-4D memory report</title>"
        "<style>body{font-family:system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}"
        "pre{background:#f6f8fa;padding:1rem;overflow:auto}</style></head><body>"
        "<h1>Persistent 4D memory report</h1>"
        "<p>Trajectories and confidence are observation histories, not ground-truth tracks.</p>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        f"<h2>Snapshot and persistence metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre>"
        "</body></html>"
    )
    return target
