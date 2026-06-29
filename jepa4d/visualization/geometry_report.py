"""Interactive depth, uncertainty, camera, and point-cloud report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jepa4d.data.schemas import RGBInputBatch
from jepa4d.models.geometry_belief import GeometryBelief


def _sample_points(belief: GeometryBelief, limit: int = 15_000) -> tuple[np.ndarray, np.ndarray]:
    if belief.pointmap_mean is None:
        return np.empty((0, 3)), np.empty((0,))
    points_tensor = belief.pointmap_mean[0].detach().float().cpu().reshape(-1, 3)
    points_array = points_tensor.numpy()
    valid = np.isfinite(points_array).all(axis=-1)
    points = points_array[valid]
    if belief.pointmap_logvar is None:
        uncertainty = np.zeros(len(points))
    else:
        uncertainty = belief.pointmap_logvar[0].detach().float().cpu().reshape(-1, 3).mean(dim=-1).numpy()[valid]
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit).astype(int)
        points, uncertainty = points[indices], uncertainty[indices]
    return points, uncertainty


def build_geometry_report(
    batch: RGBInputBatch,
    belief: GeometryBelief,
    output: str | Path,
    *,
    wandb_url: str | None = None,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    points, uncertainty = _sample_points(belief)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}], [{"type": "scene", "colspan": 2}, None]],
        subplot_titles=("Depth belief", "Depth log-variance", "Interactive world-frame point cloud"),
        vertical_spacing=0.1,
    )
    if belief.depth_mean is not None:
        figure.add_trace(go.Heatmap(z=belief.depth_mean[0, 0, 0].detach().cpu(), colorscale="Viridis"), row=1, col=1)
    if belief.depth_logvar is not None:
        figure.add_trace(go.Heatmap(z=belief.depth_logvar[0, 0, 0].detach().cpu(), colorscale="Magma"), row=1, col=2)
    if len(points):
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 1.5, "color": uncertainty, "colorscale": "Turbo", "colorbar": {"title": "logvar"}},
                name="scene points",
            ),
            row=2,
            col=1,
        )
    figure.update_layout(height=950, template="plotly_white", title="JEPA-4D Geometry Belief Diagnostics")
    metadata: dict[str, Any] = {
        "input": batch.to_serializable(),
        "belief": belief.to_serializable(),
        "wandb_url": wandb_url,
        "interpretation": {
            "scale_confidence": "confidence that distances have task-usable scale; not benchmark-calibrated yet",
            "pose_confidence": "confidence in relative/world camera poses",
            "reconstruction_confidence": "aggregate dense-geometry confidence",
        },
    }
    target.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>JEPA-4D geometry report</title>"
        "<style>body{font-family:system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem}"
        "pre{background:#f6f8fa;padding:1rem;overflow:auto}.warning{background:#fff3cd;padding:1rem}</style></head><body>"
        "<h1>JEPA-4D geometry belief</h1>"
        f"<p>Backend: <b>{html.escape(str(belief.metadata.get('backend')))}</b>; mode: <b>{html.escape(belief.mode)}</b></p>"
        "<p class='warning'>Geometry is a probabilistic belief. Low scale confidence, especially for an uncalibrated single "
        "image, must trigger verification rather than metric planning.</p>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        f"<h2>Complete metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre></body></html>"
    )
    return target
