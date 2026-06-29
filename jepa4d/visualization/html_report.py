"""Self-contained interactive feature-extraction report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jepa4d.data.schemas import JEPATokenBundle, RGBInputBatch
from jepa4d.visualization.observability import pca_rgb, temporal_cosine


def build_feature_report(
    batch: RGBInputBatch,
    bundle: JEPATokenBundle,
    output: str | Path,
    *,
    runtime: dict[str, float],
    wandb_url: str | None = None,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pca = pca_rgb(bundle.dense_tokens[0, 0, 0], bundle.patch_grid)
    cosine = temporal_cosine(bundle)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Dense-token PCA", "Temporal consistency"))
    figure.add_trace(go.Image(z=(pca * 255).astype("uint8")), row=1, col=1)
    figure.add_trace(go.Scatter(y=cosine, mode="lines+markers", name="adjacent cosine"), row=1, col=2)
    figure.update_layout(height=480, template="plotly_white", title="JEPA-4D Feature Diagnostics")
    metadata: dict[str, Any] = {
        "input_mode": batch.mode,
        "batch_size": batch.images.shape[0],
        "views": batch.images.shape[1],
        "input_timesteps": batch.images.shape[2],
        "valid_observations": int(batch.valid_mask.sum()),
        "dense_token_shape": list(bundle.dense_tokens.shape),
        "global_token_shape": list(bundle.global_tokens.shape),
        "layer_token_shapes": {str(k): list(v.shape) for k, v in bundle.layer_tokens.items()},
        "patch_grid": list(bundle.patch_grid),
        "model_config": bundle.metadata.get("model", {}),
        "runtime": runtime,
        "wandb_url": wandb_url,
    }
    figure_html = figure.to_html(full_html=False, include_plotlyjs=True)
    target.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>JEPA-4D report</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        "pre{background:#f5f5f5;padding:1rem;overflow:auto}.status{color:#176b35}</style></head><body>"
        "<h1>JEPA-4D feature extraction</h1><p class='status'>Completed successfully</p>"
        f"<p>Mode: <b>{html.escape(batch.mode)}</b> · Views: {batch.images.shape[1]} · "
        f"Timesteps: {batch.images.shape[2]}</p>{figure_html}<h2>Run metadata</h2>"
        f"<pre>{html.escape(json.dumps(metadata, indent=2))}</pre></body></html>"
    )
    return target
