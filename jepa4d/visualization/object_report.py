"""Interactive object grounding and association report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from jepa4d.data.schemas import RGBInputBatch
from jepa4d.models.object_slot_grounder import ObjectGroundingResult


def build_object_report(
    batch: RGBInputBatch,
    result: ObjectGroundingResult,
    output: str | Path,
    *,
    wandb_url: str | None = None,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = (batch.images[0, 0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    scale = min(1.0, 1200 / max(image.shape[:2]))
    if scale < 1.0:
        display_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        image = np.asarray(Image.fromarray(image).resize(display_size, Image.Resampling.BILINEAR))
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "image"}, {"type": "table"}]],
        column_widths=[0.6, 0.4],
        subplot_titles=("View 0 / time 0 detections", "Persistent object slots"),
    )
    figure.add_trace(go.Image(z=image), row=1, col=1)
    colors = ["#ef553b", "#00cc96", "#636efa", "#ab63fa", "#ffa15a"]
    visible = [value for value in result.observations if value.view_index == 0 and value.time_index == 0]
    for index, observation in enumerate(visible):
        x1, y1, x2, y2 = (value * scale for value in observation.bbox_2d)
        figure.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line={"color": colors[index % len(colors)], "width": 3},
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=x1,
            y=y1,
            text=f"{observation.category} {observation.score:.2f}",
            showarrow=False,
            bgcolor=colors[index % len(colors)],
            font={"color": "white"},
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Table(
            header={"values": ["ID", "Category", "Obs.", "Detection", "Pose"]},
            cells={
                "values": [
                    [slot.object_id[:8] for slot in result.slots],
                    [slot.category for slot in result.slots],
                    [len(slot.observations) for slot in result.slots],
                    [f"{slot.confidence.get('detection', 0):.3f}" for slot in result.slots],
                    ["available" if slot.pose_map is not None else "unknown" for slot in result.slots],
                ]
            },
        ),
        row=1,
        col=2,
    )
    figure.update_layout(height=650, template="plotly_white", title="JEPA-4D Object Grounding and Association")
    metadata = result.to_serializable()
    metadata["wandb_url"] = wandb_url
    target.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>JEPA-4D object report</title>"
        "<style>body{font-family:system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem}"
        "pre{background:#f6f8fa;padding:1rem;overflow:auto}.warning{background:#fff3cd;padding:1rem}</style></head><body>"
        "<h1>Object grounding and persistent slots</h1>"
        f"<p>Detector: <b>{html.escape(str(result.metadata['detector_backend']))}</b>; "
        f"mask backend: <b>{html.escape(str(result.metadata['mask_backend']))}</b></p>"
        "<p class='warning'>Mock detections validate interfaces only. Teacher detections remain uncertain observations "
        "until associated and verified across views, time, and geometry.</p>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        f"<h2>Complete metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre></body></html>"
    )
    return target
