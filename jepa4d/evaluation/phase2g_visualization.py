"""Deterministic local/sanitized visual evidence for formal Phase 2g-A."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from jepa4d.evaluation.phase2f_metrics import self_contained_html
from jepa4d.training.phase2g_protocol import ARMS, CANDIDATES, QUALITATIVE_IDS_PER_FAMILY


def _heatmap(values: np.ndarray, *, size: int = 180) -> Image.Image:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if array.ndim != 2 or not finite.any():
        raise ValueError("qualitative heatmap requires a finite 2D array")
    low, high = float(array[finite].min()), float(array[finite].max())
    normalized = np.zeros_like(array) if high <= low else np.clip((array - low) / (high - low), 0, 1)
    red = np.clip(1.7 * normalized, 0, 1)
    blue = np.clip(1.7 * (1 - normalized), 0, 1)
    green = np.clip(1.5 - np.abs(2 * normalized - 1) * 1.5, 0, 1)
    rgb = np.round(np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), resample=Image.Resampling.NEAREST)


def _labelled(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), "white")
    canvas.paste(image.convert("RGB"), (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), label, fill="#17202e")
    return canvas


def write_local_qualitative_panels(
    output: Path,
    *,
    family: str,
    sample_ids: Sequence[str],
    rgb_uint8: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    scale_field: torch.Tensor | None,
) -> tuple[Path, Path, list[str]]:
    """Write fixed-ID RGB/target/pred/error/uncertainty/field panels locally.

    The returned PNG and manifest are protected local artifacts and must never
    be passed to the W&B uploader.
    """

    count = len(sample_ids)
    if any(len(value) != count for value in (rgb_uint8, target_depth, valid_mask, log_depth, log_variance)):
        raise ValueError("qualitative tensors and IDs must have equal rows")
    selected = sorted(
        range(count),
        key=lambda index: (hashlib.sha256(sample_ids[index].encode()).hexdigest(), sample_ids[index]),
    )[:QUALITATIVE_IDS_PER_FAMILY]
    selected_ids = [sample_ids[index] for index in selected]
    row_width = 6 * 180
    row_height = 204
    contact = Image.new("RGB", (row_width, row_height * len(selected)), "#e9edf3")
    for row_index, source_index in enumerate(selected):
        rgb = rgb_uint8[source_index].detach().cpu()
        if rgb.shape != (3, 384, 384) or rgb.dtype != torch.uint8:
            raise ValueError("qualitative RGB must be uint8 [3,384,384]")
        rgb_image = Image.fromarray(rgb.permute(1, 2, 0).numpy()).resize((180, 180), Image.Resampling.BILINEAR)
        target = target_depth[source_index].detach().cpu().double().numpy()
        valid = valid_mask[source_index].detach().cpu().numpy().astype(bool)
        prediction = log_depth[source_index].detach().cpu().double().exp().numpy()
        uncertainty = log_variance[source_index].detach().cpu().double().mul(0.5).exp().numpy()
        target_display = np.where(valid, target, 0.0)
        error = np.where(valid, np.abs(prediction - target), 0.0)
        field = (
            np.zeros_like(target) if scale_field is None else scale_field[source_index].detach().cpu().double().numpy()
        )
        panels = (
            _labelled(rgb_image, "RGB (protected local)"),
            _labelled(_heatmap(target_display), "target depth"),
            _labelled(_heatmap(prediction), "prediction"),
            _labelled(_heatmap(error), "absolute error"),
            _labelled(_heatmap(uncertainty), "uncertainty"),
            _labelled(_heatmap(field), "scale field"),
        )
        for column, panel in enumerate(panels):
            contact.paste(panel, (column * 180, row_index * row_height))
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "qualitative_panels.png"
    contact.save(panel_path, optimize=True)
    manifest = {
        "schema_version": "jepa4d-phase2g-qualitative-panels-v1",
        "family": family,
        "selection": "lowest_sha256_sample_id_before_training",
        "count": len(selected_ids),
        "sample_ids": selected_ids,
        "contains_protected_rgb_and_target_previews": True,
        "local_only": True,
        "wandb_upload_forbidden": True,
    }
    manifest_path = output / "qualitative_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return panel_path, manifest_path, selected_ids


def _chart(title: str, rows: Sequence[tuple[str, float]], path: Path, *, lower_is_better: bool = True) -> Path:
    image = Image.new("RGB", (1120, 620), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill="#17202e")
    if not rows:
        draw.text((30, 80), "No applicable rows", fill="#6b7280")
    else:
        values = [float(value) for _, value in rows]
        minimum, maximum = min(values), max(values)
        span = max(maximum - min(0.0, minimum), 1e-12)
        height = max(18, min(48, int(500 / len(rows))))
        for index, (label, value) in enumerate(rows):
            y = 70 + index * height
            width = int(700 * (value - min(0.0, minimum)) / span)
            color = "#2f6fed" if lower_is_better else "#238636"
            draw.text((25, y + 3), label[:42], fill="#17202e")
            draw.rectangle((350, y, 350 + max(width, 2), y + height - 5), fill=color)
            draw.text((1060, y + 3), f"{value:.5g}", anchor="ra", fill="#17202e")
    image.save(path)
    return path


_PLOT_COLORS = ("#2f6fed", "#d34a4a", "#238636", "#8b5cf6", "#e18b27", "#0891b2")


def _multi_panel_lines(
    title: str,
    panels: Sequence[tuple[str, Mapping[str, Sequence[float]]]],
    path: Path,
) -> Path:
    """Draw deterministic line analyses with an independent y-scale per panel."""

    width = 1200
    panel_height = 190
    image = Image.new("RGB", (width, 70 + panel_height * len(panels)), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), title, fill="#17202e")
    for panel_index, (panel_name, series) in enumerate(panels):
        top = 55 + panel_index * panel_height
        left, right, bottom = 80, 930, top + 145
        values = [float(value) for rows in series.values() for value in rows]
        draw.text((28, top), panel_name, fill="#17202e")
        if not values:
            draw.text((100, top + 60), "No applicable values", fill="#6b7280")
            continue
        low, high = min(values), max(values)
        padding = max((high - low) * 0.08, abs(high) * 0.01, 1e-9)
        low, high = low - padding, high + padding
        draw.line((left, top + 20, left, bottom), fill="#596273", width=2)
        draw.line((left, bottom, right, bottom), fill="#596273", width=2)
        for series_index, (label, rows) in enumerate(series.items()):
            color = _PLOT_COLORS[series_index % len(_PLOT_COLORS)]
            points = []
            for index, raw in enumerate(rows):
                x = left if len(rows) == 1 else left + (right - left) * index / (len(rows) - 1)
                y = bottom - (bottom - top - 20) * (float(raw) - low) / (high - low)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            legend_y = top + 20 + series_index * 22
            draw.rectangle((960, legend_y, 975, legend_y + 10), fill=color)
            draw.text((982, legend_y - 3), label[:30], fill="#17202e")
        draw.text((left, bottom + 8), "ordered epoch/profile/coverage index", fill="#596273")
        draw.text((right, top + 2), f"range {low:.4g} .. {high:.4g}", anchor="ra", fill="#596273")
    image.save(path)
    return path


def _forest_plot(title: str, rows: Sequence[tuple[str, float, float, float]], path: Path) -> Path:
    row_height = 28 if len(rows) > 10 else 105
    image_height = max(520, 115 + row_height * len(rows))
    image = Image.new("RGB", (1320, image_height), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill="#17202e")
    values = [value for _, point, low, high in rows for value in (point, low, high)] + [0.0]
    minimum, maximum = min(values), max(values)
    padding = max((maximum - minimum) * 0.12, 1e-6)
    minimum, maximum = minimum - padding, maximum + padding
    left, right = 360, 1060

    def x_of(value: float) -> float:
        return left + (right - left) * (value - minimum) / (maximum - minimum)

    zero = x_of(0.0)
    draw.line((zero, 60, zero, image_height - 55), fill="#7b8494", width=2)
    for index, (label, point, low, high) in enumerate(rows):
        y = 85 + index * row_height
        draw.text((25, y - 8), label, fill="#17202e")
        draw.line((x_of(low), y, x_of(high), y), fill="#2f6fed", width=5)
        draw.line((x_of(low), y - 8, x_of(low), y + 8), fill="#2f6fed", width=2)
        draw.line((x_of(high), y - 8, x_of(high), y + 8), fill="#2f6fed", width=2)
        draw.ellipse((x_of(point) - 6, y - 6, x_of(point) + 6, y + 6), fill="#d34a4a")
        draw.text((1280, y - 8), f"{point:+.5f} [{low:+.5f}, {high:+.5f}]", anchor="ra", fill="#17202e")
    draw.text(
        (left, image_height - 35),
        "candidate minus paired M0 raw AbsRel; cell rows use paired-frame normal CIs, aggregate rows use 100k hierarchical CIs",
        fill="#596273",
    )
    image.save(path)
    return path


def _scatter_residual_plot(
    title: str,
    predicted: Sequence[float],
    optimal: Sequence[float],
    path: Path,
) -> Path:
    if len(predicted) != len(optimal) or not predicted:
        raise ValueError("scale scatter requires paired non-empty values")
    stride = max(1, len(predicted) // 4000)
    x_values = np.asarray(predicted[::stride], dtype=np.float64)
    y_values = np.asarray(optimal[::stride], dtype=np.float64)
    residuals = x_values - y_values
    image = Image.new("RGB", (1200, 650), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, fill="#17202e")
    low = float(min(x_values.min(), y_values.min()))
    high = float(max(x_values.max(), y_values.max()))
    padding = max((high - low) * 0.05, 1e-6)
    low, high = low - padding, high + padding
    left, top, size = 70, 75, 500

    def scale(value: float) -> float:
        return left + size * (float(value) - low) / (high - low)

    draw.rectangle((left, top, left + size, top + size), outline="#596273", width=2)
    draw.line(
        (scale(low), top + size - (scale(low) - left), scale(high), top + size - (scale(high) - left)),
        fill="#d34a4a",
        width=2,
    )
    for predicted_value, optimal_value in zip(x_values, y_values, strict=True):
        x = scale(predicted_value)
        y = top + size - (scale(optimal_value) - left)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#2f6fed")
    draw.text((left, 590), "predicted log scale", fill="#596273")
    draw.text((left, 55), "optimal log scale", fill="#596273")
    counts, edges = np.histogram(residuals, bins=30)
    hist_left, hist_right, hist_bottom = 680, 1120, 575
    maximum = max(int(counts.max()), 1)
    bar_width = (hist_right - hist_left) / len(counts)
    for index, count in enumerate(counts):
        x0 = hist_left + index * bar_width
        height = 430 * int(count) / maximum
        draw.rectangle((x0, hist_bottom - height, x0 + bar_width - 1, hist_bottom), fill="#238636")
    draw.text((hist_left, 75), "residual distribution (predicted - optimal)", fill="#17202e")
    draw.text((hist_left, 590), f"range {edges[0]:+.4g} .. {edges[-1]:+.4g}; n={len(residuals)}", fill="#596273")
    image.save(path)
    return path


def _histogram_plot(title: str, series: Mapping[str, Sequence[float]], path: Path) -> Path:
    image = Image.new("RGB", (1120, 620), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, fill="#17202e")
    values = [float(value) for rows in series.values() for value in rows]
    if not values:
        draw.text((40, 90), "No applicable values", fill="#596273")
        image.save(path)
        return path
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1e-9
    bins = np.linspace(low, high, 31)
    for series_index, (label, rows) in enumerate(series.items()):
        counts, _ = np.histogram(np.asarray(rows, dtype=np.float64), bins=bins)
        maximum = max(int(counts.max()), 1)
        baseline = 520 - series_index * 8
        color = _PLOT_COLORS[series_index % len(_PLOT_COLORS)]
        for index, count in enumerate(counts):
            x0 = 80 + index * 28
            height = 390 * int(count) / maximum
            draw.rectangle((x0, baseline - height, x0 + 22, baseline), outline=color, width=2)
        draw.text((930, 80 + series_index * 24), f"{label}: n={len(rows)}", fill=color)
    draw.text((80, 560), f"value range {low:.5g} .. {high:.5g}", fill="#596273")
    image.save(path)
    return path


def _field_analysis_plot(
    title: str,
    *,
    field_series: Mapping[str, Sequence[float]],
    performance_ratios: Mapping[str, float],
    path: Path,
) -> Path:
    """Combine M3 field distributions with full-versus-zero performance."""

    image = Image.new("RGB", (1260, 650), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, fill="#17202e")
    values = [float(value) for rows in field_series.values() for value in rows]
    if not values:
        raise ValueError("M3 field analysis requires distribution values")
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1e-9
    bins = np.linspace(low, high, 31)
    for series_index, (label, rows) in enumerate(field_series.items()):
        counts, _ = np.histogram(np.asarray(rows, dtype=np.float64), bins=bins)
        maximum = max(int(counts.max()), 1)
        color = _PLOT_COLORS[series_index]
        for index, count in enumerate(counts):
            x0 = 60 + index * 20
            height = 430 * int(count) / maximum
            baseline = 550 - series_index * 6
            draw.rectangle((x0, baseline - height, x0 + 15, baseline), outline=color, width=2)
        draw.text((60, 80 + series_index * 25), f"{label}: n={len(rows)}", fill=color)
    draw.text((60, 585), f"field statistic range {low:.5g} .. {high:.5g}", fill="#596273")
    draw.text((730, 75), "full / zero-field raw AbsRel (lower than 1 favors full field)", fill="#17202e")
    for index, (family, ratio) in enumerate(performance_ratios.items()):
        y = 125 + index * 85
        width = min(420, max(2, 350 * float(ratio)))
        draw.text((730, y + 8), family, fill="#17202e")
        draw.rectangle((850, y, 850 + width, y + 38), fill="#238636" if ratio <= 1 else "#d34a4a")
        draw.line((850 + 350, y - 5, 850 + 350, y + 43), fill="#17202e", width=2)
        draw.text((1230, y + 8), f"{ratio:.5f}", anchor="ra", fill="#17202e")
    image.save(path)
    return path


def _completeness_matrix(receipts: Sequence[Mapping[str, Any]], path: Path) -> Path:
    image = Image.new("RGB", (1200, 520), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), "10. Provenance, failure, retry, and completeness matrix", fill="#17202e")
    rotations = ("R0", "R1", "R2", "R3")
    seeds = (0, 1, 2)
    for arm_index, arm in enumerate(ARMS):
        y = 95 + arm_index * 85
        draw.text((30, y + 15), arm, fill="#17202e")
        for column, (rotation, seed) in enumerate((rotation, seed) for rotation in rotations for seed in seeds):
            x = 100 + column * 78
            matches = [
                row
                for row in receipts
                if row.get("arm") == arm and row.get("rotation") == rotation and row.get("seed") == seed
            ]
            complete = len(matches) == 1 and matches[0].get("status") == "success"
            draw.rectangle((x, y, x + 62, y + 48), fill="#238636" if complete else "#d34a4a")
            draw.text((x + 7, y + 15), f"{rotation}s{seed}", fill="white")
    job_ids = [str(row.get("execution_provenance", {}).get("slurm", {}).get("job_id", "")) for row in receipts]
    duplicates = len(job_ids) - len(set(job_ids))
    draw.text(
        (100, 455),
        f"completed={len(receipts)}/48; failed receipts={sum(row.get('status') != 'success' for row in receipts)}; "
        f"duplicate logical job IDs={duplicates}; scheduler retry history is independently audited by terminal Z",
        fill="#596273",
    )
    image.save(path)
    return path


def write_training_visualizations(output: Path, history: Sequence[Mapping[str, Any]]) -> tuple[Path, Path, Path]:
    """Write sanitized loss/gradient and resource diagnostics plus HTML."""

    diagnostics = _multi_panel_lines(
        "Phase 2g training objective and forbidden-gradient diagnostics",
        (
            ("objective", {"train total": [float(row["train_total"]) for row in history]}),
            (
                "allowed gradient norms",
                {
                    name: [float(row.get(f"train_gradient_norm_{name}", 0.0)) for row in history]
                    for name in ("shape", "scale", "field")
                },
            ),
            (
                "forbidden-gradient firewall",
                {"maximum forbidden": [float(row["gradient_firewall_max_forbidden_norm"]) for row in history]},
            ),
        ),
        output / "training_diagnostics.png",
    )
    resources = _multi_panel_lines(
        "Phase 2g descriptive epoch time, memory, and throughput",
        (
            ("epoch seconds", {"seconds": [float(row["train_epoch_seconds"]) for row in history]}),
            (
                "source-group throughput",
                {"groups/second": [float(row["throughput_source_groups_per_second"]) for row in history]},
            ),
            (
                "CUDA memory",
                {
                    "allocated GiB": [float(row["peak_cuda_allocated_bytes"]) / 2**30 for row in history],
                    "reserved GiB": [float(row["peak_cuda_reserved_bytes"]) / 2**30 for row in history],
                },
            ),
        ),
        output / "training_resources.png",
    )
    report = output / "training_report.html"
    report.write_text(
        self_contained_html(
            "Phase 2g training diagnostics",
            {
                "epochs": len(history),
                "final_train_total": history[-1]["train_total"],
                "maximum_forbidden_gradient_norm": max(
                    float(row["gradient_firewall_max_forbidden_norm"]) for row in history
                ),
                "resource_metrics_are_descriptive_only": True,
            },
            images=(("Loss and gradients", diagnostics), ("Resources", resources)),
            claim_boundary="SUN RGB-D aggregate training diagnostics; no raw data or target preview is embedded.",
        ),
        encoding="utf-8",
    )
    return diagnostics, resources, report


def write_aggregate_visualizations(
    output: Path,
    *,
    result: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[Path, ...]:
    """Write all ten frozen sanitized aggregate visualization categories."""

    output.mkdir(parents=True, exist_ok=True)
    aggregates = result["aggregates"]
    figures: list[tuple[str, Path]] = []
    bootstrap = result["hierarchical_bootstrap"]
    forest_rows: list[tuple[str, float, float, float]] = []
    for receipt in receipts:
        if receipt["arm"] not in CANDIDATES:
            continue
        reference = next(
            row
            for row in receipts
            if row["arm"] == "M0"
            and row["heldout_family"] == receipt["heldout_family"]
            and row["seed"] == receipt["seed"]
        )
        reference_by_frame = {row["frame_id"]: float(row["raw_abs_rel"]) for row in reference["metrics"]["per_frame"]}
        differences = np.asarray(
            [
                float(row["raw_abs_rel"]) - reference_by_frame[row["frame_id"]]
                for row in receipt["metrics"]["per_frame"]
            ],
            dtype=np.float64,
        )
        effect = float(differences.mean())
        half_width = 1.959963984540054 * float(differences.std(ddof=1)) / math.sqrt(len(differences))
        forest_rows.append(
            (
                f"{receipt['arm']} {receipt['heldout_family']} seed {receipt['seed']}",
                effect,
                effect - half_width,
                effect + half_width,
            )
        )
    forest_rows.extend(
        (
            f"{arm} overall hierarchical",
            float(bootstrap[arm]["raw_abs_rel"]["observed"]),
            float(bootstrap[arm]["raw_abs_rel"]["ci95"][0]),
            float(bootstrap[arm]["raw_abs_rel"]["ci95"][1]),
        )
        for arm in CANDIDATES
    )
    figures.append(
        (
            "1. Per-family/per-seed paired forest",
            _forest_plot(
                "1. Paired candidate-minus-M0 raw AbsRel with descriptive 95% intervals",
                forest_rows,
                output / "01_forest.png",
            ),
        )
    )
    figures.append(
        (
            "2. Raw/aligned AbsRel and scale error",
            _chart(
                "2. Equal-family raw/aligned AbsRel and absolute scale error",
                [
                    (f"{arm} {metric}", float(aggregates[arm][metric]))
                    for arm in ARMS
                    for metric in ("raw_abs_rel", "aligned_abs_rel", "absolute_log_scale_error")
                ],
                output / "02_quality_scale.png",
            ),
        )
    )
    predicted_scale: list[float] = []
    optimal_scale: list[float] = []
    for receipt in receipts:
        mechanism = receipt.get("scale_mechanism")
        if isinstance(mechanism, Mapping) and isinstance(mechanism.get("per_frame"), list):
            predicted_scale.extend(float(row["predicted_log_scale"]) for row in mechanism["per_frame"])
            optimal_scale.extend(float(row["optimal_log_scale"]) for row in mechanism["per_frame"])
    figures.append(
        (
            "3. Scale residual/correlation",
            _scatter_residual_plot(
                "3. Predicted-versus-optimal scale and residual distribution",
                predicted_scale,
                optimal_scale,
                output / "03_scale_mechanism.png",
            ),
        )
    )
    reliability_series: dict[str, list[float]] = {}
    risk_series: dict[str, list[float]] = {}
    risk_grid = np.linspace(0.0, 1.0, 64)
    for arm in ARMS:
        arm_receipts = [receipt for receipt in receipts if receipt["arm"] == arm]
        reliability_series[arm] = [
            float(
                np.mean(
                    [
                        next(
                            row["empirical"]
                            for row in receipt["metrics"]["coverage"]
                            if float(row["nominal"]) == nominal
                        )
                        for receipt in arm_receipts
                    ]
                )
            )
            for nominal in (0.5, 0.8, 0.9, 0.95)
        ]
        interpolated = [
            np.interp(
                risk_grid,
                np.asarray(receipt["metrics"]["risk_coverage"]["coverage"], dtype=np.float64),
                np.asarray(receipt["metrics"]["risk_coverage"]["risk"], dtype=np.float64),
            )
            for receipt in arm_receipts
        ]
        risk_series[arm] = np.mean(np.stack(interpolated), axis=0).tolist()
    figures.append(
        (
            "4. Reliability/risk coverage",
            _multi_panel_lines(
                "4. Reliability and risk-coverage curves",
                (
                    ("empirical coverage at nominal 50/80/90/95%", reliability_series),
                    ("mean risk versus retained coverage", risk_series),
                ),
                output / "04_uncertainty.png",
            ),
        )
    )
    camera_series: dict[str, list[float]] = {}
    for arm in ("M2", "M3"):
        arm_receipts = [receipt for receipt in receipts if receipt["arm"] == arm]
        for condition in ("updated", "stale", "wrong", "permuted"):
            camera_series[f"{arm} {condition}"] = [
                float(
                    np.mean(
                        [
                            receipt["camera_controls"]["profile_raw_abs_rel"][condition][f"P{profile}"]
                            for receipt in arm_receipts
                        ]
                    )
                )
                for profile in range(1, 8)
            ]
    figures.append(
        (
            "5. Camera controls",
            _multi_panel_lines(
                "5. P1-P7 camera controls by intrinsics condition",
                (("held-out raw AbsRel across profiles P1..P7", camera_series),),
                output / "05_camera_controls.png",
            ),
        )
    )
    m3_receipts = [receipt for receipt in receipts if receipt["arm"] == "M3"]
    field_means = [
        float(value) for receipt in m3_receipts for value in receipt["zero_field_intervention"]["per_frame_field_mean"]
    ]
    field_sds = [
        float(value) for receipt in m3_receipts for value in receipt["zero_field_intervention"]["per_frame_field_sd"]
    ]
    field_performance = {
        family: float(values["full"]) / float(values["zero"])
        for family, values in result["m3_zero_field_mechanism"]["per_family"].items()
    }
    field_performance["equal-family"] = float(result["m3_zero_field_mechanism"]["ratio"])
    figures.append(
        (
            "6. M3 field intervention",
            _field_analysis_plot(
                "6. M3 full-versus-zero-field performance and predicted field distributions",
                field_series={"per-frame field mean": field_means, "per-frame spatial SD": field_sds},
                performance_ratios=field_performance,
                path=output / "06_field.png",
            ),
        )
    )
    qualitative_counts = [
        (
            f"{arm} panels complete",
            float(sum(receipt["qualitative"]["count"] for receipt in receipts if receipt["arm"] == arm)),
        )
        for arm in ARMS
    ]
    figures.append(
        (
            "7. Fixed qualitative audit",
            _chart(
                "7. Local-only fixed qualitative panel completeness (16 x 12 per arm)",
                qualitative_counts,
                output / "07_qualitative_audit.png",
                lower_is_better=False,
            ),
        )
    )

    def mean_epoch_series(arm: str, key: str) -> list[float]:
        histories = [
            receipt["training_diagnostics"]["epoch_diagnostics"] for receipt in receipts if receipt["arm"] == arm
        ]
        lengths = {len(history) for history in histories}
        if len(lengths) != 1 or not histories:
            raise ValueError(f"incomplete epoch diagnostics for {arm}")
        return [float(np.mean([history[index][key] for history in histories])) for index in range(len(histories[0]))]

    figures.append(
        (
            "8. Loss/gradient curves",
            _multi_panel_lines(
                "8. Formal loss and allowed/forbidden-gradient curves",
                (
                    ("mean train objective", {arm: mean_epoch_series(arm, "train_total") for arm in ARMS}),
                    (
                        "mean sum of allowed gradient norms",
                        {
                            arm: [
                                sum(values)
                                for values in zip(
                                    mean_epoch_series(arm, "allowed_gradient_norm_shape"),
                                    mean_epoch_series(arm, "allowed_gradient_norm_scale"),
                                    mean_epoch_series(arm, "allowed_gradient_norm_field"),
                                    strict=True,
                                )
                            ]
                            for arm in ARMS
                        },
                    ),
                    (
                        "maximum forbidden gradient norm",
                        {arm: mean_epoch_series(arm, "forbidden_gradient_norm") for arm in ARMS},
                    ),
                ),
                output / "08_training_diagnostics.png",
            ),
        )
    )
    parameter_components = (
        "shape_decoder",
        "scale_projection",
        "scale_head",
        "camera_prompt",
        "coarse_scale_field",
    )
    component_parameters = {
        component: [
            float(
                next(
                    receipt["training_diagnostics"]["parameter_counts"][component]
                    for receipt in receipts
                    if receipt["arm"] == arm
                )
            )
            for arm in ARMS
        ]
        for component in parameter_components
    }
    figures.append(
        (
            "9. Resource diagnostics",
            _multi_panel_lines(
                "9. Descriptive resource and component summaries",
                (
                    ("epoch seconds", {arm: mean_epoch_series(arm, "epoch_seconds") for arm in ARMS}),
                    (
                        "source-group throughput",
                        {arm: mean_epoch_series(arm, "throughput_source_groups_per_second") for arm in ARMS},
                    ),
                    (
                        "peak allocated GiB",
                        {
                            arm: [value / 2**30 for value in mean_epoch_series(arm, "peak_cuda_allocated_bytes")]
                            for arm in ARMS
                        },
                    ),
                    ("trainable parameters by component across M0..M3", component_parameters),
                ),
                output / "09_resources.png",
            ),
        )
    )
    figures.append(
        (
            "10. Provenance/failure/completeness",
            _completeness_matrix(receipts, output / "10_completeness.png"),
        )
    )

    csv_path = output / "aggregate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arm", "raw_abs_rel", "aligned_abs_rel", "absolute_log_scale_error", "nll", "ause"))
        for arm in ARMS:
            writer.writerow(
                (
                    arm,
                    *(
                        aggregates[arm][name]
                        for name in ("raw_abs_rel", "aligned_abs_rel", "absolute_log_scale_error", "nll", "ause")
                    ),
                )
            )
    component_table = output / "component_resource_table.csv"
    with component_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "arm",
                "component",
                "trainable_parameters",
                "mean_epoch_seconds",
                "mean_source_groups_per_second",
                "mean_peak_allocated_gib",
            )
        )
        for arm_index, arm in enumerate(ARMS):
            for component in parameter_components:
                writer.writerow(
                    (
                        arm,
                        component,
                        int(component_parameters[component][arm_index]),
                        float(np.mean(mean_epoch_series(arm, "epoch_seconds"))),
                        float(np.mean(mean_epoch_series(arm, "throughput_source_groups_per_second"))),
                        float(np.mean(mean_epoch_series(arm, "peak_cuda_allocated_bytes"))) / 2**30,
                    )
                )
    manifest = output / "visualization_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "jepa4d-phase2g-aggregate-visualization-manifest-v1",
                "categories": [
                    {
                        "index": 1,
                        "plot_type": "per-family-per-seed-paired-forest-plus-hierarchical-intervals",
                        "data_points": len(forest_rows),
                    },
                    {"index": 2, "plot_type": "grouped-quality-scale-bars", "data_points": 12},
                    {
                        "index": 3,
                        "plot_type": "predicted-optimal-scatter-plus-residual-histogram",
                        "data_points": len(predicted_scale),
                    },
                    {
                        "index": 4,
                        "plot_type": "reliability-and-risk-coverage-curves",
                        "data_points": sum(len(rows) for rows in risk_series.values()),
                    },
                    {
                        "index": 5,
                        "plot_type": "p1-p7-camera-control-curves",
                        "data_points": sum(len(rows) for rows in camera_series.values()),
                    },
                    {
                        "index": 6,
                        "plot_type": "m3-full-zero-performance-and-field-distributions",
                        "data_points": len(field_means) + len(field_sds) + len(field_performance),
                    },
                    {"index": 7, "plot_type": "fixed-local-qualitative-completeness", "data_points": len(receipts)},
                    {
                        "index": 8,
                        "plot_type": "loss-allowed-forbidden-gradient-curves",
                        "data_points": sum(len(mean_epoch_series(arm, "train_total")) for arm in ARMS),
                    },
                    {
                        "index": 9,
                        "plot_type": "memory-throughput-component-curves",
                        "data_points": sum(len(mean_epoch_series(arm, "epoch_seconds")) for arm in ARMS),
                    },
                    {
                        "index": 10,
                        "plot_type": "provenance-failure-retry-completeness-matrix",
                        "data_points": len(receipts),
                    },
                ],
                "protected_pixels_embedded": False,
                "per_frame_identifiers_embedded": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = output / "report.html"
    report.write_text(
        self_contained_html(
            "Phase 2g-A development selection evidence",
            {
                "survivor": result["survivor"] or "none",
                "retained_arm": result["retained_arm"],
                "complete_evaluation_cells": len(receipts),
                "qualitative_panels": "protected local-only; aggregate completeness shown",
                "external_final_authorized": False,
            },
            images=figures,
            claim_boundary="SUN RGB-D development aggregates only; protected qualitative pixels and external-final data are absent.",
        ),
        encoding="utf-8",
    )
    return (*[path for _, path in figures], csv_path, component_table, manifest, report)
