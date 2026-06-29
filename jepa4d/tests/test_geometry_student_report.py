import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from jepa4d.visualization.geometry_student_report import (
    _peak_memory,
    build_geometry_student_report,
    write_phase2b_report,
)


def _comparison(*, failures: list[dict[str, str]] | None = None) -> dict:
    variants = [
        {
            "variant_id": "vggt_teacher",
            "family": "vggt",
            "role": "teacher_baseline",
            "seed": None,
            "metrics": {"metric_abs_rel": 0.22, "metric_rmse_m": 0.4, "metric_delta_1": 0.8},
            "runtime": {"total_ms_per_frame": 120.0, "peak_encoder_memory_gb": 8.0},
            "parameters": 1_000_000_000,
            "notes": ["Raw metric depth evaluation."],
        }
    ]
    for variant, base in (("rgb_probe", 0.42), ("vjepa_multilayer", 0.27)):
        for seed in (0, 1, 2):
            variants.append(
                {
                    "variant_id": variant,
                    "family": "rgb" if variant == "rgb_probe" else "vjepa",
                    "role": "non_jepa_baseline" if variant == "rgb_probe" else "ours",
                    "seed": seed,
                    "metrics": {
                        "metric_abs_rel": base + seed * 0.01,
                        "aligned_abs_rel": base - 0.03 + seed * 0.01,
                        "metric_rmse_m": 0.7 + seed * 0.02,
                        "metric_delta_1": 0.6 - seed * 0.01,
                        "raw_log_depth_nll": 0.8 + seed * 0.02,
                        "calibrated_log_depth_nll": 0.55 + seed * 0.01,
                        "variance_multiplier": 1.4,
                    },
                    "runtime": {
                        "encoder_ms_per_frame": 1.0 if variant == "rgb_probe" else 7.0,
                        "head_ms_per_frame": 0.5 if variant == "rgb_probe" else 1.0,
                        "total_ms_per_frame": 1.5 if variant == "rgb_probe" else 8.0,
                        "peak_encoder_memory_gb": 0.25 if variant == "rgb_probe" else 2.5,
                        "peak_head_memory_gb": 0.5 if variant == "rgb_probe" else 3.0,
                    },
                    "parameters": 12_000 if variant == "rgb_probe" else 90_000_000,
                    "notes": ["Metric-scale evaluation."],
                }
            )
    return {
        "experiment_id": "phase2b-test",
        "schema_version": "jepa4d-phase2b-comparison-v1",
        "dataset_manifest": "manifest.yaml",
        "split_hash": "a" * 64,
        "metric_policy": {"primary": "metric_abs_rel on held-out test frames"},
        "variants": variants,
        "failures": failures or [],
        "aggregates": {},
        "wandb_url": "https://wandb.ai/example/run/test",
    }


def _histories() -> list[dict]:
    return [
        {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "loss": 1.0 / (epoch + 1) + seed * 0.01,
            "validation_metric_abs_rel": base + 0.1 / (epoch + 1),
            "nll": 0.8 / (epoch + 1),
            "distillation": 0.3 / (epoch + 1),
            "gradient": 0.2 / (epoch + 1),
        }
        for variant, base in (("rgb_probe", 0.4), ("vjepa_multilayer", 0.25))
        for seed in (0, 1)
        for epoch in range(3)
    ]


def _phase2c_comparison() -> dict:
    record = _comparison()
    record["schema_version"] = "jepa4d-phase2c-cross-sequence-comparison-v1"
    for row in record["variants"]:
        if row["variant_id"] == "vjepa_multilayer":
            row["runtime"]["end_to_end_ms_per_frame"] = row["runtime"]["total_ms_per_frame"]
            row["runtime"]["peak_end_to_end_memory_gb"] = 3.2
    record["variants"][0]["sequence_metrics"] = {
        "fr3_cabinet": {
            "metric_abs_rel": 0.19,
            "aligned_abs_rel": 0.08,
            "metric_abs_log_scale_error": 0.31,
        },
        "fr3_large_cabinet": {
            "metric_abs_rel": 0.24,
            "aligned_abs_rel": 0.09,
            "metric_abs_log_scale_error": 0.39,
        },
    }
    record["variants"].append(
        {
            "variant_id": "vjepa_learned_fusion",
            "family": "vjepa",
            "role": "candidate",
            "seed": 0,
            "metrics": {
                "metric_abs_rel": 0.13,
                "aligned_abs_rel": 0.07,
                "metric_rmse_m": 0.3,
                "metric_delta_1": 0.9,
            },
            "runtime": {
                "encoder_ms_per_frame": 8.0,
                "head_ms_per_frame": 1.1,
                "total_ms_per_frame": 9.1,
                "end_to_end_ms_per_frame": 9.1,
                "peak_encoder_memory_gb": 2.7,
                "peak_end_to_end_memory_gb": 4.2,
            },
            "parameters": 90_000_003,
            "model_metadata": {
                "fusion_state": {
                    "layer_order": [2, 5, 8],
                    "final_coefficient": 0.94,
                    "coefficient_layer_2": 0.03,
                    "coefficient_layer_5": -0.01,
                    "coefficient_layer_8": 0.04,
                    "raw_gate_layer_2": 0.0902,
                    "raw_gate_layer_5": -0.03,
                    "raw_gate_layer_8": 0.1206,
                }
            },
            "sequence_metrics": {
                "fr3_cabinet": {
                    "metric_abs_rel": 0.12,
                    "aligned_abs_rel": 0.065,
                    "metric_abs_log_scale_error": 0.18,
                },
                "fr3_large_cabinet": {
                    "metric_abs_rel": 0.14,
                    "aligned_abs_rel": 0.075,
                    "metric_abs_log_scale_error": 0.22,
                },
            },
            "notes": ["Three-scalar residual fusion."],
        }
    )
    return record


def _fusion_history() -> list[dict]:
    return [
        {
            "variant": "vjepa_learned_fusion",
            "seed": 0,
            "epoch": epoch,
            "loss": 0.5 / (epoch + 1),
            "validation_metric_abs_rel": 0.2 - epoch * 0.02,
            "layer_order": [2, 5, 8],
            "final_coefficient": 1.0 - sum(coefficients),
            "coefficient_layer_2": coefficients[0],
            "coefficient_layer_5": coefficients[1],
            "coefficient_layer_8": coefficients[2],
        }
        for epoch, coefficients in enumerate(((0.0, 0.0, 0.0), (0.01, -0.005, 0.02), (0.03, -0.01, 0.04)))
    ]


def _promotion_gate() -> dict:
    return {
        "schema_version": "jepa4d-phase2c-promotion-v1",
        "decision": "retain_final_layer",
        "promoted": False,
        "conditions": {
            "primary_macro_absrel_strictly_better": False,
            "no_sequence_regression_above_5pct": True,
            "latency_at_most_1p10x_final": True,
            "peak_inference_memory_at_most_1p10x_final": True,
            "zero_failures": True,
        },
    }


def test_report_is_self_contained_and_renders_all_diagnostic_sections(tmp_path: Path) -> None:
    failure = {"variant": "vjepa_final", "seed": "2", "error": "bad <script>alert(1)</script>"}
    frames = [
        {
            "variant": "vjepa_multilayer",
            "seed": 0,
            "frame_id": "frame-007",
            "predicted_depth": [[1.0, 4.0]],
            "target_depth": [[1.0, 2.0]],
        },
        {"variant": "rgb_probe", "seed": 0, "frame_id": "frame-007", "metric_abs_rel": 0.75},
    ]
    artifacts = build_geometry_student_report(
        _comparison(failures=[failure]),
        tmp_path / "report.html",
        training_history=_histories(),
        per_frame_predictions=frames,
        static_png=False,
    )

    document = artifacts.html_path.read_text(encoding="utf-8")
    assert artifacts.html_path == tmp_path / "report.html"
    assert artifacts.png_path is None
    assert "Plotly.newPlot" in document
    assert 'src="https://cdn.plot.ly' not in document
    assert "Variant summary" in document
    assert "Training curves" in document
    assert "Per-frame diagnostics" in document
    assert "Held-out log-depth calibration" in document
    assert "batch-1 end-to-end latency" in document
    assert "Peak GPU memory GiB (co-resident when available)" in document
    assert "Phase-2b geometry student diagnostics" in document
    assert "Phase-2b geometry student: quality and resource trade-offs" in document
    assert "frame-007" in document
    assert "bad &lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "bad <script>alert(1)</script>" not in document
    assert "Held-out sequence diagnostics" not in document
    assert "Learned-fusion coefficient audit" not in document
    assert "Formal promotion decision" not in document
    assert any("recorded failure" in warning for warning in artifacts.warnings)


def test_report_renders_sequence_generalization_and_fusion_audit(tmp_path: Path) -> None:
    comparison = _phase2c_comparison()
    assert _peak_memory(comparison["variants"][-1]) == 4.2
    artifacts = build_geometry_student_report(
        comparison,
        tmp_path / "phase2c-report.html",
        training_history=[*_histories(), *_fusion_history()],
        promotion_gate=_promotion_gate(),
        static_png=False,
    )

    document = artifacts.html_path.read_text(encoding="utf-8")
    assert document.count("Plotly.newPlot") >= 4
    assert 'src="https://cdn.plot.ly' not in document
    assert "Held-out sequence diagnostics" in document
    assert "Per-sequence geometry generalization" in document
    assert "Raw metric AbsRel" in document
    assert "Median-aligned AbsRel" in document
    assert "Absolute log-scale error" in document
    assert "fr3_cabinet" in document
    assert "fr3_large_cabinet" in document
    assert "Learned-fusion coefficient audit" in document
    assert "Effective coefficient trajectories" in document
    assert "Best-checkpoint coefficients" in document
    assert "checkpoint" in document
    assert "vjepa_learned_fusion seed 0" in document
    assert "Layer 5 coefficient (raw gate)" in document
    assert "-0.01 (g=-0.03)" in document
    assert "Phase-2c geometry student diagnostics" in document
    assert "Phase-2c geometry student: quality and resource trade-offs" in document
    assert "Phase-2b geometry student" not in document
    assert "Metric Abs Rel vs reported latency" in document
    assert "Metric Abs Rel vs reported peak GPU memory" in document
    assert "measured co-resident batch-1" in document
    assert "fallback reported total (non-co-resident policy)" in document
    assert "It is not a measured batch-1 end-to-end latency" not in document
    assert "Formal promotion decision" in document
    assert "RETAIN FINAL LAYER" in document
    assert "retain_final_layer" in document
    assert "Primary Macro Absrel Strictly Better" in document
    assert "PASS" in document
    assert "FAIL" in document
    assert "4.2" in document
    assert any("Latency policies differ" in warning for warning in artifacts.warnings)
    assert not any("fusion coefficients sum" in warning for warning in artifacts.warnings)


def test_runner_wrapper_reads_jsonl_frames_and_exports_png(tmp_path: Path, monkeypatch) -> None:
    frame_path = tmp_path / "frame_metrics.jsonl"
    rows = [
        {"variant": "rgb_probe", "seed": 0, "frame_id": 1, "metric_abs_rel": 0.4},
        {"variant": "vjepa_multilayer", "seed": 0, "frame_id": 1, "metric_abs_rel": 0.2},
    ]
    frame_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    diagnostic_path = tmp_path / "vjepa_multilayer-seed0.npz"
    target = np.stack((np.ones((4, 5)), np.full((4, 5), 2.0))).astype(np.float32)
    prediction = target.copy()
    prediction[1, 1:3, 2:4] = 4.0
    np.savez_compressed(
        diagnostic_path,
        prediction_m=prediction,
        target_m=target,
        test_sample_ids=np.asarray(["fr3/office/midpoint-008", "fr3/office/worst-017"]),
        test_selection_labels=np.asarray(["deterministic-sequence-midpoint", "post-hoc-worst-by-test-AbsRel"]),
    )

    def fake_write_image(self, file, **kwargs) -> None:
        del self, kwargs
        Path(file).write_bytes(b"test-png")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    output = tmp_path / "formal-output"
    report = write_phase2b_report(
        output,
        _comparison(),
        _histories(),
        diagnostics={
            "per_frame_metrics": str(frame_path),
            "vjepa_multilayer-seed0": str(diagnostic_path),
            "debug_log": "logs/train.jsonl",
        },
    )

    document = report.read_text(encoding="utf-8")
    assert report == output / "geometry_student_report.html"
    assert (output / "geometry_student_report.png").read_bytes() == b"test-png"
    assert "frame_metrics.jsonl" in document
    assert "logs/train.jsonl" in document
    assert "Worst finite frame/variant rows" in document
    assert "Depth and relative-error grids" in document
    assert "vjepa_multilayer-seed0" in document
    # Plotly JSON escapes slashes, but preserves the audit label and sample stem.
    assert "worst-017" in document
    assert "post-hoc-worst-by-test-AbsRel" in document
    assert "frame 1 · prediction" not in document


def test_missing_static_export_backend_is_non_fatal(tmp_path: Path, monkeypatch) -> None:
    def unavailable(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("Kaleido is unavailable")

    monkeypatch.setattr(go.Figure, "write_image", unavailable)
    artifacts = build_geometry_student_report(_comparison(), tmp_path / "report-dir", static_png=True)

    assert artifacts.html_path.is_file()
    assert artifacts.png_path is None
    assert not (tmp_path / "report-dir" / "geometry_student_report.png").exists()
    assert any("Static PNG export was unavailable" in warning for warning in artifacts.warnings)
    assert "Kaleido is unavailable" in artifacts.html_path.read_text(encoding="utf-8")
