"""Run controlled and optional DAVIS identity association ablations."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from jepa4d.benchmarks.tracking4d.identity import (
    build_crossing_fixture,
    load_davis_fixture,
    run_davis_variant,
    run_identity_variant,
)
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.identity_report import build_identity_report
from jepa4d.visualization.observability import ExperimentLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vjepa2.1-vitb-fpc64-384"))
    parser.add_argument("--mock-vjepa", action="store_true")
    parser.add_argument("--davis-root", type=Path)
    parser.add_argument("--davis-sequence", default="dogs-scale")
    parser.add_argument("--davis-stride", type=int, default=4)
    parser.add_argument("--davis-max-frames", type=int, default=21)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/identity_ablation"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="online")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        enabled=args.wandb,
        name=f"identity-ablation-{'mock' if args.mock_vjepa else 'real-vjepa'}",
        mode=args.wandb_mode,
        tags=["identity", "tracking4d", "ablation", "phase-4", args.device],
        config={
            "checkpoint": str(args.checkpoint),
            "mock_vjepa": args.mock_vjepa,
            "davis_root": None if args.davis_root is None else str(args.davis_root),
            "davis_sequence": args.davis_sequence,
            "davis_stride": args.davis_stride,
            "davis_max_frames": args.davis_max_frames,
            "device": args.device,
            "tuning_policy": "DAVIS sweep is exploratory and not held out",
        },
    )
    extractor = VJEPA21FeatureExtractor(
        mock=args.mock_vjepa,
        checkpoint=None if args.mock_vjepa else args.checkpoint,
        device=args.device,
    )
    results: dict[str, dict[str, dict[str, float]]] = {}
    timings: dict[str, float] = {}
    synthetic = build_crossing_fixture()
    started = time.perf_counter()
    synthetic_tokens = extractor(synthetic.batch)
    timings["synthetic_vjepa_s"] = time.perf_counter() - started
    synthetic_variants = {
        "oracle_appearance": ("oracle", (1.0, 0.0, 0.0), 0.8),
        "rgb_appearance": ("rgb", (1.0, 0.0, 0.0), 0.8),
        "vjepa_appearance": ("vjepa", (1.0, 0.0, 0.0), 0.8),
        "vjepa_mask_appearance": ("vjepa_mask", (1.0, 0.0, 0.0), 0.8),
        "iou_only": ("ambiguous", (0.0, 1.0, 0.0), 0.2),
        "geometry_only": ("ambiguous", (0.0, 0.0, 1.0), 0.6),
        "vjepa_fused_default": ("vjepa", (0.65, 0.20, 0.15), 0.55),
        "no_appearance": ("ambiguous", (0.0, 0.57, 0.43), 0.45),
    }
    results["synthetic_crossing"] = {}
    for name, (source, weights, threshold) in synthetic_variants.items():
        _, metrics = run_identity_variant(
            synthetic,
            feature_source=source,  # type: ignore[arg-type]
            tokens=synthetic_tokens if source in {"vjepa", "vjepa_mask"} else None,
            weights=weights,
            threshold=threshold,
        )
        results["synthetic_crossing"][name] = metrics
    sweeps: list[dict[str, float | str]] = []
    if args.davis_root is not None:
        davis = load_davis_fixture(
            args.davis_root,
            sequence=args.davis_sequence,
            stride=args.davis_stride,
            max_frames=args.davis_max_frames,
        )
        started = time.perf_counter()
        davis_tokens = extractor(davis.batch)
        timings["davis_vjepa_s"] = time.perf_counter() - started
        davis_variants = {
            "oracle_appearance": ("oracle", (1.0, 0.0, 0.0), 0.8),
            "rgb_appearance": ("rgb", (1.0, 0.0, 0.0), 0.8),
            "vjepa_appearance": ("vjepa", (1.0, 0.0, 0.0), 0.8),
            "vjepa_mask_appearance": ("vjepa_mask", (1.0, 0.0, 0.0), 0.8),
            "iou_only": ("ambiguous", (0.0, 1.0, 0.0), 0.2),
            "vjepa_iou_default": ("vjepa", (0.75, 0.25, 0.0), 0.55),
            "vjepa_mask_iou_default": ("vjepa_mask", (0.75, 0.25, 0.0), 0.55),
        }
        dataset_key = f"DAVIS-{args.davis_sequence}"
        results[dataset_key] = {}
        for name, (source, weights, threshold) in davis_variants.items():
            _, metrics = run_davis_variant(
                davis,
                feature_source=source,  # type: ignore[arg-type]
                tokens=davis_tokens if source in {"vjepa", "vjepa_mask"} else None,
                weights=weights,
                threshold=threshold,
                max_time_gap=8,
            )
            results[dataset_key][name] = metrics
        candidates: list[tuple[float, float, float, float, dict[str, float]]] = []
        for appearance_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                source = "vjepa" if appearance_weight else "ambiguous"
                _, metrics = run_davis_variant(
                    davis,
                    feature_source=source,
                    tokens=davis_tokens if source == "vjepa" else None,
                    weights=(appearance_weight, 1.0 - appearance_weight, 0.0),
                    threshold=threshold,
                    max_time_gap=8,
                )
                sweeps.append(
                    {
                        "dataset": dataset_key,
                        "appearance_weight": appearance_weight,
                        "iou_weight": 1.0 - appearance_weight,
                        "threshold": threshold,
                        **metrics,
                    }
                )
                candidates.append(
                    (
                        metrics["pairwise_f1"],
                        -metrics["id_switches"],
                        appearance_weight,
                        threshold,
                        metrics,
                    )
                )
        best = max(candidates, key=lambda value: (value[0], value[1]))
        results[dataset_key][f"exploratory_best_aw{best[2]:.2f}_th{best[3]:.2f}"] = best[4]
    metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": extractor.model_config,
        "timings": timings,
        "davis_source": "https://davischallenge.org/davis2017/code.html" if args.davis_root else None,
        "claims": {
            "synthetic": "controlled diagnostic",
            "davis_default": "real-video sequence result",
            "davis_best": "exploratory same-sequence selection; not held out",
        },
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps({"results": results, "sweeps": sweeps, "metadata": metadata}, indent=2) + "\n")
    logger.log_identity_ablation(results, sweeps)  # type: ignore[arg-type]
    report_path = build_identity_report(results, args.output / "report.html", metadata=metadata, wandb_url=logger.url)
    experiment_path = args.output / "EXPERIMENT.md"
    experiment_path.write_text(
        "# Identity association ablation\n\n"
        f"- Timestamp: {metadata['timestamp']}\n"
        f"- V-JEPA backend: `{extractor.model_config['backend']}`\n"
        f"- DAVIS sequence: `{args.davis_sequence if args.davis_root else 'not run'}`\n"
        f"- Timings: `{timings}`\n"
        f"- W&B: {logger.url or 'disabled'}\n\n"
        "Synthetic results are controlled diagnostics. DAVIS default operating points are real-video sequence results. "
        "The reported exploratory best uses the same sequence for selection and evaluation and is not held-out evidence.\n"
    )
    for path, artifact_type in (
        (metrics_path, "identity-metrics"),
        (report_path, "interactive-report"),
        (experiment_path, "experiment-record"),
    ):
        logger.log_artifact(path, artifact_type)
    logger.finish({"result": "success", "datasets": list(results), "timings": timings})
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "report": str(report_path),
                "experiment": str(experiment_path),
                "wandb_url": logger.url,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
