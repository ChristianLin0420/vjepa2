"""Run CPU-safe stagewise smoke benchmarks and write JSON/HTML reports."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa4d.benchmarks.geometry.smoke import GeometrySmokeBenchmark
from jepa4d.benchmarks.memory.smoke import MemorySmokeBenchmark
from jepa4d.benchmarks.object_grounding.smoke import ObjectGroundingSmokeBenchmark
from jepa4d.benchmarks.representation.smoke import RepresentationSmokeBenchmark
from jepa4d.benchmarks.tracking4d.smoke import IdentityAssociationSmokeBenchmark
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.object_slot_grounder import ObjectSlotGrounder
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("jepa4d/config/benchmarks/smoke.yaml"))
    parser.add_argument("--mock", action="store_true", help="Required for the CPU smoke harness")
    parser.add_argument("--output", type=Path, default=Path("outputs/stagewise_smoke"))
    args = parser.parse_args()
    if not args.mock:
        raise SystemExit("The stagewise smoke runner requires --mock; real dataset adapters are introduced in Phase 6")
    config = yaml.safe_load(args.config.read_text())
    requested = config.get("stages", [config.get("stage", "representation")])
    results = []
    if "representation" in requested:
        benchmark = RepresentationSmokeBenchmark()
        predictions = benchmark.run(VJEPA21FeatureExtractor(mock=True), "tiny")
        results.append(
            {
                "benchmark": benchmark.report(),
                "predictions": predictions,
                "metrics": benchmark.compute_metrics(predictions),
            }
        )
    if "geometry" in requested:
        benchmark = GeometrySmokeBenchmark()
        predictions = benchmark.run(GeometryBeliefHead(output_size=int(config.get("output_size", 28))), "tiny")
        results.append(
            {
                "benchmark": benchmark.report(),
                "predictions": predictions,
                "metrics": benchmark.compute_metrics(predictions),
            }
        )
    if "object_grounding" in requested:
        benchmark = ObjectGroundingSmokeBenchmark()
        predictions = benchmark.run(ObjectSlotGrounder(), "tiny")
        results.append(
            {
                "benchmark": benchmark.report(),
                "predictions": predictions,
                "metrics": benchmark.compute_metrics(predictions),
            }
        )
    if "memory" in requested:
        benchmark = MemorySmokeBenchmark()
        predictions = benchmark.run(None, "tiny")
        results.append(
            {
                "benchmark": benchmark.report(),
                "predictions": predictions,
                "metrics": benchmark.compute_metrics(predictions),
            }
        )
    if "tracking4d" in requested:
        benchmark = IdentityAssociationSmokeBenchmark()
        predictions = benchmark.run(None, "tiny")
        results.append(
            {
                "benchmark": benchmark.report(),
                "predictions": predictions,
                "metrics": benchmark.compute_metrics(predictions),
            }
        )
    report = {"timestamp": datetime.now(UTC).isoformat(), "config": config, "mock": True, "results": results}
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "report.json"
    html_path = args.output / "report.html"
    markdown_path = args.output / "EXPERIMENT.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>JEPA-4D stagewise smoke</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto}pre{background:#f6f8fa;padding:1rem}</style>"
        f"</head><body><h1>Stagewise smoke benchmark</h1><pre>{html.escape(json.dumps(report, indent=2))}</pre></body></html>"
    )
    markdown_path.write_text(
        f"# Stagewise smoke benchmark\n\n- Timestamp: {report['timestamp']}\n- Config: `{args.config}`\n"
        f"- Stages: `{requested}`\n- Result: all requested mock adapters completed\n\n"
        "This experiment validates interfaces and finite outputs; it is not a model-quality benchmark.\n"
    )
    print(
        json.dumps(
            {"json": str(json_path), "html": str(html_path), "experiment": str(markdown_path), "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
