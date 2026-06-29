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
from jepa4d.benchmarks.planning.smoke import PlanningSmokeBenchmark
from jepa4d.benchmarks.representation.smoke import RepresentationSmokeBenchmark
from jepa4d.benchmarks.tracking4d.smoke import IdentityAssociationSmokeBenchmark
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.object_slot_grounder import ObjectSlotGrounder
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.experiment_record import ArtifactRecord, ExperimentRecord, StageRecord


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
    if "planning" in requested:
        benchmark = PlanningSmokeBenchmark()
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
    ExperimentRecord(
        title="Stagewise smoke benchmark",
        experiment_id="stagewise-smoke",
        stage=" + ".join(requested),
        status="complete",
        evidence_level="contract-only",
        objective="Validate every requested benchmark adapter and report interface with deterministic mock outputs.",
        hypothesis="All requested adapters complete with finite, serializable metrics.",
        decision="Use this result only as infrastructure evidence.",
        timestamp=report["timestamp"],
        config={"config_path": str(args.config), "resolved": config, "mock": True},
        stages=[StageRecord(stage, "smoke adapter", "pass", "tiny fixture", "finite metrics") for stage in requested],
        metrics={item["benchmark"]["name"]: item["metrics"] for item in results},
        artifacts=[
            ArtifactRecord(json_path, "JSON", "Machine-readable benchmark report"),
            ArtifactRecord(html_path, "HTML", "Human-readable report"),
        ],
        limitations=["Mock adapters do not provide model-quality benchmark evidence."],
        next_actions=["Replace one adapter per stage with a versioned real-data mini split."],
    ).write(markdown_path)
    print(
        json.dumps(
            {"json": str(json_path), "html": str(html_path), "experiment": str(markdown_path), "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
