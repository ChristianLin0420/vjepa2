"""Run a versioned multi-stage benchmark suite and emit Phase-6 reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from jepa4d.benchmarks.geometry.smoke import GeometrySmokeBenchmark
from jepa4d.benchmarks.manifest import DatasetManifest
from jepa4d.benchmarks.memory.smoke import MemorySmokeBenchmark
from jepa4d.benchmarks.object_grounding.smoke import ObjectGroundingSmokeBenchmark
from jepa4d.benchmarks.planning.smoke import PlanningSmokeBenchmark
from jepa4d.benchmarks.reporting import BenchmarkSuiteReport, write_suite_reports
from jepa4d.benchmarks.representation.smoke import RepresentationSmokeBenchmark
from jepa4d.benchmarks.runner import BenchmarkJob, run_job
from jepa4d.benchmarks.tracking4d.smoke import IdentityAssociationSmokeBenchmark
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.models.object_slot_grounder import ObjectSlotGrounder
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.visualization.experiment_record import ArtifactRecord, ExperimentRecord, StageRecord
from jepa4d.visualization.observability import ExperimentLogger

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _jobs(stages: list[str], config: dict[str, Any]) -> list[BenchmarkJob]:
    jobs: dict[str, BenchmarkJob] = {
        "representation": BenchmarkJob(RepresentationSmokeBenchmark(), VJEPA21FeatureExtractor(mock=True)),
        "geometry": BenchmarkJob(
            GeometrySmokeBenchmark(), GeometryBeliefHead(output_size=int(config.get("geometry_output_size", 28)))
        ),
        "object_grounding": BenchmarkJob(ObjectGroundingSmokeBenchmark(), ObjectSlotGrounder()),
        "tracking4d": BenchmarkJob(IdentityAssociationSmokeBenchmark(), None),
        "memory": BenchmarkJob(MemorySmokeBenchmark(), None),
        "planning": BenchmarkJob(PlanningSmokeBenchmark(), None),
    }
    unknown = sorted(set(stages) - set(jobs))
    if unknown:
        raise ValueError(f"unknown benchmark stages: {unknown}")
    return [jobs[stage] for stage in stages]


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("jepa4d/config/benchmarks/phase6_contract.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/phase6_benchmark"),
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb")] = False,
    wandb_project: Annotated[str, typer.Option("--wandb-project")] = "jepa4d-worldmodel",
    wandb_mode: Annotated[str, typer.Option("--wandb-mode")] = "online",
    run_name: Annotated[str, typer.Option("--run-name")] = "phase6-contract-benchmark",
) -> None:
    resolved = yaml.safe_load(config.read_text())
    logger = ExperimentLogger(
        enabled=wandb,
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        tags=["phase-6", "benchmarks", str(resolved.get("evidence_level", "contract-only"))],
        config={"config_path": str(config), **resolved},
    )
    manifest_path = Path(resolved["dataset_manifest"])
    manifest = DatasetManifest.load(manifest_path)
    issues = manifest.validate_assets()
    if issues:
        raise typer.BadParameter(f"dataset manifest failed integrity validation: {issues}")
    repetitions = int(resolved.get("repetitions", 5))
    confidence = float(resolved.get("confidence", 0.95))
    resamples = int(resolved.get("bootstrap_resamples", 2000))
    seed = int(resolved.get("seed", 0))
    stage_reports = []
    failures = []
    for index, job in enumerate(_jobs(list(resolved["stages"]), resolved)):
        report, stage_failures = run_job(
            job,
            repetitions=repetitions,
            confidence=confidence,
            bootstrap_resamples=resamples,
            seed=seed + index * 100,
        )
        stage_reports.append(report)
        failures.extend(stage_failures)
    suite = BenchmarkSuiteReport(
        suite_id=str(resolved["suite_id"]),
        timestamp=datetime.now(UTC).isoformat(),
        evidence_level=str(resolved.get("evidence_level", "contract-only")),
        config={"path": str(config), **resolved},
        dataset_manifest=manifest.to_serializable(),
        stages=stage_reports,
        failures=failures,
    )
    json_path, failures_path, html_path = write_suite_reports(suite, output)
    logger.log_benchmark_suite(suite)
    markdown_path = ExperimentRecord(
        title=f"Benchmark suite: {suite.suite_id}",
        experiment_id=suite.suite_id,
        stage="benchmark aggregation",
        status="complete" if not failures else "complete-with-failures",
        evidence_level=suite.evidence_level,
        objective="Validate versioned inputs, repeated stage adapters, confidence intervals, and failure reporting.",
        hypothesis="Every contract adapter remains deterministic and reportable through one auditable suite.",
        decision="Use this as Phase-6 infrastructure evidence; official subsets remain required for quality claims.",
        wandb_url=logger.url,
        timestamp=suite.timestamp,
        config=suite.config,
        stages=[
            StageRecord(
                value.stage,
                value.name,
                "pass" if value.failures == 0 else "failures",
                f"{value.repetitions} repetitions",
                f"{len(value.metrics)} metrics",
                f"mean latency {value.latency_ms.mean:.3f} ms",
            )
            for value in stage_reports
        ],
        metrics={
            f"{stage.name}/{metric}": {
                "mean": estimate.mean,
                "ci": [estimate.lower, estimate.upper],
                "n": estimate.samples,
            }
            for stage in stage_reports
            for metric, estimate in stage.metrics.items()
        },
        artifacts=[
            ArtifactRecord(json_path, "JSON", "Complete suite results and confidence intervals"),
            ArtifactRecord(failures_path, "JSON", "Typed per-sample failures"),
            ArtifactRecord(html_path, "HTML", "Stage, metric, and failure dashboard"),
        ],
        limitations=[
            "The bundled Robo4D-JEPA asset is a contract fixture, not an official model-quality dataset.",
            "Repeated deterministic smoke runs quantify harness variability, not population uncertainty.",
        ],
        next_actions=["Add one licensed, version-pinned official mini subset per stage."],
    ).write(output / "EXPERIMENT.md")
    for path, artifact_type in (
        (json_path, "benchmark-report"),
        (failures_path, "benchmark-failures"),
        (html_path, "benchmark-dashboard"),
        (markdown_path, "experiment-record"),
    ):
        logger.log_artifact(path, artifact_type)
    logger.finish(
        {
            "result": "success" if not failures else "completed_with_failures",
            "suite_id": suite.suite_id,
            "stages_completed": len(stage_reports),
            "failures_recorded": len(failures),
            "evidence_level": suite.evidence_level,
        }
    )
    typer.echo(
        json.dumps(
            {
                "report": str(json_path),
                "failures": str(failures_path),
                "dashboard": str(html_path),
                "experiment": str(markdown_path),
                "stages": len(stage_reports),
                "failures_recorded": len(failures),
                "wandb_url": logger.url,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
