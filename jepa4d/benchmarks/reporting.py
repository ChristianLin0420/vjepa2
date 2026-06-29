"""Statistical aggregation and machine/human-readable benchmark reports."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class FailureCategory(StrEnum):
    INPUT_DECODE = "input_decode"
    REPRESENTATION = "representation"
    GEOMETRY_DEPTH = "geometry_depth"
    GEOMETRY_POSE = "geometry_pose"
    GEOMETRY_SCALE = "geometry_scale"
    TRACKING = "tracking"
    OBJECT_GROUNDING = "object_grounding"
    MEMORY_INSERT = "memory_insert"
    MEMORY_RETRIEVAL = "memory_retrieval"
    STALE_BELIEF = "stale_belief"
    PLANNING_GROUNDING = "planning_grounding"
    VERIFICATION = "verification"
    CONTROL = "control"
    COLLISION = "collision"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureRecord:
    benchmark: str
    stage: str
    sample_id: str
    category: FailureCategory
    message: str
    contributing: list[FailureCategory] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    mean: float
    lower: float
    upper: float
    samples: int
    confidence: float


@dataclass(slots=True)
class StageBenchmarkReport:
    name: str
    stage: str
    adapter: dict[str, Any]
    metrics: dict[str, MetricEstimate]
    repetitions: int
    successes: int
    failures: int
    latency_ms: MetricEstimate
    predictions: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkSuiteReport:
    suite_id: str
    timestamp: str
    evidence_level: str
    config: dict[str, Any]
    dataset_manifest: dict[str, Any]
    stages: list[StageBenchmarkReport]
    failures: list[FailureRecord]

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_confidence_interval(
    values: list[float], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> MetricEstimate:
    if not values:
        raise ValueError("cannot estimate an empty metric")
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValueError("confidence must be in (0,1) and resamples must be positive")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("metric values must be finite")
    if len(array) == 1:
        mean = lower = upper = float(array[0])
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(array), size=(resamples, len(array)))
        means = array[indices].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        mean = float(array.mean())
        lower, upper = (float(value) for value in np.quantile(means, [alpha, 1.0 - alpha]))
    return MetricEstimate(mean, lower, upper, len(values), confidence)


def aggregate_metric_runs(
    runs: list[dict[str, float]], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> dict[str, MetricEstimate]:
    if not runs:
        return {}
    keys = set(runs[0])
    if any(set(run) != keys for run in runs):
        raise ValueError("metric keys must be identical across repetitions")
    return {
        key: bootstrap_confidence_interval(
            [float(run[key]) for run in runs], confidence=confidence, resamples=resamples, seed=seed + index
        )
        for index, key in enumerate(sorted(keys))
    }


def write_suite_reports(report: BenchmarkSuiteReport, output: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    failures_path = directory / "failures.json"
    html_path = directory / "report.html"
    payload = report.to_serializable()
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    failures_path.write_text(json.dumps([asdict(value) for value in report.failures], indent=2) + "\n")

    stage_rows: list[str] = []
    metric_rows: list[str] = []
    for stage in report.stages:
        stage_rows.append(
            f"<tr><td>{html.escape(stage.name)}</td><td>{html.escape(stage.stage)}</td>"
            f"<td>{stage.successes}</td><td>{stage.failures}</td><td>{stage.latency_ms.mean:.3f}</td></tr>"
        )
        metric_rows.extend(
            f"<tr><td>{html.escape(stage.name)}</td><td>{html.escape(name)}</td><td>{value.mean:.6g}</td>"
            f"<td>[{value.lower:.6g}, {value.upper:.6g}]</td><td>{value.samples}</td></tr>"
            for name, value in stage.metrics.items()
        )
    failure_rows = [
        f"<tr><td>{html.escape(value.benchmark)}</td><td>{html.escape(value.sample_id)}</td>"
        f"<td>{html.escape(str(value.category))}</td><td>{html.escape(value.message)}</td></tr>"
        for value in report.failures
    ] or ["<tr><td colspan='4'>No failures recorded.</td></tr>"]
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>JEPA-4D benchmark suite</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;color:#17202a}"
        "table{border-collapse:collapse;width:100%;margin-bottom:2rem}th,td{border:1px solid #ccd1d1;padding:.45rem}"
        "th{background:#eef2f7;text-align:left}.ok{color:#16794b}.warn{color:#a15c00}</style></head><body>"
        f"<h1>{html.escape(report.suite_id)}</h1><p>Evidence: <b>{html.escape(report.evidence_level)}</b>; "
        f"failures: <b class='{'ok' if not report.failures else 'warn'}'>{len(report.failures)}</b></p>"
        "<h2>Stages</h2><table><tr><th>Benchmark</th><th>Stage</th><th>Successes</th><th>Failures</th>"
        f"<th>Mean latency (ms)</th></tr>{''.join(stage_rows)}</table>"
        "<h2>Metric estimates</h2><table><tr><th>Benchmark</th><th>Metric</th><th>Mean</th>"
        f"<th>Bootstrap CI</th><th>N</th></tr>{''.join(metric_rows)}</table>"
        "<h2>Failure dashboard</h2><table><tr><th>Benchmark</th><th>Sample</th><th>Category</th>"
        f"<th>Message</th></tr>{''.join(failure_rows)}</table></body></html>"
    )
    return json_path, failures_path, html_path
