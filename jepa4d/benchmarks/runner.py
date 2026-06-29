"""Repeated benchmark execution with typed failures and confidence intervals."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.benchmarks.reporting import (
    FailureCategory,
    FailureRecord,
    StageBenchmarkReport,
    aggregate_metric_runs,
    bootstrap_confidence_interval,
)


@dataclass(slots=True)
class BenchmarkJob:
    adapter: BenchmarkAdapter
    model_or_system: object
    split: str = "tiny"


STAGE_FAILURES = {
    "representation": FailureCategory.REPRESENTATION,
    "geometry": FailureCategory.GEOMETRY_DEPTH,
    "object_grounding": FailureCategory.OBJECT_GROUNDING,
    "tracking4d": FailureCategory.TRACKING,
    "memory": FailureCategory.MEMORY_RETRIEVAL,
    "planning": FailureCategory.PLANNING_GROUNDING,
    "verification": FailureCategory.VERIFICATION,
    "control": FailureCategory.CONTROL,
}


def run_job(
    job: BenchmarkJob,
    *,
    repetitions: int,
    confidence: float = 0.95,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> tuple[StageBenchmarkReport, list[FailureRecord]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    metric_runs: list[dict[str, float]] = []
    latencies: list[float] = []
    failures: list[FailureRecord] = []
    predictions: list[Any] = []
    for repetition in range(repetitions):
        started = time.perf_counter()
        try:
            values = job.adapter.run(job.model_or_system, job.split)
            run_metrics = job.adapter.compute_metrics(values, None)
            if not all(isinstance(value, int | float) for value in run_metrics.values()):
                raise TypeError("benchmark metrics must be numeric")
            metric_runs.append({key: float(value) for key, value in run_metrics.items()})
            if repetition == 0:
                predictions = values
        except Exception as error:
            failures.append(
                FailureRecord(
                    benchmark=job.adapter.name,
                    stage=job.adapter.stage,
                    sample_id=f"{job.split}/repeat-{repetition}",
                    category=STAGE_FAILURES.get(job.adapter.stage, FailureCategory.UNKNOWN),
                    message=f"{type(error).__name__}: {error}",
                )
            )
        finally:
            latencies.append((time.perf_counter() - started) * 1000.0)
    metric_estimates = aggregate_metric_runs(
        metric_runs,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    latency = bootstrap_confidence_interval(
        latencies,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed + 10_000,
    )
    return (
        StageBenchmarkReport(
            name=job.adapter.name,
            stage=job.adapter.stage,
            adapter=job.adapter.report(),
            metrics=metric_estimates,
            repetitions=repetitions,
            successes=len(metric_runs),
            failures=len(failures),
            latency_ms=latency,
            predictions=predictions,
        ),
        failures,
    )
