"""Atomically persist the exact Phase-2d/2e Slurm submission graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer


def _ids(value: str, expected: int, label: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if len(values) != expected or len(set(values)) != expected or any(not item.isdigit() for item in values):
        raise typer.BadParameter(f"{label} must contain exactly {expected} unique numeric job IDs")
    return values


def _job(value: str, label: str) -> str:
    if not value.isdigit():
        raise typer.BadParameter(f"{label} must be a numeric Slurm job ID")
    return value


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(
    phase2d_output: Annotated[Path, typer.Option("--phase2d-output")],
    phase2e_output: Annotated[Path, typer.Option("--phase2e-output")],
    test_job: Annotated[str, typer.Option("--test-job")],
    attribution_job: Annotated[str, typer.Option("--attribution-job")],
    calibration_job: Annotated[str, typer.Option("--calibration-job")],
    latency_jobs: Annotated[str, typer.Option("--latency-jobs")],
    latency_aggregate_job: Annotated[str, typer.Option("--latency-aggregate-job")],
    phase2d_aggregate_job: Annotated[str, typer.Option("--phase2d-aggregate-job")],
    cache_job: Annotated[str, typer.Option("--cache-job")],
    pilot_job: Annotated[str, typer.Option("--pilot-job")],
    formal_jobs: Annotated[str, typer.Option("--formal-jobs")],
    final_job: Annotated[str, typer.Option("--final-job")],
) -> None:
    test = _job(test_job, "test-job")
    latency = _ids(latency_jobs, 12, "latency-jobs")
    formal = _ids(formal_jobs, 4, "formal-jobs")
    phase2d = {
        "schema_version": "jepa4d-phase2d-dependency-graph-v1",
        "test_job_id": test,
        "attribution_job_id": _job(attribution_job, "attribution-job"),
        "calibration_job_id": _job(calibration_job, "calibration-job"),
        "latency_job_ids": latency,
        "latency_aggregate_job_id": _job(latency_aggregate_job, "latency-aggregate-job"),
        "aggregate_job_id": _job(phase2d_aggregate_job, "phase2d-aggregate-job"),
    }
    phase2e = {
        "schema_version": "jepa4d-phase2e-dependency-graph-v1",
        "phase2d_test_job_id": test,
        "cache_job_id": _job(cache_job, "cache-job"),
        "pilot_job_id": _job(pilot_job, "pilot-job"),
        "formal_shard_job_ids": formal,
        "final_job_id": _job(final_job, "final-job"),
    }
    _write(phase2d_output, phase2d)
    _write(phase2e_output, phase2e)
    typer.echo(json.dumps({"phase2d": phase2d, "phase2e": phase2e}, indent=2, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
