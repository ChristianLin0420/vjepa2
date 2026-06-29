from pathlib import Path

import pytest

from scripts.aggregate_phase2d_latency import _bootstrap_interval
from scripts.run_phase2d_latency_confirmation import (
    TimingRow,
    _render_report,
    _slurm_identity,
    _summarize_gpu_telemetry,
    _summary,
)


def _rows() -> list[TimingRow]:
    rows: list[TimingRow] = []
    values = {
        "final_deployment": (10.0, 9.0),
        "final_capture_all": (10.5, 9.5),
        "fixed_deployment": (11.0, 10.0),
        "learned_deployment": (10.8, 9.8),
    }
    for block in range(3):
        for order, (variant, (wall, cuda)) in enumerate(values.items()):
            rows.append(TimingRow("e2e", block, order, variant, 100, wall + block, cuda + block, 0))
    rows.append(TimingRow("head_only", 0, 0, "final_head", 100, 0.2, 0.1, 0))
    return rows


def test_latency_summary_contains_tail_and_ratio() -> None:
    summary = _summary(_rows())
    assert summary["e2e/final_deployment"]["wall_median_ms"] == 11.0
    assert summary["e2e/learned_deployment"]["wall_p90_ms"] > 10.8
    assert 0.9 < summary["e2e/learned_deployment"]["paired_wall_median_ratio_to_final"] < 1.1


def test_latency_report_is_self_contained_and_readable(tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    rows = _rows()
    _render_report(
        output,
        rows,
        _summary(rows),
        {
            "replicate": 2,
            "gpu_name": "test-gpu",
            "slurm": {"job_id": "123"},
            "peak_cuda_memory_gb": 1.25,
            "gpu_telemetry": {
                "sample_count": 2,
                "statistics": {
                    "utilization_gpu_pct": {"mean": 70.0, "p95": 79.0},
                    "temperature_c": {"mean": 61.0, "max": 62.0},
                    "power_w": {"mean": 210.0, "max": 220.0},
                    "clocks_sm_mhz": {"mean": 1500.0, "p95": 1510.0},
                },
            },
        },
    )
    html = output.read_text()
    assert "Phase 2d" in html
    assert "Learned / final" in html
    assert "plotly" in html.lower()
    assert "GPU telemetry" in html
    assert "1.25 GiB" in html
    assert "<script src=" not in html


def test_cluster_bootstrap_interval_is_finite_and_ordered() -> None:
    low, high = _bootstrap_interval([1.01, 1.04, 1.05, 1.08])
    assert 1.0 < low <= high < 1.1


def test_latency_requires_complete_slurm_allocation_identity() -> None:
    environment = {
        "SLURM_JOB_ID": "101",
        "SLURM_JOB_NAME": "p2d-latency",
        "SLURM_JOB_PARTITION": "polar4",
        "SLURM_JOB_NODELIST": "node-a",
    }
    assert _slurm_identity(environment) == {
        "job_id": "101",
        "job_name": "p2d-latency",
        "partition": "polar4",
        "nodelist": "node-a",
    }
    environment.pop("SLURM_JOB_ID")
    with pytest.raises(RuntimeError, match="complete Slurm allocation"):
        _slurm_identity(environment)


def test_gpu_telemetry_parser_requires_and_summarizes_all_frozen_fields(tmp_path: Path) -> None:
    telemetry = tmp_path / "gpu.csv"
    telemetry.write_text(
        "timestamp, index, uuid, name, pstate, temperature.gpu, utilization.gpu [%], "
        "utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], "
        "clocks.current.sm [MHz]\n"
        "2026/06/29 08:00:00.000, 0, GPU-a, Test GPU, P0, 60, 70 %, 20 %, 1024 MiB, "
        "81920 MiB, 200 W, 1500 MHz\n"
        "2026/06/29 08:00:15.000, 0, GPU-a, Test GPU, P0, 62, 80 %, 25 %, 2048 MiB, "
        "81920 MiB, 220 W, 1520 MHz\n"
    )
    summary, rows = _summarize_gpu_telemetry(telemetry)
    assert len(rows) == 2
    assert set(summary) == {
        "utilization_gpu_pct",
        "memory_used_mib",
        "temperature_c",
        "power_w",
        "clocks_sm_mhz",
    }
    assert summary["utilization_gpu_pct"]["mean"] == 75.0
    assert summary["memory_used_mib"]["max"] == 2048.0
    assert summary["temperature_c"]["max"] == 62.0
    assert summary["power_w"]["p50"] == 210.0
    assert summary["clocks_sm_mhz"]["mean"] == 1510.0


def test_gpu_telemetry_parser_fails_closed_when_power_is_unavailable(tmp_path: Path) -> None:
    telemetry = tmp_path / "gpu.csv"
    telemetry.write_text(
        "timestamp, utilization.gpu [%], memory.used [MiB], temperature.gpu, power.draw [W], "
        "clocks.current.sm [MHz]\n"
        "now, 70 %, 1024 MiB, 60, [N/A], 1500 MHz\n"
    )
    with pytest.raises(RuntimeError, match="power_w"):
        _summarize_gpu_telemetry(telemetry)
