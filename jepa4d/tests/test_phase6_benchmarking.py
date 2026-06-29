import json
from pathlib import Path

from jepa4d.benchmarks.base import BenchmarkAdapter
from jepa4d.benchmarks.manifest import DatasetManifest
from jepa4d.benchmarks.reporting import (
    BenchmarkSuiteReport,
    FailureCategory,
    aggregate_metric_runs,
    bootstrap_confidence_interval,
    write_suite_reports,
)
from jepa4d.benchmarks.runner import BenchmarkJob, run_job


class FailingBenchmark(BenchmarkAdapter):
    name = "failing"
    stage = "verification"

    def run(self, model_or_system: object, split: str) -> list[dict]:
        raise RuntimeError(f"injected failure on {split}")

    def compute_metrics(self, predictions: object, ground_truth: object) -> dict[str, float]:
        return {}

    def report(self) -> dict:
        return {"name": self.name, "stage": self.stage}


def test_versioned_fixture_manifest_has_valid_hash() -> None:
    manifest = DatasetManifest.load("jepa4d/config/benchmarks/manifests/robo4d_jepa_v0.yaml")
    assert manifest.version == "0.1.0"
    assert not manifest.official
    assert manifest.validate_assets() == []


def test_bootstrap_estimates_are_reproducible_and_finite() -> None:
    first = bootstrap_confidence_interval([0.0, 1.0, 1.0, 1.0], seed=7, resamples=500)
    second = bootstrap_confidence_interval([0.0, 1.0, 1.0, 1.0], seed=7, resamples=500)
    assert first == second
    assert first.lower <= first.mean <= first.upper
    estimates = aggregate_metric_runs([{"score": 0.5}, {"score": 1.0}], resamples=100)
    assert estimates["score"].samples == 2


def test_runner_records_typed_failure_without_crashing_suite() -> None:
    report, failures = run_job(BenchmarkJob(FailingBenchmark(), None), repetitions=2, bootstrap_resamples=100)
    assert report.successes == 0
    assert report.failures == 2
    assert failures[0].category == FailureCategory.VERIFICATION
    assert "injected failure" in failures[0].message


def test_suite_writes_json_html_and_failure_dashboard(tmp_path: Path) -> None:
    stage, failures = run_job(BenchmarkJob(FailingBenchmark(), None), repetitions=1, bootstrap_resamples=10)
    report = BenchmarkSuiteReport(
        suite_id="test-suite",
        timestamp="2026-06-29T00:00:00+00:00",
        evidence_level="contract-only",
        config={},
        dataset_manifest={},
        stages=[stage],
        failures=failures,
    )
    json_path, failures_path, html_path = write_suite_reports(report, tmp_path)
    assert json.loads(json_path.read_text())["suite_id"] == "test-suite"
    assert json.loads(failures_path.read_text())[0]["category"] == "verification"
    assert "Failure dashboard" in html_path.read_text()
