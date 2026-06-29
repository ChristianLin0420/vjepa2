from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from jepa4d.evaluation.phase2e_feature_cache import CACHE_SCHEMA, RECEIPT_SCHEMA, sha256_file
from jepa4d.evaluation.phase2e_final import (
    ARTIFACT_MANIFEST_SCHEMA,
    BASELINE,
    CANDIDATE,
    EVALUATION_SCHEMA,
    FEATURE_WANDB_RECEIPT_SCHEMA,
    EvaluationSplit,
    compute_operational_gate,
    fit_log_variance_multiplier,
    run_final_evaluation,
    upload_evaluation_artifact,
    verify_feature_inputs_before_test,
)
from scripts.run_phase2e_factorized_shard import run_training_shard


def _split(name: str, count: int, *, paired: bool, teacher: bool) -> dict[str, Any]:
    generator = torch.Generator().manual_seed({"train": 11, "validation": 13, "test": 17}[name])
    prefix = (count, 2) if paired else (count,)
    features = torch.randn(*prefix, 768, 24, 24, generator=generator, dtype=torch.float16)
    rgb = torch.rand(*prefix, 3, 96, 96, generator=generator, dtype=torch.float16)
    targets = 0.8 + 1.8 * torch.rand(*prefix, 24, 24, generator=generator)
    intrinsics = torch.zeros(*prefix, 3, 3)
    for index in range(count):
        intrinsics[index, ..., 0, 0] = 285.0 + 35.0 * index
        intrinsics[index, ..., 1, 1] = 290.0 + 30.0 * index
    intrinsics[..., 0, 2] = 191.5
    intrinsics[..., 1, 2] = 191.5
    intrinsics[..., 2, 2] = 1.0
    value: dict[str, Any] = {
        "features": features,
        "rgb": rgb,
        "intrinsics_384": intrinsics,
        "targets": targets,
        "sample_ids": [f"{name}-{index}" for index in range(count)],
        "sensor_ids": ["kv2" if name == "test" else f"sensor-{index % 2}" for index in range(count)],
    }
    if teacher:
        log_target = targets.log()
        value["teacher_centered_shape"] = (log_target - log_target.mean(dim=(-2, -1), keepdim=True)).half()
    return value


def _metadata(split: dict[str, Any], name: str) -> list[dict[str, Any]]:
    views = ["center_square", "center_crop_0.85"] if name == "train" else ["center_square"]
    return [
        {
            "sample_id": sample_id,
            "sensor_id": sensor_id,
            "group_id": f"group-{name}-{index}",
            "views": [
                {
                    "view_name": view,
                    "crop_box_top_left_height_width": [0, 0, 384, 384],
                    "source_size_height_width": [384, 384],
                }
                for view in views
            ],
        }
        for index, (sample_id, sensor_id) in enumerate(zip(split["sample_ids"], split["sensor_ids"], strict=True))
    ]


def _write_feature_inputs(root: Path) -> tuple[Path, Path, Path]:
    train = _split("train", 1, paired=True, teacher=True)
    validation = _split("validation", 2, paired=False, teacher=False)
    test = _split("test", 2, paired=False, teacher=False)
    train_path = root / "train_validation_cache.pt"
    test_path = root / "test_cache.pt"
    torch.save({"schema_version": CACHE_SCHEMA, "splits": {"train": train, "validation": validation}}, train_path)
    torch.save({"schema_version": CACHE_SCHEMA, "splits": {"test": test}}, test_path)
    report_path = root / "feature_cache_report.html"
    report_path.write_text("<!doctype html><html><body>synthetic cache report</body></html>")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass",
        "evidence_level": "feature-cache-build",
        "created_utc": "2026-01-01T00:00:00Z",
        "dataset": {"dataset_id": "synthetic", "version": "1", "split_hash": "synthetic-split"},
        "models": {},
        "view_policy": {},
        "feature_normalization": {},
        "teacher_policy": {},
        "caches": {
            "train_validation": {
                "path": str(train_path.resolve()),
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
                "schema_version": CACHE_SCHEMA,
                "splits": ["train", "validation"],
            },
            "test": {
                "path": str(test_path.resolve()),
                "bytes": test_path.stat().st_size,
                "sha256": sha256_file(test_path),
                "schema_version": CACHE_SCHEMA,
                "splits": ["test"],
            },
        },
        "split_summaries": {},
        "sample_metadata": {
            "train": _metadata(train, "train"),
            "validation": _metadata(validation, "validation"),
            "test": _metadata(test, "test"),
        },
        "profiles": {},
        "runtime": {},
        "wandb_url": "https://wandb.invalid/cache",
        "model_metrics_computed": False,
        "large_caches_uploaded_to_wandb": False,
        "report": {
            "path": str(report_path.resolve()),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
            "self_contained": True,
        },
    }
    receipt_path = root / "feature_cache_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    wandb_receipt = {
        "schema_version": FEATURE_WANDB_RECEIPT_SCHEMA,
        "status": "uploaded",
        "mode": "online",
        "run_id": "feature-run-id",
        "artifact_id": "feature-artifact-id",
        "artifact_qualified_name": "entity/project/feature:v0",
        "artifact_digest": "feature-digest",
        "receipt_sha256": sha256_file(receipt_path),
        "report_sha256": sha256_file(report_path),
    }
    (root / "wandb_receipt.json").write_text(json.dumps(wandb_receipt, sort_keys=True) + "\n")
    return train_path, test_path, receipt_path


class _FakeTable:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[tuple[Any, ...]] = []

    def add_data(self, *values: Any) -> None:
        self.rows.append(values)


class _FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.directory: tuple[str, str] | None = None

    def add_dir(self, path: str, *, name: str) -> None:
        self.directory = (path, name)


class _FakeUploaded:
    def __init__(self, identifier: str) -> None:
        self.id = identifier
        self.name = f"artifact-{identifier}"
        self.qualified_name = f"entity/project/artifact-{identifier}:v0"
        self.version = "v0"
        self.digest = f"digest-{identifier}"
        self.wait_timeout: int | None = None

    def wait(self, timeout: int) -> _FakeUploaded:
        self.wait_timeout = timeout
        return self


class _FakeRun:
    def __init__(self, identifier: str) -> None:
        self.id = identifier
        self.url = f"https://wandb.invalid/{identifier}"
        self.path = f"entity/project/{identifier}"
        self.offline = False
        self.summary: dict[str, Any] = {}
        self.uploaded = _FakeUploaded(f"uploaded-{identifier}")
        self.logged_artifact: _FakeArtifact | None = None

    def define_metric(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def log(self, _value: Any) -> None:
        pass

    def log_artifact(self, artifact: _FakeArtifact) -> _FakeUploaded:
        self.logged_artifact = artifact
        return self.uploaded

    def finish(self, *, exit_code: int) -> None:
        assert exit_code in {0, 1}


class _FakeWandb:
    Artifact = _FakeArtifact
    Table = _FakeTable

    class Html:
        def __init__(self, path: str, *, inject: bool) -> None:
            self.path = path
            self.inject = inject


def _formal_shards(root: Path, train_cache: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    import wandb

    counter = iter(range(4))

    def init(**_kwargs: Any) -> _FakeRun:
        return _FakeRun(f"formal-run-{next(counter)}")

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Table", _FakeTable)
    monkeypatch.setattr(wandb, "Html", _FakeWandb.Html)
    monkeypatch.setattr(wandb, "Artifact", _FakeArtifact)
    groups = (
        ("monolithic_final", "factorized_bias"),
        ("factorized_vjepa", "factorized_rgb"),
        ("factorized_vjepa_rgb", "factorized_vjepa_k"),
        ("factorized_full", "factorized_full_teacher"),
    )
    directories = []
    for index, variants in enumerate(groups):
        output = root / f"shard-{index}"
        run_training_shard(
            train_cache,
            output,
            variants,
            (0, 1, 2),
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            device_name="cpu",
            wandb_enabled=True,
            run_name=f"synthetic-shard-{index}",
        )
        directories.append(output)
    return directories


@pytest.fixture
def one_cpu_thread() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_validation_variance_multiplier_is_fitted_once_from_saved_predictions() -> None:
    target = torch.full((1, 24, 24), 2.0)
    prediction = target * np.e
    split = EvaluationSplit(
        "validation",
        torch.zeros(1, 768, 24, 24),
        torch.zeros(1, 3, 96, 96),
        torch.eye(3).unsqueeze(0),
        target,
        ["validation-0"],
        ["sensor"],
    )
    payload = {
        "schema_version": "jepa4d-phase2e-validation-predictions-v1",
        "sample_ids": split.sample_ids,
        "sensor_ids": split.sensor_ids,
        "prediction_m": prediction,
        "target_m": target,
        "log_variance": torch.zeros_like(target),
    }
    assert fit_log_variance_multiplier(payload, split) == pytest.approx(1.0)


def _aggregate(
    variant: str, control: str, *, raw: float, aligned: float, scale: float, nll: float, latency: float, params: float
) -> dict[str, Any]:
    metrics = {
        key: {"mean": value, "sd": 0.01}
        for key, value in {
            "metric_abs_rel": raw,
            "aligned_abs_rel": aligned,
            "abs_log_scale_error": scale,
            "calibrated_log_depth_nll": nll,
        }.items()
    }
    return {
        "variant": variant,
        "intrinsics_control": control,
        "metrics": metrics,
        "head_latency_ms": {"mean": latency, "sd": 0.01},
        "trainable_parameters": {"mean": params, "sd": 0.0},
    }


def test_operational_gate_uses_strict_fixed_candidate_conditions() -> None:
    rows = [
        _aggregate(BASELINE, "correct", raw=0.20, aligned=0.10, scale=0.20, nll=0.50, latency=1.0, params=100),
        _aggregate(CANDIDATE, "correct", raw=0.10, aligned=0.101, scale=0.10, nll=0.40, latency=1.1, params=110),
        _aggregate(CANDIDATE, "wrong", raw=0.15, aligned=0.11, scale=0.15, nll=0.45, latency=1.1, params=110),
        _aggregate(CANDIDATE, "shuffled", raw=0.16, aligned=0.11, scale=0.16, nll=0.46, latency=1.1, params=110),
    ]
    gate = compute_operational_gate(rows)
    assert gate["passed"]
    rows[1]["metrics"]["metric_abs_rel"]["mean"] = 0.20
    assert not compute_operational_gate(rows)["conditions"]["candidate_raw_abs_rel_strictly_lower"]


def test_final_evaluator_end_to_end_outputs_and_frozen_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    one_cpu_thread: None,
) -> None:
    train_cache, test_cache, feature_receipt = _write_feature_inputs(tmp_path)
    with pytest.raises(ValueError, match="frozen manifest path"):
        verify_feature_inputs_before_test(train_cache, test_cache, feature_receipt)
    shards = _formal_shards(tmp_path, train_cache, monkeypatch)
    output = tmp_path / "final"
    result = run_final_evaluation(
        train_cache,
        test_cache,
        feature_receipt,
        shards,
        output,
        device_name="cpu",
        batch_size=2,
        latency_warmup=0,
        latency_iterations=1,
        latency_repetitions=1,
        wandb_enabled=False,
        expected_epochs=1,
        require_formal_protocol=False,
    )
    assert result["schema_version"] == EVALUATION_SCHEMA
    assert result["status"] == "success"
    assert result["counts"] == {
        "test_samples": 2,
        "formal_checkpoints": 24,
        "per_seed_rows": 42,
        "per_sample_rows": 84,
        "failures": 0,
    }
    assert len(result["aggregates"]) == 14
    candidate_controls = {row["intrinsics_control"] for row in result["aggregates"] if row["variant"] == CANDIDATE}
    assert candidate_controls == {"correct", "wrong", "shuffled"}
    assert result["gate"]["population_significance_claimed"] is False
    assert all(np.isfinite(value) for row in result["per_seed"] for value in row["metrics"].values())

    with np.load(output / "phase2e_final_predictions.npz", allow_pickle=False) as predictions:
        assert predictions["prediction_m"].shape == (42, 2, 24, 24)
        assert predictions["log_variance"].shape == (42, 2, 24, 24)
        assert predictions["target_m"].shape == (2, 24, 24)
    with (output / "phase2e_final_per_sample.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 84
    report = (output / "phase2e_final_report.html").read_text()
    assert "Operational gate" in report
    assert "Predicted vs true global log scale" in report
    assert "Same-checkpoint camera controls" in report
    assert "Candidate calibrated log-depth sigma" in report
    assert "data:image/png;base64," in report
    assert "Plotly.newPlot" in report
    assert "<script src=" not in report
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert manifest["schema_version"] == ARTIFACT_MANIFEST_SCHEMA
    assert {row["role"] for row in manifest["files"]} == {
        "canonical_evaluation",
        "full_predictions",
        "per_sample_metrics",
        "visual_report",
    }

    fake_run = _FakeRun("final-run")
    receipt = upload_evaluation_artifact(fake_run, output, result, wandb_module=_FakeWandb)
    assert fake_run.uploaded.wait_timeout == 900
    assert receipt["run_id"] == "final-run"
    assert receipt["artifact_id"] == "uploaded-final-run"
    assert json.loads((output / "wandb_receipt.json").read_text()) == receipt

    with pytest.raises(ValueError, match="exactly four"):
        run_final_evaluation(
            train_cache,
            test_cache,
            feature_receipt,
            shards[:3],
            tmp_path / "rejected",
            device_name="cpu",
            wandb_enabled=False,
            expected_epochs=1,
            require_formal_protocol=False,
        )
