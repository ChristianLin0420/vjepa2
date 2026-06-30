from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts.run_phase2g_training_smoke import (
    ARMS,
    CLAIM_BOUNDARY,
    SmokeSettings,
    _parse_args,
    _synthetic_batch,
    main,
    nvidia_smi_telemetry,
    run_training_smoke,
)


class _FakeLoggedArtifact:
    id = "artifact-id"
    version = "v0"
    digest = "artifact-digest"

    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[Path, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((Path(path), name))


class _FakeRun:
    offline = False
    id = "run-id"
    url = "https://wandb.invalid/entity/project/runs/run-id"
    project = "test-project"
    entity = "test-entity"
    group = "test-group"
    name = "test-run"

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.metrics: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.artifacts: list[_FakeArtifact] = []
        self.logged_artifact: _FakeLoggedArtifact | None = None
        self.finish_code: int | None = None

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        self.metrics.append((args, kwargs))

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        self.logs.append((values, step))

    def log_artifact(self, artifact: _FakeArtifact) -> _FakeLoggedArtifact:
        self.artifacts.append(artifact)
        self.logged_artifact = _FakeLoggedArtifact()
        return self.logged_artifact

    def finish(self, *, exit_code: int) -> None:
        self.finish_code = exit_code


class _FakeWandb:
    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_kwargs: dict[str, Any] | None = None
        self.created_artifacts: list[_FakeArtifact] = []

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run

    def Artifact(self, name: str, *, type: str, metadata: dict[str, Any]) -> _FakeArtifact:  # noqa: N802
        artifact = _FakeArtifact(name, type, metadata)
        self.created_artifacts.append(artifact)
        return artifact


@pytest.fixture
def one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _assert_finite_scalars(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_scalars(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_scalars(child)
    elif isinstance(value, float):
        assert math.isfinite(value)
    else:
        assert not isinstance(value, torch.Tensor)


def test_synthetic_batch_is_deterministic_and_has_paired_views(tmp_path: Path) -> None:
    settings = SmokeSettings(
        output=tmp_path / "unused",
        execution_id="batch-test",
        max_steps=1,
        input_dim=8,
        spatial_size=4,
        source_groups=2,
    )
    first = _synthetic_batch(settings, arm_index=2, step=0)
    second = _synthetic_batch(settings, arm_index=2, step=0)
    different = _synthetic_batch(settings, arm_index=2, step=1)
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert not torch.equal(first[0], different[0])
    assert first[0].shape == (4, 8, 4, 4)
    assert first[1].shape == (4, 4, 4)
    assert first[2].dtype == torch.bool
    assert first[3].shape == (4, 3, 3)


@pytest.mark.parametrize("steps", [0, 11])
def test_settings_rejects_unbounded_step_counts(tmp_path: Path, steps: int) -> None:
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        SmokeSettings(output=tmp_path / "run", execution_id="bounds-test", max_steps=steps)


def test_settings_rejects_unbound_git_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Git object ID"):
        SmokeSettings(output=tmp_path / "run", execution_id="git-test", git_commit="deadbeef")


def test_cli_parser_bounds_max_steps() -> None:
    assert _parse_args(["--output", "unused", "--execution-id", "parse-test"]).max_steps == 3
    assert _parse_args(["--output", "unused", "--execution-id", "parse-test", "--max-steps", "10"]).max_steps == 10
    with pytest.raises(SystemExit):
        _parse_args(["--output", "unused", "--execution-id", "parse-test", "--max-steps", "11"])


def test_nvidia_smi_telemetry_uses_allocated_gpu_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(uuid="GPU-allocated"))

    def fake_run(command, **kwargs):
        assert "--id=GPU-allocated" in command
        assert kwargs["check"] is True
        return SimpleNamespace(stdout="75, 1024, 81920, 42, 150, 1200, 1500\n")

    monkeypatch.setattr("scripts.run_phase2g_training_smoke.subprocess.run", fake_run)
    assert nvidia_smi_telemetry(torch.device("cuda:0")) == {
        "gpu_utilization_percent": 75.0,
        "gpu_memory_used_mib": 1024.0,
        "gpu_memory_total_mib": 81920.0,
        "gpu_temperature_c": 42.0,
        "gpu_power_w": 150.0,
        "gpu_sm_clock_mhz": 1200.0,
        "gpu_memory_clock_mhz": 1500.0,
    }


def test_cpu_smoke_persists_complete_safe_contract(tmp_path: Path, one_cpu_thread: None) -> None:
    output = tmp_path / "smoke"
    fake = _FakeWandb()
    telemetry_calls: list[torch.device] = []

    def telemetry(device: torch.device) -> dict[str, float]:
        telemetry_calls.append(device)
        return {}

    receipt = run_training_smoke(
        SmokeSettings(
            output=output,
            execution_id="cpu-contract-test",
            max_steps=1,
            seed=17,
            device="cpu",
            input_dim=8,
            spatial_size=4,
            source_groups=2,
            wandb_project="test-project",
            wandb_entity="test-entity",
            wandb_group="test-group",
            wandb_run_name="test-run",
        ),
        wandb_module=fake,
        telemetry_reader=telemetry,
    )

    assert receipt["status"] == "success"
    assert receipt["execution_id"] == "cpu-contract-test"
    assert receipt["evidence_level"] == "integration-smoke"
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY
    assert receipt["synthetic_inputs_only"] is True
    assert receipt["dataset_or_cache_access"] is False
    assert receipt["runtime_identity"] == receipt["config"]["runtime_identity"]
    assert receipt["runtime_identity"]["git_commit"] == "0" * 40
    assert receipt["runtime_identity"]["scheduler_job_id"] == "unit-test"
    assert set(receipt["runtime_identity"]["code"]) == {"runner", "model_module", "training_module"}
    assert all(len(identity["sha256"]) == 64 for identity in receipt["runtime_identity"]["code"].values())
    assert receipt["total_optimizer_steps"] == len(ARMS)
    assert receipt["expected_optimizer_steps"] == len(ARMS)
    assert set(receipt["checkpoints"]) == set(ARMS)
    assert set(receipt["arms"]) == set(ARMS)
    assert all(receipt["checkpoints"][arm]["exact_reload"] for arm in ARMS)
    assert all(receipt["arms"][arm]["maximum_forbidden_gradient_norm"] == 0.0 for arm in ARMS)
    _assert_finite_scalars(receipt)

    expected_files = {
        "SUCCESS",
        "steps.jsonl",
        "training_receipt.json",
        "wandb_receipt.json",
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    assert {path.name for path in (output / "checkpoints").iterdir()} == {f"{arm}.pt" for arm in ARMS}
    assert (output / "SUCCESS").read_text() == "success\n"
    assert json.loads((output / "training_receipt.json").read_text()) == receipt

    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == len(ARMS)
    assert [row["arm"] for row in rows] == list(ARMS)
    assert [row["global_step"] for row in rows] == list(range(len(ARMS)))
    for row in rows:
        _assert_finite_scalars(row)
        assert row["gradient_firewall_passed"] is True
        assert row["loss"]["total"] >= 0
        assert "norm_shape" in row["gradients"]
        assert "norm_scale" in row["gradients"]
        assert "norm_field" in row["gradients"]
        assert "firewall_max_forbidden_norm" in row["gradients"]
        assert "firewall_shape_to_scale" in row["gradients"]
        assert row["optimizer"]["parameter_update_norm"] > 0
        assert row["optimizer"]["unclipped_gradient_norm"] >= row["optimizer"]["clipped_gradient_norm"]
        assert row["timing"]["step_seconds"] > 0
        assert row["timing"]["samples_per_second"] > 0
        assert set(row["memory"]) == {
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        }
        assert row["nvidia_smi"] == {}
    assert len(telemetry_calls) == len(ARMS)
    assert all(device.type == "cpu" for device in telemetry_calls)

    assert fake.init_kwargs is not None
    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["config"]["claim_boundary"] == CLAIM_BOUNDARY
    assert fake.init_kwargs["config"]["dataset_or_cache_access"] is False
    assert fake.init_kwargs["config"]["runtime_identity"] == receipt["runtime_identity"]
    assert len(fake.run.logs) == len(ARMS)
    for row, (logged, step) in zip(rows, fake.run.logs, strict=True):
        prefix = f"arms/{row['arm']}"
        assert step == row["global_step"]
        assert f"{prefix}/loss/total" in logged
        assert f"{prefix}/gradients/firewall_max_forbidden_norm" in logged
        assert f"{prefix}/optimizer/learning_rate" in logged
        assert f"{prefix}/optimizer/unclipped_gradient_norm" in logged
        assert f"{prefix}/optimizer/parameter_update_norm" in logged
        assert f"{prefix}/throughput/samples_per_second" in logged
        assert f"{prefix}/memory/peak_reserved_bytes" in logged
    assert fake.run.finish_code == 0
    assert fake.run.logged_artifact is not None and fake.run.logged_artifact.waited
    assert len(fake.created_artifacts) == 1
    artifact_names = {name for _, name in fake.created_artifacts[0].files}
    assert artifact_names == {"steps.jsonl", *(f"checkpoints/{arm}.pt" for arm in ARMS)}
    assert not any(path.name.startswith(("features", "targets", "dataset", "cache")) for path in output.rglob("*"))


def test_cli_requires_slurm_before_wandb_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        main(
            [
                "--output",
                str(tmp_path / "run"),
                "--execution-id",
                "cli-gate-test",
                "--device",
                "cpu",
            ]
        )
