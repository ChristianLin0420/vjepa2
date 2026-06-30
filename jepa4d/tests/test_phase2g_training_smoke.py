from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import scripts.run_phase2g_training_smoke as smoke_module
from scripts.run_phase2g_training_smoke import (
    ARMS,
    CLAIM_BOUNDARY,
    SMOKE_SCHEMA,
    STEP_SCHEMA,
    WANDB_RECEIPT_SCHEMA,
    SmokeSettings,
    _parse_args,
    _runtime_identity,
    _sha256_text,
    _synthetic_batch,
    main,
    nvidia_smi_telemetry,
    run_training_smoke,
)


class _FakeLoggedArtifact:
    id = "artifact-id"
    version = "v0"
    digest = "artifact-digest"

    def __init__(self, name: str, wait_hook: Callable[[], None] | None = None) -> None:
        self.name = f"{name}:{self.version}"
        self.waited = False
        self.wait_hook = wait_hook

    def wait(self) -> None:
        self.waited = True
        if self.wait_hook is not None:
            self.wait_hook()


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
    url = "https://wandb.invalid/test-entity/test-project/runs/run-id"
    project = "test-project"
    entity = "test-entity"
    group = "test-group"
    name = "test-run"
    job_type = "phase2g-instrumentation-smoke"

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.metrics: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.artifacts: list[_FakeArtifact] = []
        self.logged_artifact: _FakeLoggedArtifact | None = None
        self.finish_code: int | None = None
        self.fail_log = False
        self.artifact_wait_hook: Callable[[_FakeArtifact], None] | None = None

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        self.metrics.append((args, kwargs))

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        if self.fail_log:
            raise RuntimeError("injected W&B log failure")
        self.logs.append((values, step))

    def log_artifact(self, artifact: _FakeArtifact) -> _FakeLoggedArtifact:
        self.artifacts.append(artifact)
        artifact_wait_hook = self.artifact_wait_hook
        wait_hook = None if artifact_wait_hook is None else lambda: artifact_wait_hook(artifact)
        self.logged_artifact = _FakeLoggedArtifact(artifact.name, wait_hook)
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


def _minimal_settings(output: Path, *, gradient_clip: float = 5.0) -> SmokeSettings:
    return SmokeSettings(
        output=output,
        execution_id="adversarial-test",
        max_steps=1,
        seed=23,
        device="cpu",
        input_dim=8,
        spatial_size=4,
        source_groups=2,
        gradient_clip=gradient_clip,
        wandb_project="test-project",
        wandb_entity="test-entity",
        wandb_group="test-group",
        wandb_run_name="test-run",
    )


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


@pytest.mark.parametrize(("setting", "match"), (("input_dim", "at least 2"), ("spatial_size", "at least 2")))
def test_settings_rejects_non_executable_synthetic_dimensions(tmp_path: Path, setting: str, match: str) -> None:
    values: dict[str, Any] = {setting: 1}
    with pytest.raises(ValueError, match=match):
        SmokeSettings(output=tmp_path / "run", execution_id="dimension-test", **values)


def test_cli_parser_bounds_max_steps() -> None:
    assert _parse_args(["--output", "unused", "--execution-id", "parse-test"]).max_steps == 3
    assert _parse_args(["--output", "unused", "--execution-id", "parse-test", "--max-steps", "10"]).max_steps == 10
    with pytest.raises(SystemExit):
        _parse_args(["--output", "unused", "--execution-id", "parse-test", "--max-steps", "11"])


def test_nvidia_smi_telemetry_uses_allocated_gpu_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(uuid="allocated"))

    def fake_run(command, **kwargs):
        assert "--id=GPU-allocated" in command
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0, stdout="75, 1024, 81920, 42, 150, 1200, 1500\n")

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


def test_nvidia_smi_failure_does_not_expose_allocated_gpu_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_uuid = "private-hardware-identity"
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(uuid=raw_uuid))
    monkeypatch.setattr(
        "scripts.run_phase2g_training_smoke.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(RuntimeError, match="telemetry query failed") as error:
        nvidia_smi_telemetry(torch.device("cuda:0"))
    assert raw_uuid not in str(error.value)


def test_hardware_uuid_is_pseudonymized_before_serialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_uuid = "GPU-private-hardware-identity"
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(
            name="Synthetic GPU",
            uuid=raw_uuid,
            major=8,
            minor=0,
            total_memory=1024,
        ),
    )
    identity = _runtime_identity(_minimal_settings(tmp_path / "unused"), torch.device("cuda:0"))
    assert "device_uuid" not in identity["hardware"]
    assert identity["hardware"]["device_uuid_sha256"] == _sha256_text(raw_uuid)


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

    assert receipt["schema_version"] == SMOKE_SCHEMA
    assert receipt["status"] == "pending-postflight"
    assert receipt["terminal_status"] == "pending-postflight"
    assert receipt["postflight_required"] is True
    assert receipt["execution_id"] == "cpu-contract-test"
    assert receipt["evidence_level"] == "integration-smoke"
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY
    assert receipt["synthetic_inputs_only"] is True
    assert receipt["dataset_or_cache_access"] is False
    assert receipt["resource_policy"] == "diagnostic-only"
    assert set(receipt["config"]["arm_configs"]) == set(ARMS)
    assert receipt["config"]["loss_config"]["global_scale_weight"] == 1.0
    assert "may be negative" in receipt["config"]["nll_convention"]
    assert receipt["config"]["optimizer"]["name"] == "torch.optim.AdamW"
    assert receipt["config"]["optimizer"]["betas"] == [0.9, 0.999]
    assert receipt["config"]["determinism"]["bitwise_reproducibility_claimed"] is False
    assert receipt["config"]["resource_policy"] == "diagnostic-only"
    assert receipt["config"]["requested_wandb_identity"] == {
        "entity": "test-entity",
        "project": "test-project",
        "group": "test-group",
        "run_name": "test-run",
        "job_type": "phase2g-instrumentation-smoke",
    }
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
        "steps.jsonl",
        "training_receipt.json",
        "wandb_receipt.json",
    }
    assert {path.name for path in output.iterdir()} == {"checkpoints", *expected_files}
    assert {path.name for path in (output / "checkpoints").iterdir()} == {f"{arm}.pt" for arm in ARMS}
    assert not (output / "SUCCESS").exists()
    assert json.loads((output / "training_receipt.json").read_text()) == receipt

    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == len(ARMS)
    assert [row["arm"] for row in rows] == list(ARMS)
    assert [row["global_step"] for row in rows] == list(range(len(ARMS)))
    for row in rows:
        assert row["schema_version"] == STEP_SCHEMA
        _assert_finite_scalars(row)
        assert row["gradient_firewall_passed"] is True
        field_name = "field_objective" if "field_objective" in row["objectives"] else "scale_field_objective"
        expected_total = (
            row["objectives"]["shape_objective"] + row["objectives"]["scale_objective"] + row["objectives"][field_name]
        )
        assert math.isfinite(row["objectives"]["total"])
        assert row["objectives"]["total"] == pytest.approx(expected_total, rel=1e-5, abs=1e-6)
        assert not (set(row["objectives"]) & set(row["diagnostics"]))
        assert "joint_nll_diagnostic_only" not in row["objectives"]
        assert "optimal_log_scale_mean" not in row["objectives"]
        assert "norm_shape" in row["gradients"]
        assert "norm_scale" in row["gradients"]
        assert "norm_field" in row["gradients"]
        assert "firewall_max_forbidden_norm" in row["gradients"]
        assert "firewall_shape_to_scale" in row["gradients"]
        assert row["optimizer"]["parameter_update_norm"] > 0
        assert row["optimizer"]["post_clip_gradient_norm"] == pytest.approx(
            row["optimizer"]["pre_clip_gradient_norm"], rel=1e-5, abs=1e-7
        )
        assert row["optimizer"]["applied_clip_coefficient"] == pytest.approx(1.0)
        assert row["optimizer"]["was_clipped"] == 0
        assert row["resources"]["policy"] == "diagnostic-only"
        assert row["resources"]["timing"]["step_seconds"] > 0
        assert row["resources"]["timing"]["samples_per_second"] > 0
        assert set(row["resources"]["memory"]) == {
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        }
        assert row["resources"]["nvidia_smi"] == {}
    assert len(telemetry_calls) == len(ARMS)
    assert all(device.type == "cpu" for device in telemetry_calls)

    assert fake.init_kwargs is not None
    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["reinit"] == "finish_previous"
    assert fake.init_kwargs["config"]["claim_boundary"] == CLAIM_BOUNDARY
    assert fake.init_kwargs["config"]["dataset_or_cache_access"] is False
    assert fake.init_kwargs["config"]["runtime_identity"] == receipt["runtime_identity"]
    assert len(fake.run.logs) == len(ARMS)
    for row, (logged, step) in zip(rows, fake.run.logs, strict=True):
        prefix = f"arms/{row['arm']}"
        assert step == row["global_step"]
        assert f"{prefix}/objective/total" in logged
        if row["diagnostics"]:
            assert any(name.startswith(f"{prefix}/diagnostic/") for name in logged)
        assert f"{prefix}/gradients/firewall_max_forbidden_norm" in logged
        assert f"{prefix}/optimizer/learning_rate" in logged
        assert f"{prefix}/optimizer/pre_clip_gradient_norm" in logged
        assert f"{prefix}/optimizer/post_clip_gradient_norm" in logged
        assert f"{prefix}/optimizer/parameter_update_norm" in logged
        assert f"{prefix}/resource_diagnostic/throughput/samples_per_second" in logged
        assert f"{prefix}/resource_diagnostic/memory/peak_reserved_bytes" in logged
    assert fake.run.finish_code == 0
    assert fake.run.summary["status"] == "pending-postflight"
    assert fake.run.logged_artifact is not None and fake.run.logged_artifact.waited
    assert len(fake.created_artifacts) == 1
    artifact_names = {name for _, name in fake.created_artifacts[0].files}
    assert artifact_names == {"steps.jsonl", *(f"checkpoints/{arm}.pt" for arm in ARMS)}
    assert fake.created_artifacts[0].metadata["terminal_status"] == "pending-postflight"
    wandb_receipt = json.loads((output / "wandb_receipt.json").read_text())
    assert wandb_receipt["schema_version"] == WANDB_RECEIPT_SCHEMA
    assert wandb_receipt["status"] == "uploaded-preliminary"
    assert wandb_receipt["terminal_status"] == "pending-postflight"
    assert ":" not in wandb_receipt["artifact_name"]
    assert {value["name"] for value in wandb_receipt["files"]} == artifact_names
    assert len(wandb_receipt["files_sha256"]) == 64
    assert not any(path.name.startswith(("features", "targets", "dataset", "cache")) for path in output.rglob("*"))


def test_controlled_small_threshold_records_actual_gradient_clipping(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = tmp_path / "forced-clipping"
    receipt = run_training_smoke(
        _minimal_settings(output, gradient_clip=1e-4),
        wandb_module=_FakeWandb(),
        telemetry_reader=lambda _device: {},
    )
    assert receipt["status"] == "pending-postflight"
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == len(ARMS)
    for row in rows:
        optimizer = row["optimizer"]
        assert optimizer["was_clipped"] == 1
        assert optimizer["pre_clip_gradient_norm"] > optimizer["post_clip_gradient_norm"]
        assert optimizer["post_clip_gradient_norm"] <= 1e-4 + 1e-6
        assert 0 < optimizer["applied_clip_coefficient"] < 1
        assert optimizer["applied_clip_coefficient"] == pytest.approx(
            optimizer["post_clip_gradient_norm"] / optimizer["pre_clip_gradient_norm"],
            rel=1e-6,
        )
    assert not (output / "SUCCESS").exists()


@pytest.mark.parametrize("failure_mode", ("train", "firewall", "nan"))
def test_training_or_nonfinite_failure_never_writes_preliminary_success(
    tmp_path: Path,
    one_cpu_thread: None,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    output = tmp_path / failure_mode
    fake = _FakeWandb()
    original = smoke_module.train_phase2f_step

    def injected_step(*args: Any, **kwargs: Any) -> Any:
        if failure_mode == "train":
            raise RuntimeError("injected training failure")
        result = original(*args, **kwargs)
        if failure_mode == "firewall":
            return SimpleNamespace(
                metrics=result.metrics,
                firewall=SimpleNamespace(passed=False, maximum_forbidden_norm=1.0),
                parameter_counts=result.parameter_counts,
            )
        result.metrics["total"] = float("nan")
        return result

    monkeypatch.setattr(smoke_module, "train_phase2f_step", injected_step)
    with pytest.raises(RuntimeError, match="injected training failure|firewall-audited|optimized objective"):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert fake.run.summary["status"] == "failed"
    assert not (output / "SUCCESS").exists()
    assert not (output / "training_receipt.json").exists()
    assert not (output / "wandb_receipt.json").exists()


def test_wandb_history_failure_never_writes_preliminary_success(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = tmp_path / "wandb-failure"
    fake = _FakeWandb()
    fake.run.fail_log = True
    with pytest.raises(RuntimeError, match="injected W&B log failure"):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert fake.run.summary["status"] == "failed"
    assert not (output / "SUCCESS").exists()
    assert not (output / "training_receipt.json").exists()
    assert not (output / "wandb_receipt.json").exists()


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    (
        ("id", None, "valid run_id"),
        ("entity", "wrong-entity", "entity differs"),
        ("project", "wrong-project", "project/group/run name/job type differs"),
        ("job_type", "wrong-job-type", "project/group/run name/job type differs"),
        ("url", "not-an-online-url", "invalid online run URL"),
        ("url", "https://wandb.invalid/entity/project/runs/different-run", "does not bind"),
    ),
)
def test_invalid_wandb_backend_identity_fails_before_training(
    tmp_path: Path,
    one_cpu_thread: None,
    attribute: str,
    value: Any,
    match: str,
) -> None:
    output = tmp_path / "missing-backend"
    fake = _FakeWandb()
    setattr(fake.run, attribute, value)
    with pytest.raises(RuntimeError, match=match):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert not (output / "SUCCESS").exists()
    assert not (output / "training_receipt.json").exists()


def test_mismatched_wandb_artifact_backend_name_fails_closed(
    tmp_path: Path,
    one_cpu_thread: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "mismatched-artifact-name"
    fake = _FakeWandb()
    original_log_artifact = fake.run.log_artifact

    def mismatched_log_artifact(artifact: _FakeArtifact) -> _FakeLoggedArtifact:
        logged = original_log_artifact(artifact)
        logged.name = "different-artifact"
        return logged

    monkeypatch.setattr(fake.run, "log_artifact", mismatched_log_artifact)
    with pytest.raises(RuntimeError, match="artifact name differs"):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert not (output / "training_receipt.json").exists()


def test_artifact_mutation_during_wandb_wait_fails_closed(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = tmp_path / "artifact-mutation"
    fake = _FakeWandb()

    def mutate_steps(artifact: _FakeArtifact) -> None:
        steps_path = next(path for path, name in artifact.files if name == "steps.jsonl")
        with steps_path.open("a", encoding="utf-8") as stream:
            stream.write('{"tampered":true}\n')

    fake.run.artifact_wait_hook = mutate_steps
    with pytest.raises(RuntimeError, match="changed during W&B upload"):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert fake.run.summary["status"] == "failed"
    assert not (output / "SUCCESS").exists()
    assert not (output / "training_receipt.json").exists()
    assert not (output / "wandb_receipt.json").exists()


def test_artifact_change_and_restore_during_wandb_wait_fails_closed(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = tmp_path / "artifact-aba-mutation"
    fake = _FakeWandb()

    def mutate_and_restore_steps(artifact: _FakeArtifact) -> None:
        steps_path = next(path for path, name in artifact.files if name == "steps.jsonl")
        original = steps_path.read_bytes()
        steps_path.write_bytes(original + b'{"tampered":true}\n')
        steps_path.write_bytes(original)

    fake.run.artifact_wait_hook = mutate_and_restore_steps
    with pytest.raises(RuntimeError, match="changed during W&B upload"):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert not (output / "training_receipt.json").exists()
    assert not (output / "wandb_receipt.json").exists()


def test_missing_wandb_artifact_backend_identity_fails_closed(
    tmp_path: Path,
    one_cpu_thread: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "missing-artifact-identity"
    fake = _FakeWandb()
    monkeypatch.setattr(_FakeLoggedArtifact, "digest", None)
    with pytest.raises(RuntimeError, match="valid artifact_digest"):
        run_training_smoke(
            _minimal_settings(output),
            wandb_module=fake,
            telemetry_reader=lambda _device: {},
        )
    assert fake.run.finish_code == 1
    assert fake.run.summary["status"] == "failed"
    assert not (output / "SUCCESS").exists()
    assert not (output / "training_receipt.json").exists()
    assert not (output / "wandb_receipt.json").exists()


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
