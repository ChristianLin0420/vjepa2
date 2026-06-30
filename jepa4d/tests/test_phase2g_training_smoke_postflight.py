from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import slurm.validate_phase2g_training_smoke as postflight_module
from jepa4d.tests.test_phase2g_training_smoke import _FakeWandb
from jepa4d.validation._content import (
    sha256_file,
    sha256_value,
    verify_content_addressed_json,
    write_content_addressed_json,
)
from scripts.run_phase2g_training_smoke import SmokeSettings, run_training_smoke
from slurm.validate_phase2g_training_smoke import (
    APPROVED_ACCOUNT,
    FINAL_WANDB_SCHEMA,
    PRELIMINARY_BACKEND_SCHEMA,
    SchedulerIdentity,
    _terminal_uploads,
    finalize_phase2g_online_run,
    validate_phase2g_training_smoke,
    verify_phase2g_preliminary_backend,
)

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
JOB_ID = "123456"
EXECUTION_ID = "p2g-postflight-test"
PRODUCTION_GOVERNED_NUMERIC = {
    "seed": postflight_module.GOVERNED_SEED,
    "input_dim": postflight_module.GOVERNED_INPUT_DIM,
    "spatial_size": postflight_module.GOVERNED_SPATIAL_SIZE,
    "source_groups": postflight_module.GOVERNED_SOURCE_GROUPS,
    "gradient_clip": postflight_module.GOVERNED_GRADIENT_CLIP,
    "learning_rate": postflight_module.GOVERNED_LEARNING_RATE,
    "weight_decay": postflight_module.GOVERNED_WEIGHT_DECAY,
}


@pytest.fixture
def one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def governed_fixture_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the governed-contract tests small while preserving exact-value validation."""

    monkeypatch.setattr(postflight_module, "GOVERNED_SEED", 17)
    monkeypatch.setattr(postflight_module, "GOVERNED_INPUT_DIM", 8)
    monkeypatch.setattr(postflight_module, "GOVERNED_SPATIAL_SIZE", 4)


def test_production_governed_numbers_match_runner_defaults(tmp_path: Path) -> None:
    settings = SmokeSettings(output=tmp_path / "unused", execution_id="governed-defaults")
    assert {
        "seed": settings.seed,
        "input_dim": settings.input_dim,
        "spatial_size": settings.spatial_size,
        "source_groups": settings.source_groups,
        "gradient_clip": settings.gradient_clip,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
    } == PRODUCTION_GOVERNED_NUMERIC


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _file_identity(path: Path, *, name: str | None = None) -> dict[str, Any]:
    return {
        "name": path.name if name is None else name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _scheduler(**changes: Any) -> SchedulerIdentity:
    values: dict[str, Any] = {
        "job_id": JOB_ID,
        "job_name": "j4d-p2g-smoke-postflight-test",
        "account": APPROVED_ACCOUNT,
        "partition": "polar4",
        "time_limit": "00:30:00",
        "nodes": 1,
        "tasks": 1,
        "gpus": 1,
        "cpus": 8,
        "allocated_cpus": 8,
        "cpus_per_task": 8,
        "memory_mib": 32 * 1024,
        "array_job_id": None,
        "array_task_id": None,
    }
    values.update(changes)
    return SchedulerIdentity(**values)


def _fake_finalizer(**kwargs: Any) -> dict[str, Any]:
    preliminary = kwargs["preliminary_receipt"]
    training_path = Path(kwargs["training_path"])
    postflight_path = Path(kwargs["postflight_path"])
    summary = kwargs["summary"]
    uploads = _terminal_uploads(
        root=training_path.parent,
        training_path=training_path,
        postflight_path=postflight_path,
    )
    return {
        "schema_version": FINAL_WANDB_SCHEMA,
        "status": "finalized",
        "terminal_status": "postflight-pass",
        "mode": "online",
        "preliminary_receipt_sha256": sha256_value(preliminary),
        "entity": preliminary["entity"],
        "project": preliminary["project"],
        "group": preliminary["group"],
        "run_name": preliminary["run_name"],
        "job_type": preliminary["job_type"],
        "run_id": preliminary["run_id"],
        "run_url": preliminary["run_url"],
        "artifact_name": f"phase2g-terminal-{preliminary['run_id']}",
        "artifact_id": "test-entity/test-project/phase2g-terminal:v0",
        "artifact_version": "v0",
        "artifact_digest": "terminal-digest",
        "summary_sha256": sha256_value(summary),
        "files": [{**_file_identity(path, name=name), "role": role} for name, path, role in uploads],
    }


def _fake_preliminary_verifier(**kwargs: Any) -> dict[str, Any]:
    preliminary = kwargs["preliminary_receipt"]
    return {
        "schema_version": PRELIMINARY_BACKEND_SCHEMA,
        "status": "verified",
        "entity": preliminary["entity"],
        "project": preliminary["project"],
        "run_id": preliminary["run_id"],
        "artifact_name": preliminary["artifact_name"],
        "artifact_type": "training-instrumentation-smoke",
        "artifact_id": preliminary["artifact_id"],
        "artifact_version": preliminary["artifact_version"],
        "artifact_digest": preliminary["artifact_digest"],
        "files_sha256": preliminary["files_sha256"],
        "files": preliminary["files"],
    }


def _rewrite_step_contract(output: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    steps_path = output / "steps.jsonl"
    rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    steps_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    training = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    wandb = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    training["steps"] = _file_identity(steps_path)
    wandb["files"][0] = _file_identity(steps_path, name="steps.jsonl")
    wandb["files_sha256"] = sha256_value({"files": wandb["files"]})
    training["wandb"] = wandb
    _write_json(output / "wandb_receipt.json", wandb)
    _write_json(output / "training_receipt.json", training)


def _rewrite_config_contract(output: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    training = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    wandb = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    mutate(training["config"])
    training["config_sha256"] = sha256_value(training["config"])
    wandb["config_sha256"] = training["config_sha256"]
    training["wandb"] = wandb
    _write_json(output / "wandb_receipt.json", wandb)
    _write_json(output / "training_receipt.json", training)


def _rewrite_checkpoint_contract(output: Path, arm: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    checkpoint = output / "checkpoints" / f"{arm}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, checkpoint)
    training = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    wandb = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    training["checkpoints"][arm].update(_file_identity(checkpoint))
    published_name = f"checkpoints/{arm}.pt"
    for index, value in enumerate(wandb["files"]):
        if value["name"] == published_name:
            wandb["files"][index] = _file_identity(checkpoint, name=published_name)
            break
    wandb["files_sha256"] = sha256_value({"files": wandb["files"]})
    training["wandb"] = wandb
    _write_json(output / "wandb_receipt.json", wandb)
    _write_json(output / "training_receipt.json", training)


def _build_output(tmp_path: Path) -> Path:
    output = tmp_path / "phase2g"
    fake = _FakeWandb()
    fake.run.group = f"phase2g-smoke-{EXECUTION_ID}"
    run_training_smoke(
        SmokeSettings(
            output=output,
            execution_id=EXECUTION_ID,
            git_commit=COMMIT,
            scheduler_job_id=JOB_ID,
            max_steps=3,
            seed=17,
            device="cpu",
            input_dim=8,
            spatial_size=4,
            source_groups=2,
            wandb_project="test-project",
            wandb_entity="test-entity",
            wandb_group=f"phase2g-smoke-{EXECUTION_ID}",
            wandb_run_name="test-run",
        ),
        wandb_module=fake,
        telemetry_reader=lambda _device: {},
    )
    assert not (output / "SUCCESS").exists()

    steps_path = output / "steps.jsonl"
    rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
    nvidia = {
        "gpu_utilization_percent": 50.0,
        "gpu_memory_used_mib": 20.0,
        "gpu_memory_total_mib": 80.0,
        "gpu_temperature_c": 42.0,
        "gpu_power_w": 100.0,
        "gpu_sm_clock_mhz": 1200.0,
        "gpu_memory_clock_mhz": 1500.0,
    }
    for row in rows:
        row["resources"]["nvidia_smi"] = nvidia
    steps_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    training = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    runtime = training["runtime_identity"]
    runtime["hardware"] = {
        "device_type": "cuda",
        "device_name": "Fixture GPU",
        "device_uuid_sha256": "b" * 64,
        "compute_capability": "8.0",
        "total_memory_bytes": 80 * 1024 * 1024,
    }
    config = training["config"]
    config["device_type"] = "cuda"
    config["runtime_identity"] = runtime
    config["determinism"]["cuda_manual_seeded_per_arm"] = True
    training["runtime_identity"] = runtime
    training["config"] = config
    training["config_sha256"] = sha256_value(config)
    training["steps"] = _file_identity(steps_path)

    wandb = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    wandb["config_sha256"] = training["config_sha256"]
    wandb["files"][0] = _file_identity(steps_path, name="steps.jsonl")
    wandb["files_sha256"] = sha256_value({"files": wandb["files"]})
    training["wandb"] = wandb
    _write_json(output / "wandb_receipt.json", wandb)
    _write_json(output / "training_receipt.json", training)
    return output


def _validate(output: Path, **changes: Any):
    return validate_phase2g_training_smoke(
        output=output,
        repo_root=ROOT,
        job_id=JOB_ID,
        expected_run_name="test-run",
        expected_wandb_project="test-project",
        expected_wandb_entity="test-entity",
        scheduler_lookup=changes.pop("scheduler_lookup", lambda _job_id: _scheduler()),
        git_lookup=changes.pop("git_lookup", lambda _root: (COMMIT, True)),
        wandb_preliminary_verifier=changes.pop("wandb_preliminary_verifier", _fake_preliminary_verifier),
        wandb_finalizer=changes.pop("wandb_finalizer", _fake_finalizer),
        **changes,
    )


class _BackendRun:
    def __init__(self, preliminary: Mapping[str, Any]) -> None:
        self.entity = preliminary["entity"]
        self.project = preliminary["project"]
        self.id = preliminary["run_id"]
        self.name = preliminary["run_name"]
        self.group = preliminary["group"]
        self.job_type = preliminary["job_type"]
        self.url = preliminary["run_url"]
        self.state = "finished"
        self.artifact: _BackendArtifact | None = None

    def logged_artifacts(self, *, per_page: int) -> list[_BackendArtifact]:
        assert per_page == 100
        assert self.artifact is not None
        return [self.artifact]


class _BackendArtifact:
    def __init__(
        self,
        preliminary: Mapping[str, Any],
        *,
        creator: _BackendRun,
        files: Mapping[str, bytes],
    ) -> None:
        self.entity = preliminary["entity"]
        self.project = preliminary["project"]
        self.name = f"{preliminary['artifact_name']}:{preliminary['artifact_version']}"
        self.type = "training-instrumentation-smoke"
        self.id = preliminary["artifact_id"]
        self.version = preliminary["artifact_version"]
        self.digest = preliminary["artifact_digest"]
        self.state = "COMMITTED"
        self.is_link = False
        self.metadata = {
            "schema_version": postflight_module.SMOKE_SCHEMA,
            "evidence_level": "integration-smoke",
            "synthetic_inputs_only": True,
            "terminal_status": "pending-postflight",
            "config_sha256": preliminary["config_sha256"],
            "files_sha256": preliminary["files_sha256"],
        }
        self.manifest = SimpleNamespace(
            entries={name: SimpleNamespace(ref=None, size=len(value)) for name, value in files.items()}
        )
        self.creator = creator
        self.files = dict(files)
        self.downloaded = False
        self.verified = False

    def logged_by(self) -> _BackendRun:
        return self.creator

    def download(
        self,
        *,
        root: str,
        allow_missing_references: bool,
        skip_cache: bool,
        multipart: bool,
    ) -> str:
        assert allow_missing_references is False
        assert skip_cache is True
        assert multipart is False
        target = Path(root)
        assert not any(target.iterdir())
        for name, value in self.files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        self.downloaded = True
        return str(target)

    def verify(self, root: str) -> None:
        assert self.downloaded
        assert Path(root).is_dir()
        self.verified = True


class _BackendApi:
    def __init__(self, preliminary: Mapping[str, Any], files: Mapping[str, bytes]) -> None:
        self.run_value = _BackendRun(preliminary)
        self.artifact_value = _BackendArtifact(preliminary, creator=self.run_value, files=files)
        self.run_value.artifact = self.artifact_value
        self.preliminary = preliminary

    def run(self, path: str) -> _BackendRun:
        assert path == f"{self.preliminary['entity']}/{self.preliminary['project']}/{self.preliminary['run_id']}"
        return self.run_value

    def artifact(self, path: str, *, type: str) -> _BackendArtifact:
        assert type == "training-instrumentation-smoke"
        assert path == (
            f"{self.preliminary['entity']}/{self.preliminary['project']}/"
            f"{self.preliminary['artifact_name']}:{self.preliminary['artifact_version']}"
        )
        return self.artifact_value


def _backend_api_snapshot(output: Path) -> _BackendApi:
    preliminary = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    files = {
        "steps.jsonl": (output / "steps.jsonl").read_bytes(),
        **{
            f"checkpoints/{arm}.pt": (output / "checkpoints" / f"{arm}.pt").read_bytes()
            for arm in ("M0", "M1", "M2", "M3")
        },
    }
    return _BackendApi(preliminary, files)


def test_postflight_validates_complete_contract_and_is_idempotent(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)
    receipt = _validate(output)
    terminal = verify_content_addressed_json(receipt.path, prefix="terminal")

    assert terminal["status"] == "pass"
    assert terminal["git_commit"] == COMMIT
    assert (output / "SUCCESS").read_text(encoding="utf-8") == f"terminal_sha256={receipt.sha256}\n"
    postflight_paths = list((output / "postflight").glob("postflight-*.json"))
    assert len(postflight_paths) == 1
    postflight = verify_content_addressed_json(postflight_paths[0], prefix="postflight")
    assert postflight["preliminary_wandb_backend"]["status"] == "verified"
    assert len(list((output / "wandb-final").glob("wandb-final-*.json"))) == 1
    assert _validate(output).path == receipt.path


@pytest.mark.parametrize(
    ("scheduler", "match"),
    (
        (_scheduler(cpus=16), "eight allocated CPUs"),
        (_scheduler(memory_mib=16 * 1024), "32 GiB"),
        (_scheduler(array_job_id="123"), "array"),
        (_scheduler(gpus=2), "one node, one task, and one GPU"),
        (_scheduler(time_limit="00:30:01"), "30-minute"),
    ),
)
def test_postflight_rejects_allocation_drift(
    tmp_path: Path,
    one_cpu_thread: None,
    scheduler: SchedulerIdentity,
    match: str,
) -> None:
    output = _build_output(tmp_path)
    with pytest.raises(ValueError, match=match):
        _validate(output, scheduler_lookup=lambda _job_id: scheduler)


def test_extra_artifact_fails_before_any_terminal_side_effect(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)
    (output / "prediction.npz").write_bytes(b"unsafe")
    finalized = False

    def unexpected_finalizer(**kwargs: Any) -> dict[str, Any]:
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    with pytest.raises(ValueError, match="artifact allowlist mismatch"):
        _validate(output, wandb_finalizer=unexpected_finalizer)
    assert finalized is False
    assert not (output / "postflight").exists()
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()
    assert not (output / "SUCCESS").exists()


def test_fifo_fails_closed_without_being_opened(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)
    os.mkfifo(output / "untrusted.pipe")
    finalized = False

    def unexpected_finalizer(**kwargs: Any) -> dict[str, Any]:
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    with pytest.raises(ValueError, match="only regular files and directories"):
        _validate(output, wandb_finalizer=unexpected_finalizer)
    assert finalized is False
    assert not (output / "postflight").exists()


def test_self_consistent_credential_key_forgery_fails_closed(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)
    training = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    wandb = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    training["runtime_identity"]["access_token"] = "opaque-redacted-value"
    training["config"]["runtime_identity"] = training["runtime_identity"]
    training["config_sha256"] = sha256_value(training["config"])
    wandb["config_sha256"] = training["config_sha256"]
    training["wandb"] = wandb
    _write_json(output / "wandb_receipt.json", wandb)
    _write_json(output / "training_receipt.json", training)

    with pytest.raises(ValueError, match="credential-like field"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_self_consistent_arbitrary_wandb_group_forgery_fails_closed(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)
    training = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    wandb = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    training["config"]["requested_wandb_identity"]["group"] = "arbitrary-unbound-group"
    training["config_sha256"] = sha256_value(training["config"])
    wandb["group"] = "arbitrary-unbound-group"
    wandb["config_sha256"] = training["config_sha256"]
    training["wandb"] = wandb
    _write_json(output / "wandb_receipt.json", wandb)
    _write_json(output / "training_receipt.json", training)

    with pytest.raises(ValueError, match="configured W&B identity differs"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_semantic_step_tamper_fails_after_hashes_are_rebound(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)

    def tamper(rows: list[dict[str, Any]]) -> None:
        rows[0]["objectives"]["total"] += 1.0

    _rewrite_step_contract(output, tamper)
    with pytest.raises(ValueError, match="objective decomposition"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_incomplete_objective_log_set_fails_after_hashes_are_rebound(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)
    _rewrite_step_contract(output, lambda rows: rows[3]["objectives"].pop("shape_nll"))
    with pytest.raises(ValueError, match="metric set mismatch"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_partial_step_log_fails_completeness_gate(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)

    def remove_last_row(rows: list[dict[str, Any]]) -> None:
        rows.pop()

    _rewrite_step_contract(output, remove_last_row)
    with pytest.raises(ValueError, match="exactly 12 optimizer steps"):
        _validate(output)
    assert not (output / "postflight").exists()


@pytest.mark.parametrize("field", ("owned-gradient", "actual-clipping", "self-consistent-clipping"))
def test_gradient_or_clipping_tamper_fails_semantic_gate(
    tmp_path: Path,
    one_cpu_thread: None,
    field: str,
) -> None:
    output = _build_output(tmp_path)

    def tamper(rows: list[dict[str, Any]]) -> None:
        if field == "owned-gradient":
            rows[0]["gradients"]["norm_scale"] = 1.0
        elif field == "actual-clipping":
            rows[0]["optimizer"]["post_clip_gradient_norm"] *= 0.5
        else:
            optimizer = rows[0]["optimizer"]
            assert optimizer["pre_clip_gradient_norm"] < optimizer["gradient_clip_threshold"]
            optimizer["post_clip_gradient_norm"] = optimizer["pre_clip_gradient_norm"] / 2
            optimizer["applied_clip_coefficient"] = 0.5
            optimizer["was_clipped"] = 1

    _rewrite_step_contract(output, tamper)
    with pytest.raises(ValueError, match="owned gradient|clipping"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_diagonal_firewall_norm_must_match_owned_gradient(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)

    def tamper(rows: list[dict[str, Any]]) -> None:
        rows[3]["gradients"]["firewall_shape_to_shape"] *= 0.5

    _rewrite_step_contract(output, tamper)
    with pytest.raises(ValueError, match="owned firewall gradient mismatch"):
        _validate(output)
    assert not (output / "postflight").exists()


@pytest.mark.parametrize("field", ("inactive-objective", "inactive-field-diagnostic", "row-index"))
def test_inactive_metrics_and_row_indices_are_exact(
    tmp_path: Path,
    one_cpu_thread: None,
    field: str,
) -> None:
    output = _build_output(tmp_path)

    def tamper(rows: list[dict[str, Any]]) -> None:
        if field == "inactive-objective":
            rows[0]["objectives"]["scale_objective"] = 1.0
            rows[0]["objectives"]["total"] += 1.0
        elif field == "inactive-field-diagnostic":
            rows[3]["diagnostics"]["scale_field_max_abs"] = 1.0
        else:
            rows[0]["global_step"] = False

    _rewrite_step_contract(output, tamper)
    with pytest.raises(ValueError, match="inactive|order or index"):
        _validate(output)
    assert not (output / "postflight").exists()


@pytest.mark.parametrize("field", ("seed", "learning-rate"))
def test_governed_numeric_configuration_is_frozen(
    tmp_path: Path,
    one_cpu_thread: None,
    field: str,
) -> None:
    output = _build_output(tmp_path)
    if field == "learning-rate":

        def change_learning_rate(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                row["optimizer"]["learning_rate"] = 2e-3

        _rewrite_step_contract(output, change_learning_rate)
        _rewrite_config_contract(output, lambda config: config["optimizer"].__setitem__("learning_rate", 2e-3))
    else:
        _rewrite_config_contract(output, lambda config: config.__setitem__("seed", 18))
    with pytest.raises(ValueError, match="frozen governed numeric smoke contract"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_checkpoint_tamper_fails_before_loading_or_finalizing(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)
    with (output / "checkpoints" / "M2.pt").open("ab") as stream:
        stream.write(b"tamper")
    finalized = False

    def unexpected_finalizer(**kwargs: Any) -> dict[str, Any]:
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        _validate(output, wandb_finalizer=unexpected_finalizer)
    assert finalized is False


def test_rehashed_checkpoint_dtype_tamper_fails_exact_reload(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)

    def change_dtype(payload: dict[str, Any]) -> None:
        name = next(name for name, tensor in payload["state_dict"].items() if tensor.is_floating_point())
        payload["state_dict"][name] = payload["state_dict"][name].double()

    _rewrite_checkpoint_contract(output, "M2", change_dtype)
    with pytest.raises(ValueError, match="strictly reconstruct|tensor contract"):
        _validate(output)
    assert not (output / "postflight").exists()


def test_backend_round_trip_rejects_self_consistent_checkpoint_rewrite(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)
    backend = _backend_api_snapshot(output)
    preliminary = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    paths = {
        "steps.jsonl": output / "steps.jsonl",
        **{f"checkpoints/{arm}.pt": output / "checkpoints" / f"{arm}.pt" for arm in ("M0", "M1", "M2", "M3")},
    }
    evidence = verify_phase2g_preliminary_backend(
        preliminary_receipt=preliminary,
        paths=paths,
        wandb_api=backend,
    )
    assert evidence["status"] == "verified"
    assert backend.artifact_value.downloaded is True
    assert backend.artifact_value.verified is True
    original_backend_files = dict(backend.artifact_value.files)

    def change_value(payload: dict[str, Any]) -> None:
        name = next(name for name, tensor in payload["state_dict"].items() if tensor.is_floating_point())
        payload["state_dict"][name].view(-1)[0] += 123.0

    _rewrite_checkpoint_contract(output, "M2", change_value)
    forged_preliminary = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    forged_backend = _BackendApi(forged_preliminary, original_backend_files)
    finalized = False

    def unexpected_finalizer(**kwargs: Any) -> dict[str, Any]:
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    with pytest.raises(RuntimeError, match="manifest size differs|backend file differs"):
        _validate(
            output,
            wandb_preliminary_verifier=lambda **kwargs: verify_phase2g_preliminary_backend(
                **kwargs,
                wandb_api=forged_backend,
            ),
            wandb_finalizer=unexpected_finalizer,
        )
    assert finalized is False
    assert not (output / "postflight").exists()
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()
    assert not (output / "SUCCESS").exists()


def test_backend_verifier_retries_only_not_yet_finished_visibility(
    tmp_path: Path,
    one_cpu_thread: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _build_output(tmp_path)
    preliminary = json.loads((output / "wandb_receipt.json").read_text(encoding="utf-8"))
    paths = {
        "steps.jsonl": output / "steps.jsonl",
        **{f"checkpoints/{arm}.pt": output / "checkpoints" / f"{arm}.pt" for arm in ("M0", "M1", "M2", "M3")},
    }
    first = _backend_api_snapshot(output)
    first.run_value.state = "running"
    second = _backend_api_snapshot(output)
    apis = iter((first, second))
    api_calls: list[int] = []
    sleeps: list[int] = []
    import wandb

    def api_factory(*, timeout: int) -> _BackendApi:
        assert timeout == 30
        api_calls.append(timeout)
        return next(apis)

    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(wandb, "Api", api_factory)
    monkeypatch.setattr(postflight_module.time, "sleep", sleeps.append)
    evidence = verify_phase2g_preliminary_backend(
        preliminary_receipt=preliminary,
        paths=paths,
    )
    assert evidence["status"] == "verified"
    assert api_calls == [30, 30]
    assert sleeps == [1]


def test_finalizer_failure_leaves_retryable_postflight_without_success(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)

    def failed_finalizer(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("injected terminal upload failure")

    with pytest.raises(RuntimeError, match="injected terminal upload failure"):
        _validate(output, wandb_finalizer=failed_finalizer)
    assert len(list((output / "postflight").glob("postflight-*.json"))) == 1
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()
    assert not (output / "SUCCESS").exists()

    receipt = _validate(output)
    assert receipt.path.is_file()
    assert (output / "SUCCESS").is_file()


def test_final_wandb_url_must_match_preliminary_run(tmp_path: Path, one_cpu_thread: None) -> None:
    output = _build_output(tmp_path)

    def mismatched_finalizer(**kwargs: Any) -> dict[str, Any]:
        receipt = _fake_finalizer(**kwargs)
        receipt["run_url"] = "https://wandb.invalid/test-entity/test-project/runs/different-run"
        return receipt

    with pytest.raises(ValueError, match="changed the preliminary run_url"):
        _validate(output, wandb_finalizer=mismatched_finalizer)
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()
    assert not (output / "SUCCESS").exists()


def test_recovery_from_final_wandb_without_terminal_never_refinalizes(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)
    original = _validate(output)
    original_payload = verify_content_addressed_json(original.path, prefix="terminal")
    (output / "SUCCESS").unlink()
    original.path.unlink()
    original.path.parent.rmdir()
    finalized = False

    def unexpected_finalizer(**kwargs: Any) -> dict[str, Any]:
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    recovered = _validate(output, wandb_finalizer=unexpected_finalizer)
    assert finalized is False
    assert recovered.sha256 == original.sha256
    assert verify_content_addressed_json(recovered.path, prefix="terminal") == original_payload
    assert (output / "SUCCESS").read_text() == f"terminal_sha256={original.sha256}\n"


class _TerminalArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[tuple[Path, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((Path(path), name))


class _TerminalRun:
    offline = False
    url = "https://wandb.invalid/test-entity/test-project/runs/run-id"

    def __init__(self, preliminary: Mapping[str, Any], *, fail_wait: bool = False) -> None:
        self.entity = preliminary["entity"]
        self.project = preliminary["project"]
        self.group = preliminary["group"]
        self.name = preliminary["run_name"]
        self.job_type = preliminary["job_type"]
        self.id = preliminary["run_id"]
        self.summary: dict[str, Any] = {}
        self.finish_codes: list[int] = []
        self.artifact: _TerminalArtifact | None = None
        self.fail_wait = fail_wait

    def log_artifact(self, artifact: _TerminalArtifact) -> SimpleNamespace:
        self.artifact = artifact

        def wait() -> None:
            if self.fail_wait:
                raise RuntimeError("injected artifact wait failure")

        return SimpleNamespace(
            name=f"{artifact.name}:v0",
            id="terminal-id",
            version="v0",
            digest="terminal-digest",
            wait=wait,
        )

    def finish(self, *, exit_code: int) -> None:
        self.finish_codes.append(exit_code)


class _TerminalWandb:
    def __init__(self, preliminary: Mapping[str, Any], *, fail_wait: bool = False) -> None:
        self.run = _TerminalRun(preliminary, fail_wait=fail_wait)
        self.init_kwargs: dict[str, Any] | None = None

    def init(self, **kwargs: Any) -> _TerminalRun:
        self.init_kwargs = kwargs
        return self.run

    def Artifact(self, name: str, *, type: str) -> _TerminalArtifact:  # noqa: N802
        return _TerminalArtifact(name, type)


def test_real_finalizer_resumes_exact_run_and_uploads_complete_terminal_evidence(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)
    preliminary = json.loads((output / "wandb_receipt.json").read_text())
    postflight = write_content_addressed_json(
        {"schema_version": "fixture-postflight", "status": "pass"},
        output / "postflight",
        prefix="postflight",
    )
    summary = {"validation/postflight/status": "pass"}
    fake = _TerminalWandb(preliminary)
    receipt = finalize_phase2g_online_run(
        preliminary_receipt=preliminary,
        artifact_root=output,
        training_path=output / "training_receipt.json",
        postflight_path=postflight.path,
        summary=summary,
        wandb_module=fake,
    )

    assert fake.init_kwargs is not None
    assert fake.init_kwargs["id"] == preliminary["run_id"]
    assert fake.init_kwargs["resume"] == "must"
    assert fake.init_kwargs["mode"] == "online"
    assert fake.run.summary == summary
    assert fake.run.finish_codes == [0]
    assert fake.run.artifact is not None
    expected_names = [
        "training_receipt.json",
        "wandb_receipt.json",
        postflight.path.name,
        "steps.jsonl",
        *(f"checkpoints/{arm}.pt" for arm in ("M0", "M1", "M2", "M3")),
    ]
    assert [name for _, name in fake.run.artifact.files] == expected_names
    assert [value["name"] for value in receipt["files"]] == expected_names
    assert receipt["status"] == "finalized"
    assert receipt["preliminary_receipt_sha256"] == sha256_value(preliminary)


def test_real_finalizer_never_publishes_pass_summary_before_artifact_wait(
    tmp_path: Path,
    one_cpu_thread: None,
) -> None:
    output = _build_output(tmp_path)
    preliminary = json.loads((output / "wandb_receipt.json").read_text())
    postflight = write_content_addressed_json(
        {"schema_version": "fixture-postflight", "status": "pass"},
        output / "postflight",
        prefix="postflight",
    )
    fake = _TerminalWandb(preliminary, fail_wait=True)
    with pytest.raises(RuntimeError, match="artifact wait failure"):
        finalize_phase2g_online_run(
            preliminary_receipt=preliminary,
            artifact_root=output,
            training_path=output / "training_receipt.json",
            postflight_path=postflight.path,
            summary={"validation/postflight/status": "pass"},
            wandb_module=fake,
        )
    assert fake.run.summary == {}
    assert fake.run.finish_codes == [1]
