from __future__ import annotations

from pathlib import Path

import pytest

import slurm.validate_geometry_official_mini as postflight_module
from jepa4d.tests.test_geometry_official_mini import _fake_publisher, _result, _settings
from jepa4d.validation._content import (
    sha256_file,
    sha256_value,
    verify_content_addressed_json,
    write_content_addressed_json,
)
from jepa4d.validation.geometry_official_mini import run_governed_geometry_official_mini
from jepa4d.validation.wandb import SAFE_WANDB_FINAL_RECEIPT_SCHEMA
from slurm.validate_geometry_official_mini import (
    APPROVED_ACCOUNT,
    SchedulerIdentity,
    _validate_report_metrics,
    validate_geometry_official_mini,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_ID = "123456"
COMMIT = "a" * 40


def _scheduler(
    job_id: str,
    *,
    account: str = APPROVED_ACCOUNT,
    partition: str = "polar4",
    time_limit: str = "02:00:00",
    nodes: int = 1,
    tasks: int = 1,
    gpus: int = 1,
) -> SchedulerIdentity:
    return SchedulerIdentity(
        job_id=job_id,
        job_name="j4d-gmini-abc12345-120000",
        account=account,
        partition=partition,
        time_limit=time_limit,
        nodes=nodes,
        tasks=tasks,
        gpus=gpus,
    )


def _build_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    settings = _settings(tmp_path)
    run_governed_geometry_official_mini(
        settings,
        evaluator=lambda _settings, _authorized: _result(),
        publisher=_fake_publisher({}),
    )
    return settings.output


def _fake_finalizer(**kwargs):
    preliminary = kwargs["preliminary_receipt"]
    files = tuple(kwargs["files"])
    return {
        "schema_version": SAFE_WANDB_FINAL_RECEIPT_SCHEMA,
        "status": "finalized",
        "terminal_status": "postflight-pass",
        "preliminary_receipt_sha256": sha256_value(preliminary),
        "entity": preliminary["entity"],
        "project": preliminary["project"],
        "group": preliminary["group"],
        "job_type": preliminary["job_type"],
        "run_name": preliminary["run_name"],
        "run_id": preliminary["run_id"],
        "run_url": preliminary["run_url"],
        "artifact_name": "terminal-mock-online-run",
        "artifact_id": "test-entity/test-project/terminal:v0",
        "artifact_version": "v0",
        "artifact_digest": "terminal-digest",
        "summary_sha256": sha256_value({**kwargs["summary"], "validation/postflight/status": "pass"}),
        "files": [
            {
                "name": item.path.name,
                "role": item.role,
                "bytes": item.path.stat().st_size,
                "sha256": sha256_file(item.path),
            }
            for item in files
        ],
    }


def _validate(
    output: Path,
    scheduler_lookup=lambda job_id: _scheduler(job_id),
    *,
    wandb_finalizer=_fake_finalizer,
):
    return validate_geometry_official_mini(
        output=output,
        repo_root=REPO_ROOT,
        job_id=JOB_ID,
        scheduler_lookup=scheduler_lookup,
        git_lookup=lambda _root: (COMMIT, True),
        wandb_finalizer=wandb_finalizer,
    )


def test_postflight_binds_scheduler_governance_dashboard_and_online_receipt(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    receipt = _validate(output)
    payload = verify_content_addressed_json(receipt.path, prefix="terminal")

    assert payload["status"] == "pass"
    postflight = verify_content_addressed_json(
        next((output / "postflight").glob("postflight-*.json")), prefix="postflight"
    )
    assert postflight["scheduler"] == {
        "job_id": JOB_ID,
        "job_name": "j4d-gmini-abc12345-120000",
        "account": APPROVED_ACCOUNT,
        "partition": "polar4",
        "time_limit": "02:00:00",
        "nodes": 1,
        "tasks": 1,
        "gpus": 1,
    }
    assert len(payload["postflight_receipt_sha256"]) == 64
    assert len(payload["final_wandb_receipt_sha256"]) == 64
    assert _validate(output).path == receipt.path


@pytest.mark.parametrize(
    ("scheduler_lookup", "match"),
    [
        (lambda job_id: _scheduler(job_id, account="another-account"), "unapproved account"),
        (lambda job_id: _scheduler(job_id, partition="unapproved"), "unapproved partition"),
        (lambda job_id: _scheduler(job_id, time_limit="04:00:01"), "four-hour"),
        (lambda job_id: _scheduler(job_id, gpus=2), "exactly one node"),
    ],
)
def test_postflight_rejects_scheduler_policy_drift(tmp_path, monkeypatch, scheduler_lookup, match) -> None:
    output = _build_output(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=match):
        _validate(output, scheduler_lookup)


def test_postflight_rejects_unbound_prediction_or_media_artifact(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    (output / "prediction.npz").write_bytes(b"unsafe")
    finalized = False

    def unexpected_finalizer(**kwargs):
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    with pytest.raises(ValueError, match="artifact allowlist mismatch"):
        _validate(output, wandb_finalizer=unexpected_finalizer)
    assert finalized is False
    assert not (output / "postflight").exists()
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda values: values.__setitem__("aligned_delta_1", 1.1), "within \\[0, 1\\]"),
        (
            lambda values: values.update({"aligned_delta_1": 0.95, "aligned_delta_2": 0.90}),
            "must be monotonic",
        ),
    ],
)
def test_postflight_revalidates_forged_quality_semantics(tmp_path, monkeypatch, mutation, match) -> None:
    output = _build_output(tmp_path, monkeypatch)
    original = postflight_module._single_content_address

    def forged_single_content_address(directory: Path, prefix: str):
        path, payload, digest = original(directory, prefix)
        if prefix == "metric-gate":
            payload = {**payload, "quality_metrics": dict(payload["quality_metrics"])}
            mutation(payload["quality_metrics"])
        return path, payload, digest

    monkeypatch.setattr(postflight_module, "_single_content_address", forged_single_content_address)
    with pytest.raises(ValueError, match=match):
        _validate(output)
    assert not (output / "postflight").exists()
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()


def test_postflight_recovers_valid_final_wandb_without_terminal_receipt(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    original = _validate(output)
    original_payload = verify_content_addressed_json(original.path, prefix="terminal")
    original.path.unlink()
    original.path.parent.rmdir()

    finalized = False

    def unexpected_finalizer(**kwargs):
        nonlocal finalized
        finalized = True
        return _fake_finalizer(**kwargs)

    recovered = _validate(output, wandb_finalizer=unexpected_finalizer)
    assert finalized is False
    assert recovered.sha256 == original.sha256
    assert verify_content_addressed_json(recovered.path, prefix="terminal") == original_payload


def test_postflight_rejects_self_consistent_preliminary_hash_forgery(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    wandb_path = next((output / "wandb").glob("wandb-receipt-*.json"))
    wandb = verify_content_addressed_json(wandb_path, prefix="wandb-receipt")
    wandb["config_sha256"] = "f" * 64
    wandb_path.unlink()
    forged_wandb = write_content_addressed_json(wandb, output / "wandb", prefix="wandb-receipt")

    execution_path = next((output / "execution").glob("execution-receipt-*.json"))
    execution = verify_content_addressed_json(execution_path, prefix="execution-receipt")
    execution["wandb_receipt_sha256"] = forged_wandb.sha256
    execution["wandb_config_sha256"] = "f" * 64
    execution_path.unlink()
    write_content_addressed_json(execution, output / "execution", prefix="execution-receipt")

    with pytest.raises(ValueError, match="execution receipt does not bind"):
        _validate(output)
    assert not (output / "postflight").exists()
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()


def test_postflight_rejects_terminal_summary_forgery_before_terminal_receipts(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)

    def forged_finalizer(**kwargs):
        receipt = _fake_finalizer(**kwargs)
        receipt["summary_sha256"] = "f" * 64
        return receipt

    with pytest.raises(ValueError, match="finalized online W&B receipt does not bind"):
        _validate(output, wandb_finalizer=forged_finalizer)
    assert (output / "postflight").is_dir()
    assert not (output / "wandb-final").exists()
    assert not (output / "terminal").exists()


def test_postflight_detects_dashboard_tampering_without_repair(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    dashboard = next((output / "dashboard").glob("validation-dashboard-*"))
    html = dashboard / "validation_dashboard.html"
    html.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match receipt"):
        _validate(output)
    assert html.read_text(encoding="utf-8") == "tampered"


def test_postflight_rejects_local_data_path_in_an_extra_receipt(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    extra = output / "extra.json"
    extra.write_text('{"note":"/lustre/restricted/raw/frame.png"}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact allowlist mismatch"):
        _validate(output)


def test_postflight_rejects_unbound_empty_directory(tmp_path, monkeypatch) -> None:
    output = _build_output(tmp_path, monkeypatch)
    (output / "unbound").mkdir()
    with pytest.raises(ValueError, match="directory allowlist mismatch"):
        _validate(output)


def test_postflight_recomputes_dashboard_metric_completeness_and_units() -> None:
    result = _result()
    rows = [
        {"domain": domain, "name": name, "value": value, "unit": "wrong", "split": "wrong"}
        for domain, metrics in (("quality", result.quality_metrics), ("resource", result.resource_metrics))
        for name, value in metrics.items()
    ]
    with pytest.raises(ValueError, match="differs from the metric gate"):
        _validate_report_metrics(rows, result.quality_metrics, result.resource_metrics)

    with pytest.raises(ValueError, match="incomplete"):
        _validate_report_metrics(rows[:-1], result.quality_metrics, result.resource_metrics)
