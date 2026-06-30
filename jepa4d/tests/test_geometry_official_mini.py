from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from jepa4d.tests.test_validation_wandb import _Artifact, _Run
from jepa4d.validation._content import sha256_file, sha256_value
from jepa4d.validation.access import DatasetAccessController
from jepa4d.validation.geometry_official_mini import (
    AGGREGATION_PROTOCOL,
    DATASET_ID,
    EXPECTED_QUALITY_METRICS,
    EXPECTED_RESOURCE_METRICS,
    EXPECTED_TEST_FRAMES,
    METRIC_UNITS,
    OPERATION,
    SPLIT_ID,
    GeometryAggregateResult,
    GeometryOfficialMiniSettings,
    _contained_regular_file,
    _verified_tum_extraction,
    main,
    run_governed_geometry_official_mini,
)
from jepa4d.validation.registry import AccessDenied, DatasetRegistry
from jepa4d.validation.wandb import SAFE_WANDB_RECEIPT_SCHEMA, SafeArtifactFile

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "configs/validation/dataset_registry.yaml"
LEDGER = REPO_ROOT / "configs/validation/consumed_test_ledger.yaml"


def _settings(tmp_path: Path) -> GeometryOfficialMiniSettings:
    return GeometryOfficialMiniSettings(
        repo_root=REPO_ROOT,
        registry_path=REGISTRY,
        ledger_path=LEDGER,
        archive=tmp_path / "must-not-be-read-archive.tgz",
        model_id=tmp_path / "must-not-be-read-model",
        output=tmp_path / "output",
        execution_id="abc12345-20260630T120000Z-mini",
        run_name="gmini-abc12345-120000",
        git_commit="a" * 40,
        wandb_entity="test-entity",
        wandb_project="test-project",
    )


def _result() -> GeometryAggregateResult:
    quality = {
        "aligned_abs_rel": 0.2,
        "aligned_log_rmse": 0.3,
        "aligned_rmse_m": 0.4,
        "aligned_delta_1": 0.8,
        "aligned_delta_2": 0.9,
        "aligned_delta_3": 0.95,
        "finite_fraction": 1.0,
        "point_error_mean_m_aligned": 0.1,
        "point_error_median_m_aligned": 0.08,
        "point_within_10cm_fraction_aligned": 0.7,
        "point_within_5cm_fraction_aligned": 0.5,
        "pose_alignment_scale": 1.2,
        "pose_ate_mean_m_sim3": 0.02,
        "pose_ate_rmse_m_sim3": 0.03,
        "pose_relative_translation_mean_m_sim3": 0.01,
        "pose_rotation_mean_deg_sim3": 2.0,
    }
    return GeometryAggregateResult(
        quality_metrics=quality,
        resource_metrics={"runtime_seconds": 2.0, "cuda_peak_memory_gib": 3.0},
        evaluated_test_frames=EXPECTED_TEST_FRAMES,
        finite_fraction=1.0,
        archive_sha256="a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
        id_manifest_sha256="84feb816ae3cabbd65a7c4d3cd2e00ad8ffbe1108fd762ce4cead59a584f5fb5",
        model_identity_sha256="c" * 64,
    )


def _fake_publisher(captured: dict[str, Any]):
    def publish(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        files = tuple(kwargs["files"])
        assert all(isinstance(item, SafeArtifactFile) for item in files)
        identities = [
            {
                "name": item.path.name,
                "role": item.role,
                "bytes": item.path.stat().st_size,
                "sha256": sha256_file(item.path),
            }
            for item in files
        ]
        return {
            "schema_version": SAFE_WANDB_RECEIPT_SCHEMA,
            "status": "uploaded-preliminary",
            "terminal_status": "pending-postflight",
            "mode": "online",
            "entity": "test-entity",
            "project": "test-project",
            "group": kwargs["group"],
            "job_type": kwargs["job_type"],
            "run_name": kwargs["run_name"],
            "run_id": "mock-online-run",
            "run_url": "https://wandb.ai/test-entity/test-project/runs/mock-online-run",
            "artifact_name": kwargs["artifact_name"],
            "artifact_id": "test-entity/test-project/governed:v0",
            "artifact_version": "v0",
            "artifact_digest": "mock-artifact-digest",
            "config_sha256": sha256_value(kwargs["config"]),
            "summary_sha256": sha256_value({**kwargs["summary"], "validation/postflight/status": "pending"}),
            "files": identities,
        }

    return publish


def test_runner_authorizes_exact_consumed_regression_before_any_data_callback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    calls: list[tuple[str, str, object]] = []
    real_authorize = DatasetAccessController.authorize

    def authorize(self, dataset_id, split_id, operation, **kwargs):
        calls.append((dataset_id, split_id, operation))
        return real_authorize(self, dataset_id, split_id, operation, **kwargs)

    monkeypatch.setattr(DatasetAccessController, "authorize", authorize)
    captured: dict[str, Any] = {}

    def evaluate(settings, authorized):
        assert calls == [(DATASET_ID, SPLIT_ID, OPERATION)]
        assert not settings.archive.exists()
        assert authorized.decision.operation is OPERATION
        return _result()

    result = run_governed_geometry_official_mini(
        _settings(tmp_path), evaluator=evaluate, publisher=_fake_publisher(captured)
    )

    assert result["status"] == "pass"
    assert captured["config"]["dataset_id"] == DATASET_ID
    assert captured["config"]["split_id"] == SPLIT_ID
    assert captured["config"]["operation"] == "regression"
    roles = {item.role for item in captured["files"]}
    assert roles == {
        "aggregate-receipt",
        "dashboard-html",
        "dashboard-json",
        "dashboard-receipt",
        "governance-receipt",
    }
    assert all(item.path.suffix in {".json", ".html"} for item in captured["files"])

    dashboard_json = next((_settings(tmp_path).output / "dashboard").rglob("validation_report.json"))
    report = json.loads(dashboard_json.read_text(encoding="utf-8"))
    observed = {(item["domain"], item["name"]): item["unit"] for item in report["metrics"]}
    assert set(name for domain, name in observed if domain == "quality") == EXPECTED_QUALITY_METRICS
    assert set(name for domain, name in observed if domain == "resource") == EXPECTED_RESOURCE_METRICS
    assert all(observed[(domain, name)] == METRIC_UNITS[name] for domain, name in observed)


def test_authorization_denial_stops_before_data_callback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    evaluated = False

    def deny(*args, **kwargs):
        raise AccessDenied("denied before data")

    def evaluate(settings, authorized):
        nonlocal evaluated
        evaluated = True
        raise AssertionError("data callback must not run")

    monkeypatch.setattr(DatasetAccessController, "authorize", deny)
    with pytest.raises(AccessDenied, match="before data"):
        run_governed_geometry_official_mini(_settings(tmp_path), evaluator=evaluate, publisher=_fake_publisher({}))
    assert evaluated is False
    assert not _settings(tmp_path).archive.exists()


def test_registered_phase2b_manifest_identity_matches_repository_bytes() -> None:
    registry = DatasetRegistry.load(REGISTRY)
    _, split = registry.split(DATASET_ID, SPLIT_ID)
    assert split.id_manifest == "jepa4d/config/benchmarks/manifests/tum_rgbd_phase2b_v1.yaml"
    assert split.id_manifest_sha256 == sha256_file(REPO_ROOT / split.id_manifest)
    assert OPERATION in split.allowed_operations


def test_aggregation_protocol_declares_mixed_frame_and_sequence_reducers() -> None:
    assert AGGREGATION_PROTOCOL == (
        "depth/point metrics preserve each metric's declared per-frame pixel reducer, then use an equal mean across "
        "exactly eight frames; pose metrics use one sequence-level Sim(3) alignment across those exactly eight frames"
    )


def test_production_entry_requires_slurm_and_canonical_authorities(tmp_path, monkeypatch) -> None:
    arguments = [
        "geometry-official-mini",
        "--repo-root",
        str(REPO_ROOT),
        "--registry",
        str(REGISTRY),
        "--ledger",
        str(LEDGER),
        "--archive",
        str(tmp_path / "archive.tgz"),
        "--model-id",
        str(tmp_path / "model"),
        "--output",
        str(tmp_path / "output"),
        "--execution-id",
        "execution",
        "--run-name",
        "run",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="inside a Slurm allocation"):
        main()

    wrong_registry = tmp_path / "registry.yaml"
    wrong_registry.write_text("schema: wrong\n", encoding="utf-8")
    arguments[arguments.index(str(REGISTRY))] = str(wrong_registry)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(RuntimeError, match="canonical registry and ledger"):
        main()


def test_production_entry_rejects_commit_change_after_submission(tmp_path, monkeypatch) -> None:
    arguments = [
        "geometry-official-mini",
        "--repo-root",
        str(REPO_ROOT),
        "--registry",
        str(REGISTRY),
        "--ledger",
        str(LEDGER),
        "--archive",
        str(tmp_path / "archive.tgz"),
        "--model-id",
        str(tmp_path / "model"),
        "--output",
        str(tmp_path / "output"),
        "--execution-id",
        "execution",
        "--run-name",
        "run",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("JEPA4D_GIT_COMMIT", "a" * 40)
    monkeypatch.setattr("jepa4d.validation.geometry_official_mini.subprocess.run", lambda *args, **kwargs: None)
    monkeypatch.setattr("jepa4d.validation.geometry_official_mini._clean_git_commit", lambda _root: "b" * 40)

    with pytest.raises(RuntimeError, match="commit changed after governed Slurm submission"):
        main()


def _write_tar_member(bundle: tarfile.TarFile, name: str, payload: bytes = b"fixture\n") -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    bundle.addfile(member, io.BytesIO(payload))


def test_ephemeral_extraction_rejects_traversal_and_never_uses_an_external_tree(tmp_path) -> None:
    archive = tmp_path / "traversal.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            _write_tar_member(bundle, f"rgbd_dataset_freiburg1_xyz/{name}")
        _write_tar_member(bundle, "rgbd_dataset_freiburg1_xyz/../escape.txt")
    manifest = {"sequence": "rgbd_dataset_freiburg1_xyz"}
    with (
        pytest.raises(ValueError, match="unsafe member"),
        archive.open("rb") as stream,
        _verified_tum_extraction(stream, manifest),
    ):
        raise AssertionError("unsafe archive must not be yielded")
    assert not (tmp_path / "escape.txt").exists()


def test_ephemeral_extraction_and_containment_accept_only_regular_members(tmp_path) -> None:
    archive = tmp_path / "safe.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "rgb/frame.png"):
            _write_tar_member(bundle, f"rgbd_dataset_freiburg1_xyz/{name}")
    with (
        archive.open("rb") as stream,
        _verified_tum_extraction(stream, {"sequence": "rgbd_dataset_freiburg1_xyz"}) as root,
    ):
        frame = _contained_regular_file(root / "rgb/frame.png", root)
        assert frame.read_bytes() == b"fixture\n"
        with pytest.raises(ValueError, match="outside"):
            _contained_regular_file(tmp_path / "safe.tgz", root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing-quality", "exact official-mini metric schema"),
        ("extra-resource", "exact official-mini resource schema"),
        ("negative-resource", "cannot be negative"),
        ("negative-quality", "cannot be negative"),
        ("invalid-fraction", "within \\[0, 1\\]"),
        ("nonmonotonic-delta", "must be monotonic"),
        ("nonmonotonic-point", "cannot exceed"),
        ("impossible-pose", "cannot exceed"),
        ("finite-mismatch", "must match"),
    ],
)
def test_aggregate_result_rejects_incomplete_or_inconsistent_schema(mutation: str, match: str) -> None:
    baseline = _result()
    quality = dict(baseline.quality_metrics)
    resources = dict(baseline.resource_metrics)
    finite_fraction = baseline.finite_fraction
    if mutation == "missing-quality":
        quality.pop("aligned_abs_rel")
    elif mutation == "extra-resource":
        resources["throughput"] = 1.0
    elif mutation == "negative-resource":
        resources["runtime_seconds"] = -1.0
    elif mutation == "negative-quality":
        quality["aligned_abs_rel"] = -1.0
    elif mutation == "invalid-fraction":
        quality["aligned_delta_1"] = 1.1
    elif mutation == "nonmonotonic-delta":
        quality["aligned_delta_1"] = 0.95
        quality["aligned_delta_2"] = 0.9
    elif mutation == "nonmonotonic-point":
        quality["point_within_5cm_fraction_aligned"] = 0.8
    elif mutation == "impossible-pose":
        quality["pose_ate_mean_m_sim3"] = 0.04
    else:
        quality["finite_fraction"] = 0.5
    with pytest.raises(ValueError, match=match):
        GeometryAggregateResult(
            quality_metrics=quality,
            resource_metrics=resources,
            evaluated_test_frames=baseline.evaluated_test_frames,
            finite_fraction=finite_fraction,
            archive_sha256=baseline.archive_sha256,
            id_manifest_sha256=baseline.id_manifest_sha256,
            model_identity_sha256=baseline.model_identity_sha256,
        )


def test_target_validity_cannot_drop_nonpositive_or_nonfinite_predictions() -> None:
    from jepa4d.validation.geometry_official_mini import _require_complete_target_support

    target = torch.ones((10, 10))
    prediction = torch.ones_like(target)
    _require_complete_target_support(prediction, target)
    for invalid in (0.0, -1.0, float("nan")):
        candidate = prediction.clone()
        candidate[0, 0] = invalid
        with pytest.raises(ValueError, match="every valid TUM target pixel"):
            _require_complete_target_support(candidate, target)


def test_generated_artifacts_never_persist_supplied_data_paths_or_sample_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    settings = _settings(tmp_path)
    run_governed_geometry_official_mini(
        settings,
        evaluator=lambda _settings, _authorized: _result(),
        publisher=_fake_publisher({}),
    )

    forbidden_values = (str(settings.archive), str(settings.model_id))
    for path in settings.output.rglob("*.json"):
        document = path.read_text(encoding="utf-8")
        assert all(value not in document for value in forbidden_values)
        payload = json.loads(document)
        flattened = json.dumps(payload, sort_keys=True).casefold()
        assert '"sample_id"' not in flattened
        assert '"sample_ids"' not in flattened
        assert '"predictions"' not in flattened
    assert not list(settings.output.rglob("*.npz"))
    assert not list(settings.output.rglob("*.ply"))
    assert not list(settings.output.rglob("*.png"))


def test_output_is_no_clobber_even_for_same_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    settings = _settings(tmp_path)
    run_governed_geometry_official_mini(
        settings,
        evaluator=lambda _settings, _authorized: _result(),
        publisher=_fake_publisher({}),
    )
    before = hashlib.sha256(next(settings.output.rglob("execution-receipt-*.json")).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="new or empty"):
        run_governed_geometry_official_mini(
            settings,
            evaluator=lambda _settings, _authorized: _result(),
            publisher=_fake_publisher({}),
        )
    after = hashlib.sha256(next(settings.output.rglob("execution-receipt-*.json")).read_bytes()).hexdigest()
    assert after == before


def test_real_safe_publisher_accepts_only_the_runner_governed_bundle_with_mocked_wandb(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEPA4D_VALIDATION_STATE_ROOT", str(tmp_path / "validation-state"))
    monkeypatch.setenv("WANDB_MODE", "online")
    run = _Run()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run, Artifact=_Artifact))

    result = run_governed_geometry_official_mini(
        _settings(tmp_path),
        evaluator=lambda _settings, _authorized: _result(),
    )

    assert result["status"] == "pass"
    assert run.logged.waited is True
    assert run.exit_codes == [0]
    assert run.artifact is not None
    assert {name for _, name in run.artifact.files} == {
        next((_settings(tmp_path).output / "governance").glob("access-decision-*.json")).name,
        next((_settings(tmp_path).output / "metrics").glob("metric-gate-*.json")).name,
        "validation_report.json",
        "validation_dashboard.html",
        "validation_dashboard.receipt.json",
    }
