from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from jepa4d.benchmarks.geometry.sun_rgbd import SUNRGBDFrame
from jepa4d.evaluation.phase2g_data import (
    ROTATIONS,
    SUN_FAMILIES,
    atomic_json,
    build_feature_shard,
    build_input_shard,
    build_sun_membership_manifest,
    build_target_shard,
    canonical_sha256,
    create_rotation_views,
    file_identity,
    materialize_sun_archive,
    validate_feature_shard,
    validate_input_shard,
    validate_rotation_view,
    validate_sun_materialization,
    validate_sun_membership_manifest,
    validate_target_shard,
    write_torch_atomic,
)
from jepa4d.evaluation.phase2g_metrics import (
    evaluate_phase2g_predictions,
    opaque_frame_id,
    paired_hierarchical_bootstrap,
)
from jepa4d.evaluation.phase2g_visualization import (
    write_aggregate_visualizations,
    write_local_qualitative_panels,
)
from jepa4d.models.phase2f_scale_geometry import Phase2fGeometryConfig, Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_training import (
    assert_strict_phase2f_reload,
    load_phase2f_checkpoint,
    save_phase2f_checkpoint,
)
from jepa4d.training.phase2g_protocol import (
    ARMS,
    EVALUATION_RECEIPT_SCHEMA,
    FAMILIES,
    FORMAL_SEEDS,
    LEARNING_RATES,
    SAMPLES_PER_FAMILY,
    TUNING_RECEIPT_SCHEMA,
    TUNING_SEED,
)
from jepa4d.training.phase2g_protocol import (
    ROTATIONS as PROTOCOL_ROTATIONS,
)
from jepa4d.training.phase2g_runtime import (
    hardened_wandb_settings,
    load_execution_provenance,
    snapshot_and_log_gpu_telemetry,
)
from scripts.select_phase2g_learning_rates import select_learning_rates
from scripts.select_phase2g_survivor import select_survivor


def _provenance() -> dict[str, object]:
    return {
        "schema_version": "jepa4d-phase2g-execution-provenance-v1",
        "execution_id": "phase2g-test",
        "git_commit": "a" * 40,
        "preregistration_sha256": "b" * 64,
        "preflight_sha256": "e" * 64,
        "test_receipt_sha256": "c" * 64,
        "dependency_graph_sha256": "d" * 64,
        "slurm": {"job_id": "123"},
        "data_access_decision": {
            "data_access_authorized": True,
            "sun_dataset_id": "sun-rgbd.geometry-development",
            "external_final_authorized": False,
            "preflight": {"sha256": "e" * 64},
            "registry": {"sha256": "f" * 64},
            "readiness": {"sha256": "1" * 64},
        },
        "git_clean": True,
        "git_pushed": True,
        "external_final_authorized": False,
    }


def test_runtime_provenance_requires_preflight_registry_and_readiness_authorization(tmp_path: Path) -> None:
    path = atomic_json(tmp_path / "provenance.json", _provenance())
    assert load_execution_provenance(path)["data_access_decision"]["data_access_authorized"] is True
    denied: dict[str, Any] = _provenance()
    denied["data_access_decision"] = {
        **denied["data_access_decision"],
        "data_access_authorized": False,
    }
    denied_path = atomic_json(tmp_path / "denied.json", denied)
    with pytest.raises(ValueError, match="authorization binding"):
        load_execution_provenance(denied_path)


def test_gpu_telemetry_is_snapshotted_logged_and_partial_rows_are_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "gpu-telemetry.csv"
    source.write_text(
        "timestamp, index, uuid, utilization.gpu [%], memory.used [MiB], power.draw [W]\n"
        "2026/06/30 19:00:00.000, 0, GPU-one, 87 %, 1234 MiB, 201.5 W\n"
        "2026/06/30 19:00:15.000, 90 %\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JEPA4D_GPU_TELEMETRY_PATH", str(source))

    class Run:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.rows: list[dict[str, object]] = []

        def log(self, row: dict[str, object]) -> None:
            self.rows.append(row)

    run = Run()
    snapshot = snapshot_and_log_gpu_telemetry(run)
    assert snapshot == tmp_path / "gpu-telemetry-wandb.csv"
    assert snapshot.read_text(encoding="utf-8").count("\n") == 2
    assert run.summary == {"gpu_telemetry_samples": 1, "gpu_telemetry_interval_seconds": 15}
    assert run.rows == [
        {
            "gpu/telemetry_sample": 0,
            "gpu/telemetry_timestamp": "2026/06/30 19:00:00.000",
            "gpu/utilization_percent": 87.0,
            "gpu/memory_used_mib": 1234.0,
            "gpu/power_w": 201.5,
        }
    ]


def test_gpu_telemetry_rejects_mixed_gpu_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "gpu-telemetry.csv"
    source.write_text(
        "timestamp, index, uuid, utilization.gpu [%]\n"
        "2026/06/30 19:00:00.000, 0, GPU-one, 50 %\n"
        "2026/06/30 19:00:00.000, 1, GPU-two, 60 %\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JEPA4D_GPU_TELEMETRY_PATH", str(source))

    class Run:
        summary: dict[str, object] = {}

        def log(self, _row: dict[str, object]) -> None:
            pass

    with pytest.raises(RuntimeError, match="mixed or invalid GPU"):
        snapshot_and_log_gpu_telemetry(Run())


def test_wandb_settings_disable_implicit_protected_metadata() -> None:
    import wandb

    settings = hardened_wandb_settings(wandb)
    assert settings.console == "off"
    assert settings.disable_git is True
    assert settings.save_code is False
    assert settings.x_disable_meta is True
    assert settings.x_disable_stats is True
    assert settings.x_disable_machine_info is True


def test_archive_materialization_binds_and_revalidates_exact_consumed_files(tmp_path: Path) -> None:
    archive = tmp_path / "SUNRGBD.zip"
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for family in SUN_FAMILIES:
            leaf = f"SUNRGBD/{family}/source/{family}-sample"
            bundle.writestr(f"{leaf}/image/frame.jpg", f"rgb-{family}".encode())
            bundle.writestr(f"{leaf}/depth_bfx/frame.png", f"depth-{family}".encode())
            bundle.writestr(f"{leaf}/intrinsics.txt", b"1 0 0 0 1 0 0 0 1\n")
            bundle.writestr(f"{leaf}/annotation/ignored.json", b"{}")
    archive_identity = file_identity(archive)
    counts = {family: 1 for family in SUN_FAMILIES}
    root, receipt_path, receipt = materialize_sun_archive(
        archive,
        tmp_path / "materialized",
        provenance=_provenance(),
        expected_archive_sha256=archive_identity["sha256"],
        expected_archive_bytes=archive_identity["bytes"],
        expected_inventory_counts=counts,
    )
    assert receipt["file_count"] == 12
    assert not any("annotation" in row["path"] for row in receipt["files"])
    validate_sun_materialization(
        root,
        receipt_path,
        provenance=_provenance(),
        expected_archive_sha256=archive_identity["sha256"],
        expected_archive_bytes=archive_identity["bytes"],
        expected_inventory_counts=counts,
    )
    mismatched_provenance = dict(_provenance())
    mismatched_provenance["git_commit"] = "f" * 40
    with pytest.raises(ValueError, match="frozen archive contract"):
        validate_sun_materialization(
            root,
            receipt_path,
            provenance=mismatched_provenance,
            expected_archive_sha256=archive_identity["sha256"],
            expected_archive_bytes=archive_identity["bytes"],
            expected_inventory_counts=counts,
        )
    consumed = next(root.rglob("*.jpg"))
    consumed.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="archive-derived identity"):
        validate_sun_materialization(
            root,
            receipt_path,
            provenance=_provenance(),
            expected_archive_sha256=archive_identity["sha256"],
            expected_archive_bytes=archive_identity["bytes"],
            expected_inventory_counts=counts,
        )


@pytest.mark.parametrize("unsafe_name", ("../escape.txt", "SUNRGBD\\escape.txt"))
def test_archive_materialization_rejects_unsafe_paths_atomically(tmp_path: Path, unsafe_name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, mode="w") as bundle:
        bundle.writestr(unsafe_name, b"unsafe")
    identity = file_identity(archive)
    output = tmp_path / "materialized"
    with pytest.raises(ValueError, match="unsafe member"):
        materialize_sun_archive(
            archive,
            output,
            provenance=_provenance(),
            expected_archive_sha256=identity["sha256"],
            expected_archive_bytes=identity["bytes"],
            expected_inventory_counts={family: 1 for family in SUN_FAMILIES},
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".materialized.tmp-*"))


def test_archive_materialization_rejects_links_and_wrong_identity(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    link = zipfile.ZipInfo("SUNRGBD/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, mode="w") as bundle:
        bundle.writestr(link, b"target")
    identity = file_identity(archive)
    with pytest.raises(ValueError, match="symbolic link"):
        materialize_sun_archive(
            archive,
            tmp_path / "linked",
            provenance=_provenance(),
            expected_archive_sha256=identity["sha256"],
            expected_archive_bytes=identity["bytes"],
            expected_inventory_counts={family: 1 for family in SUN_FAMILIES},
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_sun_archive(
            archive,
            tmp_path / "wrong-hash",
            provenance=_provenance(),
            expected_archive_sha256="0" * 64,
            expected_archive_bytes=identity["bytes"],
            expected_inventory_counts={family: 1 for family in SUN_FAMILIES},
        )
    assert not (tmp_path / "wrong-hash").exists()


def _frame(root: Path, family: str, name: str) -> SUNRGBDFrame:
    leaf = root / family / name
    (leaf / "image").mkdir(parents=True)
    (leaf / "depth_bfx").mkdir()
    image = leaf / "image" / f"{name}.jpg"
    depth = leaf / "depth_bfx" / f"{name}.png"
    intrinsics = leaf / "intrinsics.txt"
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(image)
    Image.fromarray(np.zeros((8, 10), dtype=np.uint16)).save(depth)
    intrinsics.write_text("5 0 4.5 0 5 3.5 0 0 1\n", encoding="utf-8")
    return SUNRGBDFrame(
        sample_id=f"{family}/{name}",
        group_id=f"{family}/{name}",
        sensor=family,
        leaf=leaf,
        image_path=image,
        depth_path=depth,
        intrinsics_path=intrinsics,
    )


def test_membership_freezes_sorted_ids_and_persists_only_boolean_screening(tmp_path: Path) -> None:
    inventory = {family: [_frame(tmp_path, family, name) for name in ("c", "a", "b")] for family in SUN_FAMILIES}

    def enumerate_fake(_root: Path) -> dict[str, list[SUNRGBDFrame]]:
        return inventory

    def screen(frame: SUNRGBDFrame) -> tuple[bool, str | None]:
        return (False, "fewer_than_100_valid_pixels") if frame.sample_id.endswith("/a") else (True, None)

    manifest, selected = build_sun_membership_manifest(
        tmp_path,
        samples_per_family=1,
        qualitative_ids_per_family=1,
        expected_inventory_counts={family: 3 for family in SUN_FAMILIES},
        enumerator=enumerate_fake,
        screener=screen,
    )
    validate_sun_membership_manifest(manifest, expected_per_family=1)
    assert all(frames[0].sample_id.endswith("/b") for frames in selected.values())
    for family in SUN_FAMILIES:
        rows = manifest["screening"][family]
        assert [row["sample_id"].rsplit("/", 1)[-1] for row in rows] == ["a", "b", "c"]
        assert rows[0]["failure_reasons"] == {
            "depth_decode_failed": False,
            "fewer_than_100_valid_pixels": True,
        }
        assert rows[1]["selected"] is True
        assert rows[2]["eligible"] is True and rows[2]["selected"] is False
    serialized = json.dumps(manifest)
    assert "valid_depth_pixels" not in serialized
    assert "depth_mean" not in serialized
    assert manifest["target_blindness"] == {
        "depth_values_persisted": False,
        "depth_histograms_persisted": False,
        "depth_aggregate_statistics_persisted": False,
        "target_previews_persisted": False,
        "model_outputs_used_for_membership": False,
    }
    tampered = json.loads(json.dumps(manifest))
    tampered["selected_samples"][0]["sample_id"] = "kv1/not-the-screened-selection"
    unhashed = dict(tampered)
    unhashed.pop("manifest_sha256")
    tampered["manifest_sha256"] = canonical_sha256(unhashed)
    with pytest.raises(ValueError, match="screening/selection mismatch"):
        validate_sun_membership_manifest(tampered, expected_per_family=1)


def _family_shards(family: str) -> tuple[dict, dict, dict]:
    images = torch.zeros(1, 2, 3, 384, 384, dtype=torch.uint8)
    rgb = torch.zeros(1, 2, 3, 96, 96, dtype=torch.uint8)
    intrinsics = torch.tensor([[[[300.0, 0.0, 191.5], [0.0, 310.0, 191.5], [0.0, 0.0, 1.0]]] * 2])
    input_shard = build_input_shard(
        family=family,
        sample_ids=[f"{family}/sample"],
        images_384=images,
        rgb_96=rgb,
        intrinsics_384=intrinsics,
        membership_sha256="a" * 64,
    )
    ordinary_depth = torch.ones(1, 2, 24, 24)
    ordinary_valid = torch.ones(1, 2, 24, 24, dtype=torch.bool)
    center_depth = torch.ones(1, 384, 384)
    center_valid = torch.ones(1, 384, 384, dtype=torch.bool)
    # The input hash is rebound after it is written in the rotation-view test.
    target = build_target_shard(
        input_shard,
        ordinary_depth_24=ordinary_depth,
        ordinary_valid_24=ordinary_valid,
        center_depth_384=center_depth,
        center_valid_384=center_valid,
        input_sha256="b" * 64,
    )
    feature = build_feature_shard(
        input_shard,
        ordinary_features=torch.zeros(1, 2, 768, 24, 24),
        paired_features=torch.zeros(1, 8, 768, 24, 24),
        input_sha256="b" * 64,
    )
    return input_shard, feature, target


def test_target_shard_rejects_values_outside_frozen_depth_interval() -> None:
    _, _, target = _family_shards("kv1")
    target["ordinary_targets"]["depth_24"].fill_(11.0)
    with pytest.raises(ValueError, match="values/dtypes"):
        validate_target_shard(target)


def test_input_and_feature_shards_reject_extra_targets_and_incomplete_audits() -> None:
    input_shard, feature_shard, _ = _family_shards("kv1")
    input_shard["target_depth"] = torch.ones(1, 24, 24)
    with pytest.raises(ValueError, match="input shard keys changed"):
        validate_input_shard(input_shard)

    input_shard.pop("target_depth")
    input_shard["audit"]["distinct_updated_intrinsics_per_source"] = []
    with pytest.raises(ValueError, match="audit failed"):
        validate_input_shard(input_shard)

    feature_shard["target_depth"] = torch.ones(1, 24, 24)
    with pytest.raises(ValueError, match="feature shard keys changed"):
        validate_feature_shard(feature_shard)


def test_strict_reload_detects_loadable_checkpoint_state_corruption(tmp_path: Path) -> None:
    model = Phase2fScaleGeometryProbe(Phase2fGeometryConfig(input_dim=4, arm="M0"))
    checkpoint = save_phase2f_checkpoint(model, tmp_path / "checkpoint.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_name = next(iter(payload["state_dict"]))
    payload["state_dict"][first_name] = payload["state_dict"][first_name] + 1
    torch.save(payload, checkpoint)
    corrupted, _ = load_phase2f_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match="changed M0 state tensors"):
        assert_strict_phase2f_reload(model, corrupted, torch.zeros(2, 4, 24, 24))


def test_rotation_views_physically_omit_heldout_target_paths(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    identities: dict[str, dict[str, dict]] = {}
    for family in SUN_FAMILIES:
        input_shard, feature, target = _family_shards(family)
        family_root = root / "shards" / family
        input_path = write_torch_atomic(family_root / "input.pt", input_shard)
        input_sha = file_identity(input_path)["sha256"]
        feature["input_sha256"] = input_sha
        target["input_sha256"] = input_sha
        feature_path = write_torch_atomic(family_root / "feature.pt", feature)
        target_path = write_torch_atomic(family_root / "target.pt", target)
        identities[family] = {
            "input": file_identity(input_path),
            "feature": file_identity(feature_path),
            "target": file_identity(target_path),
        }
    create_rotation_views(root, membership_sha256="a" * 64, shard_identities=identities)
    for rotation, roles in ROTATIONS.items():
        descriptor = validate_rotation_view(root / "rotations" / rotation, expected_rotation=rotation)
        assert descriptor["heldout_target_exposed"] is False
        names = {path.name for path in (root / "rotations" / rotation).iterdir()}
        assert f"{roles['heldout']}.target.pt" not in names
        assert set(descriptor["families"]) == {*roles["train"], roles["validation"]}


def test_phase2g_metrics_add_rmse_delta1_and_reliability() -> None:
    target = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    prediction = target.log()
    variance = torch.zeros_like(prediction)
    valid = torch.ones_like(target, dtype=torch.bool)
    result = evaluate_phase2g_predictions(
        prediction,
        variance,
        target,
        valid_mask=valid,
        variance_multiplier=1.0,
        frame_ids=["frame"],
        family_ids=["kv1"],
    )
    row = result["per_frame"][0]
    assert row["raw_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert row["aligned_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert row["delta1"] == pytest.approx(1.0)
    assert row["reliability_error"] >= 0.0
    macro = result["equal_family_macro"]
    assert macro["signed_log_scale_error"] == pytest.approx(0.0, abs=1e-6)
    assert [macro[f"coverage_{level}"] for level in (50, 80, 90, 95)] == [1.0, 1.0, 1.0, 1.0]


def test_paired_hierarchical_bootstrap_preserves_pairing() -> None:
    rows = []
    for family in FAMILIES:
        for seed in FORMAL_SEEDS:
            for frame in ("a", "b"):
                rows.append({"arm": "M0", "family": family, "seed": seed, "frame_id": frame, "raw_abs_rel": 1.0})
                rows.append({"arm": "M1", "family": family, "seed": seed, "frame_id": frame, "raw_abs_rel": 0.8})
    result = paired_hierarchical_bootstrap(rows, candidate="M1", resamples=200)
    assert result["observed"] == pytest.approx(-0.2)
    assert result["ci95"] == pytest.approx([-0.2, -0.2])
    assert result["descriptive_only"] is True


def _wandb() -> dict[str, object]:
    return {"mode": "online", "status": "success"}


def test_lr_selector_requires_and_ranks_complete_healthy_matrix() -> None:
    receipts = []
    for arm in ARMS:
        for rotation in PROTOCOL_ROTATIONS:
            for learning_rate in LEARNING_RATES:
                quality = {5e-4: 0.9, 1e-3: 0.8, 2e-3: 0.8}[learning_rate]
                receipts.append(
                    {
                        "schema_version": TUNING_RECEIPT_SCHEMA,
                        "status": "success",
                        "arm": arm,
                        "rotation": rotation,
                        "seed": TUNING_SEED,
                        "learning_rate": learning_rate,
                        "validation_metrics": {
                            "equal_family_macro": {
                                "raw_abs_rel": quality,
                                "absolute_log_scale_error": 0.2,
                            }
                        },
                        "checkpoint": {"sha256": "a" * 64},
                        "health": {
                            "all_finite": True,
                            "maximum_forbidden_gradient_norm": 0.0,
                            "all_expected_allowed_gradients_seen": True,
                            "model_changed_from_initialization": True,
                            "objective_decreased": True,
                        },
                        "exact_reload": True,
                        "wandb": _wandb(),
                        "execution_provenance": _provenance(),
                    }
                )
    result = select_learning_rates(receipts, _provenance())
    assert result["status"] == "pass"
    assert len(result["selected"]) == 16
    assert all(value["learning_rate"] == 1e-3 for value in result["selected"].values())


def _evaluation_receipts() -> list[dict]:
    arm_values = {
        "M0": (1.00, 0.20, 0.50, 1.00, 0.10),
        "M1": (0.97, 0.18, 0.50, 0.99, 0.10),
        "M2": (0.94, 0.17, 0.49, 0.98, 0.099),
        "M3": (0.91, 0.16, 0.48, 0.97, 0.098),
    }
    family_frame_ids = {
        family: [opaque_frame_id(f"{family}/sample-{index:04d}") for index in range(SAMPLES_PER_FAMILY)]
        for family in FAMILIES
    }
    receipts = []
    for arm in ARMS:
        raw, scale, aligned, nll, ause = arm_values[arm]
        for rotation, roles in PROTOCOL_ROTATIONS.items():
            family = roles["heldout"]
            for seed in FORMAL_SEEDS:
                macro = {
                    "raw_abs_rel": raw,
                    "absolute_log_scale_error": scale,
                    "signed_log_scale_error": scale,
                    "aligned_abs_rel": aligned,
                    "nll": nll,
                    "ause": ause,
                    "raw_rmse": raw,
                    "aligned_rmse": aligned,
                    "delta1": 0.9,
                    "reliability_error": 0.05,
                    "coverage_50": 0.5,
                    "coverage_80": 0.8,
                    "coverage_90": 0.9,
                    "coverage_95": 0.95,
                }
                if arm in {"M0", "M1"}:
                    camera = {
                        "status": "not_applicable_nonconsumer",
                        "consumes_intrinsics": False,
                        "camera_parameters": 0,
                        "evaluator_intrinsics_call_rejected": True,
                    }
                else:
                    camera = {
                        "status": "evaluated",
                        "consumes_intrinsics": True,
                        "raw_abs_rel": {"updated": raw, "stale": 1.0, "wrong": 1.0, "permuted": 1.0},
                        "profile_raw_abs_rel": {
                            condition: {f"P{profile}": value for profile in range(1, 8)}
                            for condition, value in {
                                "updated": raw,
                                "stale": 1.0,
                                "wrong": 1.0,
                                "permuted": 1.0,
                            }.items()
                        },
                        "mean_absolute_prediction_delta_m": {"stale": 1e-3, "wrong": 1e-3, "permuted": 1e-3},
                        "distinct_analytic_intrinsics_per_source_min": 8,
                        "distinct_analytic_intrinsics_per_source_max": 8,
                        "permutation_assignment_change_fraction": 1.0,
                        "permutation_matrix_change_fraction": 1.0,
                    }
                receipts.append(
                    {
                        "schema_version": EVALUATION_RECEIPT_SCHEMA,
                        "status": "success",
                        "arm": arm,
                        "rotation": rotation,
                        "seed": seed,
                        "heldout_family": family,
                        "all_values_finite": True,
                        "checkpoint_frozen_before_heldout_access": True,
                        "external_final_authorized": False,
                        "external_final_accessed": False,
                        "metrics": {
                            "frames": SAMPLES_PER_FAMILY,
                            "valid_frames": SAMPLES_PER_FAMILY,
                            "failure_count": 0,
                            "valid_pixels": SAMPLES_PER_FAMILY * 576,
                            "coverage": [
                                {"nominal": 0.5, "empirical": 0.5},
                                {"nominal": 0.8, "empirical": 0.8},
                                {"nominal": 0.9, "empirical": 0.9},
                                {"nominal": 0.95, "empirical": 0.95},
                            ],
                            "risk_coverage": {
                                "coverage": [0.0, 0.5, 1.0],
                                "risk": [0.01, raw / 2, raw],
                                "pixel_ause": ause,
                            },
                            "equal_family_macro": macro,
                            "per_frame": [
                                {"frame_id": frame_id, "group_id": family, **macro}
                                for frame_id in family_frame_ids[family]
                            ],
                        },
                        "camera_controls": camera,
                        "zero_field_intervention": None
                        if arm != "M3"
                        else {
                            "same_checkpoint": True,
                            "metrics": {"equal_family_macro": {"raw_abs_rel": 0.95}},
                            "per_frame_field_mean": [(-0.01 if index % 2 == 0 else 0.01) for index in range(1024)],
                            "per_frame_field_sd": [0.02 + index % 2 * 0.01 for index in range(1024)],
                        },
                        "scale_mechanism": {"status": "not_applicable_monolithic"}
                        if arm == "M0"
                        else {
                            "correlation": 0.8,
                            "mean_residual": 0.01,
                            "per_frame": [
                                {
                                    "predicted_log_scale": 0.1 + index / 10_000,
                                    "optimal_log_scale": 0.09 + index / 10_000,
                                    "residual": 0.01,
                                }
                                for index in range(1024)
                            ],
                        },
                        "qualitative": {"count": 16},
                        "training_diagnostics": {
                            "final_objective_decile_mean": 0.5,
                            "elapsed_seconds": 60.0,
                            "parameter_counts": {
                                "shape_decoder": 80_000,
                                "scale_projection": 5_000 if arm != "M0" else 0,
                                "scale_head": 2_000,
                                "camera_prompt": 96 if arm in {"M2", "M3"} else 0,
                                "coarse_scale_field": 769 if arm == "M3" else 0,
                                "total": 87_865,
                            },
                            "epoch_diagnostics": [
                                {
                                    "epoch": epoch,
                                    "train_total": 1.0 / (epoch + 1),
                                    "allowed_gradient_norm_shape": 0.5,
                                    "allowed_gradient_norm_scale": 0.2 if arm != "M0" else 0.0,
                                    "allowed_gradient_norm_field": 0.1 if arm == "M3" else 0.0,
                                    "forbidden_gradient_norm": 0.0,
                                    "epoch_seconds": 10.0,
                                    "throughput_source_groups_per_second": 200.0,
                                    "peak_cuda_allocated_bytes": 2**30,
                                    "peak_cuda_reserved_bytes": 2 * 2**30,
                                }
                                for epoch in range(60)
                            ],
                        },
                        "wandb": _wandb(),
                        "execution_provenance": _provenance(),
                    }
                )
    return receipts


def test_selector_applies_quality_camera_field_and_hierarchical_gates() -> None:
    result = select_survivor(_evaluation_receipts(), _provenance(), bootstrap_resamples=50)
    assert result["status"] == "success"
    assert result["survivor"] == "M3"
    assert result["retained_arm"] == "M3"
    assert result["external_final_authorized"] is False
    assert all(result["eligibility"][arm]["eligible"] for arm in ("M1", "M2", "M3"))
    assert result["camera_mechanisms"]["M2"]["passed"] is True
    assert result["m3_zero_field_mechanism"]["passed"] is True


def test_selector_rejects_incomplete_or_changed_heldout_membership() -> None:
    receipts = _evaluation_receipts()
    receipts[0]["metrics"]["per_frame"].pop()
    receipts[0]["metrics"]["frames"] -= 1
    receipts[0]["metrics"]["valid_frames"] -= 1
    with pytest.raises(ValueError, match="complete 1024-frame"):
        select_survivor(receipts, _provenance(), bootstrap_resamples=1)

    receipts = _evaluation_receipts()
    receipts[1]["metrics"]["per_frame"][0]["frame_id"] = "f" * 64
    with pytest.raises(ValueError, match="membership differs"):
        select_survivor(receipts, _provenance(), bootstrap_resamples=1)


def test_visualization_pack_covers_ten_categories_and_keeps_pixels_local(tmp_path: Path) -> None:
    ids = ["kv1/b", "kv1/a"]
    rgb = torch.zeros(2, 3, 384, 384, dtype=torch.uint8)
    target = torch.ones(2, 24, 24)
    valid = torch.ones(2, 24, 24, dtype=torch.bool)
    prediction = target.log()
    variance = torch.zeros_like(target)
    panel, manifest, selected = write_local_qualitative_panels(
        tmp_path / "protected",
        family="kv1",
        sample_ids=ids,
        rgb_uint8=rgb,
        target_depth=target,
        valid_mask=valid,
        log_depth=prediction,
        log_variance=variance,
        scale_field=None,
    )
    assert panel.is_file() and manifest.is_file()
    local = json.loads(manifest.read_text())
    assert local["local_only"] is True and local["wandb_upload_forbidden"] is True
    assert set(selected) == set(ids)

    receipts = _evaluation_receipts()
    result = select_survivor(receipts, _provenance(), bootstrap_resamples=10)
    files = write_aggregate_visualizations(tmp_path / "aggregate", result=result, receipts=receipts)
    pngs = [path for path in files if path.suffix == ".png"]
    assert len(pngs) == 10
    assert all(path.is_file() for path in files)
    report = next(path for path in files if path.name == "report.html").read_text()
    assert report.count("data:image/png;base64") == 10
    assert "protected local-only" in report
    visualization_manifest = json.loads(
        next(path for path in files if path.name == "visualization_manifest.json").read_text()
    )
    assert [row["plot_type"] for row in visualization_manifest["categories"]] == [
        "per-family-per-seed-paired-forest-plus-hierarchical-intervals",
        "grouped-quality-scale-bars",
        "predicted-optimal-scatter-plus-residual-histogram",
        "reliability-and-risk-coverage-curves",
        "p1-p7-camera-control-curves",
        "m3-full-zero-performance-and-field-distributions",
        "fixed-local-qualitative-completeness",
        "loss-allowed-forbidden-gradient-curves",
        "memory-throughput-component-curves",
        "provenance-failure-retry-completeness-matrix",
    ]
    assert visualization_manifest["protected_pixels_embedded"] is False
