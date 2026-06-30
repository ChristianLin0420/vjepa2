from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jepa4d.evaluation.phase2f_data_cache import (
    ROTATIONS,
    SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA,
    SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA,
    SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA,
    build_sun_development_feature_cache,
    build_sun_development_input_cache,
    build_sun_development_target_cache,
    reject_external_target_references,
    rotation_indices,
    sha256_file,
    validate_sun_development_feature_cache,
    validate_sun_development_input_cache,
    validate_sun_development_target_cache,
    write_cache,
)
from scripts.build_phase2f_data_cache import WANDB_RECEIPT_SCHEMA, _finish_online_wandb, build_receipt


@pytest.fixture(scope="module")
def input_payload() -> dict[str, object]:
    families = ["kv1", "xtion", "realsense", "kv2"]
    images = torch.zeros(4, 2, 3, 384, 384, dtype=torch.uint8)
    for index in range(4):
        images[index, :, 0] = index * 30
        images[index, :, 1] = torch.arange(384, dtype=torch.uint8).view(384, 1)
    rgb = torch.zeros(4, 2, 3, 96, 96, dtype=torch.uint8)
    intrinsics = torch.zeros(4, 2, 3, 3)
    for index in range(4):
        matrix = torch.tensor([[300.0 + index, 0.0, 191.5], [0.0, 305.0 + index, 191.5], [0.0, 0.0, 1.0]])
        intrinsics[index, 0] = matrix
        intrinsics[index, 1] = matrix
    return build_sun_development_input_cache(
        sample_ids=[f"{family}/sample-000" for family in families],
        family_ids=families,
        images_384=images,
        rgb_96=rgb,
        intrinsics_384=intrinsics,
        sample_manifest_sha256="a" * 64,
        expected_per_family=1,
    )


def test_input_cache_contains_exact_profiles_and_no_targets(input_payload: dict[str, object]) -> None:
    assert input_payload["schema_version"] == SUN_DEVELOPMENT_INPUT_CACHE_SCHEMA
    ordinary = input_payload["ordinary_inputs"]
    paired = input_payload["paired_inputs"]
    audit = input_payload["audit"]
    assert ordinary["images_384_uint8"].shape == (4, 2, 3, 384, 384)
    assert paired["images_384_uint8"].shape == (4, 8, 3, 384, 384)
    assert paired["profile_ids"] == [f"P{index}" for index in range(8)]
    assert paired["profile_permutation"].tolist() == [5, 6, 3, 2, 1, 7, 0, 4]
    assert audit["distinct_updated_intrinsics_per_source_min"] == 8
    assert audit["permutation_matrix_change_fraction"] == 1.0
    assert "targets" not in input_payload
    validate_sun_development_input_cache(input_payload)


def test_target_and_feature_payloads_are_physically_separate(
    tmp_path: Path,
    input_payload: dict[str, object],
) -> None:
    input_path = write_cache(tmp_path / "input.pt", input_payload)
    input_sha = sha256_file(input_path)
    ordinary_depth = torch.full((4, 2, 24, 24), 2.0)
    ordinary_valid = torch.ones((4, 2, 24, 24), dtype=torch.bool)
    center_depth = torch.full((4, 384, 384), 2.0)
    center_valid = torch.ones((4, 384, 384), dtype=torch.bool)
    target_payload = build_sun_development_target_cache(
        input_payload,
        ordinary_depth_24=ordinary_depth,
        ordinary_valid_24=ordinary_valid,
        center_depth_384=center_depth,
        center_valid_384=center_valid,
        input_cache_sha256=input_sha,
    )
    assert target_payload["schema_version"] == SUN_DEVELOPMENT_TARGET_CACHE_SCHEMA
    assert target_payload["paired_targets"]["depth_24"].shape == (4, 8, 24, 24)
    assert not ({"ordinary_inputs", "paired_inputs", "features"} & set(target_payload))
    validate_sun_development_target_cache(target_payload)

    ordinary_features = torch.zeros(4, 2, 768, 24, 24)
    paired_features = torch.zeros(4, 8, 768, 24, 24)
    feature_payload = build_sun_development_feature_cache(
        input_payload,
        ordinary_features=ordinary_features,
        paired_features=paired_features,
        input_cache_sha256=input_sha,
    )
    assert feature_payload["schema_version"] == SUN_DEVELOPMENT_FEATURE_CACHE_SCHEMA
    assert feature_payload["audit"]["normalization"].startswith("not-applied")
    assert not ({"ordinary_targets", "paired_targets", "images_384_uint8"} & set(feature_payload))
    validate_sun_development_feature_cache(feature_payload)
    feature_path = write_cache(tmp_path / "features.pt", feature_payload)
    target_path = write_cache(tmp_path / "targets.pt", target_payload)
    assert len({sha256_file(input_path), sha256_file(feature_path), sha256_file(target_path)}) == 3


def test_rotation_indices_match_frozen_family_roles(input_payload: dict[str, object]) -> None:
    assert set(ROTATIONS) == {"R0", "R1", "R2", "R3"}
    r0 = rotation_indices(input_payload, "R0")
    assert r0["train"].tolist() == [0, 1]
    assert r0["validation"].tolist() == [2]
    assert r0["development_test"].tolist() == [3]
    r3 = rotation_indices(input_payload, "R3")
    assert r3["train"].tolist() == [0, 3]
    assert r3["validation"].tolist() == [1]
    assert r3["development_test"].tolist() == [2]


def test_cache_rejects_changed_permutation_and_external_target_reference(
    input_payload: dict[str, object],
) -> None:
    tampered = copy.deepcopy(input_payload)
    tampered["paired_inputs"]["profile_permutation"] = torch.arange(8)
    with pytest.raises(ValueError, match="profile permutation changed"):
        validate_sun_development_input_cache(tampered)
    with pytest.raises(ValueError, match="external-final reference"):
        reject_external_target_references({"path": "checkpoints/DIODE/val.tar.gz"})


def test_receipt_proves_cache_separation_and_zero_sealed_archive_access(
    tmp_path: Path,
    input_payload: dict[str, object],
) -> None:
    source = tmp_path / "source.pt"
    torch.save({"sun": "development"}, source)
    input_path = write_cache(tmp_path / "input.pt", input_payload)
    # Reuse independently named valid cache files; receipt checks physical identities only.
    target_path = write_cache(tmp_path / "target-as-input-schema.pt", input_payload)
    receipt = build_receipt(
        source_bundle=source,
        input_cache=input_path,
        target_cache=target_path,
        feature_cache=None,
        input_payload=input_payload,
    )
    assert receipt["target_separation"] == {
        "rgb_k_cache_contains_targets": False,
        "feature_cache_contains_targets": False,
        "target_cache_contains_rgb_k_or_features": False,
    }
    assert receipt["sealed_archive_access_audit"]["files_opened"] == 0
    assert receipt["sealed_archive_access_audit"]["bytes_read"] == 0
    assert receipt["controls"]["permutation_matrix_change_fraction"] == 1.0
    assert receipt["status"] == "success"


def test_wandb_receipt_matches_common_parent_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from slurm.phase2f_contract import validate_wandb

    receipt_path = tmp_path / "receipt.json"
    report_path = tmp_path / "report.json"
    report_html_path = tmp_path / "report.html"
    for path in (receipt_path, report_path, report_html_path):
        path.write_text("{}\n", encoding="utf-8")

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.files: list[tuple[str, str]] = []

        def add_file(self, path: str, *, name: str) -> None:
            self.files.append((path, name))

    uploaded = SimpleNamespace(id="artifact-id", version="v0", digest="artifact-digest")
    uploaded.wait = lambda timeout: uploaded
    run = SimpleNamespace(
        entity="entity",
        project="project",
        group="group",
        name="run-name",
        id="run-id",
        url="https://wandb.invalid/run-id",
        log_artifact=lambda artifact: uploaded,
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact))
    value = _finish_online_wandb(
        run,
        receipt_path=receipt_path,
        report_path=report_path,
        report_html_path=report_html_path,
    )
    assert value["schema_version"] == WANDB_RECEIPT_SCHEMA
    assert value["mode"] == "online"
    assert value["status"] == "success"
    assert value["job_type"] == "dev-cache"
    assert all(
        value[key]
        for key in ("run_id", "run_url", "artifact_name", "artifact_version", "artifact_id", "artifact_digest")
    )
    validate_wandb({"wandb": value})
    assert all(Path(item["path"]).suffix in {".json", ".html"} for item in value["files"])
    assert value["large_caches_uploaded"] is False
    assert value["rgb_or_raw_targets_uploaded"] is False
