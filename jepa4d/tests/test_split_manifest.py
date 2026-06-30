from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jepa4d.tests.test_validation_registry import dataset_entry, registry_value
from jepa4d.validation.registry import DatasetRegistry
from jepa4d.validation.split_manifest import (
    SplitManifest,
    freeze_registered_split_manifest,
    freeze_split_manifest,
    verify_cross_manifest_disjointness,
    verify_frozen_registered_split_manifest,
    verify_frozen_split_manifest,
)


def manifest_value() -> dict:
    return {
        "schema_version": "jepa4d-split-manifest-v1",
        "manifest_version": "fixture-v1",
        "portfolio_version": "test-v1",
        "dataset_id": "fixture.dataset",
        "dataset_version": "2026.06",
        "split_id": "fixture.dataset.dev",
        "independent_unit": "video",
        "selection": {
            "algorithm": "sha256-rank-v1",
            "implementation": "jepa4d.validation.selectors.sha256_rank:v1",
            "implementation_sha256": "c" * 64,
            "seed": 20260630,
            "eligibility_rules": ["decode succeeds", "duration_seconds >= 2"],
            "parameters": {"maximum_units": 1, "hash_namespace": "fixture-dev"},
        },
        "target_isolation": {
            "reviewer": "fixture-reviewer",
            "reviewed_at": "2026-06-30T12:00:00Z",
            "targets_accessible": False,
            "selector_implementation_sha256": "c" * 64,
            "path_denial_receipt_sha256": "d" * 64,
        },
        "source_assets": [
            {
                "asset_id": "asset/video-001",
                "source_ref": "official/videos/video-001.mp4",
                "sha256": "a" * 64,
                "bytes": 101,
            },
            {
                "asset_id": "asset/video-002",
                "source_ref": "official/videos/video-002.mp4",
                "sha256": "b" * 64,
                "bytes": 202,
            },
        ],
        "clusters": [{"cluster_id": "subject-01"}, {"cluster_id": "subject-02"}],
        "selected_units": [
            {
                "disposition": "selected",
                "unit_id": "video-001",
                "cluster_id": "subject-01",
                "physical_unit_sha256": "e" * 64,
                "source_asset_ids": ["asset/video-001"],
                "metadata": {"duration_seconds": 5.0, "frame_count": 150},
            }
        ],
        "rejected_units": [
            {
                "disposition": "rejected",
                "unit_id": "video-002",
                "cluster_id": "subject-02",
                "physical_unit_sha256": "f" * 64,
                "source_asset_ids": ["asset/video-002"],
                "metadata": {"duration_seconds": 1.0, "frame_count": 30},
                "rejection_reason": "duration_seconds is below the prespecified minimum",
            }
        ],
    }


def bound_registry(manifest: SplitManifest, *, id_manifest_sha256: str | None = None) -> DatasetRegistry:
    entry = dataset_entry(dataset_id=manifest.dataset_id, split_id=manifest.split_id)
    entry["source"]["version"] = manifest.dataset_version
    entry["splits"][0]["independent_unit"] = manifest.independent_unit.value
    entry["splits"][0]["id_manifest_sha256"] = id_manifest_sha256 or manifest.sha256
    return DatasetRegistry.model_validate(registry_value(entry))


def phase_manifest_value(phase: int) -> dict:
    value = manifest_value()
    first = phase * 10 + 1
    second = phase * 10 + 2
    value["split_id"] = f"fixture.dataset.phase{phase}"
    value["selection"]["parameters"]["hash_namespace"] = f"fixture-phase{phase}"
    value["source_assets"] = [
        {
            "asset_id": f"asset/video-{first:03d}",
            "source_ref": f"official/videos/video-{first:03d}.mp4",
            "sha256": f"{phase}" * 64,
            "bytes": first,
        },
        {
            "asset_id": f"asset/video-{second:03d}",
            "source_ref": f"official/videos/video-{second:03d}.mp4",
            "sha256": f"{phase + 2}" * 64,
            "bytes": second,
        },
    ]
    value["clusters"] = [
        {"cluster_id": f"phase{phase}-scene-a"},
        {"cluster_id": f"phase{phase}-scene-b"},
    ]
    value["selected_units"][0].update(
        unit_id=f"video-{first:03d}",
        cluster_id=f"phase{phase}-scene-a",
        physical_unit_sha256=f"{phase + 4:x}" * 64,
        source_asset_ids=[f"asset/video-{first:03d}"],
    )
    value["rejected_units"][0].update(
        unit_id=f"video-{second:03d}",
        cluster_id=f"phase{phase}-scene-b",
        physical_unit_sha256=f"{phase + 6:x}" * 64,
        source_asset_ids=[f"asset/video-{second:03d}"],
    )
    return value


def test_yaml_json_loading_hash_and_freeze_are_deterministic(tmp_path: Path) -> None:
    value = manifest_value()
    yaml_path = tmp_path / "split.yaml"
    json_path = tmp_path / "split.json"
    yaml_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    json_path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    yaml_manifest = SplitManifest.load(yaml_path)
    json_manifest = SplitManifest.load(json_path)
    assert yaml_manifest.sha256 == json_manifest.sha256

    yaml_artifact = freeze_split_manifest(yaml_path, tmp_path / "frozen")
    json_artifact = freeze_split_manifest(json_path, tmp_path / "frozen")
    model_artifact = freeze_split_manifest(yaml_manifest, tmp_path / "frozen")
    assert yaml_artifact == json_artifact == model_artifact
    assert yaml_artifact.path.name == f"split-manifest-{yaml_artifact.sha256}.json"

    frozen = verify_frozen_split_manifest(yaml_artifact.path)
    assert frozen.manifest_sha256 == yaml_manifest.sha256
    assert frozen.manifest.independent_unit.value == "video"


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-key.yaml"
    path.write_text(
        yaml.safe_dump(manifest_value(), sort_keys=False) + "dataset_id: shadow.dataset\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate YAML key 'dataset_id'"):
        SplitManifest.load(path)


def test_duplicate_unit_and_cluster_identities_are_rejected() -> None:
    duplicate_unit = manifest_value()
    duplicate_unit["rejected_units"][0]["unit_id"] = "video-001"
    with pytest.raises(ValidationError, match="duplicate unit_id"):
        SplitManifest.model_validate(duplicate_unit)

    duplicate_cluster = manifest_value()
    duplicate_cluster["clusters"].append({"cluster_id": "subject-01"})
    with pytest.raises(ValidationError, match="duplicate cluster_id"):
        SplitManifest.model_validate(duplicate_cluster)


def test_duplicate_source_content_identity_is_rejected() -> None:
    value = manifest_value()
    value["source_assets"][1]["sha256"] = "a" * 64
    with pytest.raises(ValidationError, match="duplicate source asset SHA-256 identity"):
        SplitManifest.model_validate(value)

    duplicate_unit_content = manifest_value()
    duplicate_unit_content["rejected_units"][0]["physical_unit_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="duplicate physical_unit_sha256"):
        SplitManifest.model_validate(duplicate_unit_content)


def test_rejected_rows_require_an_explicit_reason() -> None:
    missing = manifest_value()
    missing["rejected_units"][0].pop("rejection_reason")
    with pytest.raises(ValidationError, match="rejection_reason"):
        SplitManifest.model_validate(missing)

    blank = manifest_value()
    blank["rejected_units"][0]["rejection_reason"] = "  "
    with pytest.raises(ValidationError, match="rejection_reason"):
        SplitManifest.model_validate(blank)


@pytest.mark.parametrize("forbidden_key", ["label", "class_balance", "mean_reward", "target-score"])
def test_target_like_metadata_keys_are_forbidden(forbidden_key: str) -> None:
    value = manifest_value()
    value["selected_units"][0]["metadata"][forbidden_key] = 1
    with pytest.raises(ValidationError, match="forbidden target-like metadata key"):
        SplitManifest.model_validate(value)


def test_selection_parameters_cannot_encode_target_statistics() -> None:
    value = manifest_value()
    value["selection"]["parameters"]["positive_label_rate"] = 0.75
    with pytest.raises(ValidationError, match="forbidden target-like metadata key"):
        SplitManifest.model_validate(value)


def test_free_text_cannot_bypass_target_like_content_screening() -> None:
    selection_rule = manifest_value()
    selection_rule["selection"]["eligibility_rules"].append("exclude when action label is unavailable")
    with pytest.raises(ValidationError, match="forbidden target-like token in selection eligibility_rules"):
        SplitManifest.model_validate(selection_rule)

    rejection_reason = manifest_value()
    rejection_reason["rejected_units"][0]["rejection_reason"] = "target annotation was unavailable"
    with pytest.raises(ValidationError, match="forbidden target-like token in rejection_reason"):
        SplitManifest.model_validate(rejection_reason)

    unit_string_value = manifest_value()
    unit_string_value["selected_units"][0]["metadata"]["audit_note"] = "derived from ground-truth labels"
    with pytest.raises(ValidationError, match="forbidden target-like token in unit metadata string value"):
        SplitManifest.model_validate(unit_string_value)

    parameter_string_value = manifest_value()
    parameter_string_value["selection"]["parameters"]["audit_note"] = "mean reward was positive"
    with pytest.raises(ValidationError, match="forbidden target-like token in selection parameters string value"):
        SplitManifest.model_validate(parameter_string_value)

    multiword_value = manifest_value()
    multiword_value["selected_units"][0]["metadata"]["audit_note"] = "derived from ground truth"
    with pytest.raises(ValidationError, match="forbidden target-like token in unit metadata string value"):
        SplitManifest.model_validate(multiword_value)

    camel_case_rule = manifest_value()
    camel_case_rule["selection"]["eligibility_rules"].append("targetScore must exist")
    with pytest.raises(ValidationError, match="forbidden target-like token in selection eligibility_rules"):
        SplitManifest.model_validate(camel_case_rule)

    camel_case_key = manifest_value()
    camel_case_key["selected_units"][0]["metadata"]["classLabel"] = "opaque-value"
    with pytest.raises(ValidationError, match="forbidden target-like metadata key"):
        SplitManifest.model_validate(camel_case_key)

    multiword_key = manifest_value()
    multiword_key["selected_units"][0]["metadata"]["ground_truth"] = "opaque-value"
    with pytest.raises(ValidationError, match="forbidden target-like metadata key"):
        SplitManifest.model_validate(multiword_key)


def test_target_isolation_attestation_is_typed_and_bound_to_selector() -> None:
    missing = manifest_value()
    missing.pop("target_isolation")
    with pytest.raises(ValidationError, match="target_isolation"):
        SplitManifest.model_validate(missing)

    accessible = manifest_value()
    accessible["target_isolation"]["targets_accessible"] = True
    with pytest.raises(ValidationError, match="targets_accessible"):
        SplitManifest.model_validate(accessible)

    coerced_false = manifest_value()
    coerced_false["target_isolation"]["targets_accessible"] = 0
    with pytest.raises(ValidationError, match="targets_accessible"):
        SplitManifest.model_validate(coerced_false)

    naive_time = manifest_value()
    naive_time["target_isolation"]["reviewed_at"] = "2026-06-30T12:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        SplitManifest.model_validate(naive_time)

    bad_receipt = manifest_value()
    bad_receipt["target_isolation"]["path_denial_receipt_sha256"] = "not-a-digest"
    with pytest.raises(ValidationError, match="path_denial_receipt_sha256"):
        SplitManifest.model_validate(bad_receipt)

    mismatched_selector = manifest_value()
    mismatched_selector["target_isolation"]["selector_implementation_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="different selector implementation"):
        SplitManifest.model_validate(mismatched_selector)


def test_independent_cluster_cannot_cross_the_selection_boundary() -> None:
    value = manifest_value()
    value["rejected_units"][0]["cluster_id"] = "subject-01"
    value["clusters"] = [{"cluster_id": "subject-01"}]
    with pytest.raises(ValidationError, match="cannot mix selected and rejected"):
        SplitManifest.model_validate(value)


def test_content_tampering_is_detected(tmp_path: Path) -> None:
    artifact = freeze_split_manifest(
        SplitManifest.model_validate(manifest_value()),
        tmp_path,
    )
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    payload["manifest"]["dataset_version"] = "tampered"
    artifact.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content digest mismatch"):
        verify_frozen_split_manifest(artifact.path)


def test_manifest_registry_binding_and_registered_freeze(tmp_path: Path) -> None:
    manifest = SplitManifest.model_validate(manifest_value())
    registry = bound_registry(manifest)
    manifest.validate_against_registry(registry)

    artifact = freeze_registered_split_manifest(manifest, registry, tmp_path)
    assert artifact.path.name == f"registered-split-manifest-{artifact.sha256}.json"
    frozen = verify_frozen_registered_split_manifest(artifact.path, registry)
    assert frozen.registry_sha256 == registry.sha256
    assert frozen.manifest_sha256 == manifest.sha256

    different_registry = registry.model_copy(update={"portfolio_version": "different-portfolio-v1"})
    with pytest.raises(ValueError, match="different registry SHA-256"):
        verify_frozen_registered_split_manifest(artifact.path, different_registry)


def test_registry_binding_rejects_portfolio_source_split_unit_and_hash_mismatches() -> None:
    manifest = SplitManifest.model_validate(manifest_value())
    registry = bound_registry(manifest)

    wrong_portfolio_value = manifest_value()
    wrong_portfolio_value["portfolio_version"] = "other-v1"
    with pytest.raises(ValueError, match="portfolio_version does not match"):
        SplitManifest.model_validate(wrong_portfolio_value).validate_against_registry(registry)

    wrong_source_value = manifest_value()
    wrong_source_value["dataset_version"] = "other-release"
    with pytest.raises(ValueError, match="dataset_version does not match"):
        SplitManifest.model_validate(wrong_source_value).validate_against_registry(registry)

    unknown_split_value = manifest_value()
    unknown_split_value["split_id"] = "fixture.dataset.unregistered"
    with pytest.raises(ValueError, match="dataset/split is not registered"):
        SplitManifest.model_validate(unknown_split_value).validate_against_registry(registry)

    wrong_unit_value = manifest_value()
    wrong_unit_value["independent_unit"] = "scene"
    with pytest.raises(ValueError, match="independent_unit does not match"):
        SplitManifest.model_validate(wrong_unit_value).validate_against_registry(registry)

    wrong_hash_registry = bound_registry(manifest, id_manifest_sha256="e" * 64)
    with pytest.raises(ValueError, match="does not match registered id_manifest_sha256"):
        manifest.validate_against_registry(wrong_hash_registry)


def test_phase5_phase6_selected_memberships_are_disjoint() -> None:
    phase5 = SplitManifest.model_validate(phase_manifest_value(5))
    phase6_value = phase_manifest_value(6)
    # A shared source archive is legal; member-level identities remain disjoint.
    phase6_value["source_assets"][0]["sha256"] = phase5.source_assets[0].sha256
    phase6 = SplitManifest.model_validate(phase6_value)
    verify_cross_manifest_disjointness((phase5, phase6))


def test_phase5_phase6_overlap_rejects_unit_and_cluster_reuse() -> None:
    phase5 = SplitManifest.model_validate(phase_manifest_value(5))

    unit_overlap_value = phase_manifest_value(6)
    unit_overlap_value["selected_units"][0]["unit_id"] = phase5.selected_units[0].unit_id
    unit_overlap = SplitManifest.model_validate(unit_overlap_value)
    with pytest.raises(ValueError, match=r"unit 'video-051'.*phase5.*phase6"):
        verify_cross_manifest_disjointness((phase5, unit_overlap))

    cluster_overlap_value = phase_manifest_value(6)
    shared_cluster = phase5.selected_units[0].cluster_id
    cluster_overlap_value["clusters"][0]["cluster_id"] = shared_cluster
    cluster_overlap_value["selected_units"][0]["cluster_id"] = shared_cluster
    cluster_overlap = SplitManifest.model_validate(cluster_overlap_value)
    with pytest.raises(ValueError, match=r"cluster 'phase5-scene-a'.*phase5.*phase6"):
        verify_cross_manifest_disjointness((phase5, cluster_overlap))

    physical_overlap_value = phase_manifest_value(6)
    physical_overlap_value["selected_units"][0]["physical_unit_sha256"] = phase5.selected_units[0].physical_unit_sha256
    physical_overlap = SplitManifest.model_validate(physical_overlap_value)
    with pytest.raises(ValueError, match=r"physical unit.*phase5.*phase6"):
        verify_cross_manifest_disjointness((phase5, physical_overlap))


def test_input_is_not_mutated_during_validation() -> None:
    value = manifest_value()
    before = deepcopy(value)
    SplitManifest.model_validate(value)
    assert value == before
