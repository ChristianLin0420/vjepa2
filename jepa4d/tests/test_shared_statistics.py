import math
from pathlib import Path

import pytest

from jepa4d.evaluation.statistics import (
    BOOTSTRAP_BIT_GENERATOR,
    BOOTSTRAP_QUANTILE_METHOD,
    MIN_POPULATION_CI_CLUSTERS,
    PAIRED_BOOTSTRAP_SCHEMA_VERSION,
    REGISTRY_RESAMPLING_UNIT,
    SEED_VARIATION_SCHEMA_VERSION,
    ClusteredMetricObservation,
    IndependentResamplingUnit,
    InferentialStatus,
    UncertaintySource,
    paired_cluster_bootstrap,
    paired_cluster_bootstrap_from_registered_manifest,
    resampling_unit_from_registry,
    summarize_optimizer_seed_variation,
)
from jepa4d.tests.test_split_manifest import bound_registry, manifest_value
from jepa4d.validation.registry import IndependentUnit
from jepa4d.validation.split_manifest import (
    SplitManifest,
    freeze_registered_split_manifest,
    verify_frozen_registered_split_manifest,
)


def _observation(pair_id: str, cluster_id: str, value: float) -> ClusteredMetricObservation:
    return ClusteredMetricObservation(pair_id=pair_id, cluster_id=cluster_id, value=value)


def test_paired_cluster_bootstrap_is_seeded_order_invariant_and_auditable() -> None:
    reference = [
        _observation("a-1", "scene-a", 1.0),
        _observation("a-2", "scene-a", 3.0),
        _observation("b-1", "scene-b", 10.0),
    ]
    candidate = [
        _observation("a-1", "scene-a", 3.0),
        _observation("a-2", "scene-a", 5.0),
        _observation("b-1", "scene-b", 9.0),
    ]

    first = paired_cluster_bootstrap(
        reference,
        candidate,
        metric_name="score",
        resampling_unit=IndependentResamplingUnit.SCENE,
        reference_name="m0",
        candidate_name="m1",
        seed=17,
        resamples=1000,
    )
    second = paired_cluster_bootstrap(
        list(reversed(reference)),
        [candidate[2], candidate[0], candidate[1]],
        metric_name="score",
        resampling_unit="scene",
        reference_name="m0",
        candidate_name="m1",
        seed=17,
        resamples=1000,
    )

    # scene-a has effect 2 and scene-b has effect -1; clusters are equal-weighted.
    assert first.effect == pytest.approx(0.5)
    assert first == second
    assert first.schema_version == PAIRED_BOOTSTRAP_SCHEMA_VERSION
    assert first.interval.lower <= first.effect <= first.interval.upper
    assert first.counts.paired_observations == 3
    assert first.counts.independent_clusters == 2
    assert first.config.resampling_unit is IndependentResamplingUnit.SCENE
    assert first.config.cluster_weighting == "equal"
    assert first.uncertainty_source is UncertaintySource.INDEPENDENT_CLUSTER_POPULATION
    assert first.is_population_confidence_interval is False
    assert first.inferential_status is InferentialStatus.DESCRIPTIVE_UNBOUND
    assert len(first.observations_sha256) == 64
    assert first.replay.implementation_version == PAIRED_BOOTSTRAP_SCHEMA_VERSION
    assert first.replay.bit_generator == BOOTSTRAP_BIT_GENERATOR
    assert first.replay.quantile_method == BOOTSTRAP_QUANTILE_METHOD
    assert first.registry_sha256 is None
    assert first.manifest_sha256 is None
    payload = first.to_serializable()
    assert payload["config"]["seed"] == 17
    assert payload["counts"]["candidate_observations"] == 3


@pytest.mark.parametrize(
    ("reference", "candidate", "match"),
    [
        (
            [_observation("pair-a", "scene-a", 1.0)],
            [_observation("pair-b", "scene-a", 2.0)],
            "not paired",
        ),
        (
            [_observation("pair-a", "scene-a", 1.0)],
            [_observation("pair-a", "scene-b", 2.0)],
            "cluster mismatch",
        ),
        (
            [_observation("pair-a", "scene-a", 1.0), _observation("pair-a", "scene-a", 2.0)],
            [_observation("pair-a", "scene-a", 2.0)],
            "duplicate reference pair_id",
        ),
        (
            [ClusteredMetricObservation("pair-a", "scene-a", 1.0, unit_id="unit-a")],
            [ClusteredMetricObservation("pair-a", "scene-a", 2.0, unit_id="unit-b")],
            "manifest unit mismatch",
        ),
    ],
)
def test_paired_cluster_bootstrap_rejects_invalid_pairing(reference, candidate, match) -> None:
    with pytest.raises(ValueError, match=match):
        paired_cluster_bootstrap(
            reference,
            candidate,
            metric_name="score",
            resampling_unit="scene",
            resamples=10,
        )


@pytest.mark.parametrize("unit", ["frame", "pixel", "timestep", "made_up_unit"])
def test_paired_cluster_bootstrap_rejects_non_independent_resampling_units(unit: str) -> None:
    values = [_observation("pair-a", "scene-a", 1.0)]
    with pytest.raises(ValueError, match="independent cluster unit"):
        paired_cluster_bootstrap(
            values,
            values,
            metric_name="score",
            resampling_unit=unit,
            resamples=10,
        )


def test_paired_cluster_bootstrap_rejects_singleton_cluster_population_ci() -> None:
    reference = [_observation("pair-a", "scene-a", 1.0), _observation("pair-b", "scene-a", 2.0)]
    candidate = [_observation("pair-a", "scene-a", 2.0), _observation("pair-b", "scene-a", 4.0)]
    with pytest.raises(ValueError, match="at least two independent clusters"):
        paired_cluster_bootstrap(
            reference,
            candidate,
            metric_name="score",
            resampling_unit="scene",
            resamples=10,
        )


def test_raw_bootstrap_remains_unbound_with_many_independent_clusters() -> None:
    reference = [_observation(f"pair-{index}", f"scene-{index}", 0.0) for index in range(5)]
    candidate = [_observation(f"pair-{index}", f"scene-{index}", 1.0) for index in range(5)]
    result = paired_cluster_bootstrap(
        reference,
        candidate,
        metric_name="score",
        resampling_unit="scene",
        resamples=100,
    )
    assert result.counts.independent_clusters == MIN_POPULATION_CI_CLUSTERS
    assert result.is_population_confidence_interval is False
    assert result.inferential_status is InferentialStatus.DESCRIPTIVE_UNBOUND


def test_observations_and_bootstrap_configuration_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _observation("pair-a", "scene-a", math.nan)
    values = [_observation("pair-a", "scene-a", 1.0)]
    with pytest.raises(ValueError, match="confidence"):
        paired_cluster_bootstrap(values, values, metric_name="score", resampling_unit="scene", confidence=1.0)
    with pytest.raises(ValueError, match="resamples"):
        paired_cluster_bootstrap(values, values, metric_name="score", resampling_unit="scene", resamples=0)


def test_optimizer_seed_spread_is_not_a_population_confidence_interval() -> None:
    summary = summarize_optimizer_seed_variation({19: 0.6, 3: 0.2, 11: 0.4}, metric_name="success_rate")
    assert summary.schema_version == SEED_VARIATION_SCHEMA_VERSION
    assert summary.mean == pytest.approx(0.4)
    assert summary.sample_standard_deviation == pytest.approx(0.2)
    assert summary.seeds == (3, 11, 19)
    assert summary.seed_count == 3
    assert summary.uncertainty_source is UncertaintySource.OPTIMIZER_SEED_VARIATION
    assert summary.interval is None
    assert summary.is_population_confidence_interval is False


def test_optimizer_seed_spread_requires_replicates_and_finite_values() -> None:
    with pytest.raises(ValueError, match="at least two"):
        summarize_optimizer_seed_variation({0: 1.0}, metric_name="score")
    with pytest.raises(ValueError, match="finite"):
        summarize_optimizer_seed_variation({0: 1.0, 1: math.inf}, metric_name="score")


@pytest.mark.parametrize(
    ("registry_unit", "statistical_unit"),
    [
        ("image", IndependentResamplingUnit.IMAGE),
        ("recording", IndependentResamplingUnit.RECORDING),
        ("trajectory", IndependentResamplingUnit.TRAJECTORY),
        ("generated-case", IndependentResamplingUnit.GENERATED_CASE),
    ],
)
def test_registry_independent_units_have_canonical_statistical_mapping(
    registry_unit: str, statistical_unit: IndependentResamplingUnit
) -> None:
    assert resampling_unit_from_registry(registry_unit) is statistical_unit


def test_every_registry_independent_unit_has_a_statistical_mapping() -> None:
    assert set(REGISTRY_RESAMPLING_UNIT) == set(IndependentUnit)


def test_registered_frozen_manifest_binds_observations_and_bootstrap_provenance(tmp_path: Path) -> None:
    value = manifest_value()
    second_selected = value["rejected_units"].pop()
    second_selected.pop("rejection_reason")
    second_selected["disposition"] = "selected"
    second_selected["metadata"]["duration_seconds"] = 3.0
    value["selected_units"].append(second_selected)
    value["selection"]["parameters"]["maximum_units"] = 2
    manifest = SplitManifest.model_validate(value)
    registry = bound_registry(manifest)
    artifact = freeze_registered_split_manifest(manifest, registry, tmp_path)
    frozen = verify_frozen_registered_split_manifest(artifact.path, registry)

    reference = [
        _observation("video-001", "subject-01", 1.0),
        _observation("video-002", "subject-02", 2.0),
    ]
    candidate = [
        _observation("video-001", "subject-01", 1.5),
        _observation("video-002", "subject-02", 2.25),
    ]
    result = paired_cluster_bootstrap_from_registered_manifest(
        reference,
        candidate,
        registered_manifest=frozen,
        registry=registry,
        metric_name="score",
        resampling_unit="video",
        seed=7,
        resamples=100,
    )
    assert result.registry_sha256 == registry.sha256
    assert result.manifest_sha256 == manifest.sha256
    assert len(result.observations_sha256) == 64
    assert result.config.resampling_unit is IndependentResamplingUnit.VIDEO
    assert result.inferential_status is InferentialStatus.DESCRIPTIVE_SMALL_CLUSTER

    with pytest.raises(ValueError, match="resampling_unit does not match"):
        paired_cluster_bootstrap_from_registered_manifest(
            reference,
            candidate,
            registered_manifest=frozen,
            registry=registry,
            metric_name="score",
            resampling_unit="scene",
            resamples=10,
        )

    wrong_cluster = [candidate[0], _observation("video-002", "wrong-cluster", 2.25)]
    with pytest.raises(ValueError, match="cluster mismatch for manifest unit"):
        paired_cluster_bootstrap_from_registered_manifest(
            reference,
            wrong_cluster,
            registered_manifest=frozen,
            registry=registry,
            metric_name="score",
            resamples=10,
        )

    with pytest.raises(ValueError, match="do not cover selected manifest units"):
        paired_cluster_bootstrap_from_registered_manifest(
            reference[:1],
            candidate[:1],
            registered_manifest=frozen,
            registry=registry,
            metric_name="score",
            resamples=10,
        )

    duplicate_unit = [
        *reference,
        ClusteredMetricObservation("video-001-extra", "subject-01", 1.0, unit_id="video-001"),
    ]
    with pytest.raises(ValueError, match="duplicate selected manifest unit"):
        paired_cluster_bootstrap_from_registered_manifest(
            duplicate_unit,
            candidate,
            registered_manifest=frozen,
            registry=registry,
            metric_name="score",
            resamples=10,
        )


def test_only_registered_manifest_bootstrap_can_promote_population_ci(tmp_path: Path) -> None:
    value = manifest_value()
    second_selected = value["rejected_units"].pop()
    second_selected.pop("rejection_reason")
    second_selected["disposition"] = "selected"
    second_selected["metadata"]["duration_seconds"] = 3.0
    value["selected_units"].append(second_selected)
    for index in range(3, 6):
        value["source_assets"].append(
            {
                "asset_id": f"asset/video-{index:03d}",
                "source_ref": f"official/videos/video-{index:03d}.mp4",
                "sha256": f"{index}" * 64,
                "bytes": 100 + index,
            }
        )
        value["clusters"].append({"cluster_id": f"subject-{index:02d}"})
        value["selected_units"].append(
            {
                "disposition": "selected",
                "unit_id": f"video-{index:03d}",
                "cluster_id": f"subject-{index:02d}",
                "physical_unit_sha256": f"{index + 3}" * 64,
                "source_asset_ids": [f"asset/video-{index:03d}"],
                "metadata": {"duration_seconds": 3.0, "frame_count": 90},
            }
        )
    value["selection"]["parameters"]["maximum_units"] = 5
    manifest = SplitManifest.model_validate(value)
    registry = bound_registry(manifest)
    artifact = freeze_registered_split_manifest(manifest, registry, tmp_path / "registered-five")
    frozen = verify_frozen_registered_split_manifest(artifact.path, registry)
    reference = [_observation(f"video-{index:03d}", f"subject-{index:02d}", float(index)) for index in range(1, 6)]
    candidate = [
        _observation(f"video-{index:03d}", f"subject-{index:02d}", float(index) + 0.5) for index in range(1, 6)
    ]

    result = paired_cluster_bootstrap_from_registered_manifest(
        reference,
        candidate,
        registered_manifest=frozen,
        registry=registry,
        metric_name="score",
        resamples=100,
    )
    assert result.counts.independent_clusters == MIN_POPULATION_CI_CLUSTERS
    assert result.is_population_confidence_interval is True
    assert result.inferential_status is InferentialStatus.POPULATION_CONFIDENCE_INTERVAL
