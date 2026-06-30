"""Shared statistical contracts for stage-wise model comparisons.

The paired bootstrap in this module operates on *independent clusters*.  It is
deliberately separate from optimizer-seed summaries: rerunning an optimizer on
one fixed evaluation population measures training variation, not uncertainty
over new scenes, subjects, videos, or episodes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from jepa4d.validation._content import sha256_value
from jepa4d.validation.registry import DatasetRegistry, IndependentUnit
from jepa4d.validation.split_manifest import FrozenRegisteredSplitManifest, SplitManifest

PAIRED_BOOTSTRAP_SCHEMA_VERSION = "jepa4d.paired-cluster-bootstrap.v2"
SEED_VARIATION_SCHEMA_VERSION = "jepa4d.optimizer-seed-variation.v1"
MIN_POPULATION_CI_CLUSTERS = 5
BOOTSTRAP_BIT_GENERATOR = "PCG64"
BOOTSTRAP_QUANTILE_METHOD: Literal["linear"] = "linear"


class IndependentResamplingUnit(StrEnum):
    """Allowed units that can represent independent evaluation clusters.

    Lower-level correlated units such as frames, pixels, crops, detections,
    tokens, and timesteps are intentionally absent.
    """

    SUBJECT = "subject"
    SCENE = "scene"
    VIDEO = "video"
    SEQUENCE = "sequence"
    RECORDING = "recording"
    TRAJECTORY = "trajectory"
    IMAGE = "image"
    EPISODE = "episode"
    TASK = "task"
    ENVIRONMENT = "environment"
    ROBOT_RUN = "robot_run"
    DATASET_ITEM = "dataset_item"
    GENERATED_CASE = "generated-case"


REGISTRY_RESAMPLING_UNIT: Mapping[IndependentUnit, IndependentResamplingUnit] = MappingProxyType(
    {
        IndependentUnit.SUBJECT: IndependentResamplingUnit.SUBJECT,
        IndependentUnit.SCENE: IndependentResamplingUnit.SCENE,
        IndependentUnit.VIDEO: IndependentResamplingUnit.VIDEO,
        IndependentUnit.SEQUENCE: IndependentResamplingUnit.SEQUENCE,
        IndependentUnit.RECORDING: IndependentResamplingUnit.RECORDING,
        IndependentUnit.TRAJECTORY: IndependentResamplingUnit.TRAJECTORY,
        IndependentUnit.IMAGE: IndependentResamplingUnit.IMAGE,
        IndependentUnit.EPISODE: IndependentResamplingUnit.EPISODE,
        IndependentUnit.ENVIRONMENT: IndependentResamplingUnit.ENVIRONMENT,
        IndependentUnit.GENERATED_CASE: IndependentResamplingUnit.GENERATED_CASE,
    }
)


def resampling_unit_from_registry(value: IndependentUnit | str) -> IndependentResamplingUnit:
    """Map every registry independent unit to the canonical statistical unit."""

    try:
        registry_unit = IndependentUnit(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown registry independent unit: {value!r}") from error
    return REGISTRY_RESAMPLING_UNIT[registry_unit]


class UncertaintySource(StrEnum):
    """What source of variation a reported interval or spread describes."""

    INDEPENDENT_CLUSTER_POPULATION = "independent_cluster_population"
    OPTIMIZER_SEED_VARIATION = "optimizer_seed_variation"


class InferentialStatus(StrEnum):
    """Whether a bootstrap interval is claimable or descriptive only."""

    DESCRIPTIVE_UNBOUND = "descriptive-unbound"
    DESCRIPTIVE_SMALL_CLUSTER = "descriptive-small-cluster"
    POPULATION_CONFIDENCE_INTERVAL = "population-confidence-interval"


@dataclass(frozen=True, slots=True)
class ClusteredMetricObservation:
    """One arm's metric value for an explicitly paired evaluation item."""

    pair_id: str
    cluster_id: str
    value: float
    unit_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ValueError("pair_id must be a non-empty string")
        if not isinstance(self.cluster_id, str) or not self.cluster_id.strip():
            raise ValueError("cluster_id must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("observation value must be a real number")
        if not math.isfinite(float(self.value)):
            raise ValueError("observation value must be finite")
        if self.unit_id is not None and (not isinstance(self.unit_id, str) or not self.unit_id.strip()):
            raise ValueError("unit_id must be a non-empty string when provided")

    @property
    def manifest_unit_id(self) -> str:
        """Return the explicit manifest unit, falling back to legacy pair IDs."""

        return self.unit_id or self.pair_id


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float
    method: str


@dataclass(frozen=True, slots=True)
class BootstrapCounts:
    paired_observations: int
    independent_clusters: int
    reference_observations: int
    candidate_observations: int


@dataclass(frozen=True, slots=True)
class PairedBootstrapConfig:
    seed: int
    resamples: int
    confidence: float
    resampling_unit: IndependentResamplingUnit
    pair_effect: str = "candidate_minus_reference"
    within_cluster_aggregation: str = "mean"
    cluster_weighting: str = "equal"
    interval_method: str = "percentile"
    minimum_population_ci_clusters: int = MIN_POPULATION_CI_CLUSTERS


@dataclass(frozen=True, slots=True)
class BootstrapReplayMetadata:
    implementation_version: str
    numpy_version: str
    bit_generator: str
    quantile_method: str


@dataclass(frozen=True, slots=True)
class PairedClusterBootstrapResult:
    schema_version: str
    metric_name: str
    reference_name: str
    candidate_name: str
    effect: float
    interval: ConfidenceInterval
    counts: BootstrapCounts
    config: PairedBootstrapConfig
    uncertainty_source: UncertaintySource
    is_population_confidence_interval: bool
    inferential_status: InferentialStatus
    observations_sha256: str
    replay: BootstrapReplayMetadata
    registry_sha256: str | None = None
    manifest_sha256: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OptimizerSeedVariationSummary:
    schema_version: str
    metric_name: str
    mean: float
    sample_standard_deviation: float
    minimum: float
    maximum: float
    seed_count: int
    seeds: tuple[int, ...]
    uncertainty_source: UncertaintySource
    interval: None
    is_population_confidence_interval: bool

    def to_serializable(self) -> dict[str, Any]:
        return asdict(self)


def _require_non_empty_label(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _coerce_resampling_unit(value: IndependentResamplingUnit | str) -> IndependentResamplingUnit:
    try:
        return IndependentResamplingUnit(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(unit.value for unit in IndependentResamplingUnit)
        raise ValueError(f"resampling_unit must be an independent cluster unit ({allowed}); got {value!r}") from error


def _validate_bootstrap_configuration(*, confidence: float, resamples: int, seed: int) -> tuple[float, int, int]:
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise TypeError("confidence must be a real number")
    confidence_value = float(confidence)
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(resamples, bool) or not isinstance(resamples, Integral):
        raise TypeError("resamples must be an integer")
    if int(resamples) < 1:
        raise ValueError("resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if not 0 <= int(seed) < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return confidence_value, int(resamples), int(seed)


def _index_arm(
    observations: Sequence[ClusteredMetricObservation], arm_name: str
) -> dict[str, ClusteredMetricObservation]:
    if isinstance(observations, str | bytes) or not isinstance(observations, Sequence):
        raise TypeError(f"{arm_name} observations must be a sequence")
    if not observations:
        raise ValueError(f"{arm_name} observations must not be empty")
    indexed: dict[str, ClusteredMetricObservation] = {}
    for observation in observations:
        if not isinstance(observation, ClusteredMetricObservation):
            raise TypeError(f"{arm_name} observations must contain ClusteredMetricObservation values")
        if observation.pair_id in indexed:
            raise ValueError(f"duplicate {arm_name} pair_id: {observation.pair_id!r}")
        indexed[observation.pair_id] = observation
    return indexed


def _observation_digest(
    reference: Sequence[ClusteredMetricObservation],
    candidate: Sequence[ClusteredMetricObservation],
) -> str:
    def rows(values: Sequence[ClusteredMetricObservation]) -> list[dict[str, str | float]]:
        return [
            {
                "pair_id": observation.pair_id,
                "unit_id": observation.manifest_unit_id,
                "cluster_id": observation.cluster_id,
                "value": float(observation.value),
            }
            for observation in sorted(values, key=lambda item: item.pair_id)
        ]

    return sha256_value({"reference": rows(reference), "candidate": rows(candidate)})


def _validate_arm_against_manifest(
    observations: Sequence[ClusteredMetricObservation],
    manifest: SplitManifest,
    arm_name: str,
) -> None:
    indexed = _index_arm(observations, arm_name)
    selected = {unit.unit_id: unit.cluster_id for unit in manifest.selected_units}
    observed_units: set[str] = set()
    for observation in indexed.values():
        unit_id = observation.manifest_unit_id
        expected_cluster = selected.get(unit_id)
        if expected_cluster is None:
            raise ValueError(
                f"{arm_name} pair {observation.pair_id!r} names unit {unit_id!r} outside selected manifest membership"
            )
        if observation.cluster_id != expected_cluster:
            raise ValueError(
                f"{arm_name} cluster mismatch for manifest unit {unit_id!r}: "
                f"{observation.cluster_id!r} != {expected_cluster!r}"
            )
        if unit_id in observed_units:
            raise ValueError(f"{arm_name} observations duplicate selected manifest unit {unit_id!r}")
        observed_units.add(unit_id)
    missing = sorted(set(selected) - observed_units)
    if missing:
        raise ValueError(f"{arm_name} observations do not cover selected manifest units: {missing}")


def _paired_cluster_effects(
    reference: Sequence[ClusteredMetricObservation],
    candidate: Sequence[ClusteredMetricObservation],
) -> tuple[np.ndarray, int]:
    reference_by_pair = _index_arm(reference, "reference")
    candidate_by_pair = _index_arm(candidate, "candidate")
    reference_pairs = set(reference_by_pair)
    candidate_pairs = set(candidate_by_pair)
    if reference_pairs != candidate_pairs:
        missing_candidate = sorted(reference_pairs - candidate_pairs)
        missing_reference = sorted(candidate_pairs - reference_pairs)
        raise ValueError(
            "candidate/reference observations are not paired: "
            f"missing_candidate={missing_candidate}, missing_reference={missing_reference}"
        )

    differences_by_cluster: dict[str, list[float]] = defaultdict(list)
    for pair_id in sorted(reference_pairs):
        reference_observation = reference_by_pair[pair_id]
        candidate_observation = candidate_by_pair[pair_id]
        if reference_observation.cluster_id != candidate_observation.cluster_id:
            raise ValueError(
                "candidate/reference cluster mismatch for pair "
                f"{pair_id!r}: {reference_observation.cluster_id!r} != {candidate_observation.cluster_id!r}"
            )
        if reference_observation.manifest_unit_id != candidate_observation.manifest_unit_id:
            raise ValueError(
                "candidate/reference manifest unit mismatch for pair "
                f"{pair_id!r}: {reference_observation.manifest_unit_id!r} != "
                f"{candidate_observation.manifest_unit_id!r}"
            )
        differences_by_cluster[reference_observation.cluster_id].append(
            float(candidate_observation.value) - float(reference_observation.value)
        )

    # Sort cluster IDs so results are invariant to the input order.
    cluster_effects = np.asarray(
        [float(np.mean(differences_by_cluster[key])) for key in sorted(differences_by_cluster)],
        dtype=np.float64,
    )
    if not np.isfinite(cluster_effects).all():  # defensive guard after subtraction/aggregation
        raise ValueError("paired cluster effects must be finite")
    return cluster_effects, len(reference_pairs)


def paired_cluster_bootstrap(
    reference: Sequence[ClusteredMetricObservation],
    candidate: Sequence[ClusteredMetricObservation],
    *,
    metric_name: str,
    resampling_unit: IndependentResamplingUnit | str,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedClusterBootstrapResult:
    """Estimate an unbound, descriptive candidate-minus-reference interval.

    Pair IDs must match exactly across arms and each pair must name the same
    independent cluster in both arms.  Paired effects are averaged within each
    cluster, then cluster means are weighted equally.  Bootstrap draws resample
    those independent cluster means with replacement.  This raw API cannot
    authorize a population-CI claim because cluster identities are asserted by
    its caller; use :func:`paired_cluster_bootstrap_from_registered_manifest`
    for manifest-bound inference.
    """

    metric = _require_non_empty_label(metric_name, "metric_name")
    reference_label = _require_non_empty_label(reference_name, "reference_name")
    candidate_label = _require_non_empty_label(candidate_name, "candidate_name")
    unit = _coerce_resampling_unit(resampling_unit)
    confidence_value, resample_count, seed_value = _validate_bootstrap_configuration(
        confidence=confidence, resamples=resamples, seed=seed
    )
    cluster_effects, pair_count = _paired_cluster_effects(reference, candidate)
    cluster_count = len(cluster_effects)
    if cluster_count < 2:
        raise ValueError("at least two independent clusters are required for a paired cluster-bootstrap interval")
    effect = float(cluster_effects.mean())

    rng = np.random.Generator(np.random.PCG64(seed_value))
    bootstrap_effects = np.empty(resample_count, dtype=np.float64)
    # Bound peak memory while preserving one deterministic RNG stream.
    chunk_size = max(1, min(resample_count, 8192))
    for start in range(0, resample_count, chunk_size):
        stop = min(start + chunk_size, resample_count)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        bootstrap_effects[start:stop] = cluster_effects[indices].mean(axis=1)
    alpha = (1.0 - confidence_value) / 2.0
    quantiles = np.asarray(
        np.quantile(
            bootstrap_effects,
            [alpha, 1.0 - alpha],
            method=BOOTSTRAP_QUANTILE_METHOD,
        ),
        dtype=np.float64,
    )
    lower, upper = float(quantiles[0]), float(quantiles[1])

    return PairedClusterBootstrapResult(
        schema_version=PAIRED_BOOTSTRAP_SCHEMA_VERSION,
        metric_name=metric,
        reference_name=reference_label,
        candidate_name=candidate_label,
        effect=effect,
        interval=ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence=confidence_value,
            method="paired_percentile_cluster_bootstrap",
        ),
        counts=BootstrapCounts(
            paired_observations=pair_count,
            independent_clusters=cluster_count,
            reference_observations=len(reference),
            candidate_observations=len(candidate),
        ),
        config=PairedBootstrapConfig(
            seed=seed_value,
            resamples=resample_count,
            confidence=confidence_value,
            resampling_unit=unit,
        ),
        uncertainty_source=UncertaintySource.INDEPENDENT_CLUSTER_POPULATION,
        is_population_confidence_interval=False,
        inferential_status=InferentialStatus.DESCRIPTIVE_UNBOUND,
        observations_sha256=_observation_digest(reference, candidate),
        replay=BootstrapReplayMetadata(
            implementation_version=PAIRED_BOOTSTRAP_SCHEMA_VERSION,
            numpy_version=np.__version__,
            bit_generator=BOOTSTRAP_BIT_GENERATOR,
            quantile_method=BOOTSTRAP_QUANTILE_METHOD,
        ),
    )


def paired_cluster_bootstrap_from_registered_manifest(
    reference: Sequence[ClusteredMetricObservation],
    candidate: Sequence[ClusteredMetricObservation],
    *,
    registered_manifest: FrozenRegisteredSplitManifest,
    registry: DatasetRegistry,
    metric_name: str,
    resampling_unit: IndependentResamplingUnit | str | None = None,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedClusterBootstrapResult:
    """Bootstrap only observations bound to a frozen registered split.

    Every observation must name a selected manifest unit and its exact cluster;
    all selected units must be covered in both arms.  The statistical unit is
    derived from the registry/manifest contract, and any caller-supplied unit
    must match it exactly.  Only this registered path can promote a result to a
    population confidence interval after the cluster-count gate.
    """

    registered_manifest.validate_against_registry(registry)
    manifest = registered_manifest.manifest
    expected_unit = resampling_unit_from_registry(manifest.independent_unit)
    if resampling_unit is not None:
        requested_unit = _coerce_resampling_unit(resampling_unit)
        if requested_unit is not expected_unit:
            raise ValueError(
                "resampling_unit does not match registered manifest independent_unit: "
                f"{requested_unit.value!r} != {expected_unit.value!r}"
            )
    _validate_arm_against_manifest(reference, manifest, "reference")
    _validate_arm_against_manifest(candidate, manifest, "candidate")
    result = paired_cluster_bootstrap(
        reference,
        candidate,
        metric_name=metric_name,
        resampling_unit=expected_unit,
        reference_name=reference_name,
        candidate_name=candidate_name,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    population_ci_claimable = result.counts.independent_clusters >= MIN_POPULATION_CI_CLUSTERS
    return replace(
        result,
        registry_sha256=registered_manifest.registry_sha256,
        manifest_sha256=registered_manifest.manifest_sha256,
        is_population_confidence_interval=population_ci_claimable,
        inferential_status=(
            InferentialStatus.POPULATION_CONFIDENCE_INTERVAL
            if population_ci_claimable
            else InferentialStatus.DESCRIPTIVE_SMALL_CLUSTER
        ),
    )


def summarize_optimizer_seed_variation(
    values_by_seed: Mapping[int, float], *, metric_name: str
) -> OptimizerSeedVariationSummary:
    """Summarize optimizer-seed spread without constructing a population CI."""

    metric = _require_non_empty_label(metric_name, "metric_name")
    if not isinstance(values_by_seed, Mapping):
        raise TypeError("values_by_seed must be a mapping from integer seed to metric value")
    if len(values_by_seed) < 2:
        raise ValueError("at least two optimizer seeds are required to estimate seed variation")

    normalized: list[tuple[int, float]] = []
    for seed, value in values_by_seed.items():
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
            raise ValueError("optimizer seeds must be non-negative integers")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("optimizer-seed metric values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("optimizer-seed metric values must be finite")
        normalized.append((int(seed), numeric_value))
    normalized.sort()
    seeds = tuple(seed for seed, _ in normalized)
    if len(set(seeds)) != len(seeds):
        # Handles distinct NumPy/Python integer keys that normalize to one seed.
        raise ValueError("optimizer seeds must remain unique after integer normalization")
    values = np.asarray([value for _, value in normalized], dtype=np.float64)
    return OptimizerSeedVariationSummary(
        schema_version=SEED_VARIATION_SCHEMA_VERSION,
        metric_name=metric,
        mean=float(values.mean()),
        sample_standard_deviation=float(values.std(ddof=1)),
        minimum=float(values.min()),
        maximum=float(values.max()),
        seed_count=len(values),
        seeds=seeds,
        uncertainty_source=UncertaintySource.OPTIMIZER_SEED_VARIATION,
        interval=None,
        is_population_confidence_interval=False,
    )
