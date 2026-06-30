"""Governed official-mini TUM RGB-D regression with aggregate-only artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np
import torch

from jepa4d.benchmarks.geometry.tum_rgbd import (
    depth_metrics,
    load_depth,
    load_tum_indices,
    point_metrics,
    pose_metrics,
)
from jepa4d.data.rgb_input import from_view_sequences
from jepa4d.evaluation.failure_taxonomy import (
    EvidenceLevel,
    ExecutionStatus,
    ProtocolStatus,
    ScientificStatus,
    ValidationStage,
    ValidationStatus,
)
from jepa4d.models.geometry_belief import GeometryBeliefHead
from jepa4d.validation._content import load_yaml_unique, sha256_file, sha256_value, write_content_addressed_json
from jepa4d.validation.access import DatasetAccessController, GovernedAccessDecision
from jepa4d.validation.ledger import ConsumedTestLedger, ConsumptionEvent
from jepa4d.validation.registry import AccessOperation, DatasetRegistry
from jepa4d.validation.report_integration import ReportDatasetRef, build_governed_validation_report
from jepa4d.validation.wandb import SafeArtifactFile, publish_safe_online_run
from jepa4d.visualization.validation_dashboard import (
    ClaimBoundary,
    GateCondition,
    GateDomain,
    MetricDomain,
    MetricRecord,
    ResourcePolicy,
    write_immutable_validation_dashboard,
)

DATASET_ID = "tum-rgbd.geometry-regression"
SPLIT_ID = "tum-rgbd.phase2b-freiburg1-xyz-test"
OPERATION = AccessOperation.REGRESSION
ACCESS_RECEIPT_SCHEMA = "jepa4d-geometry-official-mini-access-v1"
METRIC_GATE_SCHEMA = "jepa4d-geometry-official-mini-metric-gate-v1"
EXECUTION_RECEIPT_SCHEMA = "jepa4d-geometry-official-mini-execution-v1"
EXPECTED_VALIDATION_FRAMES = 16
EXPECTED_TEST_FRAMES = 8
EXPECTED_QUALITY_METRICS = frozenset(
    {
        "aligned_abs_rel",
        "aligned_log_rmse",
        "aligned_rmse_m",
        "aligned_delta_1",
        "aligned_delta_2",
        "aligned_delta_3",
        "finite_fraction",
        "point_error_mean_m_aligned",
        "point_error_median_m_aligned",
        "point_within_10cm_fraction_aligned",
        "point_within_5cm_fraction_aligned",
        "pose_alignment_scale",
        "pose_ate_mean_m_sim3",
        "pose_ate_rmse_m_sim3",
        "pose_relative_translation_mean_m_sim3",
        "pose_rotation_mean_deg_sim3",
    }
)
EXPECTED_RESOURCE_METRICS = frozenset({"cuda_peak_memory_gib", "runtime_seconds"})
METRIC_UNITS: Mapping[str, str] = {
    "aligned_abs_rel": "ratio",
    "aligned_log_rmse": "log-depth",
    "aligned_rmse_m": "metres",
    "aligned_delta_1": "fraction",
    "aligned_delta_2": "fraction",
    "aligned_delta_3": "fraction",
    "finite_fraction": "fraction",
    "point_error_mean_m_aligned": "metres",
    "point_error_median_m_aligned": "metres",
    "point_within_10cm_fraction_aligned": "fraction",
    "point_within_5cm_fraction_aligned": "fraction",
    "pose_alignment_scale": "ratio",
    "pose_ate_mean_m_sim3": "metres",
    "pose_ate_rmse_m_sim3": "metres",
    "pose_relative_translation_mean_m_sim3": "metres",
    "pose_rotation_mean_deg_sim3": "degrees",
    "cuda_peak_memory_gib": "GiB",
    "runtime_seconds": "seconds",
}
FRACTION_METRICS = frozenset(
    {
        "aligned_delta_1",
        "aligned_delta_2",
        "aligned_delta_3",
        "finite_fraction",
        "point_within_10cm_fraction_aligned",
        "point_within_5cm_fraction_aligned",
    }
)
DEPTH_ALIGNMENT_PROTOCOL = "per-frame ratio-of-medians alignment using evaluated targets; no raw metric-scale score"
DEPTH_VALIDITY_PROTOCOL = (
    "finite target depth in (0.1,10.0) metres; every such pixel requires a finite strictly-positive prediction"
)
AGGREGATION_PROTOCOL = "mean pixels within frame, then equal mean across exactly eight frames from one recording"
SUPPORTED_CLAIMS = (
    "Aggregate geometry regression behavior on the registered, consumed TUM Phase 2b split.",
    "End-to-end registry, model, metric, dashboard, and online logging integration.",
)
PROHIBITED_CLAIMS = (
    "Fresh transfer, external confirmation, or cross-dataset generalization.",
    "Sample-level, raw-target, metric-scale, deployment, or speed-based promotion claims.",
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class GeometryOfficialMiniSettings:
    repo_root: Path
    registry_path: Path
    ledger_path: Path
    archive: Path
    model_id: Path
    output: Path
    execution_id: str
    run_name: str
    git_commit: str
    device: str = "cuda:0"
    wandb_entity: str | None = None
    wandb_project: str = "jepa4d-worldmodel"

    def __post_init__(self) -> None:
        for name in ("execution_id", "run_name", "wandb_project"):
            if not _SAFE_ID.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a path-safe identifier")
        if self.wandb_entity is not None and not _SAFE_ID.fullmatch(self.wandb_entity):
            raise ValueError("wandb_entity must be a path-safe identifier")
        if not re.fullmatch(r"[0-9a-f]{40}", self.git_commit):
            raise ValueError("git_commit must be a full lowercase commit SHA")
        if not self.device.startswith("cuda"):
            raise ValueError("governed official-mini execution requires CUDA")


@dataclass(frozen=True, slots=True)
class AuthorizedGeometryRun:
    registry: DatasetRegistry
    ledger: ConsumedTestLedger
    events: tuple[ConsumptionEvent, ...]
    decision: GovernedAccessDecision


@dataclass(frozen=True, slots=True)
class GeometryAggregateResult:
    quality_metrics: Mapping[str, float]
    resource_metrics: Mapping[str, float]
    evaluated_test_frames: int
    finite_fraction: float
    archive_sha256: str
    id_manifest_sha256: str
    model_identity_sha256: str

    def __post_init__(self) -> None:
        for name, values in (("quality_metrics", self.quality_metrics), ("resource_metrics", self.resource_metrics)):
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value))
                for value in values.values()
            ):
                raise ValueError(f"{name} must contain finite aggregate values")
        if set(self.quality_metrics) != EXPECTED_QUALITY_METRICS:
            raise ValueError("quality_metrics must contain the exact official-mini metric schema")
        if set(self.resource_metrics) != EXPECTED_RESOURCE_METRICS:
            raise ValueError("resource_metrics must contain the exact official-mini resource schema")
        if any(float(self.quality_metrics[name]) < 0.0 for name in EXPECTED_QUALITY_METRICS):
            raise ValueError("official-mini error and accuracy metrics cannot be negative")
        if any(not 0.0 <= float(self.quality_metrics[name]) <= 1.0 for name in FRACTION_METRICS):
            raise ValueError("official-mini fraction metrics must be within [0, 1]")
        if float(self.quality_metrics["pose_alignment_scale"]) <= 0.0:
            raise ValueError("official-mini pose alignment scale must be positive")
        if not (
            float(self.quality_metrics["aligned_delta_1"])
            <= float(self.quality_metrics["aligned_delta_2"])
            <= float(self.quality_metrics["aligned_delta_3"])
        ):
            raise ValueError("official-mini delta accuracies must be monotonic")
        if float(self.quality_metrics["point_within_5cm_fraction_aligned"]) > float(
            self.quality_metrics["point_within_10cm_fraction_aligned"]
        ):
            raise ValueError("5 cm point accuracy cannot exceed 10 cm point accuracy")
        if (
            float(self.quality_metrics["pose_ate_mean_m_sim3"])
            > float(self.quality_metrics["pose_ate_rmse_m_sim3"]) + 1e-12
        ):
            raise ValueError("pose ATE mean cannot exceed pose ATE RMSE")
        if any(float(value) < 0.0 for value in self.resource_metrics.values()):
            raise ValueError("official-mini resource diagnostics cannot be negative")
        if self.evaluated_test_frames != EXPECTED_TEST_FRAMES:
            raise ValueError(f"official-mini result must evaluate exactly {EXPECTED_TEST_FRAMES} test frames")
        if not math.isfinite(self.finite_fraction) or not 0.0 <= self.finite_fraction <= 1.0:
            raise ValueError("finite_fraction must be within [0, 1]")
        if float(self.quality_metrics["finite_fraction"]) != float(self.finite_fraction):
            raise ValueError("quality finite_fraction must match the typed result field")
        for name in ("archive_sha256", "id_manifest_sha256", "model_identity_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


Evaluator = Callable[[GeometryOfficialMiniSettings, AuthorizedGeometryRun], GeometryAggregateResult]
Publisher = Callable[..., dict[str, Any]]


def authorize_tum_regression(settings: GeometryOfficialMiniSettings) -> AuthorizedGeometryRun:
    """Authorize the exact consumed split without touching any supplied data path."""

    registry = DatasetRegistry.load(settings.registry_path)
    ledger = ConsumedTestLedger.load(settings.ledger_path)
    controller = DatasetAccessController(registry=registry, ledger=ledger)
    decision = controller.authorize(DATASET_ID, SPLIT_ID, OPERATION)
    if (
        decision.dataset_id != DATASET_ID
        or decision.split_id != SPLIT_ID
        or decision.operation is not OPERATION
        or decision.grants_data_access is not True
    ):
        raise RuntimeError("registry controller returned an unexpected TUM regression decision")
    return AuthorizedGeometryRun(registry, ledger, tuple(controller.events), decision)


def _hash_model_tree(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("local model directory contains no files")
    identities = []
    for path in files:
        if path.is_symlink():
            raise ValueError("local model directory may not contain symbolic links")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        final = path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields) or any(
            getattr(after, field) != getattr(final, field) for field in fields
        ):
            raise ValueError("local model file changed while its identity was computed")
        identities.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": after.st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return sha256_value(identities)


def _verify_open_archive(handle: BinaryIO, manifest: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    expected = manifest.get("archive")
    if not isinstance(expected, Mapping):
        raise ValueError("registered TUM manifest lacks its archive identity")
    before = os.fstat(handle.fileno())
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    after = os.fstat(handle.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    identity = tuple(int(getattr(after, field)) for field in fields)
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError("registered TUM archive changed while it was verified")
    if after.st_size != expected.get("bytes") or digest.hexdigest() != expected.get("sha256"):
        raise ValueError("registered TUM archive differs from its governed byte identity")
    handle.seek(0)
    return digest.hexdigest(), identity


def _require_open_file_identity(handle: BinaryIO, expected: tuple[int, ...], label: str) -> None:
    stat = os.fstat(handle.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(int(getattr(stat, field)) for field in fields) != expected:
        raise ValueError(f"{label} changed during governed evaluation")


def _ground_truth_intrinsics(manifest: Mapping[str, Any], output_size: tuple[int, int]) -> torch.Tensor:
    camera = manifest["camera"]
    height, width = output_size
    return torch.tensor(
        [
            [camera["fx"] * width / camera["width"], 0.0, camera["cx"] * width / camera["width"]],
            [0.0, camera["fy"] * height / camera["height"], camera["cy"] * height / camera["height"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _mean_metrics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate an empty metric set")
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        raise ValueError("per-frame metric schemas differ")
    return {key: float(np.mean([value[key] for value in values])) for key in sorted(keys)}


@contextmanager
def _verified_tum_extraction(archive: BinaryIO, manifest: Mapping[str, Any]) -> Iterator[Path]:
    """Extract only the hash-verified official archive into ephemeral node-local storage."""

    sequence = manifest.get("sequence")
    if sequence != "rgbd_dataset_freiburg1_xyz":
        raise ValueError("unexpected TUM sequence root in the registered manifest")
    with tempfile.TemporaryDirectory(prefix="jepa4d-tum-official-mini-") as temporary:
        destination = Path(temporary)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            if not members:
                raise ValueError("registered TUM archive is empty")
            total_bytes = 0
            for member in members:
                member_path = PurePosixPath(member.name)
                if (
                    member_path.is_absolute()
                    or not member_path.parts
                    or any(part in {"", ".", ".."} for part in member_path.parts)
                    or member_path.parts[0] != sequence
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"unsafe member in the registered TUM archive: {member.name!r}")
                total_bytes += member.size
            if total_bytes <= 0 or total_bytes > 4 * 1024**3:
                raise ValueError("registered TUM extraction size is outside the frozen safety envelope")
            bundle.extractall(destination, members=members, filter="data")
        extracted = (destination / sequence).resolve(strict=True)
        if not extracted.is_dir():
            raise ValueError("registered TUM archive lacks its expected sequence directory")
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            path = extracted / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"registered TUM extraction lacks a safe {name}")
        yield extracted


def _contained_regular_file(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or root not in resolved.parents:
        raise ValueError("TUM index resolved outside the verified ephemeral extraction")
    return resolved


def _require_complete_target_support(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError("TUM prediction and target grids must match")
    target_valid = torch.isfinite(target) & (target > 0.1) & (target < 10.0)
    if int(target_valid.sum()) < 100:
        raise ValueError("TUM frame contains fewer than 100 valid target pixels")
    supported = torch.isfinite(predicted[target_valid]) & (predicted[target_valid] > 0)
    if not bool(supported.all()):
        raise ValueError("every valid TUM target pixel requires a finite strictly-positive prediction")


def _evaluate_extracted_tum(
    settings: GeometryOfficialMiniSettings,
    manifest: Mapping[str, Any],
    dataset_root: Path,
    test_indices: list[int],
    model_root: Path,
) -> tuple[dict[str, float], dict[str, float], float, int]:
    samples = load_tum_indices(dataset_root, test_indices)
    rgb_paths = [_contained_regular_file(sample.rgb_path, dataset_root) for sample in samples]
    depth_paths = [_contained_regular_file(sample.depth_path, dataset_root) for sample in samples]
    timestamps = torch.tensor([[sample.timestamp for sample in samples]], dtype=torch.float64)
    batch = from_view_sequences([rgb_paths], timestamps=timestamps)
    camera = manifest["camera"]
    batch.intrinsics = torch.tensor(
        [[[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    head = GeometryBeliefHead(backend="vggt", device=settings.device, model_id=str(model_root), precision="float32")
    belief = head(batch)
    if belief.depth_mean is None or belief.camera_extrinsics is None or belief.camera_intrinsics is None:
        raise ValueError("geometry backend omitted required official-mini outputs")
    expected_prefix = (1, 1, EXPECTED_TEST_FRAMES)
    if (
        tuple(belief.depth_mean.shape[:3]) != expected_prefix
        or tuple(belief.camera_extrinsics.shape[:3]) != expected_prefix
        or tuple(belief.camera_intrinsics.shape[:3]) != expected_prefix
        or belief.depth_mean.ndim != 5
        or belief.camera_extrinsics.shape[-2:] != (4, 4)
        or belief.camera_intrinsics.shape[-2:] != (3, 3)
    ):
        raise ValueError("geometry backend output cardinality does not match the exact eight-frame smoke")
    predictions = belief.depth_mean[0, 0].detach().cpu()
    output_resolution = tuple(predictions.shape[-2:])
    targets = torch.stack(
        [
            load_depth(path, output_resolution, depth_scale=sample.depth_scale)
            for path, sample in zip(depth_paths, samples, strict=True)
        ]
    )
    intrinsics = _ground_truth_intrinsics(manifest, output_resolution)
    frame_metrics: list[dict[str, float]] = []
    for index in range(len(samples)):
        _require_complete_target_support(predictions[index], targets[index])
        metrics, _, _ = depth_metrics(predictions[index], targets[index])
        metrics.update(point_metrics(predictions[index], targets[index], intrinsics))
        frame_metrics.append(
            {
                "aligned_abs_rel": metrics["abs_rel"],
                "aligned_rmse_m": metrics["rmse_m"],
                "aligned_log_rmse": metrics["log_rmse"],
                "aligned_delta_1": metrics["delta_1"],
                "aligned_delta_2": metrics["delta_2"],
                "aligned_delta_3": metrics["delta_3"],
                "point_error_mean_m_aligned": metrics["point_error_mean_m_aligned"],
                "point_error_median_m_aligned": metrics["point_error_median_m_aligned"],
                "point_within_5cm_fraction_aligned": metrics["point_fscore_5cm_aligned"],
                "point_within_10cm_fraction_aligned": metrics["point_fscore_10cm_aligned"],
            }
        )
    quality = _mean_metrics(frame_metrics)
    quality.update(pose_metrics(belief.camera_extrinsics[0, 0], samples))
    finite_fraction = float(torch.isfinite(predictions).float().mean())
    quality["finite_fraction"] = finite_fraction
    try:
        resources = {
            "runtime_seconds": float(belief.metadata["runtime_seconds"]),
            "cuda_peak_memory_gib": float(belief.metadata["cuda_peak_memory_bytes"]) / 1024**3,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("geometry backend omitted required resource telemetry") from error
    return quality, resources, finite_fraction, len(samples)


def evaluate_tum_regression(
    settings: GeometryOfficialMiniSettings,
    authorized: AuthorizedGeometryRun,
) -> GeometryAggregateResult:
    """Read the registered data only after an access decision exists and return aggregates."""

    _, split = authorized.registry.split(DATASET_ID, SPLIT_ID)
    if split.id_manifest is None or split.id_manifest_sha256 is None:
        raise ValueError("registered TUM regression split lacks its ID-manifest identity")

    repo_root = settings.repo_root.resolve(strict=True)
    manifest_path = (repo_root / split.id_manifest).resolve(strict=True)
    if repo_root not in manifest_path.parents or sha256_file(manifest_path) != split.id_manifest_sha256:
        raise ValueError("registered TUM ID manifest differs from its governed identity")
    archive = settings.archive.resolve(strict=True)
    model_root = settings.model_id.resolve(strict=True)
    if not archive.is_file() or not model_root.is_dir():
        raise ValueError("governed TUM archive/model inputs must be local files/directories")

    manifest_value = load_yaml_unique(manifest_path)
    if not isinstance(manifest_value, Mapping):
        raise ValueError("registered TUM Phase 2b manifest must be a mapping")
    manifest = manifest_value
    if manifest.get("dataset_id") != "tum-rgbd-fr1-xyz-phase2b" or manifest.get("official") is not True:
        raise ValueError("unexpected registered TUM Phase 2b manifest")
    train_indices = [int(value) for value in manifest.get("train_indices", [])]
    validation_indices = [int(value) for value in manifest.get("validation_indices", [])]
    test_indices = [int(value) for value in manifest.get("test_indices", [])]
    partitions = (train_indices, validation_indices, test_indices)
    if (
        len(train_indices) != 64
        or len(validation_indices) != EXPECTED_VALIDATION_FRAMES
        or len(test_indices) != EXPECTED_TEST_FRAMES
        or any(len(values) != len(set(values)) for values in partitions)
        or any(set(partitions[left]) & set(partitions[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        or manifest.get("association_max_delta_seconds") != 0.03
        or manifest.get("depth_scale") != 5000.0
        or manifest.get("split_policy") != "chronological-nonoverlapping-frame-ranges"
    ):
        raise ValueError("registered TUM Phase 2b split/association/depth protocol changed")
    # This decision authorizes the registered test split only. Validation IDs
    # are checked for manifest integrity/disjointness but are never decoded.
    model_identity = _hash_model_tree(model_root)
    with archive.open("rb") as archive_stream:
        archive_sha256, archive_identity = _verify_open_archive(archive_stream, manifest)
        with _verified_tum_extraction(archive_stream, manifest) as dataset_root:
            quality, resources, finite_fraction, evaluated_frames = _evaluate_extracted_tum(
                settings,
                manifest,
                dataset_root,
                test_indices,
                model_root,
            )
        _require_open_file_identity(archive_stream, archive_identity, "registered TUM archive")
    if _hash_model_tree(model_root) != model_identity:
        raise ValueError("local model identity changed during governed evaluation")
    return GeometryAggregateResult(
        quality_metrics=quality,
        resource_metrics=resources,
        evaluated_test_frames=evaluated_frames,
        finite_fraction=finite_fraction,
        archive_sha256=archive_sha256,
        id_manifest_sha256=split.id_manifest_sha256,
        model_identity_sha256=model_identity,
    )


def _prepare_fresh_output(output: Path) -> Path:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("official-mini output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


def _metric_records(result: GeometryAggregateResult) -> tuple[MetricRecord, ...]:
    quality = tuple(
        MetricRecord(name, float(value), METRIC_UNITS[name], MetricDomain.QUALITY, SPLIT_ID)
        for name, value in sorted(result.quality_metrics.items())
    )
    resources = tuple(
        MetricRecord(name, float(value), METRIC_UNITS[name], MetricDomain.RESOURCE, SPLIT_ID)
        for name, value in sorted(result.resource_metrics.items())
    )
    return quality + resources


def run_governed_geometry_official_mini(
    settings: GeometryOfficialMiniSettings,
    *,
    evaluator: Evaluator = evaluate_tum_regression,
    publisher: Publisher = publish_safe_online_run,
) -> dict[str, Any]:
    """Run the authorized regression and publish only aggregate governed artifacts."""

    output = _prepare_fresh_output(settings.output)
    authorized = authorize_tum_regression(settings)
    access_payload = {
        "schema_version": ACCESS_RECEIPT_SCHEMA,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "operation": OPERATION,
        "execution_id": settings.execution_id,
        "git_commit": settings.git_commit,
        "decision": authorized.decision.model_dump(mode="json", exclude_none=True),
    }
    access_receipt = write_content_addressed_json(access_payload, output / "governance", prefix="access-decision")

    result = evaluator(settings, authorized)
    conditions = (
        GateCondition(
            "all_aggregate_metrics_finite",
            GateDomain.INTEGRITY,
            all(math.isfinite(float(value)) for value in result.quality_metrics.values()),
            "Every registered aggregate quality metric is finite.",
        ),
        GateCondition(
            "complete_registered_test_frames",
            GateDomain.INTEGRITY,
            result.evaluated_test_frames == EXPECTED_TEST_FRAMES,
            f"The registered {EXPECTED_TEST_FRAMES}-frame regression subset is fully evaluated.",
        ),
        GateCondition(
            "finite_geometry_outputs",
            GateDomain.INTEGRITY,
            result.finite_fraction == 1.0,
            "All evaluated geometry predictions are finite.",
        ),
    )
    gate_passed = all(condition.passed for condition in conditions)
    metric_gate_payload = {
        "schema_version": METRIC_GATE_SCHEMA,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "operation": OPERATION,
        "execution_id": settings.execution_id,
        "git_commit": settings.git_commit,
        "registry_sha256": authorized.registry.sha256,
        "ledger_sha256": authorized.ledger.sha256,
        "access_decision_sha256": access_receipt.sha256,
        "id_manifest_sha256": result.id_manifest_sha256,
        "archive_sha256": result.archive_sha256,
        "model_identity_sha256": result.model_identity_sha256,
        "depth_alignment_protocol": DEPTH_ALIGNMENT_PROTOCOL,
        "depth_validity_protocol": DEPTH_VALIDITY_PROTOCOL,
        "aggregation_protocol": AGGREGATION_PROTOCOL,
        "evaluated_test_frames": result.evaluated_test_frames,
        "quality_metrics": dict(sorted(result.quality_metrics.items())),
        "resource_metrics": dict(sorted(result.resource_metrics.items())),
        "gate_conditions": {condition.name: condition.passed for condition in conditions},
        "gate_outcome": "pass" if gate_passed else "fail",
    }
    metric_gate = write_content_addressed_json(metric_gate_payload, output / "metrics", prefix="metric-gate")
    status = ValidationStatus(
        experiment_id=settings.execution_id,
        stage=ValidationStage.GEOMETRY,
        protocol=ProtocolStatus.FROZEN,
        execution=ExecutionStatus.COMPLETE,
        scientific=ScientificStatus.NOT_APPLICABLE,
        evidence_level=EvidenceLevel.OFFICIAL_SMOKE,
        expected_cells=1,
        successful_cells=1,
        metric_gate_receipt_sha256=metric_gate.sha256,
    )
    report = build_governed_validation_report(
        registry=authorized.registry,
        ledger=authorized.ledger,
        events=authorized.events,
        status=status,
        dataset_splits=(ReportDatasetRef(DATASET_ID, SPLIT_ID),),
        report_id=settings.execution_id,
        title="Governed TUM RGB-D geometry official smoke",
        gate_name="scientific-promotion-not-applicable",
        gate_decision=(
            "The aggregate-only regression integrity smoke passed; no scientific promotion decision is applicable."
            if gate_passed
            else "The aggregate-only regression integrity smoke failed; no scientific promotion decision is applicable."
        ),
        claim_boundary=ClaimBoundary(
            supported=SUPPORTED_CLAIMS,
            prohibited=PROHIBITED_CLAIMS,
        ),
        metrics=_metric_records(result),
        gate_conditions=conditions,
        resource_policy=ResourcePolicy.DIAGNOSTIC_ONLY,
        timestamp=datetime.now(UTC).isoformat(),
    )
    dashboard = write_immutable_validation_dashboard(report, output / "dashboard")
    upload_files = (
        SafeArtifactFile(access_receipt.path, "governance-receipt"),
        SafeArtifactFile(metric_gate.path, "aggregate-receipt"),
        SafeArtifactFile(dashboard.json_path, "dashboard-json"),
        SafeArtifactFile(dashboard.html_path, "dashboard-html"),
        SafeArtifactFile(dashboard.receipt_path, "dashboard-receipt"),
    )
    wandb_receipt_payload = publisher(
        entity=settings.wandb_entity,
        project=settings.wandb_project,
        group=settings.execution_id,
        job_type="geometry-official-mini",
        run_name=settings.run_name,
        config={
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "operation": OPERATION.value,
            "execution_id": settings.execution_id,
            "git_commit": settings.git_commit,
            "registry_sha256": authorized.registry.sha256,
            "ledger_sha256": authorized.ledger.sha256,
            "id_manifest_sha256": result.id_manifest_sha256,
            "model_identity_sha256": result.model_identity_sha256,
            "depth_alignment_protocol": DEPTH_ALIGNMENT_PROTOCOL,
            "depth_validity_protocol": DEPTH_VALIDITY_PROTOCOL,
            "aggregation_protocol": AGGREGATION_PROTOCOL,
        },
        summary=report.wandb_summary_payload(),
        artifact_name=f"geometry-official-mini-{settings.execution_id}",
        artifact_root=output,
        files=upload_files,
    )
    wandb_receipt = write_content_addressed_json(
        wandb_receipt_payload,
        output / "wandb",
        prefix="wandb-receipt",
    )
    execution_payload = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "status": "pass" if gate_passed else "fail",
        "execution_id": settings.execution_id,
        "git_commit": settings.git_commit,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "operation": OPERATION,
        "registry_sha256": authorized.registry.sha256,
        "ledger_sha256": authorized.ledger.sha256,
        "access_decision_sha256": access_receipt.sha256,
        "metric_gate_sha256": metric_gate.sha256,
        "validation_status": status.to_serializable(),
        "validation_status_sha256": sha256_value(status.to_serializable()),
        "dashboard_generation_id": dashboard.generation_id,
        "wandb_receipt_sha256": wandb_receipt.sha256,
        "wandb_run_id": wandb_receipt_payload["run_id"],
        "wandb_entity": wandb_receipt_payload["entity"],
        "wandb_project": wandb_receipt_payload["project"],
        "wandb_group": wandb_receipt_payload["group"],
        "wandb_job_type": wandb_receipt_payload["job_type"],
        "wandb_run_name": wandb_receipt_payload["run_name"],
        "wandb_artifact_name": wandb_receipt_payload["artifact_name"],
        "wandb_artifact_id": wandb_receipt_payload["artifact_id"],
        "wandb_artifact_digest": wandb_receipt_payload["artifact_digest"],
        "wandb_config_sha256": wandb_receipt_payload["config_sha256"],
        "wandb_summary_sha256": wandb_receipt_payload["summary_sha256"],
    }
    execution_receipt = write_content_addressed_json(
        execution_payload,
        output / "execution",
        prefix="execution-receipt",
    )
    return {
        "status": execution_payload["status"],
        "execution_id": settings.execution_id,
        "execution_receipt_sha256": execution_receipt.sha256,
        "dashboard_generation_id": dashboard.generation_id,
        "wandb_run_id": wandb_receipt_payload["run_id"],
    }


def _clean_git_commit(repo_root: Path) -> str:
    commit = subprocess.check_output(("git", "-C", str(repo_root), "rev-parse", "HEAD"), text=True).strip()
    status = subprocess.check_output(
        ("git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"), text=True
    )
    if status.strip():
        raise RuntimeError("governed official-mini execution requires a clean committed worktree")
    return commit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", dest="registry_path", type=Path, required=True)
    parser.add_argument("--ledger", dest="ledger_path", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve(strict=True)
    canonical_repo = Path(__file__).resolve().parents[2]
    if repo_root != canonical_repo:
        raise RuntimeError("governed official smoke must use the canonical repository checkout")
    registry_path = args.registry_path.resolve(strict=True)
    ledger_path = args.ledger_path.resolve(strict=True)
    if registry_path != repo_root / "configs/validation/dataset_registry.yaml" or ledger_path != repo_root / (
        "configs/validation/consumed_test_ledger.yaml"
    ):
        raise RuntimeError("governed official smoke requires the canonical registry and ledger")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", slurm_job_id):
        raise RuntimeError("governed official smoke must run inside a Slurm allocation")
    submitted_commit = os.environ.get("JEPA4D_GIT_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", submitted_commit):
        raise RuntimeError("governed official smoke requires the submitter's full Git commit")
    subprocess.run(
        (
            sys.executable,
            str(repo_root / "slurm/validate_geometry_official_mini.py"),
            "--job-id",
            slurm_job_id,
            "--allocation-only",
        ),
        check=True,
    )
    current_commit = _clean_git_commit(repo_root)
    if current_commit != submitted_commit:
        raise RuntimeError("repository commit changed after governed Slurm submission")
    settings = GeometryOfficialMiniSettings(
        repo_root=repo_root,
        registry_path=registry_path,
        ledger_path=ledger_path,
        archive=args.archive,
        model_id=args.model_id,
        output=args.output,
        execution_id=args.execution_id,
        run_name=args.run_name,
        git_commit=submitted_commit,
        device=args.device,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    result = run_governed_geometry_official_mini(settings)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
