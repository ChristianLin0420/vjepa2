"""Final, test-only Phase-2e evaluator for factorized metric depth probes."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from jepa4d.evaluation.phase2e_feature_cache import (
    CACHE_SCHEMA,
    RECEIPT_SCHEMA,
    sha256_file,
    validate_cache_payload,
)
from jepa4d.models.factorized_geometry import (
    FactorizedGeometryConfig,
    FactorizedGeometryOutput,
    FactorizedShapeScaleGeometryProbe,
)

EVALUATION_SCHEMA = "jepa4d-phase2e-final-evaluation-v1"
ARTIFACT_MANIFEST_SCHEMA = "jepa4d-phase2e-final-artifact-manifest-v1"
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2e-final-wandb-receipt-v1"
SHARD_SCHEMA = "jepa4d-phase2e-training-shard-v1"
SHARD_MANIFEST_SCHEMA = "jepa4d-phase2e-artifact-manifest-v1"
SHARD_WANDB_RECEIPT_SCHEMA = "jepa4d-phase2e-wandb-artifact-receipt-v1"
CHECKPOINT_SCHEMA = "jepa4d-phase2e-checkpoint-v1"
VALIDATION_PREDICTION_SCHEMA = "jepa4d-phase2e-validation-predictions-v1"
VALIDATION_METRICS_SCHEMA = "jepa4d-phase2e-validation-v1"
FEATURE_WANDB_RECEIPT_SCHEMA = "jepa4d-phase2e-cache-wandb-receipt-v1"

VARIANTS = (
    "monolithic_final",
    "factorized_bias",
    "factorized_vjepa",
    "factorized_rgb",
    "factorized_vjepa_rgb",
    "factorized_vjepa_k",
    "factorized_full",
    "factorized_full_teacher",
)
FORMAL_SHARD_GROUPS = (
    ("monolithic_final", "factorized_bias"),
    ("factorized_vjepa", "factorized_rgb"),
    ("factorized_vjepa_rgb", "factorized_vjepa_k"),
    ("factorized_full", "factorized_full_teacher"),
)
FORMAL_OPTIMIZER = {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0.0001, "gradient_clip": 5.0}
FORMAL_LOSS_WEIGHTS = {
    "geometry_probe": 1.0,
    "global_log_scale": 1.0,
    "centered_gt_shape": 0.25,
    "centered_teacher": 0.25,
    "paired_scale_consistency": 0.1,
}
SEEDS = (0, 1, 2)
CANDIDATE = "factorized_full_teacher"
BASELINE = "monolithic_final"
K_CONTROLS = ("correct", "wrong", "shuffled")
NOMINAL_COVERAGES = (0.50, 0.68, 0.80, 0.90, 0.95)
RISK_COVERAGES = tuple(float(value) for value in np.linspace(0.1, 1.0, 10))


@dataclass(slots=True)
class EvaluationSplit:
    name: str
    features: torch.Tensor
    rgb: torch.Tensor
    intrinsics: torch.Tensor
    targets: torch.Tensor
    sample_ids: list[str]
    sensor_ids: list[str]

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


@dataclass(slots=True)
class CheckpointRecord:
    variant: str
    seed: int
    shard_dir: Path
    checkpoint_path: Path
    validation_prediction_path: Path
    config: FactorizedGeometryConfig
    use_teacher: bool
    variance_multiplier: float
    parameter_count: int
    best_epoch: int
    best_validation_abs_rel: float

    @property
    def k_conditioned(self) -> bool:
        return self.config.camera_mode == "known_rays" or bool(
            {"intrinsics", "ray_summary"} & set(self.config.scale_inputs)
        )


@dataclass(slots=True)
class PredictionBundle:
    log_depth: torch.Tensor
    log_variance: torch.Tensor
    global_log_scale: torch.Tensor | None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value == "None":
        raise ValueError(f"{label} must be a non-empty durable identifier")
    return value


def _inside(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its formal shard directory: {path}") from error
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def _prepare_output(path: Path) -> Path:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be new and empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _split_from_payload(name: str, value: Mapping[str, Any]) -> EvaluationSplit:
    return EvaluationSplit(
        name=name,
        features=value["features"].detach().cpu().contiguous(),
        rgb=value["rgb"].detach().cpu().contiguous(),
        intrinsics=value["intrinsics_384"].detach().cpu().contiguous(),
        targets=value["targets"].detach().cpu().contiguous(),
        sample_ids=[str(item) for item in value["sample_ids"]],
        sensor_ids=[str(item) for item in value["sensor_ids"]],
    )


def _verify_feature_wandb_receipt(feature_receipt_path: Path, feature_receipt: Mapping[str, Any]) -> None:
    path = feature_receipt_path.parent / "wandb_receipt.json"
    receipt = _load_json(path)
    if receipt.get("schema_version") != FEATURE_WANDB_RECEIPT_SCHEMA:
        raise ValueError("unexpected feature-cache W&B receipt schema")
    if receipt.get("status") != "uploaded" or receipt.get("mode") != "online":
        raise ValueError("feature-cache W&B receipt is not a completed online upload")
    for key in ("run_id", "artifact_id", "artifact_qualified_name", "artifact_digest"):
        _require_nonempty_string(receipt.get(key), f"feature W&B {key}")
    if receipt.get("receipt_sha256") != sha256_file(feature_receipt_path):
        raise ValueError("feature-cache W&B receipt does not bind feature_cache_receipt.json")
    report = Path(str(feature_receipt["report"]["path"])).resolve(strict=True)
    if receipt.get("report_sha256") != sha256_file(report):
        raise ValueError("feature-cache report hash differs from its W&B receipt")


def verify_feature_inputs_before_test(
    train_validation_cache: Path,
    test_cache: Path,
    feature_receipt_path: Path,
    *,
    require_formal_protocol: bool = True,
) -> tuple[EvaluationSplit, dict[str, Any], str, str]:
    """Verify all feature evidence without deserializing the isolated test cache."""
    train_path = train_validation_cache.resolve(strict=True)
    test_path = test_cache.resolve(strict=True)
    receipt_path = feature_receipt_path.resolve(strict=True)
    if train_path == test_path:
        raise ValueError("train/validation and test caches must be physically separate files")
    train_sha, test_sha = sha256_file(train_path), sha256_file(test_path)
    if train_sha == test_sha:
        raise ValueError("physically separate cache files unexpectedly have identical content hashes")

    payload = torch.load(train_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("train/validation cache root must be a mapping")
    validate_cache_payload(payload, expected_splits={"train", "validation"})
    validation = _split_from_payload("validation", payload["splits"]["validation"])
    train_ids = {str(item) for item in payload["splits"]["train"]["sample_ids"]}
    if train_ids & set(validation.sample_ids):
        raise ValueError("train and validation IDs overlap")

    receipt = _load_json(receipt_path)
    required = {
        "schema_version",
        "status",
        "evidence_level",
        "dataset",
        "models",
        "view_policy",
        "feature_normalization",
        "teacher_policy",
        "caches",
        "split_summaries",
        "sample_metadata",
        "profiles",
        "runtime",
        "wandb_url",
        "model_metrics_computed",
        "large_caches_uploaded_to_wandb",
        "report",
    }
    if not required <= set(receipt) or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("feature-cache receipt is incomplete or has an unexpected schema")
    if receipt.get("status") != "pass" or receipt.get("evidence_level") != "feature-cache-build":
        raise ValueError("feature-cache receipt is not a passing build receipt")
    if (
        receipt.get("model_metrics_computed") is not False
        or receipt.get("large_caches_uploaded_to_wandb") is not False
    ):
        raise ValueError("feature-cache receipt violates the frozen no-test-metrics/no-large-upload policy")
    if require_formal_protocol:
        from slurm.validate_phase2e_cache import validate_phase2e_cache

        dataset = receipt.get("dataset")
        if not isinstance(dataset, Mapping) or not isinstance(dataset.get("manifest"), str):
            raise ValueError("formal feature-cache receipt has no frozen manifest path")
        strict = validate_phase2e_cache(receipt_path.parent, Path(dataset["manifest"]))
        if strict.get("train_validation_cache_sha256") != train_sha or strict.get("test_cache_sha256") != test_sha:
            raise ValueError("strict feature-cache postflight disagrees with final evaluator identities")
    if set(receipt["caches"]) != {"train_validation", "test"}:
        raise ValueError("feature-cache receipt must bind exactly train_validation and test caches")
    for name, path, digest, splits in (
        ("train_validation", train_path, train_sha, ["train", "validation"]),
        ("test", test_path, test_sha, ["test"]),
    ):
        record = receipt["caches"][name]
        if Path(str(record["path"])).resolve() != path:
            raise ValueError(f"feature-cache receipt path mismatch for {name}")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != digest:
            raise ValueError(f"feature-cache receipt size/hash mismatch for {name}")
        if record.get("schema_version") != CACHE_SCHEMA or record.get("splits") != splits:
            raise ValueError(f"feature-cache receipt schema/split mismatch for {name}")
    if set(receipt["sample_metadata"]) != {"train", "validation", "test"}:
        raise ValueError("feature-cache sample metadata must cover exactly all three splits")
    for name, expected_ids in (
        ("train", train_ids),
        ("validation", set(validation.sample_ids)),
    ):
        metadata = receipt["sample_metadata"][name]
        found = {str(row["sample_id"]) for row in metadata}
        if found != expected_ids or len(found) != len(metadata):
            raise ValueError(f"feature receipt {name} sample metadata differs from the cache")
    _verify_feature_wandb_receipt(receipt_path, receipt)
    return validation, receipt, train_sha, test_sha


def load_verified_test_cache(
    test_cache: Path,
    feature_receipt: Mapping[str, Any],
    forbidden_ids: set[str],
) -> EvaluationSplit:
    """Open test only after all pre-test evidence has passed verification."""
    payload = torch.load(test_cache.resolve(strict=True), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("test cache root must be a mapping")
    validate_cache_payload(payload, expected_splits={"test"})
    split = _split_from_payload("test", payload["splits"]["test"])
    if split.size < 2:
        raise ValueError("test cache requires at least two examples for the shuffled-K control")
    if set(split.sample_ids) & forbidden_ids:
        raise ValueError("test sample IDs overlap train/validation")
    if set(split.sensor_ids) != {"kv2"}:
        raise ValueError(f"formal Phase2e test must be untouched kv2 only, found {sorted(set(split.sensor_ids))}")
    metadata = feature_receipt["sample_metadata"]["test"]
    if [str(row["sample_id"]) for row in metadata] != split.sample_ids:
        raise ValueError("test cache order differs from feature receipt sample metadata")
    if [str(row["sensor_id"]) for row in metadata] != split.sensor_ids:
        raise ValueError("test sensor IDs differ from feature receipt")
    if any(row.get("views", [{}])[0].get("view_name") != "center_square" for row in metadata):
        raise ValueError("formal test cache contains a non-original crop view")
    return split


def _variant_config(name: str, hidden_dim: int) -> tuple[FactorizedGeometryConfig, bool]:
    definitions: dict[str, tuple[str, str, tuple[str, ...], bool]] = {
        "monolithic_final": ("monolithic", "none", (), False),
        "factorized_bias": ("factorized", "none", (), False),
        "factorized_vjepa": ("factorized", "none", ("vjepa",), False),
        "factorized_rgb": ("factorized", "none", ("rgb",), False),
        "factorized_vjepa_rgb": ("factorized", "none", ("vjepa", "rgb"), False),
        "factorized_vjepa_k": (
            "factorized",
            "known_rays",
            ("vjepa", "intrinsics", "ray_summary"),
            False,
        ),
        "factorized_full": (
            "factorized",
            "known_rays",
            ("vjepa", "rgb", "intrinsics", "ray_summary"),
            False,
        ),
        "factorized_full_teacher": (
            "factorized",
            "known_rays",
            ("vjepa", "rgb", "intrinsics", "ray_summary"),
            True,
        ),
    }
    mode, camera_mode, scale_inputs, teacher = definitions[name]
    config = FactorizedGeometryConfig(
        input_dim=768,
        hidden_dim=hidden_dim,
        mode=mode,  # type: ignore[arg-type]
        camera_mode=camera_mode,  # type: ignore[arg-type]
        scale_inputs=scale_inputs,  # type: ignore[arg-type]
    )
    return config, teacher


def _config_from_payload(value: Mapping[str, Any]) -> FactorizedGeometryConfig:
    config = dict(value)
    config["scale_inputs"] = tuple(config["scale_inputs"])
    return FactorizedGeometryConfig(**config)


def fit_log_variance_multiplier(prediction: Mapping[str, Any], validation: EvaluationSplit) -> float:
    if prediction.get("schema_version") != VALIDATION_PREDICTION_SCHEMA:
        raise ValueError("unexpected validation prediction schema")
    if prediction.get("sample_ids") != validation.sample_ids or prediction.get("sensor_ids") != validation.sensor_ids:
        raise ValueError("saved validation prediction identities differ from validation cache")
    predicted = prediction.get("prediction_m")
    target = prediction.get("target_m")
    log_variance = prediction.get("log_variance")
    expected = tuple(validation.targets.shape)
    if not all(
        isinstance(value, torch.Tensor) and tuple(value.shape) == expected
        for value in (predicted, target, log_variance)
    ):
        raise ValueError("saved validation prediction tensors have unexpected shapes")
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(log_variance, torch.Tensor)
    if not torch.equal(target.cpu(), validation.targets):
        raise ValueError("saved validation targets differ from the immutable validation cache")
    valid = torch.isfinite(target) & (target > 0.1) & (target < 10.0)
    if not bool(valid.any()) or not torch.isfinite(predicted).all() or not bool((predicted > 0).all()):
        raise ValueError("saved validation predictions are incomplete or invalid")
    residual = predicted.float().clamp_min(1e-8).log() - target.float().clamp_min(1e-8).log()
    variance = log_variance.float().exp().clamp_min(1e-8)
    multiplier = float((residual.square() / variance)[valid].mean().clamp(1e-4, 1e4))
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("validation-fitted variance multiplier is invalid")
    return multiplier


def _verify_shard_manifest(shard_dir: Path, shard: Mapping[str, Any]) -> None:
    path = shard_dir / "artifact_manifest.json"
    manifest = _load_json(path)
    if manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected artifact manifest schema in {shard_dir}")
    if manifest.get("config_sha256") != shard.get("config_sha256") or manifest.get("selection_split") != "validation":
        raise ValueError(f"artifact manifest protocol mismatch in {shard_dir}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"empty artifact manifest in {shard_dir}")
    expected_paths: dict[str, Path] = {
        "shard": shard_dir / "phase2e_shard.json",
        "resolved_config": shard_dir / "resolved_config.json",
        "html_report": Path(str(shard["report"])),
    }
    telemetry = shard.get("gpu_telemetry")
    if isinstance(telemetry, Mapping) and telemetry.get("available"):
        expected_paths["gpu_telemetry"] = Path(str(telemetry["snapshot"]))
        expected_paths["gpu_telemetry_summary"] = shard_dir / "gpu_telemetry_summary.json"
    for result in shard["results"]:
        identity = f"{result['variant']}:seed{result['seed']}"
        for key in ("checkpoint", "history", "validation_predictions", "validation_metrics_path"):
            expected_paths[f"{key}:{identity}"] = Path(str(result[key]))
    roles: set[str] = set()
    for entry in files:
        role = str(entry["role"])
        if role in roles:
            raise ValueError(f"duplicate artifact role {role} in {shard_dir}")
        roles.add(role)
        file_path = _inside(shard_dir, shard_dir / str(entry["path"]), role)
        if role not in expected_paths or file_path != expected_paths[role].resolve():
            raise ValueError(f"artifact manifest path/role mismatch for {role}")
        if not file_path.is_file() or entry.get("bytes") != file_path.stat().st_size:
            raise ValueError(f"artifact manifest size mismatch for {role}")
        if entry.get("sha256") != sha256_file(file_path):
            raise ValueError(f"artifact manifest hash mismatch for {role}")
    if roles != set(expected_paths):
        raise ValueError(f"artifact manifest has missing or unexpected roles in {shard_dir}")
    if sha256_file(expected_paths["html_report"]) != shard.get("report_sha256"):
        raise ValueError(f"shard HTML report hash mismatch in {shard_dir}")


def _verify_shard_wandb_receipt(shard_dir: Path, shard_path: Path) -> tuple[str, str]:
    receipt = _load_json(shard_dir / "wandb_receipt.json")
    if receipt.get("schema_version") != SHARD_WANDB_RECEIPT_SCHEMA:
        raise ValueError(f"unexpected shard W&B receipt schema in {shard_dir}")
    if receipt.get("status") != "uploaded" or receipt.get("mode") != "online":
        raise ValueError(f"shard W&B artifact was not durably uploaded in {shard_dir}")
    run_id = _require_nonempty_string(receipt.get("run_id"), "shard W&B run_id")
    artifact_id = _require_nonempty_string(receipt.get("artifact_id"), "shard W&B artifact_id")
    for key in ("artifact_qualified_name", "artifact_digest"):
        _require_nonempty_string(receipt.get(key), f"shard W&B {key}")
    if receipt.get("phase2e_shard_sha256") != sha256_file(shard_path):
        raise ValueError(f"shard W&B receipt does not bind phase2e_shard.json in {shard_dir}")
    if receipt.get("artifact_manifest_sha256") != sha256_file(shard_dir / "artifact_manifest.json"):
        raise ValueError(f"shard W&B receipt does not bind artifact_manifest.json in {shard_dir}")
    return run_id, artifact_id


def _verify_result_files(
    shard_dir: Path,
    result: Mapping[str, Any],
    validation: EvaluationSplit,
    train_cache_sha256: str,
    shard_config_sha256: str,
    expected_epochs: int,
) -> CheckpointRecord:
    variant, seed = str(result["variant"]), int(result["seed"])
    checkpoint_path = _inside(shard_dir, str(result["checkpoint"]), "checkpoint")
    prediction_path = _inside(shard_dir, str(result["validation_predictions"]), "validation predictions")
    history_path = _inside(shard_dir, str(result["history"]), "history")
    metrics_path = _inside(shard_dir, str(result["validation_metrics_path"]), "validation metrics")
    for path, hash_key in (
        (checkpoint_path, "checkpoint_sha256"),
        (prediction_path, "validation_predictions_sha256"),
        (history_path, "history_sha256"),
        (metrics_path, "validation_metrics_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != result.get(hash_key):
            raise ValueError(f"{variant}/seed{seed} {hash_key} verification failed")
    if result.get("checkpoint_reload") != "strict-prediction-equality-pass":
        raise ValueError(f"{variant}/seed{seed} lacks strict checkpoint reload evidence")
    if result.get("run_config_sha256") != shard_config_sha256:
        raise ValueError(f"{variant}/seed{seed} run config hash mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected checkpoint schema for {variant}/seed{seed}")
    if checkpoint.get("variant") != variant or checkpoint.get("seed") != seed:
        raise ValueError(f"checkpoint identity mismatch for {variant}/seed{seed}")
    if checkpoint.get("feature_cache_sha256") != train_cache_sha256:
        raise ValueError(f"checkpoint feature-cache hash mismatch for {variant}/seed{seed}")
    if checkpoint.get("run_config_sha256") != shard_config_sha256:
        raise ValueError(f"checkpoint config hash mismatch for {variant}/seed{seed}")
    config = _config_from_payload(checkpoint["config"])
    expected_config, expected_teacher = _variant_config(variant, config.hidden_dim)
    if asdict(config) != asdict(expected_config) or bool(checkpoint.get("use_teacher")) != expected_teacher:
        raise ValueError(f"checkpoint architecture differs from registered variant {variant}")
    if _canonical_sha256(result["config"]) != _canonical_sha256(asdict(config)):
        raise ValueError(f"shard/checkpoint architecture mismatch for {variant}/seed{seed}")
    if bool(result.get("use_teacher")) != expected_teacher:
        raise ValueError(f"teacher flag mismatch for {variant}/seed{seed}")
    model = FactorizedShapeScaleGeometryProbe(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    parameter_count = model.trainable_parameter_count
    if result.get("trainable_parameters") != parameter_count:
        raise ValueError(f"parameter count mismatch for {variant}/seed{seed}")

    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if len(history) != expected_epochs:
        raise ValueError(f"history epoch count differs for {variant}/seed{seed}: {len(history)} != {expected_epochs}")
    if any(row.get("variant") != variant or row.get("seed") != seed for row in history):
        raise ValueError(f"history identity mismatch for {variant}/seed{seed}")
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(len(history))):
        raise ValueError(f"history epochs are incomplete for {variant}/seed{seed}")
    selected = min(history, key=lambda row: (float(row["validation_metric_abs_rel"]), int(row["epoch"])))
    best_epoch = int(selected["epoch"])
    best_abs_rel = float(selected["validation_metric_abs_rel"])
    if checkpoint.get("best_epoch") != best_epoch or result.get("best_epoch") != best_epoch:
        raise ValueError(f"checkpoint was not selected at minimum validation raw AbsRel for {variant}/seed{seed}")
    for value in (checkpoint.get("best_validation_metric_abs_rel"), result.get("best_validation_metric_abs_rel")):
        if not isinstance(value, (int, float)):
            raise ValueError(f"selected validation raw AbsRel is absent for {variant}/seed{seed}")
        if not math.isclose(float(value), best_abs_rel, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"selected validation raw AbsRel mismatch for {variant}/seed{seed}")

    prediction = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if not isinstance(prediction, Mapping) or prediction.get("variant") != variant or prediction.get("seed") != seed:
        raise ValueError(f"validation prediction identity mismatch for {variant}/seed{seed}")
    multiplier = fit_log_variance_multiplier(prediction, validation)
    metrics = _load_json(metrics_path)
    if (
        metrics.get("schema_version") != VALIDATION_METRICS_SCHEMA
        or metrics.get("variant") != variant
        or metrics.get("seed") != seed
    ):
        raise ValueError(f"validation metrics identity mismatch for {variant}/seed{seed}")
    predicted_depth = prediction["prediction_m"].float()
    target_depth = prediction["target_m"].float()
    validation_abs_rel = []
    for predicted_sample, target_sample in zip(predicted_depth, target_depth, strict=True):
        valid = _valid_depth(target_sample)
        validation_abs_rel.append(
            float(((predicted_sample[valid] - target_sample[valid]).abs() / target_sample[valid]).mean())
        )
    reproduced_abs_rel = float(np.mean(validation_abs_rel))
    recorded_values = (
        metrics.get("metrics", {}).get("metric_abs_rel"),
        result.get("validation_metrics", {}).get("metric_abs_rel"),
        best_abs_rel,
    )
    if any(
        not isinstance(value, (int, float))
        or not math.isclose(float(value), reproduced_abs_rel, rel_tol=1e-6, abs_tol=1e-7)
        for value in recorded_values
    ):
        raise ValueError(f"saved validation prediction/metric mismatch for {variant}/seed{seed}")
    return CheckpointRecord(
        variant=variant,
        seed=seed,
        shard_dir=shard_dir,
        checkpoint_path=checkpoint_path,
        validation_prediction_path=prediction_path,
        config=config,
        use_teacher=expected_teacher,
        variance_multiplier=multiplier,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        best_validation_abs_rel=best_abs_rel,
    )


def verify_formal_shards(
    shard_directories: Sequence[Path],
    train_validation_cache: Path,
    train_cache_sha256: str,
    validation: EvaluationSplit,
    *,
    expected_epochs: int = 60,
    require_formal_protocol: bool = True,
) -> list[CheckpointRecord]:
    if len(shard_directories) != 4 or len({path.resolve() for path in shard_directories}) != 4:
        raise ValueError("formal Phase2e evaluation requires exactly four distinct shard directories")
    records: list[CheckpointRecord] = []
    run_ids: set[str] = set()
    artifact_ids: set[str] = set()
    observed_groups: set[tuple[str, ...]] = set()
    for directory in shard_directories:
        shard_dir = directory.resolve(strict=True)
        shard_path = shard_dir / "phase2e_shard.json"
        shard = _load_json(shard_path)
        if shard.get("schema_version") != SHARD_SCHEMA or shard.get("status") != "success":
            raise ValueError(f"formal shard is not successful: {shard_dir}")
        if shard.get("selection_split") != "validation" or shard.get("feature_cache_sha256") != train_cache_sha256:
            raise ValueError(f"formal shard used an invalid split/cache: {shard_dir}")
        resolved_path = shard_dir / "resolved_config.json"
        resolved = _load_json(resolved_path)
        if sha256_file(resolved_path) != shard.get("resolved_config_file_sha256"):
            raise ValueError(f"resolved config file hash mismatch in {shard_dir}")
        config_without_hash = {key: value for key, value in resolved.items() if key != "config_sha256"}
        if _canonical_sha256(config_without_hash) != resolved.get("config_sha256"):
            raise ValueError(f"resolved config canonical hash mismatch in {shard_dir}")
        if resolved.get("config_sha256") != shard.get("config_sha256"):
            raise ValueError(f"shard/resolved config hash mismatch in {shard_dir}")
        feature_record = resolved.get("feature_cache", {})
        if (
            Path(str(feature_record.get("path"))).resolve() != train_validation_cache.resolve()
            or feature_record.get("sha256") != train_cache_sha256
            or feature_record.get("schema_version") != CACHE_SCHEMA
        ):
            raise ValueError(f"formal shard resolved a different feature cache in {shard_dir}")
        if resolved.get("seeds") != list(SEEDS):
            raise ValueError(f"formal shard does not contain exactly seeds 0/1/2: {shard_dir}")
        variants = tuple(str(value) for value in resolved.get("variants", []))
        if variants not in FORMAL_SHARD_GROUPS or variants in observed_groups:
            raise ValueError(f"formal shard variant grouping differs from the frozen four-shard map: {shard_dir}")
        observed_groups.add(variants)
        if require_formal_protocol and (
            int(resolved.get("epochs", -1)) != expected_epochs
            or int(resolved.get("batch_size", -1)) != 8
            or int(resolved.get("hidden_dim", -1)) != 64
            or resolved.get("optimizer") != FORMAL_OPTIMIZER
            or resolved.get("loss_weights") != FORMAL_LOSS_WEIGHTS
        ):
            raise ValueError(f"formal shard optimization protocol differs from the preregistration: {shard_dir}")
        if require_formal_protocol:
            telemetry = shard.get("gpu_telemetry")
            required_statistics = {"utilization_gpu", "memory_used_mib", "temperature_c", "power_w"}
            if (
                not isinstance(telemetry, Mapping)
                or telemetry.get("available") is not True
                or int(telemetry.get("samples", 0)) < 1
                or not isinstance(telemetry.get("statistics"), Mapping)
                or not required_statistics <= set(telemetry["statistics"])
            ):
                raise ValueError(f"formal shard lacks required GPU telemetry: {shard_dir}")
        if resolved.get("checkpoint_selection") != "minimum validation raw metric_abs_rel only":
            raise ValueError(f"formal shard checkpoint-selection policy differs in {shard_dir}")
        if resolved.get("wandb", {}).get("enabled") is not True or resolved.get("wandb", {}).get("mode") != "online":
            raise ValueError(f"formal shard was not run with online W&B in {shard_dir}")
        _verify_shard_manifest(shard_dir, shard)
        run_id, artifact_id = _verify_shard_wandb_receipt(shard_dir, shard_path)
        if run_id in run_ids or artifact_id in artifact_ids:
            raise ValueError("formal shards reuse a W&B run or artifact ID")
        run_ids.add(run_id)
        artifact_ids.add(artifact_id)
        results = shard.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"formal shard has no result rows: {shard_dir}")
        expected_rows = {(str(name), int(seed)) for name in resolved["variants"] for seed in resolved["seeds"]}
        found_rows = {(str(row["variant"]), int(row["seed"])) for row in results}
        if len(found_rows) != len(results) or found_rows != expected_rows:
            raise ValueError(f"formal shard result rows are duplicated or incomplete: {shard_dir}")
        records.extend(
            _verify_result_files(
                shard_dir,
                row,
                validation,
                train_cache_sha256,
                str(shard["config_sha256"]),
                int(resolved["epochs"]),
            )
            for row in results
        )
    identities = {(record.variant, record.seed) for record in records}
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if len(identities) != len(records) or identities != expected:
        missing, extra = sorted(expected - identities), sorted(identities - expected)
        raise ValueError(f"formal shard coverage differs: missing={missing}, extra={extra}")
    if observed_groups != set(FORMAL_SHARD_GROUPS):
        raise ValueError("formal shard set does not contain each frozen variant group exactly once")
    return sorted(records, key=lambda record: (VARIANTS.index(record.variant), record.seed))


def _load_model(record: CheckpointRecord, device: torch.device) -> FactorizedShapeScaleGeometryProbe:
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    model = FactorizedShapeScaleGeometryProbe(record.config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _forward_model(
    model: FactorizedShapeScaleGeometryProbe,
    config: FactorizedGeometryConfig,
    features: torch.Tensor,
    rgb: torch.Tensor,
    intrinsics: torch.Tensor,
    control: str,
) -> FactorizedGeometryOutput:
    kwargs: dict[str, Any] = {}
    if "rgb" in config.scale_inputs:
        kwargs["rgb"] = rgb
    k_conditioned = config.camera_mode == "known_rays" or bool(
        {"intrinsics", "ray_summary"} & set(config.scale_inputs)
    )
    if k_conditioned:
        kwargs["intrinsics"] = intrinsics
        kwargs["intrinsics_image_size"] = (384, 384)
        if control == "wrong":
            kwargs["intrinsics_control"] = "wrong"
            kwargs["wrong_focal_scale"] = 1.25
            kwargs["wrong_principal_shift"] = (19.2, -19.2)
        elif control not in {"correct", "shuffled"}:
            raise ValueError(f"unknown K control: {control}")
    elif control != "correct":
        raise ValueError(f"non-K-conditioned model cannot run {control} control")
    return model(features, **kwargs)


def predict_split(
    model: FactorizedShapeScaleGeometryProbe,
    record: CheckpointRecord,
    split: EvaluationSplit,
    device: torch.device,
    batch_size: int,
    control: str,
) -> PredictionBundle:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if control not in K_CONTROLS:
        raise ValueError(f"unknown K control: {control}")
    if control != "correct" and not record.k_conditioned:
        raise ValueError(f"{record.variant} is not K-conditioned")
    permutation = torch.arange(split.size).roll(1)
    log_depth, log_variance, global_scale = [], [], []
    with torch.inference_mode():
        for offset in range(0, split.size, batch_size):
            indices = torch.arange(offset, min(offset + batch_size, split.size))
            camera_indices = permutation.index_select(0, indices) if control == "shuffled" else indices
            features = split.features.index_select(0, indices).to(device)
            rgb = split.rgb.index_select(0, indices).to(device)
            intrinsics = split.intrinsics.index_select(0, camera_indices).to(device)
            output = _forward_model(model, record.config, features, rgb, intrinsics, control)
            log_depth.append(output.log_depth.detach().float().cpu())
            log_variance.append(output.log_variance.detach().float().cpu())
            if output.global_log_scale is not None:
                global_scale.append(output.global_log_scale.detach().flatten().float().cpu())
    return PredictionBundle(
        log_depth=torch.cat(log_depth),
        log_variance=torch.cat(log_variance),
        global_log_scale=None if not global_scale else torch.cat(global_scale),
    )


def verify_saved_validation_prediction(
    model: FactorizedShapeScaleGeometryProbe,
    record: CheckpointRecord,
    validation: EvaluationSplit,
    device: torch.device,
    batch_size: int,
) -> None:
    saved = torch.load(record.validation_prediction_path, map_location="cpu", weights_only=True)
    predicted = predict_split(model, record, validation, device, batch_size, "correct")
    saved_log_depth = saved["prediction_m"].float().clamp_min(1e-8).log()
    if not torch.allclose(predicted.log_depth, saved_log_depth, rtol=5e-4, atol=5e-5):
        maximum = float((predicted.log_depth - saved_log_depth).abs().max())
        raise ValueError(
            f"checkpoint does not reproduce saved validation predictions for {record.variant}/seed{record.seed}; "
            f"max_abs_diff={maximum:.8g}"
        )
    if not torch.allclose(predicted.log_variance, saved["log_variance"].float(), rtol=5e-4, atol=5e-5):
        raise ValueError(f"checkpoint does not reproduce validation variance for {record.variant}/seed{record.seed}")


def synchronized_head_latency(
    model: FactorizedShapeScaleGeometryProbe,
    record: CheckpointRecord,
    split: EvaluationSplit,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    repetitions: int,
) -> dict[str, Any]:
    if warmup < 0 or iterations <= 0 or repetitions <= 0:
        raise ValueError("latency warmup must be non-negative and iterations/repetitions positive")
    features = split.features[:1].to(device)
    rgb = split.rgb[:1].to(device)
    intrinsics = split.intrinsics[:1].to(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _forward_model(model, record.config, features, rgb, intrinsics, "correct")
        synchronize()
        values = []
        for _ in range(repetitions):
            synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                _forward_model(model, record.config, features, rgb, intrinsics, "correct")
            synchronize()
            values.append(1000.0 * (time.perf_counter() - started) / iterations)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise RuntimeError("head-only latency produced a non-positive or non-finite value")
    return {
        "synchronized_head_only_ms": float(statistics.median(values)),
        "repetitions_ms": values,
        "warmup": warmup,
        "iterations": iterations,
        "repetitions": repetitions,
        "batch_size": 1,
        "device": str(device),
        "synchronization": "torch.cuda.synchronize before/after each block"
        if device.type == "cuda"
        else "CPU wall clock",
    }


def _valid_depth(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target) & (target > 0.1) & (target < 10.0)


def _normal_quantile(coverage: float) -> float:
    probability = torch.tensor((1.0 + coverage) / 2.0, dtype=torch.float64)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(probability))


def sample_metrics(
    log_depth: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    variance_multiplier: float,
    global_log_scale: float | None,
) -> dict[str, Any]:
    valid = _valid_depth(target)
    if int(valid.sum()) < 1:
        raise ValueError("test sample has no valid depth pixels")
    prediction = log_depth[valid].float().exp()
    truth = target[valid].float()
    residual = log_depth[valid].float() - truth.log()
    raw_variance = log_variance[valid].float().exp().clamp_min(1e-8)
    calibrated_variance = raw_variance * variance_multiplier
    if not torch.isfinite(prediction).all() or not bool((prediction > 0).all()):
        raise ValueError("test prediction is non-finite or non-positive")
    error = prediction - truth
    ratio = torch.maximum(prediction / truth, truth / prediction.clamp_min(1e-8))
    alignment_scale = truth.median() / prediction.median().clamp_min(1e-8)
    aligned = prediction * alignment_scale
    aligned_error = aligned - truth
    aligned_ratio = torch.maximum(aligned / truth, truth / aligned.clamp_min(1e-8))
    raw_nll = 0.5 * (raw_variance.log() + residual.square() / raw_variance)
    calibrated_nll = 0.5 * (calibrated_variance.log() + residual.square() / calibrated_variance)

    raw_coverage, calibrated_coverage = [], []
    for nominal in NOMINAL_COVERAGES:
        quantile = _normal_quantile(nominal)
        raw_coverage.append(float((residual.abs() <= quantile * raw_variance.sqrt()).float().mean()))
        calibrated_coverage.append(float((residual.abs() <= quantile * calibrated_variance.sqrt()).float().mean()))
    nominal_array = np.asarray(NOMINAL_COVERAGES)
    raw_calibration_error = float(np.abs(np.asarray(raw_coverage) - nominal_array).mean())
    calibrated_calibration_error = float(np.abs(np.asarray(calibrated_coverage) - nominal_array).mean())

    relative_error = error.abs() / truth
    uncertainty_order = torch.argsort(calibrated_variance)
    oracle_order = torch.argsort(relative_error)
    risk, oracle_risk = [], []
    for coverage in RISK_COVERAGES:
        count = max(1, int(math.ceil(relative_error.numel() * coverage)))
        risk.append(float(relative_error[uncertainty_order[:count]].mean()))
        oracle_risk.append(float(relative_error[oracle_order[:count]].mean()))
    excess_risk = np.asarray(risk) - np.asarray(oracle_risk)
    coverage_array = np.asarray(RISK_COVERAGES)
    ause = float(np.sum(0.5 * (excess_risk[:-1] + excess_risk[1:]) * np.diff(coverage_array)))

    predicted_scale = float(log_depth[valid].mean()) if global_log_scale is None else float(global_log_scale)
    true_scale = float(truth.log().mean())
    result: dict[str, Any] = {
        "metric_abs_rel": float(relative_error.mean()),
        "metric_rmse_m": float(error.square().mean().sqrt()),
        "metric_delta_1": float((ratio < 1.25).float().mean()),
        "aligned_abs_rel": float((aligned_error.abs() / truth).mean()),
        "aligned_rmse_m": float(aligned_error.square().mean().sqrt()),
        "aligned_delta_1": float((aligned_ratio < 1.25).float().mean()),
        "abs_log_scale_error": abs(float(alignment_scale.clamp_min(1e-12).log())),
        "predicted_global_log_scale": predicted_scale,
        "true_global_log_scale": true_scale,
        "global_log_scale_abs_error": abs(predicted_scale - true_scale),
        "raw_log_depth_nll": float(raw_nll.mean()),
        "calibrated_log_depth_nll": float(calibrated_nll.mean()),
        "raw_reliability_error": raw_calibration_error,
        "calibrated_reliability_error": calibrated_calibration_error,
        "uncertainty_ause": ause,
        "valid_pixel_fraction": float(valid.float().mean()),
        "nominal_coverage": list(NOMINAL_COVERAGES),
        "raw_observed_coverage": raw_coverage,
        "calibrated_observed_coverage": calibrated_coverage,
        "risk_coverage": list(RISK_COVERAGES),
        "uncertainty_risk": risk,
        "oracle_risk": oracle_risk,
    }
    scalar_values = [value for value in result.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("test metric computation produced a non-finite value")
    return result


SCALAR_METRICS = (
    "metric_abs_rel",
    "metric_rmse_m",
    "metric_delta_1",
    "aligned_abs_rel",
    "aligned_rmse_m",
    "aligned_delta_1",
    "abs_log_scale_error",
    "predicted_global_log_scale",
    "true_global_log_scale",
    "global_log_scale_abs_error",
    "raw_log_depth_nll",
    "calibrated_log_depth_nll",
    "raw_reliability_error",
    "calibrated_reliability_error",
    "uncertainty_ause",
    "valid_pixel_fraction",
)


def _mean_curve(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"curve {key} is incomplete or non-finite")
    return values.mean(axis=0).tolist()


def summarize_seed_rows(
    record: CheckpointRecord,
    control: str,
    rows: Sequence[Mapping[str, Any]],
    latency: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty test row collection")
    metrics = {key: float(np.mean([float(row[key]) for row in rows])) for key in SCALAR_METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("per-seed summary contains a non-finite value")
    return {
        "variant": record.variant,
        "seed": record.seed,
        "intrinsics_control": control,
        "test_samples": len(rows),
        "validation_variance_multiplier": record.variance_multiplier,
        "best_epoch": record.best_epoch,
        "best_validation_metric_abs_rel": record.best_validation_abs_rel,
        "trainable_parameters": record.parameter_count,
        "head_latency": dict(latency),
        "metrics": metrics,
        "curves": {
            "nominal_coverage": list(NOMINAL_COVERAGES),
            "raw_observed_coverage": _mean_curve(rows, "raw_observed_coverage"),
            "calibrated_observed_coverage": _mean_curve(rows, "calibrated_observed_coverage"),
            "risk_coverage": list(RISK_COVERAGES),
            "uncertainty_risk": _mean_curve(rows, "uncertainty_risk"),
            "oracle_risk": _mean_curve(rows, "oracle_risk"),
        },
    }


def _estimate_three_seed_values(values: Sequence[float], label: str) -> dict[str, float]:
    if len(values) != len(SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError(f"aggregate {label} is incomplete or non-finite")
    return {"mean": float(statistics.fmean(values)), "sd": float(statistics.stdev(values))}


def aggregate_seed_summaries(per_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault((str(row["variant"]), str(row["intrinsics_control"])), []).append(row)
    aggregates = []
    for (variant, control), rows in sorted(grouped.items(), key=lambda item: (VARIANTS.index(item[0][0]), item[0][1])):
        if {int(row["seed"]) for row in rows} != set(SEEDS):
            raise ValueError(f"aggregate {variant}/{control} is missing a formal seed")
        aggregate_label = f"{variant}/{control}"

        metrics = {
            key: _estimate_three_seed_values(
                [float(row["metrics"][key]) for row in rows],
                aggregate_label,
            )
            for key in SCALAR_METRICS
        }
        latency = _estimate_three_seed_values(
            [float(row["head_latency"]["synchronized_head_only_ms"]) for row in rows],
            aggregate_label,
        )
        parameters = _estimate_three_seed_values(
            [float(row["trainable_parameters"]) for row in rows],
            aggregate_label,
        )
        curves: dict[str, Any] = {}
        for key in ("raw_observed_coverage", "calibrated_observed_coverage", "uncertainty_risk", "oracle_risk"):
            matrix = np.asarray([row["curves"][key] for row in rows], dtype=np.float64)
            curves[key] = {
                "mean": matrix.mean(axis=0).tolist(),
                "sd": matrix.std(axis=0, ddof=1).tolist(),
            }
        curves["nominal_coverage"] = list(NOMINAL_COVERAGES)
        curves["risk_coverage"] = list(RISK_COVERAGES)
        aggregates.append(
            {
                "variant": variant,
                "intrinsics_control": control,
                "seeds": list(SEEDS),
                "metrics": metrics,
                "head_latency_ms": latency,
                "trainable_parameters": parameters,
                "curves": curves,
            }
        )
    return aggregates


def compute_operational_gate(aggregates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {(str(row["variant"]), str(row["intrinsics_control"])): row for row in aggregates}
    candidate = lookup[(CANDIDATE, "correct")]
    baseline = lookup[(BASELINE, "correct")]
    wrong = lookup[(CANDIDATE, "wrong")]
    shuffled = lookup[(CANDIDATE, "shuffled")]

    def metric(row: Mapping[str, Any], key: str) -> float:
        return float(row["metrics"][key]["mean"])

    candidate_latency = float(candidate["head_latency_ms"]["mean"])
    baseline_latency = float(baseline["head_latency_ms"]["mean"])
    candidate_parameters = float(candidate["trainable_parameters"]["mean"])
    baseline_parameters = float(baseline["trainable_parameters"]["mean"])
    conditions = {
        "candidate_raw_abs_rel_strictly_lower": metric(candidate, "metric_abs_rel")
        < metric(baseline, "metric_abs_rel"),
        "candidate_scale_error_strictly_lower": metric(candidate, "abs_log_scale_error")
        < metric(baseline, "abs_log_scale_error"),
        "candidate_aligned_abs_rel_no_more_than_2pct_regression": metric(candidate, "aligned_abs_rel")
        <= 1.02 * metric(baseline, "aligned_abs_rel"),
        "candidate_calibrated_nll_strictly_lower": metric(candidate, "calibrated_log_depth_nll")
        < metric(baseline, "calibrated_log_depth_nll"),
        "candidate_correct_k_beats_wrong_k_raw_abs_rel": metric(candidate, "metric_abs_rel")
        < metric(wrong, "metric_abs_rel"),
        "candidate_correct_k_beats_shuffled_k_raw_abs_rel": metric(candidate, "metric_abs_rel")
        < metric(shuffled, "metric_abs_rel"),
        "candidate_head_latency_at_most_1p10x_baseline": candidate_latency <= 1.10 * baseline_latency,
        "candidate_parameters_at_most_1p10x_baseline": candidate_parameters <= 1.10 * baseline_parameters,
        "all_finite_complete_zero_failures": True,
    }
    values = [
        metric(candidate, "metric_abs_rel"),
        metric(baseline, "metric_abs_rel"),
        metric(candidate, "abs_log_scale_error"),
        metric(baseline, "abs_log_scale_error"),
        candidate_latency,
        baseline_latency,
        candidate_parameters,
        baseline_parameters,
    ]
    conditions["all_finite_complete_zero_failures"] = all(math.isfinite(value) for value in values)
    return {
        "schema_version": "jepa4d-phase2e-operational-gate-v1",
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "population_significance_claimed": False,
        "interpretation": "operational gate on the fixed formal runs; not a population-significance test",
        "conditions": conditions,
        "ratios": {
            "aligned_abs_rel_candidate_over_baseline": metric(candidate, "aligned_abs_rel")
            / max(metric(baseline, "aligned_abs_rel"), 1e-12),
            "head_latency_candidate_over_baseline": candidate_latency / max(baseline_latency, 1e-12),
            "parameters_candidate_over_baseline": candidate_parameters / max(baseline_parameters, 1e-12),
        },
        "passed": all(conditions.values()),
    }


def _depth_png(value: torch.Tensor, *, error: bool = False) -> str:
    array = value.detach().float().cpu().numpy()
    finite = np.isfinite(array)
    if not finite.any():
        array = np.zeros_like(array)
    else:
        low = 0.0 if error else float(np.percentile(array[finite], 2))
        high = float(np.percentile(array[finite], 98))
        high = max(high, low + 1e-6)
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
    red = (255.0 * array).astype(np.uint8)
    green = (255.0 * np.sqrt(array)).astype(np.uint8)
    blue = (255.0 * (1.0 - array)).astype(np.uint8)
    image = Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB").resize((240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def build_visual_report(
    path: Path,
    evaluation: Mapping[str, Any],
    per_sample: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, int, str], PredictionBundle],
    test: EvaluationSplit,
) -> Path:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    aggregates = evaluation["aggregates"]
    correct = [row for row in aggregates if row["intrinsics_control"] == "correct"]
    correct.sort(key=lambda row: float(row["metrics"]["metric_abs_rel"]["mean"]))
    names = [str(row["variant"]) for row in correct]
    raw_values = [float(row["metrics"]["metric_abs_rel"]["mean"]) for row in correct]
    raw_sd = [float(row["metrics"]["metric_abs_rel"]["sd"]) for row in correct]

    ranking = go.Figure(
        go.Bar(
            x=names,
            y=raw_values,
            error_y={"type": "data", "array": raw_sd},
            marker_color=[
                "#4ade80" if name == CANDIDATE else "#60a5fa" if name == BASELINE else "#a78bfa" for name in names
            ],
            hovertemplate="%{x}<br>raw AbsRel=%{y:.5f}<extra></extra>",
        )
    )
    ranking.update_layout(title="Formal kv2 ranking · raw AbsRel (mean ± seed SD)", yaxis_title="lower is better")

    raw_aligned = go.Figure()
    raw_aligned.add_trace(
        go.Scatter(
            x=[float(row["metrics"]["metric_abs_rel"]["mean"]) for row in correct],
            y=[float(row["metrics"]["aligned_abs_rel"]["mean"]) for row in correct],
            mode="markers+text",
            text=names,
            textposition="top center",
            marker={"size": 13, "color": raw_values, "colorscale": "Viridis", "showscale": True},
            hovertemplate="%{text}<br>raw=%{x:.5f}<br>aligned=%{y:.5f}<extra></extra>",
        )
    )
    raw_aligned.update_layout(
        title="Raw vs median-aligned geometry", xaxis_title="raw AbsRel", yaxis_title="aligned AbsRel"
    )

    scale = go.Figure()
    for variant, color in ((BASELINE, "#60a5fa"), (CANDIDATE, "#4ade80")):
        rows = [row for row in per_sample if row["variant"] == variant and row["intrinsics_control"] == "correct"]
        scale.add_trace(
            go.Scatter(
                x=[row["true_global_log_scale"] for row in rows],
                y=[row["predicted_global_log_scale"] for row in rows],
                mode="markers",
                name=variant,
                marker={"color": color, "opacity": 0.55},
                hovertemplate="true log scale=%{x:.4f}<br>predicted=%{y:.4f}<extra></extra>",
            )
        )
    scale_values = [float(row["true_global_log_scale"]) for row in per_sample]
    low, high = min(scale_values), max(scale_values)
    scale.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="ideal", line={"dash": "dash"}))
    scale.update_layout(title="Predicted vs true global log scale", xaxis_title="true", yaxis_title="predicted")

    reliability = make_subplots(
        rows=1, cols=2, subplot_titles=("Reliability / interval coverage", "Uncertainty risk–coverage")
    )
    for variant, color in ((BASELINE, "#60a5fa"), (CANDIDATE, "#4ade80")):
        row = next(item for item in correct if item["variant"] == variant)
        reliability.add_trace(
            go.Scatter(
                x=row["curves"]["nominal_coverage"],
                y=row["curves"]["calibrated_observed_coverage"]["mean"],
                mode="lines+markers",
                name=f"{variant} calibrated",
                line={"color": color},
            ),
            row=1,
            col=1,
        )
        reliability.add_trace(
            go.Scatter(
                x=row["curves"]["risk_coverage"],
                y=row["curves"]["uncertainty_risk"]["mean"],
                mode="lines+markers",
                name=f"{variant} uncertainty",
                line={"color": color},
            ),
            row=1,
            col=2,
        )
        reliability.add_trace(
            go.Scatter(
                x=row["curves"]["risk_coverage"],
                y=row["curves"]["oracle_risk"]["mean"],
                mode="lines",
                name=f"{variant} oracle",
                line={"color": color, "dash": "dot"},
            ),
            row=1,
            col=2,
        )
    reliability.add_trace(
        go.Scatter(x=[0.5, 0.95], y=[0.5, 0.95], mode="lines", name="ideal coverage", line={"dash": "dash"}),
        row=1,
        col=1,
    )
    reliability.update_xaxes(title_text="nominal coverage", row=1, col=1)
    reliability.update_yaxes(title_text="observed coverage", row=1, col=1)
    reliability.update_xaxes(title_text="retained pixels", row=1, col=2)
    reliability.update_yaxes(title_text="AbsRel risk", row=1, col=2)
    reliability.update_layout(title="Validation-fitted uncertainty on untouched kv2 test")

    k_rows = [row for row in aggregates if row["intrinsics_control"] in K_CONTROLS]
    k_variants = [
        name
        for name in VARIANTS
        if any(row["variant"] == name and row["intrinsics_control"] == "wrong" for row in k_rows)
    ]
    k_figure = go.Figure()
    for control, color in zip(K_CONTROLS, ("#4ade80", "#fb7185", "#f59e0b"), strict=True):
        k_figure.add_trace(
            go.Bar(
                name=control,
                x=k_variants,
                y=[
                    next(
                        float(row["metrics"]["metric_abs_rel"]["mean"])
                        for row in k_rows
                        if row["variant"] == variant and row["intrinsics_control"] == control
                    )
                    for variant in k_variants
                ],
                marker_color=color,
            )
        )
    k_figure.update_layout(barmode="group", title="Same-checkpoint camera controls", yaxis_title="raw AbsRel")

    resources = make_subplots(
        rows=1, cols=2, subplot_titles=("Synchronized head-only latency", "Trainable parameters")
    )
    resources.add_trace(
        go.Bar(x=names, y=[float(row["head_latency_ms"]["mean"]) for row in correct], marker_color="#38bdf8"),
        row=1,
        col=1,
    )
    resources.add_trace(
        go.Bar(x=names, y=[float(row["trainable_parameters"]["mean"]) for row in correct], marker_color="#c084fc"),
        row=1,
        col=2,
    )
    resources.update_yaxes(title_text="ms / sample", row=1, col=1)
    resources.update_yaxes(title_text="parameters", row=1, col=2)
    resources.update_layout(title="Head-only operational cost")

    figure_html = []
    for index, figure in enumerate((ranking, raw_aligned, scale, reliability, k_figure, resources)):
        figure.update_layout(template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827", height=520)
        figure_html.append(
            pio.to_html(
                figure,
                include_plotlyjs=index == 0,
                full_html=False,
                config={"displaylogo": False, "responsive": True},
            )
        )

    gate = evaluation["gate"]
    cards = []
    for name, passed in gate["conditions"].items():
        label = name.replace("_", " ")
        cards.append(
            f'<div class="card {"pass" if passed else "fail"}"><span>{"PASS" if passed else "FAIL"}</span>'
            f"<p>{html.escape(label)}</p></div>"
        )
    ranking_rows = "".join(
        f"<tr><td>{index + 1}</td><td>{html.escape(str(row['variant']))}</td>"
        f"<td>{row['metrics']['metric_abs_rel']['mean']:.6f} ± {row['metrics']['metric_abs_rel']['sd']:.6f}</td>"
        f"<td>{row['metrics']['aligned_abs_rel']['mean']:.6f}</td>"
        f"<td>{row['metrics']['calibrated_log_depth_nll']['mean']:.6f}</td>"
        f"<td>{row['head_latency_ms']['mean']:.4f}</td><td>{row['trainable_parameters']['mean']:.0f}</td></tr>"
        for index, row in enumerate(correct)
    )

    qualitative = []
    candidate_bundle = predictions[(CANDIDATE, 0, "correct")]
    candidate = candidate_bundle.log_depth.exp()
    baseline = predictions[(BASELINE, 0, "correct")].log_depth.exp()
    candidate_multiplier = next(
        float(row["validation_variance_multiplier"])
        for row in evaluation["per_seed"]
        if row["variant"] == CANDIDATE and row["seed"] == 0 and row["intrinsics_control"] == "correct"
    )
    candidate_sigma = (0.5 * candidate_bundle.log_variance).exp() * math.sqrt(candidate_multiplier)
    for index in range(min(3, test.size)):
        target = test.targets[index]
        panels = (
            ("Target", target, False),
            ("Candidate", candidate[index], False),
            ("Baseline", baseline[index], False),
            ("Candidate |relative error|", (candidate[index] - target).abs() / target.clamp_min(1e-4), True),
            ("Candidate calibrated log-depth sigma", candidate_sigma[index], True),
        )
        images = "".join(
            f'<figure><img src="data:image/png;base64,{_depth_png(values, error=is_error)}" '
            f'alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'
            for label, values, is_error in panels
        )
        qualitative.append(
            f"<article><h3>{html.escape(test.sample_ids[index])}</h3><div class=panels>{images}</div></article>"
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phase2e final kv2 evaluation</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;background:#07101f;color:#e5eefc;margin:0}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:2.4rem;margin-bottom:.2rem}}
.subtitle{{color:#9fb0ca}}.hero{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:24px 0}}
.card{{border-radius:14px;padding:16px;background:#152238;border:1px solid #334155}}.card span{{font-weight:800}}
.card.pass{{border-color:#22c55e}}.card.pass span{{color:#4ade80}}.card.fail{{border-color:#ef4444}}.card.fail span{{color:#fb7185}}
section{{background:#111827;border:1px solid #26364e;border-radius:16px;padding:18px;margin:20px 0;overflow:hidden}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #334155;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}
.panels{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0}}img{{width:190px;height:190px;image-rendering:pixelated;border-radius:8px}}figcaption{{font-size:.85rem;color:#a9b8ce}}
.decision{{font-size:1.4rem;font-weight:800;color:{"#4ade80" if gate["passed"] else "#fb7185"}}}
</style></head><body><main>
<h1>Phase2e · Final untouched-kv2 evaluation</h1>
<p class=subtitle>Exact validation-selected checkpoints · one validation-fitted variance multiplier per seed · test opened only after evidence verification</p>
<p class=decision>Operational gate: {"PASS" if gate["passed"] else "FAIL"}</p><div class=hero>{"".join(cards)}</div>
<section><h2>Ranking</h2><table><thead><tr><th>Rank</th><th>Variant</th><th>Raw AbsRel</th><th>Aligned AbsRel</th><th>Calibrated NLL</th><th>Head ms</th><th>Params</th></tr></thead><tbody>{ranking_rows}</tbody></table></section>
{"".join(f"<section>{value}</section>" for value in figure_html)}
<section><h2>Fixed qualitative panels · seed 0, correct K</h2>{"".join(qualitative)}</section>
<section><h2>Interpretation boundary</h2><p>This is a fixed operational gate over the registered formal runs. It is not a population-significance claim.</p></section>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")
    return path


def _write_per_sample_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    identity = (
        "variant",
        "seed",
        "intrinsics_control",
        "sample_index",
        "sample_id",
        "sensor_id",
        "validation_variance_multiplier",
        "trainable_parameters",
        "synchronized_head_only_ms",
    )
    curves = (
        "nominal_coverage",
        "raw_observed_coverage",
        "calibrated_observed_coverage",
        "risk_coverage",
        "uncertainty_risk",
        "oracle_risk",
    )
    fields = (*identity, *SCALAR_METRICS, *curves)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {key: row[key] for key in (*identity, *SCALAR_METRICS)}
            flattened.update({key: json.dumps(row[key], separators=(",", ":")) for key in curves})
            writer.writerow(flattened)


def _write_predictions_npz(
    path: Path,
    order: Sequence[tuple[str, int, str]],
    predictions: Mapping[tuple[str, int, str], PredictionBundle],
    per_sample: Sequence[Mapping[str, Any]],
    test: EvaluationSplit,
) -> None:
    scales: dict[tuple[str, int, str], list[float]] = {}
    for key in order:
        scales[key] = [
            float(row["predicted_global_log_scale"])
            for row in per_sample
            if (str(row["variant"]), int(row["seed"]), str(row["intrinsics_control"])) == key
        ]
        if len(scales[key]) != test.size:
            raise ValueError(f"prediction NPZ scale rows are incomplete for {key}")
    np.savez_compressed(
        path,
        schema_version=np.asarray(["jepa4d-phase2e-final-predictions-v1"]),
        variants=np.asarray([key[0] for key in order]),
        seeds=np.asarray([key[1] for key in order], dtype=np.int64),
        intrinsics_controls=np.asarray([key[2] for key in order]),
        sample_ids=np.asarray(test.sample_ids),
        sensor_ids=np.asarray(test.sensor_ids),
        prediction_m=np.stack([predictions[key].log_depth.exp().numpy() for key in order]),
        log_variance=np.stack([predictions[key].log_variance.numpy() for key in order]),
        predicted_global_log_scale=np.asarray([scales[key] for key in order], dtype=np.float32),
        target_m=test.targets.numpy(),
        intrinsics_384=test.intrinsics.numpy(),
    )


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    roles = {
        "canonical_evaluation": output / "phase2e_final_evaluation.json",
        "full_predictions": output / "phase2e_final_predictions.npz",
        "per_sample_metrics": output / "phase2e_final_per_sample.csv",
        "visual_report": output / "phase2e_final_report.html",
    }
    files = []
    for role, path in roles.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"final artifact file missing or empty: {path}")
        files.append(
            {
                "role": role,
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {"schema_version": ARTIFACT_MANIFEST_SCHEMA, "files": files}
    _write_json(output / "artifact_manifest.json", manifest)
    return manifest


def upload_evaluation_artifact(
    run: Any,
    output: Path,
    evaluation: Mapping[str, Any],
    *,
    wandb_module: Any | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    if wandb_module is None:
        import wandb

        wandb_module = wandb
    manifest_path = output / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError("final artifact manifest has an unexpected schema")
    for entry in manifest["files"]:
        path = _inside(output, output / str(entry["path"]), str(entry["role"]))
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"final artifact changed after manifest creation: {entry['role']}")
    artifact = wandb_module.Artifact(
        name=f"{run.id}-phase2e-final-evaluation",
        type="phase2e-final-evaluation",
        metadata={
            "schema_version": EVALUATION_SCHEMA,
            "candidate": CANDIDATE,
            "baseline": BASELINE,
            "gate_passed": evaluation["gate"]["passed"],
            "test_cache_sha256": evaluation["inputs"]["test_cache_sha256"],
        },
    )
    artifact.add_dir(str(output), name="phase2e_final")
    uploaded = run.log_artifact(artifact).wait(timeout=timeout_seconds)
    receipt = {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "status": "uploaded",
        "mode": "online",
        "run_id": _require_nonempty_string(str(run.id), "final W&B run_id"),
        "run_url": str(run.url),
        "run_path": str(run.path),
        "artifact_id": _require_nonempty_string(str(uploaded.id), "final W&B artifact_id"),
        "artifact_name": str(uploaded.name),
        "artifact_qualified_name": str(uploaded.qualified_name),
        "artifact_version": str(uploaded.version),
        "artifact_digest": str(uploaded.digest),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "evaluation_sha256": sha256_file(output / "phase2e_final_evaluation.json"),
        "report_sha256": sha256_file(output / "phase2e_final_report.html"),
    }
    _write_json(output / "wandb_receipt.json", receipt)
    return receipt


def run_final_evaluation(
    train_validation_cache: Path,
    test_cache: Path,
    feature_cache_receipt: Path,
    shard_directories: Sequence[Path],
    output_path: Path,
    *,
    test_receipt: Path | None = None,
    device_name: str = "cuda:0",
    batch_size: int = 8,
    latency_warmup: int = 20,
    latency_iterations: int = 100,
    latency_repetitions: int = 5,
    wandb_enabled: bool = True,
    wandb_project: str = "jepa4d-worldmodel",
    wandb_entity: str | None = None,
    run_name: str = "phase2e-final-evaluation",
    expected_epochs: int = 60,
    require_formal_protocol: bool = True,
) -> dict[str, Any]:
    """Verify formal evidence, then and only then evaluate the isolated kv2 test cache."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    started = time.perf_counter()
    output = _prepare_output(output_path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
    run = None
    try:
        repo_root = Path(__file__).resolve().parents[2]
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip()
        dependency_graph_path = os.getenv("JEPA4D_DEPENDENCY_GRAPH", "")
        dependency_graph_identity = None
        test_receipt_identity = None
        if test_receipt is not None:
            resolved_test_receipt = test_receipt.resolve(strict=True)
            receipt_payload = _load_json(resolved_test_receipt)
            if (
                receipt_payload.get("schema_version") != "jepa4d-phase2d-test-receipt-v1"
                or receipt_payload.get("status") != "pass"
                or receipt_payload.get("git_commit") != git_commit
            ):
                raise ValueError("formal final evaluator test receipt does not match the execution commit")
            test_receipt_identity = {
                "path": str(resolved_test_receipt),
                "sha256": sha256_file(resolved_test_receipt),
                "test_job_id": receipt_payload.get("slurm", {}).get("SLURM_JOB_ID"),
            }
        if dependency_graph_path:
            resolved_graph = Path(dependency_graph_path).resolve(strict=True)
            graph = _load_json(resolved_graph)
            expected_graph_keys = {
                "schema_version",
                "phase2d_test_job_id",
                "cache_job_id",
                "pilot_job_id",
                "formal_shard_job_ids",
                "final_job_id",
            }
            shard_job_ids = [str(value) for value in graph.get("formal_shard_job_ids", [])]
            if (
                set(graph) != expected_graph_keys
                or graph.get("schema_version") != "jepa4d-phase2e-dependency-graph-v1"
                or len(shard_job_ids) != 4
                or len(set(shard_job_ids)) != 4
                or str(graph.get("final_job_id", "")) != os.getenv("SLURM_JOB_ID", "")
                or (
                    test_receipt_identity is not None
                    and str(graph.get("phase2d_test_job_id", "")) != str(test_receipt_identity["test_job_id"])
                )
                or any(
                    not str(graph.get(key, ""))
                    for key in ("phase2d_test_job_id", "cache_job_id", "pilot_job_id", "final_job_id")
                )
            ):
                raise ValueError("formal Phase2e dependency graph is incomplete or inconsistent")
            dependency_graph_identity = {
                "path": str(resolved_graph),
                "bytes": resolved_graph.stat().st_size,
                "sha256": sha256_file(resolved_graph),
                "graph": graph,
            }
        if require_formal_protocol and (
            git_status or test_receipt_identity is None or dependency_graph_identity is None
        ):
            raise ValueError(
                "formal final evaluation requires a clean commit, passing test receipt, and dependency graph"
            )
        validation, feature_receipt, train_sha, test_sha = verify_feature_inputs_before_test(
            train_validation_cache,
            test_cache,
            feature_cache_receipt,
            require_formal_protocol=require_formal_protocol,
        )
        records = verify_formal_shards(
            shard_directories,
            train_validation_cache,
            train_sha,
            validation,
            expected_epochs=expected_epochs,
            require_formal_protocol=require_formal_protocol,
        )

        # Reproduce every saved validation prediction before opening test_cache.pt.
        for record in records:
            model = _load_model(record, device)
            verify_saved_validation_prediction(model, record, validation, device, batch_size)
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_ids = {
            str(row["sample_id"])
            for split in ("train", "validation")
            for row in feature_receipt["sample_metadata"][split]
        }
        if wandb_enabled:
            import wandb

            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                job_type="phase2e-final-evaluation",
                mode="online",
                config={
                    "schema_version": EVALUATION_SCHEMA,
                    "train_validation_cache_sha256": train_sha,
                    "test_cache_sha256": test_sha,
                    "variants": list(VARIANTS),
                    "seeds": list(SEEDS),
                    "candidate": CANDIDATE,
                    "baseline": BASELINE,
                    "batch_size": batch_size,
                    "device": str(device),
                    "latency_warmup": latency_warmup,
                    "latency_iterations": latency_iterations,
                    "latency_repetitions": latency_repetitions,
                    "git_commit": git_commit,
                    "test_receipt_sha256": None if test_receipt_identity is None else test_receipt_identity["sha256"],
                    "dependency_graph_sha256": None
                    if dependency_graph_identity is None
                    else dependency_graph_identity["sha256"],
                },
                tags=["phase-2e", "final-evaluation", "kv2-test", "operational-gate"],
            )
            if run.offline:
                raise RuntimeError("formal Phase2e final evaluation requires online W&B")
        test = load_verified_test_cache(test_cache, feature_receipt, train_ids)

        per_sample: list[dict[str, Any]] = []
        per_seed: list[dict[str, Any]] = []
        predictions: dict[tuple[str, int, str], PredictionBundle] = {}
        prediction_order: list[tuple[str, int, str]] = []
        for record_index, record in enumerate(records):
            model = _load_model(record, device)
            latency = synchronized_head_latency(
                model,
                record,
                test,
                device,
                warmup=latency_warmup,
                iterations=latency_iterations,
                repetitions=latency_repetitions,
            )
            controls = K_CONTROLS if record.k_conditioned else ("correct",)
            for control in controls:
                bundle = predict_split(model, record, test, device, batch_size, control)
                key = (record.variant, record.seed, control)
                predictions[key] = bundle
                prediction_order.append(key)
                rows = []
                for index, (log_depth, log_variance, target) in enumerate(
                    zip(bundle.log_depth, bundle.log_variance, test.targets, strict=True)
                ):
                    global_scale = None if bundle.global_log_scale is None else float(bundle.global_log_scale[index])
                    metrics = sample_metrics(
                        log_depth,
                        log_variance,
                        target,
                        record.variance_multiplier,
                        global_scale,
                    )
                    row = {
                        "variant": record.variant,
                        "seed": record.seed,
                        "intrinsics_control": control,
                        "sample_index": index,
                        "sample_id": test.sample_ids[index],
                        "sensor_id": test.sensor_ids[index],
                        "validation_variance_multiplier": record.variance_multiplier,
                        "trainable_parameters": record.parameter_count,
                        "synchronized_head_only_ms": latency["synchronized_head_only_ms"],
                        **metrics,
                    }
                    rows.append(row)
                    per_sample.append(row)
                per_seed.append(summarize_seed_rows(record, control, rows, latency))
            if run is not None:
                run.log(
                    {
                        "evaluation/checkpoints_completed": record_index + 1,
                        "evaluation/checkpoints_total": len(records),
                        "evaluation/current_variant": record.variant,
                        "evaluation/current_seed": record.seed,
                    }
                )
            del model

        expected_seed_rows = len(VARIANTS) * len(SEEDS) + 2 * 3 * len(SEEDS)
        if len(per_seed) != expected_seed_rows or len(per_sample) != expected_seed_rows * test.size:
            raise RuntimeError("final evaluation produced incomplete seed/sample rows")
        aggregates = aggregate_seed_summaries(per_seed)
        gate = compute_operational_gate(aggregates)
        evaluation: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA,
            "status": "success",
            "inputs": {
                "train_validation_cache": str(train_validation_cache.resolve()),
                "train_validation_cache_sha256": train_sha,
                "test_cache": str(test_cache.resolve()),
                "test_cache_sha256": test_sha,
                "feature_cache_receipt": str(feature_cache_receipt.resolve()),
                "feature_cache_receipt_sha256": sha256_file(feature_cache_receipt),
                "formal_shards": [
                    {
                        "path": str(path.resolve()),
                        "phase2e_shard_sha256": sha256_file(path.resolve() / "phase2e_shard.json"),
                        "wandb_receipt_sha256": sha256_file(path.resolve() / "wandb_receipt.json"),
                    }
                    for path in shard_directories
                ],
            },
            "protocol": {
                "dataset": "SUN RGB-D kv2 untouched test split",
                "checkpoint_selection": "validation raw AbsRel only; exact saved selected checkpoints",
                "uncertainty_calibration": "one scalar log-variance multiplier fit from each saved validation prediction",
                "test_intrinsics": "correct K for every model; deterministic wrong and cyclic-shuffled K for every K-conditioned model",
                "wrong_k": {"focal_scale": 1.25, "principal_shift_px_at_384": [19.2, -19.2]},
                "shuffled_k": "global one-sample cyclic roll over immutable test-cache order",
                "aggregation": "per-sample metrics; equal-weight sample macro per seed; mean and sample SD over seeds 0/1/2",
                "latency": "synchronized batch-1 head only; excludes cached V-JEPA feature extraction",
                "gate": "fixed operational gate; no population-significance claim",
            },
            "counts": {
                "test_samples": test.size,
                "formal_checkpoints": len(records),
                "per_seed_rows": len(per_seed),
                "per_sample_rows": len(per_sample),
                "failures": 0,
            },
            "provenance": {
                "created_utc": datetime.now(UTC).isoformat(),
                "git_commit": git_commit,
                "git_status": git_status,
                "test_receipt": test_receipt_identity,
                "slurm_dependency_graph": dependency_graph_identity,
                "slurm": {
                    key: os.environ.get(key)
                    for key in (
                        "SLURM_JOB_ID",
                        "SLURM_JOB_NAME",
                        "SLURM_JOB_PARTITION",
                        "SLURM_JOB_NODELIST",
                        "SLURM_JOB_DEPENDENCY",
                    )
                },
                "runtime": {
                    "seconds": time.perf_counter() - started,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda_build": torch.version.cuda,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                },
            },
            "per_seed": per_seed,
            "per_sample": per_sample,
            "aggregates": aggregates,
            "gate": gate,
            "wandb_url": None if run is None else str(run.url),
        }
        _write_json(output / "phase2e_final_evaluation.json", evaluation)
        _write_per_sample_csv(output / "phase2e_final_per_sample.csv", per_sample)
        _write_predictions_npz(
            output / "phase2e_final_predictions.npz",
            prediction_order,
            predictions,
            per_sample,
            test,
        )
        report = build_visual_report(output / "phase2e_final_report.html", evaluation, per_sample, predictions, test)
        write_artifact_manifest(output)

        if run is not None:
            import wandb

            aggregate_table = wandb.Table(
                columns=[
                    "variant",
                    "intrinsics_control",
                    "raw_absrel_mean",
                    "raw_absrel_sd",
                    "aligned_absrel_mean",
                    "scale_error_mean",
                    "calibrated_nll_mean",
                    "ause_mean",
                    "head_latency_ms",
                    "parameters",
                ]
            )
            for row in aggregates:
                aggregate_table.add_data(
                    row["variant"],
                    row["intrinsics_control"],
                    row["metrics"]["metric_abs_rel"]["mean"],
                    row["metrics"]["metric_abs_rel"]["sd"],
                    row["metrics"]["aligned_abs_rel"]["mean"],
                    row["metrics"]["abs_log_scale_error"]["mean"],
                    row["metrics"]["calibrated_log_depth_nll"]["mean"],
                    row["metrics"]["uncertainty_ause"]["mean"],
                    row["head_latency_ms"]["mean"],
                    row["trainable_parameters"]["mean"],
                )
            sample_table = wandb.Table(
                columns=[
                    "variant",
                    "seed",
                    "intrinsics_control",
                    "sample_id",
                    "raw_absrel",
                    "aligned_absrel",
                    "scale_error",
                    "calibrated_nll",
                    "ause",
                ]
            )
            for row in per_sample:
                sample_table.add_data(
                    row["variant"],
                    row["seed"],
                    row["intrinsics_control"],
                    row["sample_id"],
                    row["metric_abs_rel"],
                    row["aligned_abs_rel"],
                    row["abs_log_scale_error"],
                    row["calibrated_log_depth_nll"],
                    row["uncertainty_ause"],
                )
            candidate_prediction = predictions[(CANDIDATE, 0, "correct")].log_depth[0].exp().numpy()
            candidate_multiplier = next(
                float(row["validation_variance_multiplier"])
                for row in per_seed
                if row["variant"] == CANDIDATE and row["seed"] == 0 and row["intrinsics_control"] == "correct"
            )
            candidate_uncertainty = (
                (0.5 * predictions[(CANDIDATE, 0, "correct")].log_variance[0]).exp() * math.sqrt(candidate_multiplier)
            ).numpy()
            run.log(
                {
                    "evaluation/aggregate": aggregate_table,
                    "evaluation/per_sample": sample_table,
                    "evaluation/report": wandb.Html(str(report), inject=False),
                    "evaluation/qualitative_target": wandb.Image(test.targets[0].numpy(), caption=test.sample_ids[0]),
                    "evaluation/qualitative_candidate": wandb.Image(
                        candidate_prediction,
                        caption=f"{CANDIDATE} seed0 correct-K · {test.sample_ids[0]}",
                    ),
                    "evaluation/qualitative_candidate_calibrated_sigma": wandb.Image(
                        candidate_uncertainty,
                        caption=f"{CANDIDATE} seed0 calibrated log-depth sigma · {test.sample_ids[0]}",
                    ),
                    "gate/passed": gate["passed"],
                    **{f"gate/{key}": value for key, value in gate["conditions"].items()},
                }
            )
            receipt = upload_evaluation_artifact(run, output, evaluation, wandb_module=wandb)
            run.summary.update(
                {
                    "status": "success",
                    "gate_passed": gate["passed"],
                    "test_cache_sha256": test_sha,
                    "failure_count": 0,
                    "artifact_id": receipt["artifact_id"],
                    "artifact_qualified_name": receipt["artifact_qualified_name"],
                    "artifact_digest": receipt["artifact_digest"],
                }
            )
            run.finish(exit_code=0)
        return evaluation
    except Exception as error:
        for name in (
            "phase2e_final_evaluation.json",
            "phase2e_final_predictions.npz",
            "phase2e_final_per_sample.csv",
            "phase2e_final_report.html",
            "artifact_manifest.json",
            "wandb_receipt.json",
        ):
            (output / name).unlink(missing_ok=True)
        _write_json(
            output / "run_failure.json",
            {
                "schema_version": EVALUATION_SCHEMA,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        if run is not None:
            run.summary.update({"status": "failed", "failure_count": 1, "error": str(error)})
            run.finish(exit_code=1)
        raise
