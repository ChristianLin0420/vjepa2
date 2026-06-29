"""Strictly validate the immutable Phase-2e feature-cache handoff."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from jepa4d.evaluation.phase2e_feature_cache import (
    FROZEN_DATASET_ID,
    FROZEN_DATASET_VERSION,
    FROZEN_MANIFEST_SHA256,
    FROZEN_MODEL_IDENTITIES,
    FROZEN_SPLIT_COUNTS,
    FROZEN_SPLIT_SHA256,
    VIEW_POLICY,
)

CACHE_SCHEMA = "jepa4d-phase2e-feature-cache-v1"
RECEIPT_SCHEMA = "jepa4d-phase2e-feature-cache-receipt-v1"
WANDB_SCHEMA = "jepa4d-phase2e-cache-wandb-receipt-v1"
MANIFEST_SCHEMA = "jepa4d-sunrgbd-sensor-blocked-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required cache artifact is absent or empty: {path}")
    return path.resolve()


def _check_identity(path: Path, identity: Any, label: str) -> None:
    if not isinstance(identity, dict):
        raise RuntimeError(f"{label} identity is missing")
    recorded_path = Path(str(identity.get("path", ""))).resolve()
    if recorded_path != path.resolve():
        raise RuntimeError(f"{label} path differs from its receipt: {recorded_path} != {path}")
    if int(identity.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} byte count differs from its receipt")
    if str(identity.get("sha256", "")) != _sha256(path):
        raise RuntimeError(f"{label} SHA-256 differs from its receipt")


def _require_backend_identity(receipt: dict[str, Any]) -> None:
    required = (
        "run_id",
        "run_url",
        "run_path",
        "artifact_id",
        "artifact_name",
        "artifact_qualified_name",
        "artifact_version",
        "artifact_digest",
    )
    missing = [key for key in required if not str(receipt.get(key, "")).strip() or str(receipt.get(key)) == "None"]
    if missing:
        raise RuntimeError(f"Phase-2e cache W&B receipt lacks backend identities: {missing}")


def validate_phase2e_cache(cache_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate hashes, frozen split identity, separation, and online upload receipt."""

    root = cache_root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unexpected Phase-2e SUN RGB-D manifest schema")
    split_hash = str(manifest.get("integrity", {}).get("split_hash", ""))
    split_counts = manifest.get("selected_counts_by_split")
    if len(split_hash) != 64 or not isinstance(split_counts, dict):
        raise RuntimeError("frozen Phase-2e manifest lacks split identity/counts")
    if (
        _sha256(manifest_path) != FROZEN_MANIFEST_SHA256
        or split_hash != FROZEN_SPLIT_SHA256
        or split_counts != FROZEN_SPLIT_COUNTS
        or manifest.get("dataset_id") != FROZEN_DATASET_ID
        or manifest.get("version") != FROZEN_DATASET_VERSION
    ):
        raise RuntimeError("manifest differs from the preregistered Phase-2e dataset/split identity")

    train_cache = _require_file(root, "train_validation_cache.pt")
    test_cache = _require_file(root, "test_cache.pt")
    normalizer = _require_file(root, "feature_normalization.pt")
    report = _require_file(root, "feature_cache_report.html")
    receipt_path = _require_file(root, "feature_cache_receipt.json")
    wandb_path = _require_file(root, "wandb_receipt.json")
    if train_cache == test_cache:
        raise RuntimeError("train/validation and test caches must be physically separate files")

    receipt = _json(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "pass":
        raise RuntimeError("Phase-2e feature-cache receipt does not pass")
    dataset = receipt.get("dataset")
    if not isinstance(dataset, dict):
        raise RuntimeError("Phase-2e feature-cache receipt lacks dataset identity")
    if Path(str(dataset.get("manifest", ""))).resolve() != manifest_path:
        raise RuntimeError("cache receipt points at a different manifest")
    if dataset.get("manifest_sha256") != _sha256(manifest_path) or dataset.get("split_hash") != split_hash:
        raise RuntimeError("cache receipt manifest/split hash differs from the frozen manifest")
    if (
        dataset.get("dataset_id") != FROZEN_DATASET_ID
        or dataset.get("version") != FROZEN_DATASET_VERSION
        or dataset.get("split_policy") != "train-kv1-plus-xtion-validation-realsense-untouched-test-kv2"
    ):
        raise RuntimeError("cache receipt dataset protocol differs from the preregistration")
    models = receipt.get("models")
    if not isinstance(models, dict) or set(models) != set(FROZEN_MODEL_IDENTITIES):
        raise RuntimeError("cache receipt model identities are incomplete")
    for name, expected in FROZEN_MODEL_IDENTITIES.items():
        if not isinstance(models[name], dict) or {key: models[name].get(key) for key in expected} != expected:
            raise RuntimeError(f"cache receipt model identity changed for {name}")
    view_policy = receipt.get("view_policy")
    if (
        not isinstance(view_policy, dict)
        or view_policy.get("name") != VIEW_POLICY
        or view_policy.get("train") != ["center_square", "center_crop_0.85"]
        or view_policy.get("validation") != ["center_square"]
        or view_policy.get("test") != ["center_square"]
        or view_policy.get("resize")
        != {
            "vjepa_rgb": [384, 384],
            "cached_rgb": [96, 96],
            "target_depth": [24, 24],
            "rgb_mode": "bilinear-align_corners_false",
            "depth_mode": "nearest",
        }
        or view_policy.get("intrinsics") != "update_intrinsics_for_crop_resize with half-pixel centres"
    ):
        raise RuntimeError("cache receipt view/camera policy differs from the preregistration")

    caches = receipt.get("caches")
    if not isinstance(caches, dict) or set(caches) != {"train_validation", "test"}:
        raise RuntimeError("cache receipt must contain exactly train_validation and test identities")
    _check_identity(train_cache, caches["train_validation"], "train/validation cache")
    _check_identity(test_cache, caches["test"], "test cache")
    if caches["train_validation"].get("schema_version") != CACHE_SCHEMA or caches["train_validation"].get(
        "splits"
    ) != ["train", "validation"]:
        raise RuntimeError("train/validation cache schema or split boundary changed")
    if caches["test"].get("schema_version") != CACHE_SCHEMA or caches["test"].get("splits") != ["test"]:
        raise RuntimeError("test cache schema or split boundary changed")

    normalization = receipt.get("feature_normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError("cache receipt lacks feature-normalization identity")
    if (
        normalization.get("policy")
        != "channel mean/std over training samples, both train views, and 24x24 spatial grid only"
    ):
        raise RuntimeError("cache feature normalization is not training-only under the frozen policy")
    _check_identity(
        normalizer,
        {
            "path": normalization.get("artifact"),
            "bytes": normalizer.stat().st_size,
            "sha256": normalization.get("sha256"),
        },
        "feature normalization",
    )
    if len(str(normalization.get("content_sha256", ""))) != 64:
        raise RuntimeError("feature-normalization content identity is incomplete")

    report_identity = receipt.get("report")
    if not isinstance(report_identity, dict):
        raise RuntimeError("feature-cache report identity is missing")
    _check_identity(report, report_identity, "feature-cache report")
    if report_identity.get("self_contained") is not True or re.search(
        r"<script[^>]+src\s*=", report.read_text(), flags=re.IGNORECASE
    ):
        raise RuntimeError("feature-cache report is not self-contained")

    summaries = receipt.get("split_summaries")
    if not isinstance(summaries, dict) or set(summaries) != {"train", "validation", "test"}:
        raise RuntimeError("cache receipt split summaries are incomplete")
    for split in ("train", "validation", "test"):
        if int(summaries[split].get("samples", -1)) != int(split_counts[split]):
            raise RuntimeError(f"cache sample count differs from frozen manifest for {split}")
    if summaries["train"].get("views") != 2 or any(
        summaries[name].get("views") != 1 for name in ("validation", "test")
    ):
        raise RuntimeError("cache view policy differs from the frozen two-view/single-view protocol")
    test_summary = summaries["test"]
    if {"target_depth_m", "intrinsics_384", "raw_features", "normalized_features"} & set(test_summary):
        raise RuntimeError("cache receipt exposes held-out target/input statistics before final evaluation")
    for label in ("input_tensors", "target_tensor"):
        boundary = test_summary.get(label)
        if (
            not isinstance(boundary, dict)
            or boundary.get("access") != "opaque_until_final_evaluation"
            or boundary.get("statistics_computed") is not False
            or boundary.get("preview_generated") is not False
        ):
            raise RuntimeError(f"cache receipt does not preserve the held-out {label} boundary")
    test_access = receipt.get("test_target_access")
    if (
        not isinstance(test_access, dict)
        or test_access.get("statistics_computed") is not False
        or test_access.get("preview_generated") is not False
        or test_access.get("logged_to_wandb") is not False
    ):
        raise RuntimeError("cache receipt does not prove the no-test-target-logging policy")
    teacher = receipt.get("teacher_policy")
    if (
        not isinstance(teacher, dict)
        or teacher.get("backend") != "official VGGT-1B"
        or teacher.get("precision") != "bfloat16"
        or teacher.get("split") != "train-only"
        or teacher.get("views") != ["center_square", "center_crop_0.85"]
        or teacher.get("target") != "spatially centered log-depth at 24x24"
        or teacher.get("metric_scale_fitted") is not False
    ):
        raise RuntimeError("VGGT teacher boundary is not train-only")
    if teacher.get("validation_teacher_computed") is not False or teacher.get("test_teacher_computed") is not False:
        raise RuntimeError("VGGT teacher was computed outside the training split")

    wandb = _json(wandb_path)
    if (
        wandb.get("schema_version") != WANDB_SCHEMA
        or wandb.get("status") != "uploaded"
        or wandb.get("mode") != "online"
    ):
        raise RuntimeError("Phase-2e cache W&B receipt is not a completed online upload")
    _require_backend_identity(wandb)
    if wandb.get("receipt_sha256") != _sha256(receipt_path) or wandb.get("report_sha256") != _sha256(report):
        raise RuntimeError("Phase-2e cache W&B receipt is not bound to the local receipt/report")

    return {
        "schema_version": "jepa4d-phase2e-cache-validation-v1",
        "status": "pass",
        "cache_root": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "split_hash": split_hash,
        "train_validation_cache_sha256": _sha256(train_cache),
        "test_cache_sha256": _sha256(test_cache),
        "wandb_artifact_id": wandb["artifact_id"],
        "wandb_artifact_digest": wandb["artifact_digest"],
    }


def main(
    cache_root: Annotated[Path, typer.Option("--cache-root")],
    manifest: Annotated[Path, typer.Option("--manifest")],
) -> None:
    typer.echo(json.dumps(validate_phase2e_cache(cache_root, manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
