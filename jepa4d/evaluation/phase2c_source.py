"""Fail-closed identity checks for the completed formal Phase-2c source.

Phase 2d is diagnostic work over one immutable Phase-2c execution.  These
checks bind every consumer to the completed output, the dataset manifest, and
the exact V-JEPA weights/implementation recorded by that execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PHASE2C_COMPARISON_SCHEMA = "jepa4d-phase2c-cross-sequence-comparison-v1"
PHASE2C_PROMOTION_SCHEMA = "jepa4d-phase2c-promotion-v1"
PHASE2C_WANDB_SCHEMA = "jepa4d-phase2c-wandb-artifact-v1"
SOURCE_IDENTITY_SCHEMA = "jepa4d-phase2c-source-identity-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_output_manifest(root: Path, manifest: dict[str, Any]) -> None:
    if not manifest:
        raise ValueError("Phase-2c artifact_manifest.json is empty")
    for relative, identity in manifest.items():
        if not isinstance(relative, str) or not isinstance(identity, dict):
            raise ValueError("Phase-2c artifact manifest has an invalid row")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase-2c artifact: {path}")
        if path.stat().st_size != int(identity.get("bytes", -1)):
            raise ValueError(f"Phase-2c artifact byte count changed: {relative}")
        if sha256_file(path) != identity.get("sha256"):
            raise ValueError(f"Phase-2c artifact SHA-256 changed: {relative}")


def _verify_asset(label: str, root: Path, asset_manifest: dict[str, Any]) -> dict[str, Any]:
    recorded = asset_manifest.get(label)
    if not isinstance(recorded, dict) or not isinstance(recorded.get("files"), list):
        raise ValueError(f"Phase-2c asset manifest has no valid {label!r} entry")
    verified_files = []
    for identity in recorded["files"]:
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError(f"Phase-2c {label} asset row is invalid")
        relative = identity["path"]
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase-2c {label} file: {path}")
        if path.stat().st_size != int(identity.get("bytes", -1)):
            raise ValueError(f"Phase-2c {label} byte count changed: {relative}")
        digest = sha256_file(path)
        if digest != identity.get("sha256"):
            raise ValueError(f"Phase-2c {label} SHA-256 changed: {relative}")
        verified_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    return {"path": str(root), "files": verified_files}


def validate_phase2c_source(
    phase2c_output: Path,
    *,
    dataset_manifest: Path,
    vjepa_checkpoint: Path,
    vjepa_implementation: Path,
) -> dict[str, Any]:
    """Validate and summarize the immutable Phase-2c evidence chain."""
    root = phase2c_output.resolve(strict=True)
    dataset_manifest = dataset_manifest.resolve(strict=True)
    vjepa_checkpoint = vjepa_checkpoint.resolve(strict=True)
    vjepa_implementation = vjepa_implementation.resolve(strict=True)
    required = {
        name: root / name
        for name in (
            "artifact_manifest.json",
            "asset_manifest.json",
            "comparison.json",
            "completion_gate.json",
            "dataset_fingerprint.json",
            "environment.json",
            "promotion_gate.json",
            "resolved_config.json",
            "wandb_artifact_receipt.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-2c source is incomplete: {missing}")

    artifact_manifest = _json(required["artifact_manifest.json"])
    _verify_output_manifest(root, artifact_manifest)
    comparison = _json(required["comparison.json"])
    config = _json(required["resolved_config.json"])
    completion = _json(required["completion_gate.json"])
    fingerprint = _json(required["dataset_fingerprint.json"])
    environment = _json(required["environment.json"])
    promotion = _json(required["promotion_gate.json"])
    wandb_receipt = _json(required["wandb_artifact_receipt.json"])

    if comparison.get("schema_version") != PHASE2C_COMPARISON_SCHEMA or comparison.get("failures"):
        raise ValueError("Phase-2c comparison is not a complete failure-free formal result")
    split_hash = comparison.get("split_hash")
    if not isinstance(split_hash, str) or len(split_hash) != 64 or config.get("split_hash") != split_hash:
        raise ValueError("Phase-2c split identity is missing or inconsistent")
    if (
        config.get("protocol") != "phase2c-cross-sequence-v1"
        or config.get("seeds") != [0, 1, 2]
        or int(config.get("epochs", -1)) != 60
        or config.get("split_counts") != {"train": 128, "validation": 64, "test": 128}
    ):
        raise ValueError("Phase-2c resolved protocol differs from the frozen formal protocol")
    if (
        completion.get("status") != "success"
        or int(completion.get("seed_failures", -1)) != 0
        or int(completion.get("probe_checkpoints", -1)) != 12
        or int(completion.get("result_rows", -1)) != 13
    ):
        raise ValueError("Phase-2c completion gate does not pass")
    if (
        promotion.get("schema_version") != PHASE2C_PROMOTION_SCHEMA
        or promotion.get("decision") != "retain_final_layer"
        or promotion.get("promoted") is not False
    ):
        raise ValueError("Phase-2c promotion record differs from the frozen decision")
    if (
        wandb_receipt.get("schema_version") != PHASE2C_WANDB_SCHEMA
        or wandb_receipt.get("status") != "success"
        or wandb_receipt.get("mode") != "online"
        or not wandb_receipt.get("run_id")
        or not wandb_receipt.get("artifact_name")
        or not wandb_receipt.get("artifact_version")
        or not wandb_receipt.get("artifact_digest")
    ):
        raise ValueError("Phase-2c online W&B artifact receipt is incomplete")
    recorded_manifest = fingerprint.get("manifest")
    if not isinstance(recorded_manifest, dict) or sha256_file(dataset_manifest) != recorded_manifest.get("sha256"):
        raise ValueError("current dataset manifest differs from the formal Phase-2c manifest")

    asset_manifest = _json(required["asset_manifest.json"])
    assets = {
        "vjepa_checkpoint": _verify_asset("vjepa_checkpoint", vjepa_checkpoint, asset_manifest),
        "vjepa_implementation": _verify_asset("vjepa_implementation", vjepa_implementation, asset_manifest),
    }
    git_commit = environment.get("git_commit")
    if not isinstance(git_commit, str) or len(git_commit) != 40 or environment.get("git_status") != "":
        raise ValueError("Phase-2c environment is not bound to a clean execution commit")
    return {
        "schema_version": SOURCE_IDENTITY_SCHEMA,
        "phase2c_output": str(root),
        "phase2c_git_commit": git_commit,
        "split_hash": split_hash,
        "dataset_manifest": {"path": str(dataset_manifest), "sha256": sha256_file(dataset_manifest)},
        "source_files": {name: sha256_file(path) for name, path in sorted(required.items())},
        "assets": assets,
        "wandb": {
            key: wandb_receipt[key]
            for key in ("run_id", "run_path", "run_url", "artifact_name", "artifact_version", "artifact_digest")
        },
    }
