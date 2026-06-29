"""Strict local and W&B-receipt validation for a Phase 2e training shard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from scripts.run_phase2e_factorized_shard import validate_shard_artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(
    output: Annotated[Path, typer.Option("--output")],
    expected_variants: Annotated[str, typer.Option("--expected-variants")],
    expected_epochs: Annotated[int, typer.Option("--expected-epochs")],
    expected_feature_cache: Annotated[Path, typer.Option("--expected-feature-cache")],
) -> None:
    output = output.resolve(strict=True)
    if (output / "run_failure.json").exists():
        raise RuntimeError("Phase 2e shard contains a run_failure.json")
    expected_feature_cache = expected_feature_cache.resolve(strict=True)
    shard = json.loads((output / "phase2e_shard.json").read_text())
    config = json.loads((output / "resolved_config.json").read_text())
    receipt = json.loads((output / "wandb_receipt.json").read_text())
    expected = [value.strip() for value in expected_variants.split(",") if value.strip()]
    if shard.get("schema_version") != "jepa4d-phase2e-training-shard-v1" or shard.get("status") != "success":
        raise RuntimeError("Phase 2e shard does not report success")
    if config.get("variants") != expected or config.get("seeds") != [0, 1, 2]:
        raise RuntimeError("Phase 2e shard variants/seeds differ from the submitted protocol")
    if int(config.get("epochs", -1)) != expected_epochs:
        raise RuntimeError("Phase 2e shard epoch count differs from the submitted protocol")
    telemetry = shard.get("gpu_telemetry")
    required_statistics = {"utilization_gpu", "memory_used_mib", "temperature_c", "power_w"}
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("available") is not True
        or int(telemetry.get("samples", 0)) < 1
        or not isinstance(telemetry.get("statistics"), dict)
        or not required_statistics <= set(telemetry["statistics"])
    ):
        raise RuntimeError("Phase 2e shard lacks required numeric GPU telemetry")
    expected_cache_sha256 = _sha256(expected_feature_cache)
    cache_config = config.get("feature_cache")
    if (
        not isinstance(cache_config, dict)
        or Path(str(cache_config.get("path", ""))).resolve() != expected_feature_cache
        or cache_config.get("schema_version") != "jepa4d-phase2e-feature-cache-v1"
        or cache_config.get("sha256") != expected_cache_sha256
        or shard.get("feature_cache_sha256") != expected_cache_sha256
    ):
        raise RuntimeError("Phase 2e shard is not bound to the expected feature cache")
    rows = shard.get("results", [])
    if len(rows) != len(expected) * 3:
        raise RuntimeError("Phase 2e shard result count is incomplete")
    observed = {(str(row["variant"]), int(row["seed"])) for row in rows}
    required = {(variant, seed) for variant in expected for seed in (0, 1, 2)}
    if observed != required:
        raise RuntimeError("Phase 2e shard is missing a variant/seed result")
    for row in rows:
        history_path = Path(str(row.get("history", ""))).resolve(strict=True)
        if len([line for line in history_path.read_text().splitlines() if line.strip()]) != expected_epochs:
            raise RuntimeError(f"Phase 2e history is not exactly {expected_epochs} epochs: {history_path}")
    expected_artifact_manifest = validate_shard_artifacts(output, shard)
    artifact_manifest_path = output / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    if _canonical(artifact_manifest) != _canonical(expected_artifact_manifest):
        raise RuntimeError("Phase 2e artifact manifest differs from strict local reconstruction")
    if (
        receipt.get("schema_version") != "jepa4d-phase2e-wandb-artifact-receipt-v1"
        or receipt.get("status") != "uploaded"
        or receipt.get("mode") != "online"
        or not receipt.get("run_id")
        or not receipt.get("artifact_id")
        or not receipt.get("artifact_version")
        or not receipt.get("artifact_digest")
    ):
        raise RuntimeError("Phase 2e W&B artifact receipt is incomplete")
    if receipt.get("artifact_manifest_sha256") != _sha256(artifact_manifest_path):
        raise RuntimeError("Phase 2e W&B receipt is not bound to artifact_manifest.json")
    if receipt.get("phase2e_shard_sha256") != _sha256(output / "phase2e_shard.json"):
        raise RuntimeError("Phase 2e W&B receipt is not bound to phase2e_shard.json")


if __name__ == "__main__":
    typer.run(main)
