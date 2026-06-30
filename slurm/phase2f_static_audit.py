#!/usr/bin/env python3
"""Run the Phase 2f static architecture/cache qualification before latency."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from jepa4d.evaluation.phase2f_data_cache import (
    validate_sun_development_feature_cache,
    validate_sun_development_input_cache,
    validate_sun_development_target_cache,
)
from jepa4d.evaluation.phase2f_metrics import publish_online_wandb
from jepa4d.models.phase2f_scale_geometry import Phase2fScaleGeometryProbe
from jepa4d.training.phase2f_training import phase2f_arm_configs
from slurm.phase2f_contract import atomic_json, file_identity, load_json, reject_secrets

SCHEMA = "jepa4d-phase2f-static-qualification-v1"
EXPECTED_PARAMETERS = {"M0": 86_402, "M1": 92_820, "M2": 92_916, "M3": 93_685}
HARD_PARAMETER_LIMIT = 95_042


def static_audit(input_cache: Path, feature_cache: Path, target_cache: Path, cache_receipt: Path) -> dict[str, Any]:
    receipt = load_json(cache_receipt)
    separation = receipt.get("target_separation")
    if separation != {
        "rgb_k_cache_contains_targets": False,
        "feature_cache_contains_targets": False,
        "target_cache_contains_rgb_k_or_features": False,
    }:
        raise ValueError("development cache target separation did not validate")
    sealed = receipt.get("sealed_archive_access_audit")
    if not isinstance(sealed, dict) or any(
        sealed.get(key) not in (0, False)
        for key in ("files_opened", "bytes_read", "values_loaded", "statistics_computed", "previews_generated")
    ):
        raise ValueError("development cache receipt does not prove zero sealed-final access")
    for path, validator in (
        (input_cache, validate_sun_development_input_cache),
        (feature_cache, validate_sun_development_feature_cache),
        (target_cache, validate_sun_development_target_cache),
    ):
        payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"cache must contain a mapping: {path}")
        validator(payload)
    configs = phase2f_arm_configs(768)
    parameters: dict[str, int] = {}
    components: dict[str, Any] = {}
    for arm in ("M0", "M1", "M2", "M3"):
        model = Phase2fScaleGeometryProbe(configs[arm])
        count = model.trainable_parameter_count
        if count != EXPECTED_PARAMETERS[arm] or count > HARD_PARAMETER_LIMIT:
            raise ValueError(f"{arm} parameter contract failed: {count}")
        parameters[arm] = count
        components[arm] = model.parameter_counts()
    return {
        "parameters": parameters,
        "component_parameters": components,
        "hard_parameter_limit": HARD_PARAMETER_LIMIT,
        "cache_receipt": file_identity(cache_receipt),
        "caches": {
            "input": file_identity(input_cache),
            "feature": file_identity(feature_cache),
            "target": file_identity(target_cache),
        },
        "sealed_final_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = load_json(args.provenance)
    result = {
        "schema_version": SCHEMA,
        "status": "success",
        "created_utc": datetime.now(UTC).isoformat(),
        **static_audit(args.input_cache, args.feature_cache, args.target_cache, args.cache_receipt),
        "execution_provenance": provenance,
    }
    reject_secrets(result)
    receipt = atomic_json(args.output, result)
    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    wandb = publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type="static-audit",
        run_name=f"{execution_id}-static-audit-{job_id}",
        config={"execution_id": execution_id, "git_commit": provenance["git_commit"]},
        summary={"status": "success", **result["parameters"]},
        artifact_name=f"phase2f-static-audit-{execution_id}",
        artifact_files=(receipt,),
    )
    result["wandb"] = wandb
    atomic_json(receipt, result)
    atomic_json(receipt.parent / "wandb_receipt.json", wandb)
    (receipt.parent / "SUCCESS").write_text("success\n", encoding="utf-8")
    print(json.dumps({"status": "success", "parameters": result["parameters"]}, sort_keys=True))


if __name__ == "__main__":
    main()
