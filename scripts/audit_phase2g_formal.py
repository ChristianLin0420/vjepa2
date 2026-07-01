#!/usr/bin/env python3
"""Audit Phase 2g architecture identities, membership, shards, and target isolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from jepa4d.evaluation.phase2g_data import (
    SUN_FAMILIES,
    assert_expected_parameter_counts,
    file_identity,
    load_torch,
    validate_feature_shard,
    validate_input_shard,
    validate_rotation_view,
    validate_sun_materialization,
    validate_sun_membership_manifest,
    validate_target_shard,
)
from jepa4d.training.phase2g_protocol import (
    CACHE_AUDIT_SCHEMA,
    CACHE_RECEIPT_SCHEMA,
    ROTATIONS,
    SAMPLES_PER_FAMILY,
)
from jepa4d.training.phase2g_runtime import (
    assert_same_execution,
    atomic_json,
    complete_output,
    finish_wandb_run,
    load_execution_provenance,
    load_json,
    prepare_output,
    start_wandb_run,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2g formal audit may run only inside Slurm")
    output = prepare_output(args.output)
    root = args.cache_root.resolve(strict=True)
    provenance = load_execution_provenance(args.provenance)
    cache_receipt = load_json(root / "cache_receipt.json")
    if cache_receipt.get("schema_version") != CACHE_RECEIPT_SCHEMA or cache_receipt.get("status") != "success":
        raise ValueError("formal audit requires a successful Phase 2g cache receipt")
    assert_same_execution((cache_receipt,), provenance)
    materialization_root = args.materialization_root.resolve(strict=True)
    materialization_receipt_path = materialization_root / "materialization_receipt.json"
    materialization = validate_sun_materialization(
        materialization_root / "SUNRGBD",
        materialization_receipt_path,
        provenance=provenance,
    )
    materialization_identity = file_identity(materialization_receipt_path)
    if cache_receipt.get("source_materialization", {}).get("sha256") != materialization_identity["sha256"]:
        raise ValueError("cache receipt is not bound to the revalidated archive-derived materialization")
    manifest = load_json(root / "sun_membership.json")
    validate_sun_membership_manifest(manifest)
    if file_identity(root / "sun_membership.json")["sha256"] != cache_receipt["membership"]["sha256"]:
        raise ValueError("cache receipt membership file identity changed")
    materialized_files = {row["path"]: row for row in materialization["files"]}
    for selected in manifest["selected_samples"]:
        for source in selected["files"].values():
            observed = materialized_files.get(source["path"])
            if observed != {key: source[key] for key in ("path", "bytes", "sha256")}:
                raise ValueError("selected membership source identity differs from archive materialization")
    shard_hashes: dict[str, dict[str, str]] = {}
    for family in SUN_FAMILIES:
        values = {kind: load_torch(root / "shards" / family / f"{kind}.pt") for kind in ("input", "feature", "target")}
        validate_input_shard(values["input"], expected_count=SAMPLES_PER_FAMILY)
        validate_feature_shard(values["feature"], expected_count=SAMPLES_PER_FAMILY)
        validate_target_shard(values["target"], expected_count=SAMPLES_PER_FAMILY)
        if not (values["input"]["samples"] == values["feature"]["samples"] == values["target"]["samples"]):
            raise ValueError(f"cache shard row identity mismatch for {family}")
        input_identity = file_identity(root / "shards" / family / "input.pt")
        if (
            values["feature"]["input_sha256"] != input_identity["sha256"]
            or values["target"]["input_sha256"] != input_identity["sha256"]
        ):
            raise ValueError(f"cache shard input hash binding failed for {family}")
        shard_hashes[family] = {}
        for kind in ("input", "feature", "target"):
            identity = file_identity(root / "shards" / family / f"{kind}.pt")
            expected = cache_receipt.get("shards", {}).get(family, {}).get(kind, {})
            if identity["sha256"] != expected.get("sha256") or identity["bytes"] != expected.get("bytes"):
                raise ValueError(f"cache receipt {family}/{kind} shard identity changed")
            shard_hashes[family][kind] = identity["sha256"]
    view_hashes = {
        rotation: validate_rotation_view(root / "rotations" / rotation, expected_rotation=rotation)["view_sha256"]
        for rotation in ROTATIONS
    }
    for rotation, view_sha256 in view_hashes.items():
        if cache_receipt.get("rotation_views", {}).get(rotation, {}).get("view_sha256") != view_sha256:
            raise ValueError(f"cache receipt rotation view identity changed for {rotation}")
    parameter_counts = assert_expected_parameter_counts()
    run = start_wandb_run(
        provenance=provenance,
        job_type="formal-audit",
        semantic_name="formal-audit",
        config={"families": list(SUN_FAMILIES), "samples_per_family": SAMPLES_PER_FAMILY},
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    receipt = {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "status": "pass",
        "membership_sha256": manifest["manifest_sha256"],
        "membership_file_sha256": file_identity(root / "sun_membership.json")["sha256"],
        "family_counts": {family: SAMPLES_PER_FAMILY for family in SUN_FAMILIES},
        "source_materialization": {
            "receipt_sha256": materialization_identity["sha256"],
            "archive_sha256": materialization["archive"]["sha256"],
            "files_manifest_sha256": materialization["files_manifest_sha256"],
            "file_count": materialization["file_count"],
            "selected_source_identities_revalidated": 3 * len(manifest["selected_samples"]),
        },
        "shard_sha256": shard_hashes,
        "rotation_view_sha256": view_hashes,
        "parameter_counts": parameter_counts,
        "target_separation": {
            "input_contains_targets": False,
            "feature_contains_targets": False,
            "each_training_view_contains_exactly_train_plus_validation_targets": True,
            "heldout_target_paths_absent_from_training_views": True,
        },
        "external_final": {"path_exposed": False, "accessed": False},
        "cache_receipt": file_identity(root / "cache_receipt.json", schema=CACHE_RECEIPT_SCHEMA),
        "execution_provenance": provenance,
    }
    atomic_json(output / "audit_receipt.json", receipt)
    report = atomic_json(
        output / "audit_report.json",
        {
            "schema_version": "jepa4d-phase2g-cache-audit-report-v1",
            "status": "pass",
            "membership_sha256": file_identity(root / "sun_membership.json")["sha256"],
            "membership_canonical_sha256": manifest["manifest_sha256"],
            "family_counts": receipt["family_counts"],
            "source_archive_sha256": materialization["archive"]["sha256"],
            "materialization_manifest_sha256": materialization["files_manifest_sha256"],
            "materialized_file_count": materialization["file_count"],
            "selected_source_identities_revalidated": receipt["source_materialization"][
                "selected_source_identities_revalidated"
            ],
            "parameter_counts": parameter_counts,
            "target_separation": receipt["target_separation"],
            "source_paths_uploaded": False,
        },
    )
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-formal-audit-{provenance['execution_id']}",
        job_type="formal-audit",
        files=(report,),
        summary={"status": "pass", "samples": 4 * SAMPLES_PER_FAMILY},
    )
    complete_output(output, receipt_name="audit_receipt.json", receipt=receipt, wandb_receipt=wandb_receipt)
    print(json.dumps({"status": "pass", "membership_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
