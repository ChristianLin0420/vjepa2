#!/usr/bin/env python3
"""Build the frozen 4x1024 SUN RGB-D membership and isolated Phase 2g shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.phase2e_feature_cache import FROZEN_MODEL_IDENTITIES
from jepa4d.evaluation.phase2f_data_cache import prepare_sun_development_frames
from jepa4d.evaluation.phase2g_data import (
    FEATURE_SHARD_SCHEMA,
    INPUT_SHARD_SCHEMA,
    MATERIALIZATION_SCHEMA,
    MEMBERSHIP_SCHEMA,
    SUN_FAMILIES,
    TARGET_SHARD_SCHEMA,
    atomic_json,
    build_feature_shard,
    build_input_shard,
    build_sun_membership_manifest,
    build_target_shard,
    create_rotation_views,
    file_identity,
    sha256_file,
    validate_sun_materialization,
    write_torch_atomic,
)
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.training.phase2g_protocol import CACHE_RECEIPT_SCHEMA, SAMPLES_PER_FAMILY
from jepa4d.training.phase2g_runtime import (
    complete_output,
    finish_wandb_run,
    load_execution_provenance,
    prepare_output,
    start_wandb_run,
)


def _path_identity(path: Path) -> dict[str, Any]:
    root = path.resolve(strict=True)
    files = (
        [root]
        if root.is_file()
        else sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
        )
    )
    combined = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest = sha256_file(item)
        size = item.stat().st_size
        combined.update(relative.encode())
        combined.update(str(size).encode())
        combined.update(digest.encode())
        total_bytes += size
    return {
        "path": str(root),
        "files": len(files),
        "bytes": total_bytes,
        "content_manifest_sha256": combined.hexdigest(),
    }


def _single_image_batch(images: torch.Tensor) -> Any:
    return collate_rgb_inputs([from_view_sequences([[image]]) for image in images])


def _extract_features(
    extractor: VJEPA21FeatureExtractor,
    images_uint8: torch.Tensor,
    *,
    prefix: tuple[int, ...],
    chunk_size: int,
) -> torch.Tensor:
    flattened = images_uint8.flatten(0, len(prefix) - 1)
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, len(flattened), chunk_size):
            images = flattened[offset : offset + chunk_size].float().div(255)
            bundle = extractor(_single_image_batch(images))
            grid = bundle.dense_tokens[:, 0, 0].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2)
            rows.append(grid.detach().cpu().half().contiguous())
    return torch.cat(rows).reshape(*prefix, 768, 24, 24)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sun-root", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal Phase 2g cache building may run only in Slurm")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase 2g cache building requires an allocated CUDA device")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    output = prepare_output(args.output)
    provenance = load_execution_provenance(args.provenance)
    materialization = validate_sun_materialization(
        args.sun_root,
        args.materialization_receipt,
        provenance=provenance,
    )
    run = start_wandb_run(
        provenance=provenance,
        job_type="cache",
        semantic_name="cache",
        config={
            "samples_per_family": SAMPLES_PER_FAMILY,
            "families": list(SUN_FAMILIES),
            "device": args.device,
            "chunk_size": args.chunk_size,
            "archive_derived_materialization": True,
            "materialization_manifest_sha256": materialization["files_manifest_sha256"],
            "raw_data_uploaded": False,
            "membership_manifest_uploaded": False,
        },
        entity=args.wandb_entity,
        project=args.wandb_project,
    )
    started = time.perf_counter()
    run.log({"cache/progress": 0.01, "cache/stage": "freeze-membership"})
    manifest, selected = build_sun_membership_manifest(args.sun_root)
    manifest_path = atomic_json(output / "sun_membership.json", manifest)
    checkpoint_identity = _path_identity(args.vjepa_checkpoint)
    implementation_identity = _path_identity(args.vjepa_implementation)
    if (
        checkpoint_identity["content_manifest_sha256"]
        != FROZEN_MODEL_IDENTITIES["vjepa_checkpoint"]["content_manifest_sha256"]
    ):
        raise ValueError("frozen V-JEPA checkpoint identity changed")
    if (
        implementation_identity["content_manifest_sha256"]
        != FROZEN_MODEL_IDENTITIES["vjepa_implementation"]["content_manifest_sha256"]
    ):
        raise ValueError("matched V-JEPA implementation identity changed")
    extractor = VJEPA21FeatureExtractor(
        model_name="vjepa2_1_vit_base_384",
        checkpoint=args.vjepa_checkpoint,
        implementation_path=args.vjepa_implementation,
        backend="hf_compat",
        device=args.device,
        frozen=True,
        capture_layers=(),
    )
    shard_identities: dict[str, dict[str, dict[str, Any]]] = {}
    for family_index, family in enumerate(SUN_FAMILIES):
        run.log(
            {
                "cache/progress": 0.05 + 0.2 * family_index,
                "cache/stage": f"build-family-{family}",
            }
        )
        prepared = prepare_sun_development_frames(selected[family], clamp_max_depth_m=None)
        input_payload = build_input_shard(
            family=family,
            sample_ids=prepared.sample_ids,
            images_384=prepared.images_384,
            rgb_96=prepared.rgb_96,
            intrinsics_384=prepared.intrinsics_384,
            membership_sha256=manifest["manifest_sha256"],
        )
        shard_root = output / "shards" / family
        input_path = write_torch_atomic(shard_root / "input.pt", input_payload)
        input_sha256 = file_identity(input_path)["sha256"]
        ordinary_features = _extract_features(
            extractor,
            input_payload["ordinary_inputs"]["images_384_uint8"],
            prefix=(SAMPLES_PER_FAMILY, 2),
            chunk_size=args.chunk_size,
        )
        paired_features = _extract_features(
            extractor,
            input_payload["paired_inputs"]["images_384_uint8"],
            prefix=(SAMPLES_PER_FAMILY, 8),
            chunk_size=args.chunk_size,
        )
        feature_payload = build_feature_shard(
            input_payload,
            ordinary_features=ordinary_features,
            paired_features=paired_features,
            input_sha256=input_sha256,
        )
        feature_path = write_torch_atomic(shard_root / "feature.pt", feature_payload)
        target_payload = build_target_shard(
            input_payload,
            ordinary_depth_24=prepared.ordinary_depth_24,
            ordinary_valid_24=prepared.ordinary_valid_24,
            center_depth_384=prepared.center_depth_384,
            center_valid_384=prepared.center_valid_384,
            input_sha256=input_sha256,
        )
        target_path = write_torch_atomic(shard_root / "target.pt", target_payload)
        shard_identities[family] = {
            "input": file_identity(input_path, schema=INPUT_SHARD_SCHEMA),
            "feature": file_identity(feature_path, schema=FEATURE_SHARD_SCHEMA),
            "target": file_identity(target_path, schema=TARGET_SHARD_SCHEMA),
        }
        del prepared, input_payload, feature_payload, target_payload, ordinary_features, paired_features
        gc.collect()
    views = create_rotation_views(
        output,
        membership_sha256=manifest["manifest_sha256"],
        shard_identities=shard_identities,
    )
    receipt = {
        "schema_version": CACHE_RECEIPT_SCHEMA,
        "status": "success",
        "membership": file_identity(manifest_path, schema=MEMBERSHIP_SCHEMA),
        "membership_sha256": manifest["manifest_sha256"],
        "source_materialization": {
            **file_identity(args.materialization_receipt, schema=MATERIALIZATION_SCHEMA),
            "archive_sha256": materialization["archive"]["sha256"],
            "files_manifest_sha256": materialization["files_manifest_sha256"],
            "file_count": materialization["file_count"],
        },
        "samples_per_family": SAMPLES_PER_FAMILY,
        "family_counts": {family: SAMPLES_PER_FAMILY for family in SUN_FAMILIES},
        "qualitative_ids": manifest["qualitative_ids"],
        "shards": shard_identities,
        "rotation_views": {
            rotation: {
                "path": f"rotations/{rotation}",
                "view_sha256": descriptor["view_sha256"],
                "heldout_family": descriptor["heldout_family"],
                "heldout_target_exposed": False,
            }
            for rotation, descriptor in views.items()
        },
        "model_identity": {
            "vjepa_checkpoint": checkpoint_identity,
            "vjepa_implementation": implementation_identity,
        },
        "target_separation": {
            "input_shards_contain_targets": False,
            "feature_shards_contain_targets": False,
            "training_views_expose_heldout_targets": False,
        },
        "external_final": {
            "path_exposed": False,
            "files_opened": 0,
            "bytes_read": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "execution_provenance": provenance,
    }
    atomic_json(output / "cache_receipt.json", receipt)
    # This W&B report contains counts/hashes only.  The local membership file,
    # selected source paths, RGB, targets, and large feature caches are excluded.
    report = atomic_json(
        output / "cache_report.json",
        {
            "schema_version": "jepa4d-phase2g-cache-report-v1",
            "status": "success",
            "membership_sha256": receipt["membership"]["sha256"],
            "membership_canonical_sha256": manifest["manifest_sha256"],
            "source_archive_sha256": materialization["archive"]["sha256"],
            "materialization_manifest_sha256": materialization["files_manifest_sha256"],
            "materialized_file_count": materialization["file_count"],
            "family_counts": receipt["family_counts"],
            "qualitative_ids_sha256": {
                family: __import__("hashlib")
                .sha256(json.dumps(manifest["qualitative_ids"][family], separators=(",", ":")).encode())
                .hexdigest()
                for family in SUN_FAMILIES
            },
            "target_separation": receipt["target_separation"],
            "membership_manifest_uploaded": False,
            "source_paths_uploaded": False,
            "raw_rgb_or_targets_uploaded": False,
            "large_caches_uploaded": False,
        },
    )
    wandb_receipt = finish_wandb_run(
        run,
        artifact_name=f"phase2g-cache-{provenance['execution_id']}",
        job_type="cache",
        files=(report,),
        summary={"families": 4, "samples": 4 * SAMPLES_PER_FAMILY, "membership_manifest_uploaded": False},
    )
    complete_output(
        output,
        receipt_name="cache_receipt.json",
        receipt=receipt,
        wandb_receipt=wandb_receipt,
    )
    print(json.dumps({"status": "success", "membership_sha256": manifest["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
