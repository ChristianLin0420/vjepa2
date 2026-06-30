#!/usr/bin/env python3
"""Write separated Phase 2f SUN development input/target/feature caches."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from jepa4d.benchmarks.geometry.sun_rgbd import (
    load_sensor_blocked_manifest,
    rank_midpoint_selection,
    validate_sunrgbd_frame,
)
from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.phase2e_feature_cache import (
    FROZEN_MANIFEST_SHA256,
    FROZEN_MODEL_IDENTITIES,
    FROZEN_SPLIT_SHA256,
)
from jepa4d.evaluation.phase2f_data_cache import (
    CLAIM_BOUNDARY,
    SUN_DEVELOPMENT_RECEIPT_SCHEMA,
    SUN_FAMILIES,
    build_sun_development_feature_cache,
    build_sun_development_input_cache,
    build_sun_development_target_cache,
    prepare_sun_development_frames,
    reject_external_target_references,
    sha256_file,
    write_cache,
)
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor

SOURCE_BUNDLE_SCHEMA = "jepa4d-phase2f-sun-dev-source-bundle-v1"
WANDB_RECEIPT_SCHEMA = "jepa4d-phase2f-wandb-artifact-receipt-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-bundle", type=Path)
    mode.add_argument("--sun-root", type=Path)
    parser.add_argument("--sun-manifest", type=Path)
    parser.add_argument("--vjepa-checkpoint", type=Path)
    parser.add_argument("--vjepa-implementation", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--execution-provenance-json", type=Path)
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="phase2f-scale-camera")
    parser.add_argument("--wandb-run-name", default="phase2f-sun-dev-cache")
    parser.add_argument("--wandb-receipt-output", type=Path)
    parser.add_argument("--input-cache-output", type=Path, required=True)
    parser.add_argument("--target-cache-output", type=Path, required=True)
    parser.add_argument("--feature-cache-output", type=Path)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> Path:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _write_html_report(path: Path, report: dict[str, Any]) -> Path:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        f"<section><strong>{html.escape(str(key))}</strong><span>{html.escape(str(value))}</span></section>"
        for key, value in report.items()
        if key != "cache_sha256"
    )
    raw = html.escape(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    document = f"""<!doctype html><html lang='en'><meta charset='utf-8'>
<title>Phase 2f SUN development-cache audit</title><style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:28px}}
main{{max-width:1100px;margin:auto}} .boundary{{background:#fff1d6;border-left:5px solid #d98216;padding:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}}
section{{background:white;border-radius:10px;padding:14px;box-shadow:0 1px 5px #cbd2df;display:flex;flex-direction:column}}
section span{{font-size:1.08rem;margin-top:8px;overflow-wrap:anywhere}} pre{{background:#172033;color:#f4f7fc;padding:16px;overflow:auto}}
</style><main><h1>Phase 2f SUN development-cache audit</h1>
<p class='boundary'>{html.escape(CLAIM_BOUNDARY)}. This report contains aggregate cache/control gates only; no RGB or target preview.</p>
<div class='cards'>{cards}</div><details><summary>Machine-readable summary</summary><pre>{raw}</pre></details>
</main></html>"""
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return output


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


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
    grids = []
    with torch.inference_mode():
        for offset in range(0, len(flattened), chunk_size):
            images = flattened[offset : offset + chunk_size].float().div(255)
            bundle = extractor(_single_image_batch(images))
            grid = bundle.dense_tokens[:, 0, 0].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2)
            grids.append(grid.detach().cpu().half().contiguous())
    return torch.cat(grids).reshape(*prefix, 768, 24, 24)


def _build_direct_source(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = {
        "sun_manifest": args.sun_manifest,
        "vjepa_checkpoint": args.vjepa_checkpoint,
        "vjepa_implementation": args.vjepa_implementation,
        "execution_provenance_json": args.execution_provenance_json,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"direct SUN mode is missing arguments: {missing}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("direct SUN cache building requires an allocated CUDA device")
    manifest_identity = _file_identity(args.sun_manifest)
    if manifest_identity["sha256"] != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen Phase 2e SUN manifest SHA-256 changed")
    bundle = load_sensor_blocked_manifest(
        args.sun_root,
        args.sun_manifest,
        verify_file_hashes=True,
        validate_depth=False,
    )
    if bundle.split_hash != FROZEN_SPLIT_SHA256:
        raise ValueError("frozen Phase 2e SUN split hash changed")
    selected = []
    for family in SUN_FAMILIES:
        candidates = sorted(
            (frame for frame in bundle.samples if frame.sensor == family), key=lambda frame: frame.sample_id
        )
        selected.extend(rank_midpoint_selection(candidates, 128))
    clamp_value = bundle.manifest["protocol"]["depth_decode"]["clamp_max_depth_m"]
    clamp = None if clamp_value is None else float(clamp_value)
    validated = [validate_sunrgbd_frame(frame, clamp_max_depth_m=clamp)[0] for frame in selected]
    selection_rows = [{"sample_id": frame.sample_id, "family": frame.sensor} for frame in validated]
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    prepared = prepare_sun_development_frames(validated, clamp_max_depth_m=clamp)
    input_payload = build_sun_development_input_cache(
        sample_ids=prepared.sample_ids,
        family_ids=prepared.family_ids,
        images_384=prepared.images_384,
        rgb_96=prepared.rgb_96,
        intrinsics_384=prepared.intrinsics_384,
        sample_manifest_sha256=selection_sha256,
    )
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
    started = time.perf_counter()
    extractor = VJEPA21FeatureExtractor(
        model_name="vjepa2_1_vit_base_384",
        checkpoint=args.vjepa_checkpoint,
        implementation_path=args.vjepa_implementation,
        backend="hf_compat",
        device=args.device,
        frozen=True,
        capture_layers=(),
    )
    ordinary_features = _extract_features(
        extractor,
        input_payload["ordinary_inputs"]["images_384_uint8"],
        prefix=(len(validated), 2),
        chunk_size=args.chunk_size,
    )
    paired_features = _extract_features(
        extractor,
        input_payload["paired_inputs"]["images_384_uint8"],
        prefix=(len(validated), 8),
        chunk_size=args.chunk_size,
    )
    source = {
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "sample_manifest_sha256": selection_sha256,
        "sample_ids": prepared.sample_ids,
        "family_ids": prepared.family_ids,
        "images_384": prepared.images_384,
        "rgb_96": prepared.rgb_96,
        "intrinsics_384": prepared.intrinsics_384,
        "ordinary_depth_24": prepared.ordinary_depth_24,
        "ordinary_valid_24": prepared.ordinary_valid_24,
        "center_depth_384": prepared.center_depth_384,
        "center_valid_384": prepared.center_valid_384,
        "ordinary_features": ordinary_features,
        "paired_features": paired_features,
    }
    provenance = {
        "sun_manifest": manifest_identity,
        "sun_split_sha256": bundle.split_hash,
        "selection_sha256": selection_sha256,
        "family_counts": dict(Counter(prepared.family_ids)),
        "vjepa_checkpoint": checkpoint_identity,
        "vjepa_implementation": implementation_identity,
        "feature_extraction_seconds": time.perf_counter() - started,
        "device": args.device,
        "features_raw_unnormalized": True,
    }
    return source, provenance, input_payload


def load_source_bundle(path: Path) -> dict[str, Any]:
    """Load only the exact SUN development tensor contract; reject external-target references."""

    source = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "sample_manifest_sha256",
        "sample_ids",
        "family_ids",
        "images_384",
        "rgb_96",
        "intrinsics_384",
        "ordinary_depth_24",
        "ordinary_valid_24",
        "center_depth_384",
        "center_valid_384",
    }
    optional = {"ordinary_features", "paired_features"}
    if not isinstance(source, dict) or set(source) not in (required, required | optional):
        raise ValueError("SUN development source bundle has unexpected fields")
    if source.get("schema_version") != SOURCE_BUNDLE_SCHEMA:
        raise ValueError(f"unexpected SUN development source schema: {source.get('schema_version')}")
    if ("ordinary_features" in source) != ("paired_features" in source):
        raise ValueError("ordinary_features and paired_features must be present together")
    reject_external_target_references(source)
    return source


def build_receipt(
    *,
    source_bundle: Path | None,
    input_cache: Path,
    target_cache: Path,
    feature_cache: Path | None,
    input_payload: dict[str, Any],
    source_provenance: dict[str, Any] | None = None,
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit = input_payload["audit"]
    return {
        "schema_version": SUN_DEVELOPMENT_RECEIPT_SCHEMA,
        "status": "success",
        "created_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": "sun-rgbd-official",
        "source_bundle": None if source_bundle is None else _file_identity(source_bundle),
        "source_provenance": {} if source_provenance is None else source_provenance,
        "execution_provenance": {} if execution_provenance is None else execution_provenance,
        "caches": {
            "input": _file_identity(input_cache),
            "target": _file_identity(target_cache),
            "feature": None if feature_cache is None else _file_identity(feature_cache),
        },
        "controls": {
            "profile_ids": [f"P{index}" for index in range(8)],
            "profile_permutation": [5, 6, 3, 2, 1, 7, 0, 4],
            "distinct_updated_intrinsics_per_source_min": audit["distinct_updated_intrinsics_per_source_min"],
            "permutation_assignment_change_fraction": audit["permutation_assignment_change_fraction"],
            "permutation_matrix_change_fraction": audit["permutation_matrix_change_fraction"],
        },
        "target_separation": {
            "rgb_k_cache_contains_targets": False,
            "feature_cache_contains_targets": False,
            "target_cache_contains_rgb_k_or_features": False,
        },
        "sealed_archive_access_audit": {
            "paths_present_in_source_or_caches": False,
            "files_opened": 0,
            "bytes_read": 0,
            "values_loaded": False,
            "statistics_computed": False,
            "previews_generated": False,
        },
    }


def _finish_online_wandb(
    run: Any,
    *,
    receipt_path: Path,
    report_path: Path,
    report_html_path: Path,
) -> dict[str, Any]:
    import wandb

    artifact = wandb.Artifact(f"phase2f-sun-dev-cache-{run.id}-receipt", type="phase2f-dev-cache")
    artifact.add_file(str(receipt_path.resolve(strict=True)), name=receipt_path.name)
    artifact.add_file(str(report_path.resolve(strict=True)), name=report_path.name)
    artifact.add_file(str(report_html_path.resolve(strict=True)), name=report_html_path.name)
    uploaded = run.log_artifact(artifact).wait(timeout=900)
    files = [receipt_path, report_path, report_html_path]
    return {
        "schema_version": WANDB_RECEIPT_SCHEMA,
        "status": "success",
        "mode": "online",
        "entity": str(run.entity),
        "project": str(run.project),
        "group": str(run.group),
        "job_type": "dev-cache",
        "run_name": str(run.name),
        "run_id": str(run.id),
        "run_url": str(run.url),
        "artifact_id": str(uploaded.id),
        "artifact_name": f"phase2f-sun-dev-cache-{run.id}-receipt",
        "artifact_version": str(uploaded.version),
        "artifact_digest": str(uploaded.digest),
        "files": [_file_identity(path) for path in files],
        "large_caches_uploaded": False,
        "rgb_or_raw_targets_uploaded": False,
    }


def main() -> None:
    args = _parse_args()
    execution_provenance: dict[str, Any] = {}
    if args.execution_provenance_json is not None:
        loaded_provenance = json.loads(args.execution_provenance_json.resolve(strict=True).read_text(encoding="utf-8"))
        if not isinstance(loaded_provenance, dict) or not loaded_provenance:
            raise ValueError("--execution-provenance-json must contain a non-empty JSON object")
        reject_external_target_references(loaded_provenance)
        execution_provenance = loaded_provenance

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name,
        mode="online",
        job_type="dev-cache",
        tags=["phase-2f", "SUN-RGBD", "feature-cache", "camera-controls"],
        config={
            "mode": "prebuilt" if args.source_bundle is not None else "direct",
            "device": args.device,
            "chunk_size": args.chunk_size,
            "claim_boundary": CLAIM_BOUNDARY,
            "large_caches_uploaded": False,
            "rgb_or_raw_targets_uploaded": False,
        },
    )
    if run.offline:
        run.finish(exit_code=1)
        raise RuntimeError("Phase 2f cache building requires online W&B")
    started = time.perf_counter()
    try:
        run.log({"cache/progress": 0.05, "cache/stage": "validate-source"})
        if args.source_bundle is not None:
            source = load_source_bundle(args.source_bundle)
            source_provenance: dict[str, Any] = {
                "mode": "prebuilt-sun-development-source-bundle",
                "identity": _file_identity(args.source_bundle),
            }
            input_payload = build_sun_development_input_cache(
                sample_ids=source["sample_ids"],
                family_ids=source["family_ids"],
                images_384=source["images_384"],
                rgb_96=source["rgb_96"],
                intrinsics_384=source["intrinsics_384"],
                sample_manifest_sha256=source["sample_manifest_sha256"],
            )
        else:
            source, source_provenance, input_payload = _build_direct_source(args)
            source_provenance = {"mode": "direct-frozen-sun-and-vjepa", **source_provenance}
        has_features = "ordinary_features" in source
        if has_features != (args.feature_cache_output is not None):
            raise ValueError("--feature-cache-output is required exactly when source features are present")
        run.log({"cache/progress": 0.55, "cache/stage": "write-separated-caches"})
        input_path = write_cache(args.input_cache_output, input_payload)
        input_sha256 = sha256_file(input_path)
        target_payload = build_sun_development_target_cache(
            input_payload,
            ordinary_depth_24=source["ordinary_depth_24"],
            ordinary_valid_24=source["ordinary_valid_24"],
            center_depth_384=source["center_depth_384"],
            center_valid_384=source["center_valid_384"],
            input_cache_sha256=input_sha256,
        )
        target_path = write_cache(args.target_cache_output, target_payload)
        feature_path = None
        if has_features:
            feature_payload = build_sun_development_feature_cache(
                input_payload,
                ordinary_features=source["ordinary_features"],
                paired_features=source["paired_features"],
                input_cache_sha256=input_sha256,
            )
            feature_path = write_cache(args.feature_cache_output, feature_payload)
        receipt = build_receipt(
            source_bundle=args.source_bundle,
            input_cache=input_path,
            target_cache=target_path,
            feature_cache=feature_path,
            input_payload=input_payload,
            source_provenance=source_provenance,
            execution_provenance=execution_provenance,
        )
        receipt_path = _write_json(args.receipt_output, receipt)
        elapsed_seconds = time.perf_counter() - started
        report = {
            "schema_version": "jepa4d-phase2f-cache-report-v1",
            "status": "success",
            "sample_count": input_payload["audit"]["sample_count"],
            "family_counts": input_payload["audit"]["family_counts"],
            "profiles_per_sample": input_payload["audit"]["profiles_per_sample"],
            "distinct_updated_intrinsics_per_source_min": input_payload["audit"][
                "distinct_updated_intrinsics_per_source_min"
            ],
            "permutation_matrix_change_fraction": input_payload["audit"]["permutation_matrix_change_fraction"],
            "elapsed_seconds": elapsed_seconds,
            "cache_sha256": {name: value["sha256"] for name, value in receipt["caches"].items() if value},
            "sealed_archive_files_opened": 0,
            "large_caches_uploaded": False,
            "rgb_or_raw_targets_uploaded": False,
        }
        report_path = _write_json(
            receipt_path.with_name(f"{receipt_path.stem}.report.json"),
            report,
        )
        report_html_path = _write_html_report(
            receipt_path.with_name(f"{receipt_path.stem}.report.html"),
            report,
        )
        family_table = wandb.Table(columns=["family", "samples"])
        for family in SUN_FAMILIES:
            family_table.add_data(family, input_payload["audit"]["family_counts"][family])
        run.log(
            {
                "cache/progress": 1.0,
                "cache/stage": "complete",
                "cache/elapsed_seconds": elapsed_seconds,
                "cache/sample_count": input_payload["audit"]["sample_count"],
                "cache/family_counts": family_table,
                "cache/distinct_updated_intrinsics_per_source_min": input_payload["audit"][
                    "distinct_updated_intrinsics_per_source_min"
                ],
                "cache/permutation_matrix_change_fraction": input_payload["audit"][
                    "permutation_matrix_change_fraction"
                ],
                "cache/report": wandb.Html(str(report_html_path), inject=False),
            }
        )
        wandb_receipt = _finish_online_wandb(
            run,
            receipt_path=receipt_path,
            report_path=report_path,
            report_html_path=report_html_path,
        )
        receipt["wandb"] = wandb_receipt
        receipt_path = _write_json(args.receipt_output, receipt)
        wandb_receipt_path = _write_json(
            args.wandb_receipt_output or receipt_path.with_name(f"{receipt_path.stem}.wandb.json"),
            wandb_receipt,
        )
        run.summary.update(
            {
                "status": "success",
                "input_cache_sha256": input_sha256,
                "large_caches_uploaded": False,
                "rgb_or_raw_targets_uploaded": False,
            }
        )
        run.finish(exit_code=0)
    except Exception:
        run.summary["status"] = "failure"
        run.finish(exit_code=1)
        raise
    print(
        json.dumps(
            {
                "input_cache": str(input_path),
                "receipt": str(receipt_path),
                "wandb_receipt": str(wandb_receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
