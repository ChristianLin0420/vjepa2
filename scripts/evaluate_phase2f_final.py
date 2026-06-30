#!/usr/bin/env python3
"""Run the single guarded Phase 2f M0-versus-survivor DIODE evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from jepa4d.data.camera_geometry import update_intrinsics_for_crop_resize
from jepa4d.data.rgb_input import collate_rgb_inputs, from_view_sequences
from jepa4d.evaluation.phase2f_metrics import (
    array_to_png_bytes,
    atomic_json,
    cuda_hardware_identity,
    evaluate_depth_predictions,
    file_identity,
    require_finite_tree,
    self_contained_html,
)
from jepa4d.models.vjepa21_adapter import VJEPA21FeatureExtractor
from jepa4d.training.phase2f_training import load_phase2f_checkpoint
from scripts.run_phase2f_training import _finish_wandb, _normalize, _validate_provenance

SELECTOR_SCHEMA = "jepa4d-phase2f-development-selector-v1"
FINAL_SCHEMA = "jepa4d-phase2f-external-final-v1"
ASSET_SEAL_SCHEMA = "jepa4d-phase2f-diode-asset-seal-v1"
NORMALIZATION_SCHEMA = "jepa4d-phase2f-rotation-feature-normalization-v1"
ARCHIVE_BYTES = 2_774_625_282
ARCHIVE_MD5 = "5c895d09201b88973c8fe4552a67dd85"
META_SHA256 = "ea293e1e8eb5615430353291ea9b798d8e75b6672abfd90d185069a3f53b1288"
INTRINSICS_SHA256 = "ba3c845f0ca40173196bcdf8ce66b03be431840b077665bf85172f156b930b02"
LICENSE_SHA256 = "bb83d5a21f4b0d0dd6a024a41e4f3719cda0fcf0093b03f1c536931a7f396a58"
EXPECTED_DOMAIN_COUNTS = {"indoor": 220, "outdoor": 392}
ROTATIONS = ("R0", "R1", "R2", "R3")
SEEDS = (0, 1, 2)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    require_finite_tree(value, str(path))
    return value


def validate_selector(selector_path: Path) -> dict[str, Any]:
    """Require one frozen survivor and exactly 12 checkpoints for each arm."""

    selector = _load_json(selector_path)
    if selector.get("schema_version") != SELECTOR_SCHEMA or selector.get("status") != "success":
        raise ValueError("external final requires a valid development selector")
    if selector.get("wandb", {}).get("mode") != "online" or selector.get("wandb", {}).get("status") != "success":
        raise ValueError("selector lacks a successful online W&B receipt")
    if selector.get("final_authorized") is not True:
        return selector
    survivor = selector.get("survivor")
    if survivor not in {"M1", "M2", "M3"}:
        raise ValueError("authorized selector must name exactly one M1-M3 survivor")
    checkpoint_set = selector.get("checkpoint_set")
    if not isinstance(checkpoint_set, dict) or set(checkpoint_set) != {"M0", survivor}:
        raise ValueError("selector checkpoint set must contain only M0 and the survivor")
    for arm in ("M0", survivor):
        rows = checkpoint_set[arm]
        if not isinstance(rows, list) or len(rows) != 12:
            raise ValueError(f"selector must bind 12 {arm} checkpoints")
        identities = {(row.get("rotation"), row.get("seed")) for row in rows}
        expected = {(rotation, seed) for rotation in ROTATIONS for seed in SEEDS}
        if identities != expected:
            raise ValueError(f"selector {arm} checkpoint identities are incomplete")
    latency_identity = selector.get("latency_gate")
    if not isinstance(latency_identity, Mapping) or not isinstance(latency_identity.get("path"), str):
        raise ValueError("selector lacks its frozen latency-gate identity")
    latency_path = Path(latency_identity["path"])
    if file_identity(latency_path)["sha256"] != latency_identity.get("sha256"):
        raise ValueError("frozen latency-gate bytes changed after selection")
    latency = _load_json(latency_path)
    latency_arm = latency.get("arms", {}).get(survivor, {})
    if (
        latency.get("schema_version") != "jepa4d-phase2f-latency-qualification-v1"
        or latency.get("status") != "pass"
        or latency_arm.get("qualified") is not True
        or float(latency_arm.get("ratio_ci95", [float("inf"), float("inf")])[1]) > 1.10
        or int(latency_arm.get("parameter_count", 10**18)) > 95_042
    ):
        raise ValueError("survivor no longer satisfies the frozen efficiency/static gate")
    return selector


def _hash_archive(path: Path) -> tuple[str, str, int]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - official immutable dataset identity is MD5.
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def validate_sealed_asset(
    archive: Path,
    seal_path: Path,
    meta_path: Path,
    intrinsics_path: Path,
    license_path: Path,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate only compressed bytes and public metadata before consuming final."""

    seal = _load_json(seal_path)
    if seal.get("schema_version") != ASSET_SEAL_SCHEMA or seal.get("status") != "success":
        raise ValueError("DIODE archive lacks a passing compressed-byte asset seal")
    if seal.get("wandb", {}).get("mode") != "online" or seal.get("wandb", {}).get("status") != "success":
        raise ValueError("DIODE asset seal lacks a successful online W&B upload")
    if provenance is not None:
        keys = (
            "execution_id",
            "git_commit",
            "preregistration_sha256",
            "test_receipt_sha256",
            "dependency_graph_sha256",
        )
        parent = seal.get("execution_provenance", {})
        if any(parent.get(key) != provenance.get(key) for key in keys):
            raise ValueError("DIODE asset seal belongs to a different execution")
    if seal.get("target_opacity") != {
        "compressed_stream_only": True,
        "tar_listed": False,
        "tar_extracted": False,
        "target_array_loaded": False,
        "target_statistics_computed": False,
        "target_preview_generated": False,
    }:
        raise ValueError("DIODE asset seal violated target opacity")
    identities = {
        "diode_meta": (meta_path, META_SHA256),
        "intrinsics": (intrinsics_path, INTRINSICS_SHA256),
        "license": (license_path, LICENSE_SHA256),
    }
    for label, (path, expected) in identities.items():
        actual = file_identity(path)
        if actual["sha256"] != expected:
            raise ValueError(f"DIODE {label} hash differs from preregistration")
    archive_path = archive.resolve(strict=True)
    sha256, md5, size = _hash_archive(archive_path)
    if size != ARCHIVE_BYTES or md5 != ARCHIVE_MD5:
        raise ValueError("DIODE compressed archive identity differs from preregistration")
    sealed_archive = seal.get("archive", {})
    if (
        sealed_archive.get("bytes") != size
        or sealed_archive.get("md5") != md5
        or sealed_archive.get("sha256") != sha256
    ):
        raise ValueError("DIODE archive bytes no longer match the asset-seal receipt")
    return {
        "path": str(archive_path),
        "bytes": size,
        "md5": md5,
        "sha256": sha256,
        "seal": file_identity(seal_path, schema=ASSET_SEAL_SCHEMA),
        **{label: file_identity(path) for label, (path, _) in identities.items()},
    }


def create_open_sentinel(
    path: Path,
    *,
    selector_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically consume the preregistered final before extracting any member."""

    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "jepa4d-phase2f-fresh-final-opened-v1",
        "fresh_final_opened": True,
        "opened_utc": datetime.now(UTC).isoformat(),
        "execution_id": provenance["execution_id"],
        "git_commit": provenance["git_commit"],
        "selector": file_identity(selector_path, schema=SELECTOR_SCHEMA),
        "slurm": provenance["slurm"],
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    output.chmod(0o444)
    return payload


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, mode="r:gz") as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise ValueError(f"unsafe DIODE archive member: {member.name}") from error
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported DIODE archive member type: {member.name}")
        stream.extractall(destination, filter="data")


def _domain(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "indoors" in lowered or "indoor" in lowered:
        return "indoor"
    if "outdoor" in lowered or "outdoors" in lowered:
        return "outdoor"
    raise ValueError(f"cannot infer DIODE domain from {path}")


def discover_diode_samples(root: Path) -> list[dict[str, Any]]:
    """Pair public metadata paths without opening any target array."""

    depth_files = sorted(path for path in root.rglob("*_depth.npy") if not path.name.endswith("_depth_mask.npy"))
    rows: list[dict[str, Any]] = []
    for depth in depth_files:
        prefix = depth.name.removesuffix("_depth.npy")
        mask = depth.with_name(prefix + "_depth_mask.npy")
        candidates = (depth.with_name(prefix + ".png"), depth.with_name(prefix + ".jpg"))
        rgb = next((path for path in candidates if path.is_file()), None)
        if rgb is None or not mask.is_file():
            raise ValueError(f"DIODE sample files are incomplete for {depth}")
        relative = depth.relative_to(root).as_posix()
        domain = _domain(depth.relative_to(root))
        parts = depth.relative_to(root).parts
        domain_index = next(index for index, part in enumerate(parts) if part.lower().rstrip("s") == domain)
        scene = parts[domain_index + 1] if domain_index + 1 < len(parts) else "unknown"
        rows.append(
            {
                "sample_id": relative.removesuffix("_depth.npy"),
                "domain": domain,
                "scene": f"{domain}/{scene}",
                "rgb": rgb,
                "depth": depth,
                "mask": mask,
            }
        )
    counts = {domain: sum(row["domain"] == domain for row in rows) for domain in EXPECTED_DOMAIN_COUNTS}
    if counts != EXPECTED_DOMAIN_COUNTS or len(rows) != sum(EXPECTED_DOMAIN_COUNTS.values()):
        raise ValueError(f"DIODE sample counts {counts} differ from preregistered {EXPECTED_DOMAIN_COUNTS}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("DIODE sample IDs are not unique")
    return rows


def _parse_intrinsics(path: Path) -> torch.Tensor:
    numbers = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", path.read_text())]
    if len(numbers) >= 9:
        matrix = torch.tensor(numbers[:9], dtype=torch.float32).reshape(3, 3)
    elif len(numbers) >= 4:
        fx, fy, cx, cy = numbers[:4]
        matrix = torch.tensor(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=torch.float32)
    else:
        raise ValueError("DIODE intrinsics file contains fewer than four numeric values")
    if not bool(torch.isfinite(matrix).all()) or float(matrix[0, 0]) <= 0 or float(matrix[1, 1]) <= 0:
        raise ValueError("DIODE intrinsics are invalid")
    return matrix


def _preprocess_sample(row: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    with Image.open(row["rgb"]) as image:
        rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    depth_array = np.load(row["depth"], allow_pickle=False)
    mask_array = np.load(row["mask"], allow_pickle=False)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array[..., 0]
    if rgb_array.shape != (768, 1024, 3) or depth_array.shape != (768, 1024) or mask_array.shape != (768, 1024):
        raise ValueError(f"DIODE sample {row['sample_id']} has unexpected dimensions")
    valid = np.asarray(mask_array, dtype=bool) & np.isfinite(depth_array) & (depth_array > 0)
    depth = torch.from_numpy(np.asarray(depth_array, dtype=np.float32).copy())[:, 128:896]
    valid_tensor = torch.from_numpy(valid.copy())[:, 128:896]
    rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).float().div(255)[:, :, 128:896]
    safe_depth = torch.where(valid_tensor, depth, torch.zeros_like(depth))
    weighted = F.interpolate(safe_depth.view(1, 1, 768, 768), (24, 24), mode="area")[0, 0]
    mass = F.interpolate(valid_tensor.float().view(1, 1, 768, 768), (24, 24), mode="area")[0, 0]
    valid_24 = mass >= 0.25
    depth_24 = torch.where(valid_24, weighted / mass.clamp_min(1e-12), torch.zeros_like(weighted))
    if int(valid_24.sum()) == 0 or not bool(torch.isfinite(depth_24[valid_24]).all()):
        raise ValueError(f"DIODE sample {row['sample_id']} has no finite reduced target")
    rgb_384 = F.interpolate(rgb.unsqueeze(0), (384, 384), mode="bilinear", align_corners=False, antialias=True)[0]
    return rgb_384, depth_24, valid_24, rgb_array[:, 128:896]


def _batch(images: torch.Tensor) -> Any:
    return collate_rgb_inputs([from_view_sequences([[image]]) for image in images])


def _load_models(selector: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    survivor = str(selector["survivor"])
    models: list[dict[str, Any]] = []
    for arm in ("M0", survivor):
        for row in selector["checkpoint_set"][arm]:
            checkpoint_info = row["checkpoint"]
            checkpoint = Path(checkpoint_info["path"])
            if file_identity(checkpoint)["sha256"] != checkpoint_info["sha256"]:
                raise ValueError("formal checkpoint hash differs from selector")
            normalization_info = row["feature_normalization"]
            normalization_path = Path(normalization_info["path"])
            if file_identity(normalization_path)["sha256"] != normalization_info["sha256"]:
                raise ValueError("feature normalization hash differs from selector")
            normalization = torch.load(normalization_path, map_location="cpu", weights_only=True)
            if not isinstance(normalization, dict) or normalization.get("schema_version") != NORMALIZATION_SCHEMA:
                raise ValueError("formal feature normalization schema changed")
            model, _ = load_phase2f_checkpoint(checkpoint, device=device)
            if model.config.arm != arm:
                raise ValueError("checkpoint arm differs from selector")
            model.eval()
            models.append(
                {
                    "arm": arm,
                    "rotation": row["rotation"],
                    "seed": row["seed"],
                    "model": model,
                    "normalization": normalization,
                    "variance_multiplier": float(row["validation_variance_calibration"]["multiplier"]),
                    "checkpoint": checkpoint_info,
                    "feature_normalization": normalization_info,
                }
            )
    if len(models) != 24:
        raise RuntimeError("external final must load exactly 24 checkpoints")
    return models


def _panel(path: Path, panels: Sequence[Mapping[str, Any]], survivor: str) -> None:
    cell = 192
    include_field = any("field" in panel for panel in panels)
    labels = ["RGB", "target", "M0", survivor, "absolute error", "uncertainty"]
    keys = ["target", "m0", "survivor", "error", "uncertainty"]
    if include_field:
        labels.append("M3 scale field")
        keys.append("field")
    columns = len(labels)
    rows = len(panels)
    canvas = Image.new("RGB", (columns * cell, rows * (cell + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * cell + 6, 4), label, fill="#17202e")
    for row_index, panel in enumerate(panels):
        y = row_index * (cell + 28) + 28
        arrays = [panel["rgb"]]
        for key in keys:
            png = array_to_png_bytes(np.asarray(panel[key]))
            arrays.append(np.asarray(Image.open(__import__("io").BytesIO(png)).resize((cell, cell))))
        rgb_image = Image.fromarray(panel["rgb"]).resize((cell, cell))
        canvas.paste(rgb_image, (0, y))
        for column, array in enumerate(arrays[1:], start=1):
            canvas.paste(Image.fromarray(array).resize((cell, cell)), (column * cell, y))
        draw.text((4, y + cell), str(panel["sample_id"]), fill="#17202e")
    canvas.save(path)


def evaluate_external_final(
    rows: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
    *,
    extractor: VJEPA21FeatureExtractor,
    intrinsics: torch.Tensor,
    device: torch.device,
    progress: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all 24 frozen checkpoints in one streaming target pass."""

    models = _load_models(selector, device)
    crop_k = update_intrinsics_for_crop_resize(
        intrinsics,
        (768, 1024),
        (384, 384),
        crop=(0, 128, 768, 768),
        half_pixel_centers=True,
    ).to(device)
    predictions: dict[tuple[str, str, int], list[torch.Tensor]] = defaultdict(list)
    variances: dict[tuple[str, str, int], list[torch.Tensor]] = defaultdict(list)
    scale_fields: dict[tuple[str, str, int], list[torch.Tensor]] = defaultdict(list)
    targets: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    panel_ids = {
        row["sample_id"]
        for domain in ("indoor", "outdoor")
        for row in sorted(
            (item for item in rows if item["domain"] == domain),
            key=lambda item: hashlib.sha256(item["sample_id"].encode()).hexdigest(),
        )[:6]
    }
    panel_source: dict[str, dict[str, Any]] = {}
    batch_size = 8
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            prepared = [_preprocess_sample(row) for row in batch_rows]
            images = torch.stack([item[0] for item in prepared])
            target_batch = torch.stack([item[1] for item in prepared])
            valid_batch = torch.stack([item[2] for item in prepared])
            bundle = extractor(_batch(images))
            raw = bundle.dense_tokens[:, 0, 0].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2).float()
            targets.extend(target_batch)
            valid_masks.extend(valid_batch)
            for item in models:
                features = _normalize(raw, item["normalization"], device)
                camera = crop_k.unsqueeze(0).expand(len(features), -1, -1) if item["arm"] in {"M2", "M3"} else None
                output = item["model"](
                    features,
                    intrinsics=camera,
                    intrinsics_image_size=(384, 384) if camera is not None else None,
                )
                key = (item["arm"], item["rotation"], item["seed"])
                predictions[key].extend(output.log_depth.detach().cpu())
                variances[key].extend(output.log_variance.detach().cpu())
                if output.scale_field is not None:
                    scale_fields[key].extend(output.scale_field.detach().cpu())
            for row, prepared_row in zip(batch_rows, prepared, strict=True):
                if row["sample_id"] in panel_ids:
                    panel_source[row["sample_id"]] = {"rgb": prepared_row[3], "target": prepared_row[1]}
            if progress is not None:
                progress(min(offset + batch_size, len(rows)), len(rows))
    target_tensor = torch.stack(targets)
    valid_tensor = torch.stack(valid_masks)
    frame_ids = [str(row["sample_id"]) for row in rows]
    domains = [str(row["domain"]) for row in rows]
    scenes = [str(row["scene"]) for row in rows]
    checkpoint_metrics: list[dict[str, Any]] = []
    metrics_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in models:
        key = (item["arm"], item["rotation"], item["seed"])
        means = torch.stack(predictions[key])
        logs = torch.stack(variances[key])
        metrics = evaluate_depth_predictions(
            means,
            logs,
            target_tensor,
            valid_mask=valid_tensor,
            variance_multiplier=item["variance_multiplier"],
            frame_ids=frame_ids,
            group_ids=domains,
        )
        scene_metrics = evaluate_depth_predictions(
            means,
            logs,
            target_tensor,
            valid_mask=valid_tensor,
            variance_multiplier=item["variance_multiplier"],
            frame_ids=frame_ids,
            group_ids=scenes,
        )
        metrics["per_scene"] = scene_metrics["per_group"]
        metrics_by_key[key] = metrics
        checkpoint_metrics.append(
            {
                "arm": key[0],
                "rotation": key[1],
                "seed": key[2],
                "equal_domain_macro": metrics["group_macro"],
                "per_domain": metrics["per_group"],
                "per_scene": metrics["per_scene"],
                "pooled_frame": metrics["frame_macro"],
                "coverage": metrics["coverage"],
                "risk_coverage": metrics["risk_coverage"],
            }
        )
    survivor = str(selector["survivor"])
    aggregate: dict[str, dict[str, float]] = {}
    for arm in ("M0", survivor):
        selected = [row["equal_domain_macro"] for row in checkpoint_metrics if row["arm"] == arm]
        aggregate[arm] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in ("raw_abs_rel", "absolute_log_scale_error", "aligned_abs_rel", "nll", "ause")
        }
    m0, candidate = aggregate["M0"], aggregate[survivor]
    scientific_checks = {
        "raw_abs_rel_lower": candidate["raw_abs_rel"] < m0["raw_abs_rel"],
        "scale_error_lower": candidate["absolute_log_scale_error"] < m0["absolute_log_scale_error"],
        "aligned_abs_rel_noninferior": candidate["aligned_abs_rel"] <= 1.02 * m0["aligned_abs_rel"],
        "nll_lower": candidate["nll"] < m0["nll"],
        "ause_no_worse": candidate["ause"] <= m0["ause"],
        "frozen_latency_ratio": selector["eligibility"][survivor]["checks"]["latency_frozen"] is True,
        "frozen_parameter_ratio": selector["eligibility"][survivor]["checks"]["parameters_frozen"] is True,
        "all_samples_and_checkpoints_complete": len(checkpoint_metrics) == 24
        and len(rows) == sum(EXPECTED_DOMAIN_COUNTS.values()),
    }
    panels: list[dict[str, Any]] = []
    m0_key = ("M0", "R0", 0)
    survivor_key = (survivor, "R0", 0)
    m0_prediction = torch.stack(predictions[m0_key]).exp().numpy()
    survivor_prediction = torch.stack(predictions[survivor_key]).exp().numpy()
    survivor_uncertainty = torch.stack(variances[survivor_key]).mul(0.5).exp().numpy()
    for index, row in enumerate(rows):
        if row["sample_id"] not in panel_ids:
            continue
        source = panel_source[row["sample_id"]]
        target = source["target"].numpy()
        panels.append(
            {
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "rgb": source["rgb"],
                "target": target,
                "m0": m0_prediction[index],
                "survivor": survivor_prediction[index],
                "error": np.abs(survivor_prediction[index] - target),
                "uncertainty": survivor_uncertainty[index],
            }
        )
        if survivor == "M3":
            panels[-1]["field"] = torch.stack(scale_fields[survivor_key])[index].numpy()
    result = {
        "checkpoint_metrics": checkpoint_metrics,
        "paired_12_checkpoint_means": aggregate,
        "scientific_checks": scientific_checks,
        "scientific_gate": all(scientific_checks.values()),
        "sample_counts": EXPECTED_DOMAIN_COUNTS,
        "target_scored_k_ablations": False,
    }
    require_finite_tree(result, "external_final_metrics")
    return result, panels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--asset-seal", type=Path, required=True)
    parser.add_argument("--diode-meta", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--devkit-license", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="crlc112358")
    parser.add_argument("--wandb-project", default="jepa4d-worldmodel")
    return parser.parse_args()


def _initialize_run(args: argparse.Namespace, provenance: Mapping[str, Any], status: str) -> Any:
    import wandb

    execution_id = str(provenance["execution_id"])
    slurm_id = str(provenance["slurm"].get("job_id", os.environ.get("SLURM_JOB_ID", "unknown")))
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"phase2f-{execution_id}",
        job_type="external-final",
        name=f"{execution_id}-external-final-{slurm_id}",
        mode="online",
        reinit=True,
        config={"git_commit": provenance["git_commit"], "initial_status": status},
    )
    if run is None or run.offline:
        raise RuntimeError("Phase 2f external final requires online W&B")
    return run


def main() -> None:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Phase 2f experiment CLIs may run only inside a Slurm job")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = _validate_provenance(args.provenance)
    selector = validate_selector(args.selector)
    execution_keys = (
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "test_receipt_sha256",
        "dependency_graph_sha256",
    )
    selector_provenance = selector.get("execution_provenance", {})
    if any(selector_provenance.get(key) != provenance.get(key) for key in execution_keys):
        raise ValueError("external-final current-job provenance differs from the selector execution")
    run = _initialize_run(args, provenance, "authorized" if selector.get("final_authorized") else "skip")
    execution_id = str(provenance["execution_id"])
    if selector.get("final_authorized") is not True:
        # Deliberately do not resolve, stat, hash, list, or open args.archive.
        result = {
            "schema_version": FINAL_SCHEMA,
            "status": "skipped_no_survivor",
            "fresh_final_opened": False,
            "archive_touched": False,
            "selector": file_identity(args.selector, schema=SELECTOR_SCHEMA),
            "outcome": "no_survivor",
            "execution_provenance": provenance,
        }
        report = output / "report.html"
        report.write_text(
            self_contained_html(
                "Phase 2f external final guard",
                {"outcome": "no_survivor", "archive touched": False},
                claim_boundary="No development survivor; DIODE remained unopened and unconsumed.",
            ),
            encoding="utf-8",
        )
        run.log(
            {
                "terminal/status": "skipped_no_survivor",
                "terminal/scientific_gate": False,
                "terminal/outcome": "no_survivor",
            }
        )
        wandb_receipt = _finish_wandb(
            run,
            artifact_name=f"phase2f-external-final-{execution_id}-skip",
            job_type="external-final",
            files=(report,),
        )
        result["wandb"] = wandb_receipt
        atomic_json(output / "wandb_receipt.json", wandb_receipt)
        atomic_json(output / "final_receipt.json", result)
        (output / "SUCCESS").write_text("skipped_no_survivor\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "archive_touched": False}, sort_keys=True))
        return

    if os.path.lexists(args.sentinel):
        raise RuntimeError("FRESH_FINAL_OPENED sentinel already exists; archive access and rerun are forbidden")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 2f external final must run on a Slurm CUDA allocation")
    asset_identity = validate_sealed_asset(
        args.archive,
        args.asset_seal,
        args.diode_meta,
        args.intrinsics,
        args.devkit_license,
        provenance=provenance,
    )
    sentinel = create_open_sentinel(args.sentinel, selector_path=args.selector, provenance=provenance)
    started = time.perf_counter()
    tmp_root = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())).resolve()
    extraction = tmp_root / f"phase2f-diode-{execution_id}"
    try:
        _safe_extract(args.archive.resolve(strict=True), extraction)
        rows = discover_diode_samples(extraction)
        intrinsics = _parse_intrinsics(args.intrinsics)
        extractor = VJEPA21FeatureExtractor(
            checkpoint=args.vjepa_checkpoint,
            implementation_path=args.vjepa_implementation,
            backend="hf_compat",
            device=device,
            capture_layers=(),
        )

        def progress(done: int, total: int) -> None:
            run.log({"external_final/frames_processed": done, "external_final/frames_total": total})

        metrics, panels = evaluate_external_final(
            rows,
            selector,
            extractor=extractor,
            intrinsics=intrinsics,
            device=device,
            progress=progress,
        )
        survivor = str(selector["survivor"])
        outcome = f"promote_{survivor}" if metrics["scientific_gate"] else "retain_M0"
        metrics_path = atomic_json(output / "metrics.json", metrics)
        with (output / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("arm", "rotation", "seed", "raw_abs_rel", "scale_error", "aligned_abs_rel", "nll", "ause")
            )
            for row in metrics["checkpoint_metrics"]:
                values = row["equal_domain_macro"]
                writer.writerow(
                    (
                        row["arm"],
                        row["rotation"],
                        row["seed"],
                        values["raw_abs_rel"],
                        values["absolute_log_scale_error"],
                        values["aligned_abs_rel"],
                        values["nll"],
                        values["ause"],
                    )
                )
        np.savez_compressed(
            output / "summary.npz",
            arms=np.asarray(("M0", survivor)),
            raw_abs_rel=np.asarray(
                [metrics["paired_12_checkpoint_means"][arm]["raw_abs_rel"] for arm in ("M0", survivor)]
            ),
            scientific_gate=np.asarray([metrics["scientific_gate"]]),
        )
        panel_path = output / "panels.png"
        _panel(panel_path, panels, survivor)
        report = output / "report.html"
        report.write_text(
            self_contained_html(
                "Phase 2f fresh external final",
                {
                    "outcome": outcome,
                    "survivor": survivor,
                    "scientific gate": metrics["scientific_gate"],
                    "M0 raw AbsRel": metrics["paired_12_checkpoint_means"]["M0"]["raw_abs_rel"],
                    f"{survivor} raw AbsRel": metrics["paired_12_checkpoint_means"][survivor]["raw_abs_rel"],
                },
                images=(("Preselected indoor/outdoor panels", panel_path),),
                claim_boundary="One-shot DIODE confirmation of M0 versus exactly one frozen survivor; no K ablation.",
            ),
            encoding="utf-8",
        )
        run.log(
            {
                "terminal/status": "success",
                "terminal/scientific_gate": metrics["scientific_gate"],
                "terminal/outcome": outcome,
            }
        )
        wandb_receipt = _finish_wandb(
            run,
            artifact_name=f"phase2f-external-final-{execution_id}",
            job_type="external-final",
            files=(metrics_path, output / "checkpoint_metrics.csv", output / "summary.npz", panel_path, report),
        )
        wandb_path = atomic_json(output / "wandb_receipt.json", wandb_receipt)
        output_identities = {
            name: file_identity(path)
            for name, path in {
                "metrics": metrics_path,
                "checkpoint_metrics_csv": output / "checkpoint_metrics.csv",
                "summary_npz": output / "summary.npz",
                "panels": panel_path,
                "report": report,
                "wandb_receipt": wandb_path,
            }.items()
        }
        result = {
            "schema_version": FINAL_SCHEMA,
            "status": "success",
            "fresh_final_opened": True,
            "sentinel": {**sentinel, **file_identity(args.sentinel)},
            "asset": asset_identity,
            "selector": file_identity(args.selector, schema=SELECTOR_SCHEMA),
            "survivor": survivor,
            "models_evaluated": ["M0", survivor],
            "checkpoint_count_per_arm": 12,
            "target_scored_k_ablations": False,
            "sample_counts": EXPECTED_DOMAIN_COUNTS,
            "scientific_gate": metrics["scientific_gate"],
            "scientific_checks": metrics["scientific_checks"],
            "outcome": outcome,
            "elapsed_seconds": time.perf_counter() - started,
            "hardware": {
                **cuda_hardware_identity(device),
                "peak_allocation_bytes": torch.cuda.max_memory_allocated(device),
            },
            "outputs": output_identities,
            "wandb": wandb_receipt,
            "execution_provenance": provenance,
        }
        require_finite_tree(result, "final_receipt")
        atomic_json(output / "final_receipt.json", result)
        (output / "SUCCESS").write_text("success\n", encoding="utf-8")
        print(json.dumps({"status": "success", "outcome": outcome}, sort_keys=True))
    except BaseException as error:
        atomic_json(
            output / "external_final_consumed_inconclusive.json",
            {
                "schema_version": FINAL_SCHEMA,
                "status": "external_final_consumed_inconclusive",
                "fresh_final_opened": True,
                "sentinel": file_identity(args.sentinel),
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
                "rerun_authorized": False,
                "execution_provenance": provenance,
            },
        )
        run.summary["status"] = "external_final_consumed_inconclusive"
        run.finish(exit_code=1)
        raise
    finally:
        if extraction.exists():
            shutil.rmtree(extraction)


if __name__ == "__main__":
    main()
