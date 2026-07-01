"""Frozen SUN RGB-D membership and isolated cache contracts for Phase 2g-A.

The membership builder deliberately freezes sorted sample identities before it
decodes depth.  Depth is used only to produce one eligibility boolean under the
registered ``>=100`` finite pixels in ``0.1 < d < 10`` rule.  No depth value,
histogram, preview, or aggregate is serialized into the membership manifest.

Training consumes a *rotation view*, not the full cache root.  A rotation view
contains input/feature/target shards for exactly its two training families and
one validation family.  Its held-out family appears only as an identifier; no
held-out target path is present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from jepa4d.benchmarks.geometry.sun_rgbd import (
    DATASET_ID,
    OFFICIAL_ARCHIVE_BYTES,
    OFFICIAL_ARCHIVE_SHA256,
    SUNRGBDFrame,
    decode_sunrgbd_depth,
    enumerate_sunrgbd_frames,
    load_intrinsics,
    sha256_file,
)
from jepa4d.evaluation.phase2f_camera_controls import (
    CAMERA_CONTROL_SCHEMA,
    PROFILE_COUNT,
    PROFILE_IDS,
    PROFILE_PERMUTATION,
    apply_profile_to_rgb,
    build_paired_camera_controls,
    frozen_camera_profiles,
    transform_and_reduce_depth,
)
from jepa4d.training.phase2g_protocol import (
    EXPECTED_INVENTORY_COUNTS,
    FEATURE_SHARD_SCHEMA,
    INPUT_SHARD_SCHEMA,
    MATERIALIZATION_SCHEMA,
    MEMBERSHIP_SCHEMA,
    MINIMUM_VALID_PIXELS,
    QUALITATIVE_IDS_PER_FAMILY,
    ROTATION_VIEW_SCHEMA,
    ROTATIONS,
    SAMPLES_PER_FAMILY,
    TARGET_SHARD_SCHEMA,
    VALID_DEPTH_INTERVAL_M,
)
from jepa4d.training.phase2g_protocol import (
    FAMILIES as SUN_FAMILIES,
)

DEPTH_MIN_M, DEPTH_MAX_M = VALID_DEPTH_INTERVAL_M
CLAIM_BOUNDARY = "SUN RGB-D development cache only; the sealed evaluation archive is absent"

_SEALED_TOKENS = ("diode", "external_final", "external-target", "final_target", "val.tar.gz")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_record_sha256(value: Mapping[str, Any], *, excluded_fields: Sequence[str]) -> str:
    """Hash a JSON record after removing explicitly post-hash operational fields."""

    payload = dict(value)
    for field in excluded_fields:
        payload.pop(field, None)
    return canonical_sha256(payload)


def require_canonical_record_sha256(
    value: Mapping[str, Any],
    *,
    field: str,
    excluded_fields: Sequence[str],
) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"record lacks {field}")
    if claimed != canonical_record_sha256(value, excluded_fields=excluded_fields):
        raise ValueError(f"record {field} mismatch")
    return claimed


def _execution_binding(provenance: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "execution_id",
        "git_commit",
        "preregistration_sha256",
        "preflight_sha256",
        "dependency_graph_sha256",
    )
    binding = {key: provenance.get(key) for key in keys}
    if any(value is None or value == "" for value in binding.values()):
        raise ValueError("SUN materialization requires complete execution provenance")
    return binding


def _safe_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("SUN archive contains an unsafe member name")
    member = PurePosixPath(name.rstrip("/"))
    if member.is_absolute() or not member.parts or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"SUN archive contains an unsafe member path: {name!r}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise ValueError(f"SUN archive contains a symbolic link: {name!r}")
    if info.flag_bits & 0x1:
        raise ValueError(f"SUN archive contains an encrypted member: {name!r}")
    return member


def _sun_archive_leaf_files(
    archive: zipfile.ZipFile,
    *,
    expected_inventory_counts: Mapping[str, int],
) -> dict[str, list[tuple[PurePosixPath, zipfile.ZipInfo]]]:
    """Select only the three runtime files for every official SUN leaf."""

    file_infos: dict[PurePosixPath, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        member = _safe_zip_member(info)
        if info.is_dir():
            continue
        if member in file_infos:
            raise ValueError(f"SUN archive contains a duplicate file member: {member}")
        file_infos[member] = info

    images: dict[tuple[str, PurePosixPath], list[PurePosixPath]] = {}
    depths: dict[tuple[str, PurePosixPath], list[PurePosixPath]] = {}
    for member in file_infos:
        if len(member.parts) < 5 or member.parts[0] != "SUNRGBD" or member.parts[1] not in SUN_FAMILIES:
            continue
        family = member.parts[1]
        leaf = PurePosixPath(*member.parts[:-2])
        key = (family, leaf)
        if member.parts[-2] == "image" and member.suffix == ".jpg":
            images.setdefault(key, []).append(member)
        elif member.parts[-2] == "depth_bfx" and member.suffix == ".png":
            depths.setdefault(key, []).append(member)

    selected: dict[str, list[tuple[PurePosixPath, zipfile.ZipInfo]]] = {family: [] for family in SUN_FAMILIES}
    leaves = sorted(set(images) | set(depths), key=lambda item: (item[0], item[1].as_posix()))
    for family, leaf in leaves:
        key = (family, leaf)
        rgb = images.get(key, [])
        depth = depths.get(key, [])
        intrinsics = leaf / "intrinsics.txt"
        if len(rgb) != 1 or len(depth) != 1 or intrinsics not in file_infos:
            raise ValueError(
                f"SUN archive leaf is incomplete: {leaf} (images={len(rgb)}, depths={len(depth)}, "
                f"intrinsics={intrinsics in file_infos})"
            )
        for member in (rgb[0], depth[0], intrinsics):
            selected[family].append((member, file_infos[member]))

    observed = {family: len(selected[family]) // 3 for family in SUN_FAMILIES}
    if observed != dict(expected_inventory_counts) or any(len(selected[family]) % 3 for family in SUN_FAMILIES):
        raise ValueError(
            "SUN archive leaf inventory differs from the frozen official counts: "
            f"observed={observed}, expected={dict(expected_inventory_counts)}"
        )
    return selected


def materialize_sun_archive(
    archive_path: Path,
    output: Path,
    *,
    provenance: Mapping[str, Any],
    expected_archive_sha256: str = OFFICIAL_ARCHIVE_SHA256,
    expected_archive_bytes: int = OFFICIAL_ARCHIVE_BYTES,
    expected_inventory_counts: Mapping[str, int] = EXPECTED_INVENTORY_COUNTS,
) -> tuple[Path, Path, dict[str, Any]]:
    """Materialize the formal SUN tree from the same verified archive file descriptor.

    Only direct leaf RGB, ``depth_bfx``, and intrinsics files are extracted.  The
    archive is hashed before extraction without closing or reopening its file
    descriptor, eliminating the preflight-root provenance gap and path-swap
    race.  The output remains local protected data.
    """

    if archive_path.is_symlink():
        raise ValueError("SUN archive may not be a symbolic link")
    archive_resolved = archive_path.resolve(strict=True)
    if not archive_resolved.is_file():
        raise ValueError("SUN archive must be a regular file")
    destination = output.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"SUN materialization output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"SUN materialization temporary path already exists: {temporary}")
    temporary.mkdir(mode=0o750)
    rows: list[dict[str, Any]] = []
    archive_digest = hashlib.sha256()
    archive_bytes = 0
    try:
        with archive_resolved.open("rb") as stream:
            descriptor_size = os.fstat(stream.fileno()).st_size
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                archive_digest.update(chunk)
                archive_bytes += len(chunk)
            observed_sha256 = archive_digest.hexdigest()
            if descriptor_size != expected_archive_bytes or archive_bytes != expected_archive_bytes:
                raise ValueError(
                    f"SUN archive byte mismatch: descriptor={descriptor_size}, read={archive_bytes}, "
                    f"expected={expected_archive_bytes}"
                )
            if observed_sha256 != expected_archive_sha256:
                raise ValueError(f"SUN archive SHA-256 mismatch: {observed_sha256}")
            stream.seek(0)
            with zipfile.ZipFile(stream, mode="r") as archive:
                selected = _sun_archive_leaf_files(
                    archive,
                    expected_inventory_counts=expected_inventory_counts,
                )
                for family in SUN_FAMILIES:
                    for member, info in selected[family]:
                        relative = member.relative_to(PurePosixPath("SUNRGBD"))
                        target = temporary / "SUNRGBD" / Path(*relative.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        digest = hashlib.sha256()
                        size = 0
                        with archive.open(info, mode="r") as source, target.open("xb") as sink:
                            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                digest.update(chunk)
                                sink.write(chunk)
                                size += len(chunk)
                        if size != info.file_size:
                            raise ValueError(f"SUN archive member byte mismatch: {member}")
                        target.chmod(0o640)
                        rows.append(
                            {
                                "path": relative.as_posix(),
                                "bytes": size,
                                "sha256": digest.hexdigest(),
                            }
                        )
        rows.sort(key=lambda row: str(row["path"]))
        receipt: dict[str, Any] = {
            "schema_version": MATERIALIZATION_SCHEMA,
            "status": "success",
            "archive": {
                "name": "SUNRGBD.zip",
                "bytes": archive_bytes,
                "sha256": archive_digest.hexdigest(),
            },
            "dataset_root": "SUNRGBD",
            "selection": "direct-leaf-image-depth_bfx-intrinsics-only",
            "family_counts": dict(expected_inventory_counts),
            "file_count": len(rows),
            "files_manifest_sha256": canonical_sha256(rows),
            "files": rows,
            "execution_identity": _execution_binding(provenance),
            "raw_data_uploaded": False,
        }
        receipt_path = atomic_json(temporary / "materialization_receipt.json", receipt)
        temporary.replace(destination)
        return (
            destination / "SUNRGBD",
            destination / receipt_path.relative_to(temporary),
            receipt,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_sun_materialization(
    dataset_root: Path,
    receipt_path: Path,
    *,
    provenance: Mapping[str, Any],
    expected_archive_sha256: str = OFFICIAL_ARCHIVE_SHA256,
    expected_archive_bytes: int = OFFICIAL_ARCHIVE_BYTES,
    expected_inventory_counts: Mapping[str, int] = EXPECTED_INVENTORY_COUNTS,
) -> dict[str, Any]:
    """Rehash the consumed extraction and validate it against C's archive receipt."""

    if dataset_root.is_symlink() or receipt_path.is_symlink():
        raise ValueError("SUN materialization paths may not be symbolic links")
    root = dataset_root.resolve(strict=True)
    receipt_resolved = receipt_path.resolve(strict=True)
    if not root.is_dir() or not receipt_resolved.is_file() or receipt_resolved.parent != root.parent:
        raise ValueError("SUN materialization root/receipt layout is invalid")
    value = json.loads(receipt_resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("SUN materialization receipt must be an object")
    archive = value.get("archive")
    rows = value.get("files")
    if (
        value.get("schema_version") != MATERIALIZATION_SCHEMA
        or value.get("status") != "success"
        or value.get("dataset_root") != "SUNRGBD"
        or value.get("selection") != "direct-leaf-image-depth_bfx-intrinsics-only"
        or value.get("raw_data_uploaded") is not False
        or value.get("family_counts") != dict(expected_inventory_counts)
        or value.get("execution_identity") != _execution_binding(provenance)
        or not isinstance(archive, Mapping)
        or archive.get("bytes") != expected_archive_bytes
        or archive.get("sha256") != expected_archive_sha256
        or not isinstance(rows, list)
        or value.get("file_count") != 3 * sum(expected_inventory_counts.values())
        or len(rows) != value.get("file_count")
        or value.get("files_manifest_sha256") != canonical_sha256(rows)
    ):
        raise ValueError("SUN materialization receipt differs from the frozen archive contract")
    expected_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError("SUN materialization file row is invalid")
        relative = PurePosixPath(str(row["path"]))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("SUN materialization receipt contains an unsafe relative path")
        name = relative.as_posix()
        if name in expected_paths:
            raise ValueError("SUN materialization receipt contains a duplicate path")
        expected_paths.add(name)
        candidate = root / Path(*relative.parts)
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != row.get("bytes")
            or sha256_file(candidate) != row.get("sha256")
        ):
            raise ValueError(f"SUN materialized file differs from its archive-derived identity: {name}")
    actual_paths: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or (not candidate.is_file() and not candidate.is_dir()):
            raise ValueError("SUN materialization contains a link or special file")
        if candidate.is_file():
            actual_paths.add(candidate.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("SUN materialization file set differs from its archive-derived receipt")
    return value


def atomic_json(path: Path, value: Any) -> Path:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def reject_sealed_references(value: Any, *, location: str = "payload") -> None:
    """Fail closed when development inputs expose an external-final reference."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in _SEALED_TOKENS):
                if child is False:
                    continue
                raise ValueError(f"sealed external field is forbidden at {location}.{key}")
            reject_sealed_references(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_sealed_references(child, location=f"{location}[{index}]")
    elif isinstance(value, (str, Path)):
        lowered = str(value).casefold()
        if any(token in lowered for token in _SEALED_TOKENS):
            raise ValueError(f"sealed external reference is forbidden at {location}")


def file_identity(path: Path, *, root: Path | None = None, schema: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected regular file: {resolved}")
    display = resolved
    if root is not None:
        root_value = root.resolve(strict=True)
        if not resolved.is_relative_to(root_value):
            raise ValueError(f"file escapes registered root: {resolved}")
        display = resolved.relative_to(root_value)
    result: dict[str, Any] = {
        "path": display.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if schema is not None:
        result["schema"] = schema
    return result


def _screen_frame(frame: SUNRGBDFrame) -> tuple[bool, str | None]:
    """Return only the frozen eligibility boolean and categorical failure."""

    try:
        depth = decode_sunrgbd_depth(frame.depth_path, clamp_max_depth_m=None)
    except (OSError, ValueError):
        return False, "depth_decode_failed"
    valid = np.isfinite(depth) & (depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M)
    if int(valid.sum()) < MINIMUM_VALID_PIXELS:
        return False, "fewer_than_100_valid_pixels"
    return True, None


def _validate_selected_frame(frame: SUNRGBDFrame, dataset_root: Path) -> SUNRGBDFrame:
    """Validate selected RGB/K alignment without returning target statistics."""

    with Image.open(frame.image_path) as image:
        image_size = (image.height, image.width)
    with Image.open(frame.depth_path) as image:
        raw = np.asarray(image).copy()
    if raw.ndim != 2 or raw.dtype != np.uint16 or tuple(raw.shape) != image_size:
        raise ValueError(f"selected RGB/depth alignment failed: {frame.sample_id}")
    intrinsics = load_intrinsics(frame.intrinsics_path)
    height, width = image_size
    if not (-0.5 <= intrinsics[0, 2] <= width - 0.5 and -0.5 <= intrinsics[1, 2] <= height - 0.5):
        raise ValueError(f"selected principal point is outside RGB frame: {frame.sample_id}")
    for path in (frame.image_path, frame.depth_path, frame.intrinsics_path):
        if not path.resolve(strict=True).is_relative_to(dataset_root):
            raise ValueError(f"selected source path escapes SUN root: {path}")
    return replace(frame, intrinsics=intrinsics, image_size_hw=image_size, depth_size_hw=image_size)


def _selected_row(frame: SUNRGBDFrame, *, rank: int, root: Path) -> dict[str, Any]:
    assert frame.intrinsics is not None and frame.image_size_hw is not None
    return {
        "sample_id": frame.sample_id,
        "family": frame.sensor,
        "rank": rank,
        "group_id": frame.group_id,
        "files": {
            "rgb": file_identity(frame.image_path, root=root),
            "depth_bfx": file_identity(frame.depth_path, root=root),
            "intrinsics": file_identity(frame.intrinsics_path, root=root),
        },
        "image_size_hw": list(frame.image_size_hw),
        "intrinsics": frame.intrinsics.tolist(),
    }


def build_sun_membership_manifest(
    dataset_root: Path,
    *,
    samples_per_family: int = SAMPLES_PER_FAMILY,
    qualitative_ids_per_family: int = QUALITATIVE_IDS_PER_FAMILY,
    expected_inventory_counts: Mapping[str, int] | None = EXPECTED_INVENTORY_COUNTS,
    enumerator: Callable[[Path], Mapping[str, Sequence[SUNRGBDFrame]]] = enumerate_sunrgbd_frames,
    screener: Callable[[SUNRGBDFrame], tuple[bool, str | None]] = _screen_frame,
) -> tuple[dict[str, Any], dict[str, tuple[SUNRGBDFrame, ...]]]:
    """Build the sorted, target-value-blind membership manifest.

    The returned frame objects are an in-memory runtime convenience.  Only the
    manifest is persisted; it intentionally contains no target-derived scalar.
    """

    if isinstance(samples_per_family, bool) or not isinstance(samples_per_family, int) or samples_per_family <= 0:
        raise ValueError("samples_per_family must be a positive integer")
    if not 0 < qualitative_ids_per_family <= samples_per_family:
        raise ValueError("qualitative_ids_per_family must be in [1,samples_per_family]")
    root = dataset_root.resolve(strict=True)
    reject_sealed_references(root)
    inventory = enumerator(root)
    if set(inventory) != set(SUN_FAMILIES):
        raise ValueError(f"SUN inventory must contain exactly {SUN_FAMILIES}")
    observed_counts = {family: len(inventory[family]) for family in SUN_FAMILIES}
    if expected_inventory_counts is not None and observed_counts != dict(expected_inventory_counts):
        raise ValueError(
            f"SUN extraction inventory differs from the frozen official counts: "
            f"observed={observed_counts}, expected={dict(expected_inventory_counts)}"
        )
    frozen_ids: dict[str, list[str]] = {}
    for family in SUN_FAMILIES:
        ordered = sorted(inventory[family], key=lambda frame: frame.sample_id)
        ids = [frame.sample_id for frame in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample IDs in {family}")
        if len(ids) < samples_per_family:
            raise ValueError(f"{family} has only {len(ids)} rows; requires {samples_per_family}")
        frozen_ids[family] = ids
    rank_order_sha256 = canonical_sha256(frozen_ids)

    screening: dict[str, list[dict[str, Any]]] = {}
    selected_rows: list[dict[str, Any]] = []
    selected_frames: dict[str, tuple[SUNRGBDFrame, ...]] = {}
    for family in SUN_FAMILIES:
        ordered = sorted(inventory[family], key=lambda frame: frame.sample_id)
        selected: list[tuple[int, SUNRGBDFrame]] = []
        family_screening: list[dict[str, Any]] = []
        for rank, frame in enumerate(ordered):
            eligible, failure = screener(frame)
            if not isinstance(eligible, bool) or (eligible and failure is not None) or (not eligible and not failure):
                raise ValueError("eligibility screener returned an invalid boolean/reason pair")
            choose = eligible and len(selected) < samples_per_family
            if choose:
                selected.append((rank, frame))
            family_screening.append(
                {
                    "sample_id": frame.sample_id,
                    "rank": rank,
                    "eligible": eligible,
                    "selected": choose,
                    "failure_reasons": {
                        "depth_decode_failed": failure == "depth_decode_failed",
                        "fewer_than_100_valid_pixels": failure == "fewer_than_100_valid_pixels",
                    },
                }
            )
        if len(selected) != samples_per_family:
            raise ValueError(f"{family} has {len(selected)} mechanically eligible rows; requires {samples_per_family}")
        validated = tuple(_validate_selected_frame(frame, root) for _, frame in selected)
        selected_frames[family] = validated
        selected_rows.extend(
            _selected_row(frame, rank=rank, root=root) for (rank, _), frame in zip(selected, validated, strict=True)
        )
        screening[family] = family_screening

    qualitative: dict[str, list[str]] = {}
    for family in SUN_FAMILIES:
        ids = [frame.sample_id for frame in selected_frames[family]]
        qualitative[family] = sorted(
            ids, key=lambda sample_id: (hashlib.sha256(sample_id.encode()).hexdigest(), sample_id)
        )[:qualitative_ids_per_family]
    manifest: dict[str, Any] = {
        "schema_version": MEMBERSHIP_SCHEMA,
        "dataset_id": DATASET_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_root_name": root.name,
        "families": list(SUN_FAMILIES),
        "samples_per_family": samples_per_family,
        "available_counts_by_family": observed_counts,
        "selection_rule": {
            "rank_order": "sample_id_ascii_ascending_before_depth_decode",
            "rank_order_sha256": rank_order_sha256,
            "eligibility": "at_least_100_finite_pixels_strictly_between_0.1_and_10.0_metres",
            "take": "first_eligible_rows_without_replacement",
            "abort_if_insufficient": True,
        },
        "selected_samples": selected_rows,
        "screening": screening,
        "qualitative_ids": qualitative,
        "target_blindness": {
            "depth_values_persisted": False,
            "depth_histograms_persisted": False,
            "depth_aggregate_statistics_persisted": False,
            "target_previews_persisted": False,
            "model_outputs_used_for_membership": False,
        },
        "external_final_exposed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_sun_membership_manifest(manifest, expected_per_family=samples_per_family)
    return manifest, selected_frames


def validate_sun_membership_manifest(
    manifest: Mapping[str, Any], *, expected_per_family: int = SAMPLES_PER_FAMILY
) -> None:
    reject_sealed_references(manifest)
    if manifest.get("schema_version") != MEMBERSHIP_SCHEMA or manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected Phase 2g membership schema/dataset")
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("membership manifest lacks SHA-256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if canonical_sha256(unhashed) != claimed:
        raise ValueError("membership manifest SHA-256 mismatch")
    if manifest.get("families") != list(SUN_FAMILIES) or manifest.get("samples_per_family") != expected_per_family:
        raise ValueError("membership family/count policy changed")
    if (
        expected_per_family == SAMPLES_PER_FAMILY
        and manifest.get("available_counts_by_family") != EXPECTED_INVENTORY_COUNTS
    ):
        raise ValueError("membership available inventory differs from frozen official counts")
    rows = manifest.get("selected_samples")
    screening = manifest.get("screening")
    if not isinstance(rows, list) or not isinstance(screening, Mapping):
        raise ValueError("membership rows/screening are missing")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("selected membership rows must be objects")
    selected_rows = cast(list[Mapping[str, Any]], rows)
    ids = [row.get("sample_id") for row in selected_rows]
    if (
        len(rows) != len(SUN_FAMILIES) * expected_per_family
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("selected membership count/uniqueness failed")
    frozen_ids: dict[str, list[str]] = {}
    for family in SUN_FAMILIES:
        family_rows = [row for row in selected_rows if row.get("family") == family]
        if len(family_rows) != expected_per_family:
            raise ValueError(f"membership count failed for {family}")
        inventory = screening.get(family)
        if not isinstance(inventory, list) or not inventory or any(not isinstance(row, Mapping) for row in inventory):
            raise ValueError(f"screening inventory missing for {family}")
        inventory_rows = cast(list[Mapping[str, Any]], inventory)
        raw_inventory_ids = [row.get("sample_id") for row in inventory_rows]
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in raw_inventory_ids):
            raise ValueError(f"screening sample-ID order changed for {family}")
        inventory_ids = cast(list[str], raw_inventory_ids)
        if len(inventory_ids) != len(set(inventory_ids)) or inventory_ids != sorted(inventory_ids):
            raise ValueError(f"screening sample-ID order changed for {family}")
        frozen_ids[family] = inventory_ids
        if [row.get("rank") for row in inventory_rows] != list(range(len(inventory_rows))):
            raise ValueError(f"screening rank order changed for {family}")
        if any(
            not isinstance(row.get("eligible"), bool) or not isinstance(row.get("selected"), bool)
            for row in inventory_rows
        ):
            raise ValueError("screening eligibility/selection values must be booleans")
        expected_trace = [row for row in inventory_rows if row.get("eligible") is True][:expected_per_family]
        selected_inventory = [row for row in inventory_rows if row.get("selected") is True]
        expected_keys = [(row.get("sample_id"), row.get("rank")) for row in expected_trace]
        selected_keys = [(row.get("sample_id"), row.get("rank")) for row in selected_inventory]
        row_keys = [(row.get("sample_id"), row.get("rank")) for row in family_rows]
        if len(expected_trace) != expected_per_family or selected_keys != expected_keys or row_keys != expected_keys:
            raise ValueError(f"screening/selection mismatch for {family}")
        expected_key_set = set(expected_keys)
        if any(
            bool(row.get("selected")) != ((row.get("sample_id"), row.get("rank")) in expected_key_set)
            for row in inventory_rows
        ):
            raise ValueError(f"screening first-eligible trace changed for {family}")
        for row in inventory_rows:
            reasons = row.get("failure_reasons")
            if not isinstance(reasons, Mapping) or set(reasons) != {
                "depth_decode_failed",
                "fewer_than_100_valid_pixels",
            }:
                raise ValueError("screening rows may contain only frozen boolean failure reasons")
            if any(not isinstance(value, bool) for value in reasons.values()):
                raise ValueError("screening failure reasons must be booleans")
            expected_failures = 0 if row.get("eligible") is True else 1
            if sum(bool(value) for value in reasons.values()) != expected_failures:
                raise ValueError("eligibility and failure-reason booleans disagree")
    selection_rule = manifest.get("selection_rule")
    if (
        not isinstance(selection_rule, Mapping)
        or selection_rule.get("rank_order") != "sample_id_ascii_ascending_before_depth_decode"
        or selection_rule.get("rank_order_sha256") != canonical_sha256(frozen_ids)
        or selection_rule.get("take") != "first_eligible_rows_without_replacement"
        or selection_rule.get("abort_if_insufficient") is not True
    ):
        raise ValueError("membership rank/selection rule changed")
    blindness = manifest.get("target_blindness")
    if (
        not isinstance(blindness, Mapping)
        or any(blindness.values())
        or manifest.get("external_final_exposed") is not False
    ):
        raise ValueError("membership target-blindness audit failed")
    qualitative = manifest.get("qualitative_ids")
    if not isinstance(qualitative, Mapping):
        raise ValueError("qualitative sample IDs are missing")
    for family in SUN_FAMILIES:
        expected = sorted(
            [str(row["sample_id"]) for row in selected_rows if row.get("family") == family],
            key=lambda sample_id: (hashlib.sha256(sample_id.encode()).hexdigest(), sample_id),
        )[:QUALITATIVE_IDS_PER_FAMILY]
        if expected_per_family >= QUALITATIVE_IDS_PER_FAMILY and qualitative.get(family) != expected:
            raise ValueError(f"qualitative sample policy changed for {family}")


def _sample_block(sample_ids: Sequence[str], family: str) -> dict[str, Any]:
    ids = list(sample_ids)
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("shard sample IDs must be unique non-empty strings")
    if family not in SUN_FAMILIES:
        raise ValueError(f"unknown SUN family: {family}")
    return {"sample_ids": ids, "family": family, "count": len(ids)}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys changed: missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _validate_sample_block(value: Any, label: str) -> tuple[Mapping[str, Any], int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} sample block is invalid")
    _require_exact_keys(value, {"sample_ids", "family", "count"}, f"{label} sample block")
    ids = value.get("sample_ids")
    count = value.get("count")
    if (
        value.get("family") not in SUN_FAMILIES
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(ids, list)
        or len(ids) != count
        or len(set(ids)) != count
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
    ):
        raise ValueError(f"{label} sample IDs/count are invalid")
    return value, count


def _validate_intrinsics_tensor(value: Any, shape: tuple[int, ...], label: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all())
        or not bool((value[..., 0, 0] > 0).all())
        or not bool((value[..., 1, 1] > 0).all())
        or not bool(torch.allclose(value[..., 2, :], value.new_tensor((0.0, 0.0, 1.0)).expand_as(value[..., 2, :])))
        or not bool(torch.allclose(value[..., 0, 1], torch.zeros_like(value[..., 0, 1])))
        or not bool(torch.allclose(value[..., 1, 0], torch.zeros_like(value[..., 1, 0])))
    ):
        raise ValueError(f"{label} is not a finite pinhole intrinsics tensor")


def build_input_shard(
    *,
    family: str,
    sample_ids: Sequence[str],
    images_384: torch.Tensor,
    rgb_96: torch.Tensor,
    intrinsics_384: torch.Tensor,
    membership_sha256: str,
) -> dict[str, Any]:
    samples = _sample_block(sample_ids, family)
    count = samples["count"]
    expected = {
        "images": (count, 2, 3, 384, 384),
        "rgb": (count, 2, 3, 96, 96),
        "intrinsics": (count, 2, 3, 3),
    }
    if tuple(images_384.shape) != expected["images"] or tuple(rgb_96.shape) != expected["rgb"]:
        raise ValueError("ordinary RGB tensors have unexpected shape")
    images = images_384.detach().cpu()
    rgb = rgb_96.detach().cpu()
    for label, value in (("images_384", images), ("rgb_96", rgb)):
        if value.dtype != torch.uint8:
            if (
                not torch.is_floating_point(value)
                or not bool(torch.isfinite(value).all())
                or float(value.min()) < 0
                or float(value.max()) > 1
            ):
                raise ValueError(f"{label} must be uint8 or finite [0,1]")
            value = torch.round(value * 255).to(torch.uint8)
        if label == "images_384":
            images = value.contiguous()
        else:
            rgb = value.contiguous()
    k = intrinsics_384.detach().cpu().float().contiguous()
    if tuple(k.shape) != expected["intrinsics"] or not bool(torch.isfinite(k).all()):
        raise ValueError("ordinary intrinsics have unexpected shape/values")
    paired_images = torch.empty((count, PROFILE_COUNT, 3, 384, 384), dtype=torch.uint8)
    profiles = frozen_camera_profiles()
    for source_index, image_uint8 in enumerate(images[:, 0]):
        image = image_uint8.float().div(255)
        for profile_index, profile in enumerate(profiles):
            paired_images[source_index, profile_index] = torch.round(
                apply_profile_to_rgb(image, profile).clamp(0, 1) * 255
            ).to(torch.uint8)
    controls = build_paired_camera_controls(k[:, 0])
    payload = {
        "schema_version": INPUT_SHARD_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": DATASET_ID,
        "membership_sha256": membership_sha256,
        "samples": samples,
        "ordinary_inputs": {
            "view_ids": ["center_square", "center_crop_0.85"],
            "images_384_uint8": images,
            "rgb_96_uint8": rgb,
            "intrinsics_384": k,
        },
        "paired_inputs": {
            "profile_ids": list(PROFILE_IDS),
            "profile_permutation": controls.permutation.cpu(),
            "images_384_uint8": paired_images,
            "updated_k": controls.updated.cpu().float(),
            "stale_k": controls.stale.cpu().float(),
            "wrong_k": controls.wrong.cpu().float(),
            "permuted_k": controls.permuted.cpu().float(),
        },
        "audit": {
            "camera_control_schema": CAMERA_CONTROL_SCHEMA,
            "profiles_per_sample": PROFILE_COUNT,
            "distinct_updated_intrinsics_per_source": list(controls.distinct_updated_per_source),
            "permutation_assignment_change_fraction": controls.permutation_assignment_change_fraction,
            "permutation_matrix_change_fraction": controls.permutation_matrix_change_fraction,
            "targets_present": False,
            "external_final_present": False,
        },
    }
    validate_input_shard(payload)
    return payload


def validate_input_shard(payload: Mapping[str, Any], *, expected_count: int | None = None) -> None:
    reject_sealed_references(payload)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "claim_boundary",
            "dataset_id",
            "membership_sha256",
            "samples",
            "ordinary_inputs",
            "paired_inputs",
            "audit",
        },
        "input shard",
    )
    if payload.get("schema_version") != INPUT_SHARD_SCHEMA or payload.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected Phase 2g input shard")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY or not isinstance(payload.get("membership_sha256"), str):
        raise ValueError("input shard identity/claim boundary changed")
    samples, count = _validate_sample_block(payload.get("samples"), "input shard")
    if expected_count is not None and count != expected_count:
        raise ValueError("input shard sample count mismatch")
    ordinary = payload.get("ordinary_inputs")
    paired = payload.get("paired_inputs")
    if not isinstance(ordinary, Mapping) or not isinstance(paired, Mapping):
        raise ValueError("input shard blocks are invalid")
    _require_exact_keys(
        ordinary,
        {"view_ids", "images_384_uint8", "rgb_96_uint8", "intrinsics_384"},
        "ordinary input block",
    )
    _require_exact_keys(
        paired,
        {
            "profile_ids",
            "profile_permutation",
            "images_384_uint8",
            "updated_k",
            "stale_k",
            "wrong_k",
            "permuted_k",
        },
        "paired input block",
    )
    if ordinary.get("view_ids") != ["center_square", "center_crop_0.85"] or paired.get("profile_ids") != list(
        PROFILE_IDS
    ):
        raise ValueError("input shard view/profile policy changed")
    tensors = (
        (ordinary.get("images_384_uint8"), (count, 2, 3, 384, 384), torch.uint8),
        (ordinary.get("rgb_96_uint8"), (count, 2, 3, 96, 96), torch.uint8),
        (ordinary.get("intrinsics_384"), (count, 2, 3, 3), None),
        (paired.get("images_384_uint8"), (count, 8, 3, 384, 384), torch.uint8),
    )
    for value, shape, dtype in tensors:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or (dtype is not None and value.dtype != dtype)
        ):
            raise ValueError("input shard tensor shape/dtype changed")
    _validate_intrinsics_tensor(ordinary.get("intrinsics_384"), (count, 2, 3, 3), "ordinary intrinsics")
    permutation = paired.get("profile_permutation")
    if not isinstance(permutation, torch.Tensor) or not torch.equal(
        permutation.cpu(), torch.tensor(PROFILE_PERMUTATION)
    ):
        raise ValueError("input shard profile permutation changed")
    for name in ("updated_k", "stale_k", "wrong_k", "permuted_k"):
        value = paired.get(name)
        _validate_intrinsics_tensor(value, (count, 8, 3, 3), f"input shard {name}")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("input shard audit is invalid")
    _require_exact_keys(
        audit,
        {
            "camera_control_schema",
            "profiles_per_sample",
            "distinct_updated_intrinsics_per_source",
            "permutation_assignment_change_fraction",
            "permutation_matrix_change_fraction",
            "targets_present",
            "external_final_present",
        },
        "input shard audit",
    )
    distinct = audit.get("distinct_updated_intrinsics_per_source")
    if (
        audit.get("camera_control_schema") != CAMERA_CONTROL_SCHEMA
        or audit.get("profiles_per_sample") != 8
        or audit.get("permutation_assignment_change_fraction") != 1.0
        or audit.get("permutation_matrix_change_fraction") != 1.0
        or audit.get("targets_present") is not False
        or audit.get("external_final_present") is not False
        or not isinstance(distinct, list)
        or len(distinct) != count
        or any(value != 8 for value in distinct)
    ):
        raise ValueError("input shard audit failed")


def build_target_shard(
    input_shard: Mapping[str, Any],
    *,
    ordinary_depth_24: torch.Tensor,
    ordinary_valid_24: torch.Tensor,
    center_depth_384: torch.Tensor,
    center_valid_384: torch.Tensor,
    input_sha256: str,
) -> dict[str, Any]:
    validate_input_shard(input_shard)
    samples = dict(input_shard["samples"])
    count = int(samples["count"])
    if tuple(ordinary_depth_24.shape) != (count, 2, 24, 24) or tuple(ordinary_valid_24.shape) != (count, 2, 24, 24):
        raise ValueError("ordinary targets have unexpected shape")
    if tuple(center_depth_384.shape) != (count, 384, 384) or tuple(center_valid_384.shape) != (count, 384, 384):
        raise ValueError("center targets have unexpected shape")
    paired_depth: list[torch.Tensor] = []
    paired_valid: list[torch.Tensor] = []
    profiles = frozen_camera_profiles()
    for depth, valid in zip(center_depth_384, center_valid_384, strict=True):
        transformed = [transform_and_reduce_depth(depth, valid, profile) for profile in profiles]
        paired_depth.append(torch.stack([item[0] for item in transformed]))
        paired_valid.append(torch.stack([item[1] for item in transformed]))
    payload = {
        "schema_version": TARGET_SHARD_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": DATASET_ID,
        "membership_sha256": input_shard["membership_sha256"],
        "input_sha256": input_sha256,
        "samples": samples,
        "ordinary_targets": {
            "view_ids": ["center_square", "center_crop_0.85"],
            "depth_24": ordinary_depth_24.detach().cpu().float().contiguous(),
            "valid_24": ordinary_valid_24.detach().cpu().bool().contiguous(),
        },
        "paired_targets": {
            "profile_ids": list(PROFILE_IDS),
            "depth_24": torch.stack(paired_depth).float().contiguous(),
            "valid_24": torch.stack(paired_valid).bool().contiguous(),
        },
        "audit": {"target_reduction": "mask-weighted-area-valid-mass-ge-0.25", "external_final_present": False},
    }
    validate_target_shard(payload)
    return payload


def validate_target_shard(payload: Mapping[str, Any], *, expected_count: int | None = None) -> None:
    reject_sealed_references(payload)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "claim_boundary",
            "dataset_id",
            "membership_sha256",
            "input_sha256",
            "samples",
            "ordinary_targets",
            "paired_targets",
            "audit",
        },
        "target shard",
    )
    if payload.get("schema_version") != TARGET_SHARD_SCHEMA or payload.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected Phase 2g target shard")
    _, count = _validate_sample_block(payload.get("samples"), "target shard")
    if expected_count is not None and count != expected_count:
        raise ValueError("target shard count mismatch")
    for root, views, shape in (
        ("ordinary_targets", ["center_square", "center_crop_0.85"], (count, 2, 24, 24)),
        ("paired_targets", list(PROFILE_IDS), (count, 8, 24, 24)),
    ):
        block = payload.get(root)
        if not isinstance(block, Mapping):
            raise ValueError("target shard block is invalid")
        key = "view_ids" if root == "ordinary_targets" else "profile_ids"
        _require_exact_keys(block, {key, "depth_24", "valid_24"}, f"{root} block")
        if block.get(key) != views:
            raise ValueError("target shard view/profile policy changed")
        depth, valid = block.get("depth_24"), block.get("valid_24")
        if (
            not isinstance(depth, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or tuple(depth.shape) != shape
            or tuple(valid.shape) != shape
        ):
            raise ValueError("target shard tensor shape changed")
        if (
            valid.dtype != torch.bool
            or not torch.is_floating_point(depth)
            or not bool(torch.isfinite(depth[valid]).all())
            or bool((valid.flatten(2).sum(dim=2) == 0).any())
            or bool((depth[valid] <= DEPTH_MIN_M).any())
            or bool((depth[valid] >= DEPTH_MAX_M).any())
        ):
            raise ValueError("target shard values/dtypes are invalid")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("target shard audit is invalid")
    _require_exact_keys(audit, {"target_reduction", "external_final_present"}, "target shard audit")
    if (
        audit.get("target_reduction") != "mask-weighted-area-valid-mass-ge-0.25"
        or audit.get("external_final_present") is not False
    ):
        raise ValueError("target shard external-final audit failed")


def build_feature_shard(
    input_shard: Mapping[str, Any],
    *,
    ordinary_features: torch.Tensor,
    paired_features: torch.Tensor,
    input_sha256: str,
) -> dict[str, Any]:
    validate_input_shard(input_shard)
    count = int(input_shard["samples"]["count"])
    if tuple(ordinary_features.shape) != (count, 2, 768, 24, 24) or tuple(paired_features.shape) != (
        count,
        8,
        768,
        24,
        24,
    ):
        raise ValueError("feature shard tensors have unexpected shape")
    if any(
        not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
        for value in (ordinary_features, paired_features)
    ):
        raise ValueError("feature shard tensors must be finite floating point")
    payload = {
        "schema_version": FEATURE_SHARD_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset_id": DATASET_ID,
        "membership_sha256": input_shard["membership_sha256"],
        "input_sha256": input_sha256,
        "samples": dict(input_shard["samples"]),
        "ordinary_features": ordinary_features.detach().cpu().half().contiguous(),
        "paired_features": paired_features.detach().cpu().half().contiguous(),
        "audit": {
            "normalization": "not-applied-fit-per-rotation-on-two-train-families-only",
            "targets_present": False,
            "external_final_present": False,
        },
    }
    validate_feature_shard(payload)
    return payload


def validate_feature_shard(payload: Mapping[str, Any], *, expected_count: int | None = None) -> None:
    reject_sealed_references(payload)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "claim_boundary",
            "dataset_id",
            "membership_sha256",
            "input_sha256",
            "samples",
            "ordinary_features",
            "paired_features",
            "audit",
        },
        "feature shard",
    )
    if payload.get("schema_version") != FEATURE_SHARD_SCHEMA or payload.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected Phase 2g feature shard")
    _, count = _validate_sample_block(payload.get("samples"), "feature shard")
    if expected_count is not None and count != expected_count:
        raise ValueError("feature shard count mismatch")
    for name, shape in (
        ("ordinary_features", (count, 2, 768, 24, 24)),
        ("paired_features", (count, 8, 768, 24, 24)),
    ):
        value = payload.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"feature shard {name} is invalid")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("feature shard audit is invalid")
    _require_exact_keys(audit, {"normalization", "targets_present", "external_final_present"}, "feature shard audit")
    if (
        audit.get("normalization") != "not-applied-fit-per-rotation-on-two-train-families-only"
        or audit.get("targets_present") is not False
        or audit.get("external_final_present") is not False
    ):
        raise ValueError("feature shard separation audit failed")


def write_torch_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)
    return output


def load_torch(path: Path) -> dict[str, Any]:
    reject_sealed_references(path)
    resolved = path.resolve(strict=True)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError):
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"cache shard must contain a mapping: {path}")
    return value


def create_rotation_views(
    cache_root: Path,
    *,
    membership_sha256: str,
    shard_identities: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Create hard-linked, target-isolated rotation views below ``rotations``."""

    root = cache_root.resolve(strict=True)
    results: dict[str, dict[str, Any]] = {}
    for rotation, roles in ROTATIONS.items():
        view_root = root / "rotations" / rotation
        view_root.mkdir(parents=True, exist_ok=False)
        allowed = [*roles["train"], roles["validation"]]
        family_records: dict[str, Any] = {}
        for family in allowed:
            role = "train" if family in roles["train"] else "validation"
            files: dict[str, Any] = {}
            for kind in ("input", "feature", "target"):
                source = root / "shards" / family / f"{kind}.pt"
                destination = view_root / f"{family}.{kind}.pt"
                os.link(source, destination)
                identity = file_identity(destination)
                expected = shard_identities[family][kind]
                if identity["sha256"] != expected["sha256"] or identity["bytes"] != expected["bytes"]:
                    raise RuntimeError("rotation hard link identity changed")
                files[kind] = {**identity, "path": destination.name}
            family_records[family] = {"role": role, "files": files}
        descriptor: dict[str, Any] = {
            "schema_version": ROTATION_VIEW_SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "rotation": rotation,
            "membership_sha256": membership_sha256,
            "train_families": list(roles["train"]),
            "validation_family": roles["validation"],
            "heldout_family": roles["heldout"],
            "heldout_target_exposed": False,
            "families": family_records,
            "external_final_exposed": False,
        }
        descriptor["view_sha256"] = canonical_sha256(descriptor)
        atomic_json(view_root / "view.json", descriptor)
        validate_rotation_view(view_root, expected_rotation=rotation)
        results[rotation] = descriptor
    return results


def validate_rotation_view(view_root: Path, *, expected_rotation: str | None = None) -> dict[str, Any]:
    """Validate and return a target-isolated training view descriptor."""

    root = view_root.resolve(strict=True)
    reject_sealed_references(root)
    descriptor = json.loads((root / "view.json").read_text(encoding="utf-8"))
    reject_sealed_references(descriptor)
    rotation = descriptor.get("rotation")
    if descriptor.get("schema_version") != ROTATION_VIEW_SCHEMA or rotation not in ROTATIONS:
        raise ValueError("unexpected Phase 2g rotation view")
    if expected_rotation is not None and rotation != expected_rotation:
        raise ValueError("rotation view identity mismatch")
    claimed = descriptor.get("view_sha256")
    unhashed = dict(descriptor)
    unhashed.pop("view_sha256", None)
    if canonical_sha256(unhashed) != claimed:
        raise ValueError("rotation view SHA-256 mismatch")
    roles = ROTATIONS[str(rotation)]
    if (
        descriptor.get("train_families") != list(roles["train"])
        or descriptor.get("validation_family") != roles["validation"]
        or descriptor.get("heldout_family") != roles["heldout"]
        or descriptor.get("heldout_target_exposed") is not False
        or descriptor.get("external_final_exposed") is not False
    ):
        raise ValueError("rotation role/isolation contract changed")
    expected_families = {*roles["train"], roles["validation"]}
    records = descriptor.get("families")
    if not isinstance(records, Mapping) or set(records) != expected_families or roles["heldout"] in records:
        raise ValueError("rotation view exposes wrong family set")
    expected_files = {"view.json"}
    for family in expected_families:
        expected_role = "train" if family in roles["train"] else "validation"
        record = records[family]
        if record.get("role") != expected_role or set(record.get("files", {})) != {"input", "feature", "target"}:
            raise ValueError("rotation family role/files changed")
        loaded: dict[str, dict[str, Any]] = {}
        for kind, schema, validator in (
            ("input", INPUT_SHARD_SCHEMA, validate_input_shard),
            ("feature", FEATURE_SHARD_SCHEMA, validate_feature_shard),
            ("target", TARGET_SHARD_SCHEMA, validate_target_shard),
        ):
            relative = record["files"][kind].get("path")
            if relative != f"{family}.{kind}.pt":
                raise ValueError("rotation view permits only flat, canonical shard names")
            path = (root / relative).resolve(strict=True)
            if path.parent != root:
                raise ValueError("rotation shard path escapes view root")
            identity = file_identity(path, schema=schema)
            expected_files.add(relative)
            if any(identity[key] != record["files"][kind].get(key) for key in ("bytes", "sha256")):
                raise ValueError("rotation shard identity mismatch")
            payload = load_torch(path)
            validator(payload)
            if payload.get("samples", {}).get("family") != family:
                raise ValueError("rotation shard family identity mismatch")
            loaded[kind] = payload
        if not (
            loaded["input"]["samples"] == loaded["feature"]["samples"] == loaded["target"]["samples"]
            and loaded["input"]["membership_sha256"]
            == loaded["feature"]["membership_sha256"]
            == loaded["target"]["membership_sha256"]
            == descriptor["membership_sha256"]
        ):
            raise ValueError("rotation shard row/membership binding mismatch")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_dir() for path in root.iterdir()):
        raise ValueError("rotation view contains unregistered files/directories")
    return descriptor


def load_rotation_training_data(
    view_root: Path, rotation: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load only registered train/validation shards from one isolated view."""

    descriptor = validate_rotation_view(view_root, expected_rotation=rotation)
    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    root = view_root.resolve(strict=True)
    for _family, record in descriptor["families"].items():
        bundle = {kind: load_torch(root / record["files"][kind]["path"]) for kind in ("input", "feature", "target")}
        result[record["role"]].append(bundle)
    result["train"].sort(key=lambda value: descriptor["train_families"].index(value["input"]["samples"]["family"]))
    return result, descriptor


def validate_heldout_shards(
    *,
    rotation: str,
    input_path: Path,
    feature_path: Path,
    target_path: Path,
    expected_count: int = SAMPLES_PER_FAMILY,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load explicit evaluation-only held-out shards and prove row/hash binding."""

    if rotation not in ROTATIONS:
        raise ValueError(f"unknown rotation: {rotation}")
    values = {
        "input": load_torch(input_path),
        "feature": load_torch(feature_path),
        "target": load_torch(target_path),
    }
    validate_input_shard(values["input"], expected_count=expected_count)
    validate_feature_shard(values["feature"], expected_count=expected_count)
    validate_target_shard(values["target"], expected_count=expected_count)
    expected_family = ROTATIONS[rotation]["heldout"]
    if any(value["samples"]["family"] != expected_family for value in values.values()):
        raise ValueError("evaluation shards are not the rotation's held-out family")
    if not (values["input"]["samples"] == values["feature"]["samples"] == values["target"]["samples"]):
        raise ValueError("evaluation shard rows differ")
    input_identity = file_identity(input_path, schema=INPUT_SHARD_SCHEMA)
    if (
        values["feature"]["input_sha256"] != input_identity["sha256"]
        or values["target"]["input_sha256"] != input_identity["sha256"]
    ):
        raise ValueError("evaluation shards are not bound to the supplied input shard")
    identities = {
        "input": input_identity,
        "feature": file_identity(feature_path, schema=FEATURE_SHARD_SCHEMA),
        "target": file_identity(target_path, schema=TARGET_SHARD_SCHEMA),
    }
    return values["input"], values["feature"], values["target"], identities


def assert_expected_parameter_counts() -> dict[str, int]:
    """Return and enforce the frozen M0--M3 architecture identities."""

    from jepa4d.models.phase2f_scale_geometry import Phase2fArm, Phase2fScaleGeometryProbe
    from jepa4d.training.phase2f_training import phase2f_arm_configs

    expected = {"M0": 86_402, "M1": 92_820, "M2": 92_916, "M3": 93_685}
    configs = phase2f_arm_configs(768)
    actual = {
        arm: Phase2fScaleGeometryProbe(configs[cast(Phase2fArm, arm)]).trainable_parameter_count for arm in expected
    }
    if actual != expected:
        raise RuntimeError(f"Phase 2g architecture identity changed: {actual} != {expected}")
    return actual


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
