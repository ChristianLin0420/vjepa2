"""Download, safely extract, and content-verify the Phase 2c TUM bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slurm.phase2b_gate import asset_inventory  # noqa: E402
from slurm.phase2c_gate import atomic_json, bundle_identity, protocol_contract, validated_bundle  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid(path: Path, expected_bytes: int, expected_sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == expected_bytes and _sha256(path) == expected_sha256


def _download(url: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    if _is_valid(destination, expected_bytes, expected_sha256):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        os.getenv("JEPA4D_DOWNLOAD_STAGING_ROOT", f"/tmp/{os.getenv('USER', 'jepa4d')}/jepa4d-phase2c")
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f"{destination.name}.partial"
    request = urllib.request.Request(url, headers={"User-Agent": "jepa4d-phase2c-asset-setup/1"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
    if not _is_valid(temporary, expected_bytes, expected_sha256):
        raise RuntimeError(f"download identity mismatch for {destination.name}")
    shutil.copyfile(temporary, destination)
    if not _is_valid(destination, expected_bytes, expected_sha256):
        raise RuntimeError(f"Lustre copy identity mismatch for {destination}")


def _safe_extract(archive: Path, destination: Path, expected_root: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    parent = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(parent):
                raise RuntimeError(f"archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"archive contains unsupported member type: {member.name}")
        roots = {Path(member.name.removeprefix("./")).parts[0] for member in members if member.name.removeprefix("./")}
        if roots != {expected_root}:
            raise RuntimeError(f"archive roots {sorted(roots)} do not match {expected_root}")
        bundle.extractall(destination, filter="data")
    if not (destination / expected_root).is_dir():
        raise RuntimeError(f"archive did not create expected root: {expected_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.manifest.read_text())
    entries = raw.get("sequences")
    if raw.get("schema_version") != "jepa4d-tum-cross-sequence-v1" or not isinstance(entries, list):
        raise RuntimeError("unexpected Phase 2c bundle manifest")
    if len(entries) != 5:
        raise RuntimeError("formal Phase 2c requires exactly five sequence archives")
    downloads: list[dict[str, Any]] = []
    for entry in entries:
        archive_spec = entry["archive"]
        archive = args.data_parent / str(archive_spec["filename"])
        expected_bytes = int(archive_spec["bytes"])
        expected_sha256 = str(archive_spec["sha256"])
        _download(str(entry["source_url"]), archive, expected_bytes, expected_sha256)
        _safe_extract(archive, args.data_parent, str(entry["root_name"]))
        downloads.append(
            {
                "sequence_id": str(entry["sequence_id"]),
                "archive": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": _sha256(archive),
                "root": str((args.data_parent / str(entry["root_name"])).resolve()),
            }
        )
    bundle = validated_bundle(args.data_parent, args.manifest)
    report = {
        "schema_version": "jepa4d-phase2c-asset-setup-v1",
        "status": "pass",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": protocol_contract(),
        "bundle": bundle_identity(bundle),
        "downloads": downloads,
        "models": {
            "vjepa_checkpoint": asset_inventory(args.vjepa_checkpoint),
            "vjepa_implementation": asset_inventory(args.vjepa_implementation),
            "vggt_checkpoint": asset_inventory(args.vggt_checkpoint),
        },
        "recommended_environment": {
            "JEPA4D_DATASET_PARENT": str(args.data_parent.resolve()),
            "JEPA4D_MANIFEST": str(args.manifest.resolve()),
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
