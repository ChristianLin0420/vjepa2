#!/usr/bin/env python3
"""Audit only compressed DIODE bytes and pinned devkit metadata; never list or extract targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2f_metrics import publish_online_wandb
from slurm.phase2f_contract import atomic_json, file_identity, load_json, reject_secrets

SCHEMA = "jepa4d-phase2f-diode-asset-seal-v1"
ARCHIVE_BYTES = 2_774_625_282
ARCHIVE_MD5 = "5c895d09201b88973c8fe4552a67dd85"
DEVKIT_COMMIT = "8b1765b7d801a5f5e2877c434ffe164e62ce8c90"
DEVKIT_FILES = {
    "diode_meta.json": "ea293e1e8eb5615430353291ea9b798d8e75b6672abfd90d185069a3f53b1288",
    "intrinsics.txt": "ba3c845f0ca40173196bcdf8ce66b03be431840b077665bf85172f156b930b02",
    "LICENSE": "bb83d5a21f4b0d0dd6a024a41e4f3719cda0fcf0093b03f1c536931a7f396a58",
}


def hash_archive(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


def audit_assets(
    archive: Path,
    devkit_root: Path,
    *,
    expected_bytes: int = ARCHIVE_BYTES,
    expected_md5: str = ARCHIVE_MD5,
    expected_commit: str = DEVKIT_COMMIT,
    expected_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    archive = archive.resolve(strict=True)
    if not archive.is_file():
        raise ValueError("DIODE validation archive must be one regular compressed file")
    md5, sha256, size = hash_archive(archive)
    if size != expected_bytes or md5 != expected_md5:
        raise ValueError(f"DIODE archive identity mismatch: bytes={size}, md5={md5}")
    root = devkit_root.resolve(strict=True)
    commit = subprocess.check_output(("git", "-C", str(root), "rev-parse", "HEAD"), text=True).strip()
    if commit != expected_commit:
        raise ValueError(f"DIODE devkit commit {commit} != {expected_commit}")
    identities: dict[str, Any] = {}
    for relative, expected_sha256 in (expected_files or DEVKIT_FILES).items():
        identity = file_identity(root / relative)
        if identity["sha256"] != expected_sha256:
            raise ValueError(f"DIODE devkit metadata identity changed: {relative}")
        identities[relative] = identity
    return {
        "archive": {"path": str(archive), "bytes": size, "md5": md5, "sha256": sha256},
        "devkit": {"path": str(root), "git_commit": commit, "files": identities},
        "target_opacity": {
            "compressed_stream_only": True,
            "tar_listed": False,
            "tar_extracted": False,
            "target_array_loaded": False,
            "target_statistics_computed": False,
            "target_preview_generated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--devkit-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = load_json(args.provenance)
    result = {
        "schema_version": SCHEMA,
        "status": "success",
        "created_utc": datetime.now(UTC).isoformat(),
        **audit_assets(args.archive, args.devkit_root),
        "execution_provenance": provenance,
        "claim_boundary": "Compressed-byte and public-metadata audit only; no DIODE target value was opened.",
    }
    reject_secrets(result)
    receipt = atomic_json(args.output, result)
    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    wandb = publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type="asset-seal",
        run_name=f"{execution_id}-asset-seal-{job_id}",
        config={"execution_id": execution_id, "git_commit": provenance["git_commit"]},
        summary={"status": "success", "target_values_accessed": False},
        artifact_name=f"phase2f-asset-seal-{execution_id}",
        artifact_files=(receipt,),
    )
    result["wandb"] = wandb
    atomic_json(receipt, result)
    atomic_json(receipt.parent / "wandb_receipt.json", wandb)
    (receipt.parent / "SUCCESS").write_text("success\n", encoding="utf-8")
    print(json.dumps({"status": "success", "archive_sha256": result["archive"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
