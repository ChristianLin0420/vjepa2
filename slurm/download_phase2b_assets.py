"""Download and verify the public model and TUM assets required by Phase 2b."""

from __future__ import annotations

import argparse
import fnmatch
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import yaml  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from jepa4d.benchmarks.geometry.tum_rgbd import load_tum_indices, validate_archive  # noqa: E402
from scripts.run_phase2b_geometry_distillation import _dataset_fingerprint  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    existing_is_valid = (
        destination.is_file()
        and destination.stat().st_size == expected_bytes
        and _sha256(destination) == expected_sha256
    )
    if existing_is_valid:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        os.getenv("JEPA4D_DOWNLOAD_STAGING_ROOT", f"/tmp/{os.getenv('USER', 'jepa4d')}/jepa4d-phase2b-downloads")
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f"{destination.name}.partial"
    request = urllib.request.Request(url, headers={"User-Agent": "jepa4d-phase2b-asset-setup/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(f"downloaded TUM archive has {temporary.stat().st_size} bytes, expected {expected_bytes}")
    actual_sha256 = _sha256(temporary)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"downloaded TUM SHA-256 is {actual_sha256}, expected {expected_sha256}")
    shutil.copyfile(temporary, destination)
    if destination.stat().st_size != expected_bytes or _sha256(destination) != expected_sha256:
        raise RuntimeError(f"copied TUM archive failed verification: {destination}")


def _extract_verified_archive(archive: Path, data_parent: Path, dataset_root: Path) -> None:
    data_parent.mkdir(parents=True, exist_ok=True)
    parent = data_parent.resolve()
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            target = (data_parent / member.name).resolve()
            if not target.is_relative_to(parent):
                raise RuntimeError(f"archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"archive contains unsupported member type: {member.name}")
        bundle.extractall(data_parent, filter="data")
    if not dataset_root.is_dir():
        raise RuntimeError(f"archive did not create expected dataset root: {dataset_root}")


def _snapshot(
    api: HfApi,
    repo_id: str,
    requested_revision: str,
    destination: Path,
    allow_patterns: list[str],
    token: str | None,
) -> dict[str, Any]:
    info = api.model_info(repo_id, revision=requested_revision, token=token, files_metadata=True)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not resolve a commit for {repo_id}@{requested_revision}")
    resolved_revision = info.sha
    destination.mkdir(parents=True, exist_ok=True)
    selected = [
        sibling
        for sibling in (info.siblings or [])
        if any(fnmatch.fnmatch(str(sibling.rfilename), pattern) for pattern in allow_patterns)
    ]
    if not selected:
        raise RuntimeError(f"allow patterns selected no files from {repo_id}@{resolved_revision}")
    if not all(_verify_hf_file(destination / str(sibling.rfilename), sibling) for sibling in selected):
        staging_root = Path(
            os.getenv("JEPA4D_HF_STAGING_ROOT", f"/tmp/{os.getenv('USER', 'jepa4d')}/jepa4d-phase2b-hf")
        )
        staging = staging_root / repo_id.replace("/", "--") / resolved_revision
        staging.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            local_dir=staging,
            allow_patterns=allow_patterns,
            token=token,
            # Finalize on node-local storage. Hugging Face's atomic rename can
            # block indefinitely when local_dir is on this cluster's Lustre.
            max_workers=1,
        )
        for sibling in selected:
            filename = str(sibling.rfilename)
            source = staging / filename
            if not _verify_hf_file(source, sibling):
                raise RuntimeError(f"staged Hugging Face file failed identity verification: {source}")
            final = destination / filename
            if not _verify_hf_file(final, sibling):
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, final)
            if not _verify_hf_file(final, sibling):
                raise RuntimeError(f"copied Hugging Face file failed identity verification: {final}")
    verified_files = []
    for sibling in selected:
        path = destination / str(sibling.rfilename)
        if not _verify_hf_file(path, sibling):
            raise RuntimeError(f"downloaded Hugging Face file failed identity verification: {path}")
        size, identity, is_lfs = _sibling_identity(sibling)
        verified_files.append(
            {
                "path": str(sibling.rfilename),
                "bytes": size,
                "identity": identity,
                "identity_type": "sha256" if is_lfs else "git_blob_sha1",
            }
        )
    marker = {
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "allow_patterns": allow_patterns,
        "staging_root": str(
            Path(os.getenv("JEPA4D_HF_STAGING_ROOT", f"/tmp/{os.getenv('USER', 'jepa4d')}/jepa4d-phase2b-hf"))
        ),
        "verified_files": verified_files,
    }
    _atomic_json(destination / "jepa4d_huggingface_revision.json", marker)
    return {**marker, "path": str(destination.resolve())}


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sibling_identity(sibling: Any) -> tuple[int, str, bool]:
    size = int(sibling.size)
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        return size, str(lfs.sha256), True
    blob_id = getattr(sibling, "blob_id", None)
    if not blob_id:
        raise RuntimeError(f"Hugging Face did not provide an identity for {sibling.rfilename}")
    return size, str(blob_id), False


def _verify_hf_file(path: Path, sibling: Any) -> bool:
    expected_size, expected_identity, is_lfs = _sibling_identity(sibling)
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    actual_identity = _sha256(path) if is_lfs else _git_blob_sha1(path)
    return actual_identity == expected_identity


def _require_weight(path: Path, minimum_bytes: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise RuntimeError(f"missing or truncated model weight: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vjepa-repo", default="davevanveen/vjepa2.1-vitb-fpc64-384")
    parser.add_argument("--vjepa-revision", default="main")
    parser.add_argument("--implementation-repo", default="Dev-Jahn/vjepa2.1-vitl-fpc64-384")
    parser.add_argument("--implementation-revision", default="b22f310ee1ed02126842983d9a3adc4e296d9284")
    parser.add_argument("--vggt-repo", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-revision", default="d88e637d32a505f4a64de03f8588547b7f7d3ba6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    checkpoint_root = args.asset_root / "checkpoints" / "phase2b_assets"
    vjepa_path = checkpoint_root / "vjepa2.1-vitb-fpc64-384"
    implementation_path = checkpoint_root / "vjepa21_hf_impl"
    vggt_path = checkpoint_root / "VGGT-1B"

    models = {
        "vjepa": _snapshot(
            api,
            args.vjepa_repo,
            args.vjepa_revision,
            vjepa_path,
            ["config.json", "model.safetensors", "README.md"],
            token,
        ),
        "vjepa_implementation": _snapshot(
            api,
            args.implementation_repo,
            args.implementation_revision,
            implementation_path,
            ["*.py"],
            token,
        ),
        "vggt": _snapshot(
            api,
            args.vggt_repo,
            args.vggt_revision,
            vggt_path,
            ["config.json", "model.safetensors", "README.md"],
            token,
        ),
    }
    models["vjepa"]["weight"] = _require_weight(vjepa_path / "model.safetensors", 100 * 1024 * 1024)
    models["vggt"]["weight"] = _require_weight(vggt_path / "model.safetensors", 4 * 1024**3)
    for filename in ("configuration_vjepa21.py", "modeling_vjepa21.py"):
        path = implementation_path / filename
        if not path.is_file():
            raise RuntimeError(f"missing V-JEPA compatibility source: {path}")

    raw_manifest = yaml.safe_load(args.manifest.read_text())
    archive_info = raw_manifest["archive"]
    archive = args.data_parent / f"{raw_manifest['sequence']}.tgz"
    _download_verified(
        str(raw_manifest["source_url"]),
        archive,
        int(archive_info["bytes"]),
        str(archive_info["sha256"]),
    )
    validated_manifest = validate_archive(archive, args.manifest)
    dataset_root = args.data_parent / str(validated_manifest["sequence"])
    _extract_verified_archive(archive, args.data_parent, dataset_root)
    splits = {
        split: load_tum_indices(dataset_root, [int(value) for value in validated_manifest[f"{split}_indices"]])
        for split in ("train", "validation", "test")
    }
    dataset_fingerprint = _dataset_fingerprint(dataset_root, splits, archive)

    report = {
        "schema_version": "jepa4d-phase2b-asset-setup-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models,
        "dataset": {
            "root": str(dataset_root.resolve()),
            "archive": str(archive.resolve()),
            "archive_sha256": archive_info["sha256"],
            "manifest": str(args.manifest.resolve()),
            "fingerprint": dataset_fingerprint,
        },
        "recommended_environment": {
            "JEPA4D_ASSET_ROOT": str(args.asset_root.resolve()),
            "JEPA4D_DATASET_ROOT": str(dataset_root.resolve()),
            "JEPA4D_TUM_ARCHIVE": str(archive.resolve()),
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
