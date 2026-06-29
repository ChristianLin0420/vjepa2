"""Deterministic identities shared by the Phase 2b Slurm gates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
        timeout=120,
    )
    return result.stdout


def repository_fingerprint(repo_root: Path) -> dict[str, Any]:
    """Hash every tracked or relevant untracked file, including dirty state."""
    root = repo_root.resolve(strict=True)
    listed = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", binary=True)
    assert isinstance(listed, bytes)
    relative_paths = sorted(Path(value.decode()) for value in listed.split(b"\0") if value)
    files: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink():
            target = os.readlink(path)
            files.append(
                {
                    "path": relative.as_posix(),
                    "kind": "symlink",
                    "target": target,
                    "sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
            )
        elif path.is_file():
            files.append(
                {
                    "path": relative.as_posix(),
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        else:
            files.append({"path": relative.as_posix(), "kind": "missing"})
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    commit = _git(root, "rev-parse", "HEAD")
    assert isinstance(status, str)
    assert isinstance(commit, str)
    identity = {
        "git_commit": commit.strip(),
        "git_status": status.splitlines(),
        "file_count": len(files),
        "files": files,
    }
    return {**identity, "sha256": canonical_sha256(identity)}


def environment_fingerprint() -> dict[str, Any]:
    """Capture the exact Python distribution set used by a gate."""
    distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or "<unnamed>"
        distributions.append(f"{name.lower()}=={distribution.version}")
    distributions.sort()
    identity = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "distributions": distributions,
    }
    return {**identity, "sha256": canonical_sha256(identity)}


def asset_inventory(path: Path) -> dict[str, Any]:
    """Return a content-complete inventory for one checkpoint or source tree."""
    root = path.resolve(strict=True)
    files = [root] if root.is_file() else sorted(value for value in root.rglob("*") if value.is_file())
    rows: list[dict[str, Any]] = [
        {
            "path": str(value.relative_to(root) if root.is_dir() else value.name),
            "bytes": value.stat().st_size,
            "sha256": sha256(value),
        }
        for value in files
    ]
    identity = {
        "path": str(root),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    return {**identity, "sha256": canonical_sha256(identity)}
