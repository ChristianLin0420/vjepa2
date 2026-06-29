"""Versioned dataset manifests with optional local-asset integrity checks."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AssetManifest:
    path: str
    sha256: str | None = None
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    asset: str
    reason: str


@dataclass(slots=True)
class DatasetManifest:
    dataset_id: str
    version: str
    revision: str
    split: str
    license: str
    evidence_level: str
    official: bool
    source_url: str | None = None
    description: str = ""
    assets: list[AssetManifest] = field(default_factory=list)
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("dataset_id", "version", "revision", "split", "license", "evidence_level"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"manifest field {name!r} cannot be empty")
        if self.official and not self.source_url:
            raise ValueError("official manifests require source_url")

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        source = Path(path)
        raw = yaml.safe_load(source.read_text())
        if not isinstance(raw, dict):
            raise ValueError("dataset manifest must contain a mapping")
        raw["assets"] = [AssetManifest(**value) for value in raw.get("assets", [])]
        value = cls(**raw)
        value.manifest_path = source.resolve()
        return value

    @property
    def root(self) -> Path:
        if self.manifest_path is None:
            raise ValueError("manifest root is unavailable before loading from a file")
        return self.manifest_path.parent

    def validate_assets(self, *, verify_hashes: bool = True) -> list[ManifestIssue]:
        issues: list[ManifestIssue] = []
        for asset in self.assets:
            path = self.root / asset.path
            if not path.is_file():
                issues.append(ManifestIssue(asset.path, "missing"))
                continue
            if asset.bytes is not None and path.stat().st_size != asset.bytes:
                issues.append(ManifestIssue(asset.path, "size_mismatch"))
            if verify_hashes and asset.sha256:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != asset.sha256:
                    issues.append(ManifestIssue(asset.path, "sha256_mismatch"))
        return issues

    def to_serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["manifest_path"] = None if self.manifest_path is None else str(self.manifest_path)
        return value
