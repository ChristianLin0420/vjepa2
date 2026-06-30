"""Canonical serialization and content-addressed artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ContentAddress:
    path: Path
    sha256: str
    bytes: int


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that fails closed on duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_unique(path: str | Path) -> Any:
    source = Path(path)
    try:
        return yaml.load(source.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ValueError(f"could not load YAML {source}: {error}") from error


def json_value(value: Any) -> Any:
    """Normalize a value into deterministic JSON-compatible primitives."""
    if isinstance(value, BaseModel):
        return json_value(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, Set) and not isinstance(value, str | bytes):
        normalized = [json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if isinstance(value, tuple | list):
        return [json_value(item) for item in value]
    return value


def canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes without a trailing newline."""
    return json.dumps(
        json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_content_addressed_json(
    value: Any,
    output_dir: str | Path,
    *,
    prefix: str,
) -> ContentAddress:
    """Atomically create an immutable JSON object named by its payload digest."""
    payload = canonical_json(value) + b"\n"
    digest = sha256_value(value)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{prefix}-{digest}.json"
    if target.exists():
        verify_content_addressed_json(target, prefix=prefix)
        return ContentAddress(target, digest, target.stat().st_size)

    descriptor, temporary = tempfile.mkstemp(prefix=f".{prefix}-", suffix=".tmp", dir=directory)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            verify_content_addressed_json(target, prefix=prefix)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
    return ContentAddress(target, digest, target.stat().st_size)


def verify_content_addressed_json(path: str | Path, *, prefix: str) -> dict[str, Any]:
    source = Path(path)
    marker = f"{prefix}-"
    if not source.name.startswith(marker) or source.suffix != ".json":
        raise ValueError(f"expected content-addressed {prefix!r} JSON filename: {source.name}")
    expected = source.name[len(marker) : -len(".json")]
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"invalid SHA-256 filename: {source.name}")
    try:
        payload = source.read_bytes()
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact {source}: {error}") from error
    actual = sha256_value(value)
    if actual != expected:
        raise ValueError(f"content digest mismatch for {source}: expected {expected}, got {actual}")
    canonical_payload = canonical_json(value) + b"\n"
    if payload != canonical_payload:
        raise ValueError(f"non-canonical JSON encoding for {source}")
    return value
