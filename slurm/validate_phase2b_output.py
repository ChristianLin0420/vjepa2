"""Fail a Phase 2b Slurm job when its output contract is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_finite(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{location}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"non-finite value at {location}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output.resolve()
    destination = args.report or root / "postflight-validation.json"
    errors: list[str] = []
    comparison_path = root / "comparison.json"
    failures_path = root / "failures.json"
    comparison: dict[str, Any] = {}

    if not comparison_path.is_file():
        errors.append(f"missing comparison report: {comparison_path}")
    else:
        try:
            comparison = json.loads(comparison_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"cannot read comparison report: {error}")
    failures: list[Any] = []
    if not failures_path.is_file():
        errors.append(f"missing failures report: {failures_path}")
    else:
        try:
            failures = json.loads(failures_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"cannot read failures report: {error}")

    variants = comparison.get("variants", [])
    if len(variants) != 10:
        errors.append(f"expected 10 result rows (one teacher plus nine probes), found {len(variants)}")
    expected_counts = {"vggt_teacher": 1, "rgb_probe": 3, "vjepa_final": 3, "vjepa_multilayer": 3}
    actual_counts = Counter(row.get("variant_id") for row in variants)
    if dict(actual_counts) != expected_counts:
        errors.append(f"unexpected variant counts: {dict(actual_counts)}")
    seeds: dict[str, set[Any]] = defaultdict(set)
    for index, row in enumerate(variants):
        seeds[str(row.get("variant_id"))].add(row.get("seed"))
        _check_finite(row.get("metrics", {}), f"variants[{index}].metrics", errors)
        _check_finite(row.get("runtime", {}), f"variants[{index}].runtime", errors)
        required_metrics = {"metric_abs_rel", "metric_rmse_m", "metric_delta_1", "aligned_abs_rel"}
        missing_metrics = required_metrics - set(row.get("metrics", {}))
        if missing_metrics:
            errors.append(f"variants[{index}] is missing metrics: {sorted(missing_metrics)}")
        required_runtime = {"encoder_ms_per_frame", "head_ms_per_frame", "total_ms_per_frame"}
        missing_runtime = required_runtime - set(row.get("runtime", {}))
        if missing_runtime:
            errors.append(f"variants[{index}] is missing runtime fields: {sorted(missing_runtime)}")
    for variant in ("rgb_probe", "vjepa_final", "vjepa_multilayer"):
        if seeds[variant] != {0, 1, 2}:
            errors.append(f"{variant} seeds are {sorted(seeds[variant], key=str)}, expected [0, 1, 2]")
    if seeds["vggt_teacher"] != {None}:
        errors.append(f"teacher seed must be null, found {seeds['vggt_teacher']}")
    if comparison.get("failures"):
        errors.append(f"comparison contains {len(comparison['failures'])} failure(s)")
    if failures:
        errors.append(f"failures.json contains {len(failures)} failure(s)")
    _check_finite(comparison.get("aggregates", {}), "aggregates", errors)
    if args.require_wandb and not comparison.get("wandb_url"):
        errors.append("online W&B was requested but comparison.wandb_url is empty")

    checkpoints = sorted((root / "checkpoints").glob("*.pt"))
    if len(checkpoints) != 9:
        errors.append(f"expected 9 probe checkpoints, found {len(checkpoints)}")
    checkpoint_hashes = {path.name: _sha256(path) for path in checkpoints}
    recorded_checkpoints = [row.get("checkpoint") for row in variants if row.get("variant_id") != "vggt_teacher"]
    for checkpoint in recorded_checkpoints:
        if not checkpoint or not Path(checkpoint).is_file():
            errors.append(f"recorded checkpoint is missing: {checkpoint}")
    for row in variants:
        checkpoint = row.get("checkpoint")
        recorded_hash = row.get("checkpoint_sha256")
        if checkpoint and recorded_hash and Path(checkpoint).is_file() and _sha256(Path(checkpoint)) != recorded_hash:
            errors.append(f"recorded checkpoint hash mismatch: {checkpoint}")
    recorded_artifacts = comparison.get("artifacts", {})
    for name in ("rgb_probe-normalization.pt", "vjepa_final-normalization.pt", "vjepa_multilayer-normalization.pt"):
        normalization = root / name
        if not normalization.is_file():
            errors.append(f"missing normalization artifact: {name}")
        elif recorded_artifacts.get(name) != _sha256(normalization):
            errors.append(f"normalization hash is missing or mismatched: {name}")

    completion_path = root / "completion_gate.json"
    if not completion_path.is_file():
        errors.append("missing completion_gate.json")
    else:
        try:
            completion = json.loads(completion_path.read_text())
            if completion.get("status") != "success":
                errors.append(f"runner completion gate did not pass: {completion}")
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"cannot read completion gate: {error}")
    html_report = root / "geometry_student_report.html"
    if not html_report.is_file() or html_report.stat().st_size == 0:
        errors.append("missing or empty geometry_student_report.html")
    if (root / "run_failure.json").exists():
        errors.append("run_failure.json exists")

    artifact_manifest_path = root / "artifact_manifest.json"
    artifact_manifest: dict[str, Any] = {}
    if not artifact_manifest_path.is_file():
        errors.append("missing artifact_manifest.json")
    else:
        try:
            artifact_manifest = json.loads(artifact_manifest_path.read_text())
            required_artifacts = ["comparison.json", "geometry_student_report.html"] + [
                f"checkpoints/{path.name}" for path in checkpoints
            ]
            for relative in required_artifacts:
                entry = artifact_manifest.get(relative)
                artifact = root / relative
                if entry is None:
                    errors.append(f"artifact manifest is missing {relative}")
                elif artifact.is_file() and entry.get("sha256") != _sha256(artifact):
                    errors.append(f"artifact manifest hash mismatch: {relative}")
            excluded = {
                "artifact_manifest.json",
                "wandb_artifact_receipt.json",
            }
            actual_files = {
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file() and str(path.relative_to(root)) not in excluded
            }
            manifest_files = set(artifact_manifest)
            missing_from_manifest = sorted(actual_files - manifest_files)
            missing_from_disk = sorted(manifest_files - actual_files)
            if missing_from_manifest:
                errors.append(f"files missing from artifact manifest: {missing_from_manifest}")
            if missing_from_disk:
                errors.append(f"manifest entries missing from disk: {missing_from_disk}")
            for relative, entry in artifact_manifest.items():
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    errors.append(f"unsafe artifact manifest path: {relative}")
                    continue
                artifact = root / relative_path
                if not artifact.is_file():
                    continue
                if not isinstance(entry, dict):
                    errors.append(f"invalid artifact manifest entry: {relative}")
                    continue
                if entry.get("bytes") != artifact.stat().st_size:
                    errors.append(f"artifact manifest byte count mismatch: {relative}")
                if entry.get("sha256") != _sha256(artifact):
                    errors.append(f"artifact manifest hash mismatch: {relative}")
        except (json.JSONDecodeError, OSError, AttributeError) as error:
            errors.append(f"cannot validate artifact manifest: {error}")

    if args.require_wandb:
        receipt_path = root / "wandb_artifact_receipt.json"
        if not receipt_path.is_file():
            errors.append("missing W&B artifact upload receipt")
        else:
            try:
                receipt = json.loads(receipt_path.read_text())
                required_receipt = (
                    "run_id",
                    "run_url",
                    "artifact_name",
                    "artifact_version",
                    "artifact_digest",
                    "artifact_manifest_sha256",
                )
                if receipt.get("schema_version") != "jepa4d-phase2b-wandb-artifact-v1":
                    errors.append("unexpected W&B artifact receipt schema")
                if receipt.get("status") != "success" or receipt.get("mode") != "online":
                    errors.append(f"W&B artifact receipt is not an online success: {receipt.get('status')}")
                if any(not receipt.get(key) for key in required_receipt):
                    errors.append("W&B artifact receipt is incomplete")
                if receipt.get("run_url") != comparison.get("wandb_url"):
                    errors.append("W&B artifact receipt run URL differs from comparison")
                if artifact_manifest_path.is_file() and receipt.get("artifact_manifest_sha256") != _sha256(
                    artifact_manifest_path
                ):
                    errors.append("W&B artifact receipt refers to a different artifact manifest")
            except (json.JSONDecodeError, OSError, AttributeError) as error:
                errors.append(f"cannot validate W&B artifact receipt: {error}")

    report = {
        "schema_version": "jepa4d-phase2b-postflight-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pass" if not errors else "fail",
        "output": str(root),
        "comparison_sha256": _sha256(comparison_path) if comparison_path.is_file() else None,
        "checkpoint_sha256": checkpoint_hashes,
        "result_rows": len(variants),
        "variant_counts": dict(actual_counts),
        "failures_count": len(failures),
        "errors": errors,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
