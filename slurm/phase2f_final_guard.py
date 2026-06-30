#!/usr/bin/env python3
"""Enforce one-shot DIODE opening or write the registered no-survivor final skip."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2f_metrics import publish_online_wandb
from slurm.phase2f_asset_audit import ARCHIVE_BYTES, ARCHIVE_MD5
from slurm.phase2f_asset_audit import SCHEMA as ASSET_SCHEMA
from slurm.phase2f_contract import atomic_json, file_identity, load_json, reject_secrets, sha256_file

FINAL_SCHEMA = "jepa4d-phase2f-external-final-v1"
SELECTOR_SCHEMA = "jepa4d-phase2f-development-selector-v1"
SENTINEL_SCHEMA = "jepa4d-phase2f-fresh-final-opened-v1"


def _sentinel_preregistration_sha256(value: dict[str, Any]) -> str | None:
    direct = value.get("preregistration_sha256")
    if isinstance(direct, str):
        return direct
    provenance = value.get("execution_provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("preregistration_sha256"), str):
        return str(provenance["preregistration_sha256"])
    selector = value.get("selector")
    if isinstance(selector, dict) and isinstance(selector.get("path"), str):
        try:
            selector_value = load_json(Path(selector["path"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        selector_provenance = selector_value.get("execution_provenance")
        if isinstance(selector_provenance, dict) and isinstance(
            selector_provenance.get("preregistration_sha256"), str
        ):
            return str(selector_provenance["preregistration_sha256"])
    return None


def validate_selector(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != SELECTOR_SCHEMA or value.get("status") != "success":
        raise ValueError("external final requires a successful development selector")
    survivor = value.get("survivor")
    authorized = value.get("final_authorized")
    if authorized is True:
        if survivor not in {"M1", "M2", "M3"} or set(value.get("checkpoint_set", {})) != {"M0", survivor}:
            raise ValueError("selector authorized an invalid survivor/checkpoint set")
    elif authorized is False:
        if survivor is not None or value.get("checkpoint_set") != {}:
            raise ValueError("no-survivor selector must expose no final checkpoint set")
    else:
        raise ValueError("selector final_authorized is not boolean")
    return value


def validate_asset_receipt(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != ASSET_SCHEMA or value.get("status") != "success":
        raise ValueError("external final requires a successful DIODE asset seal")
    archive = value.get("archive")
    opacity = value.get("target_opacity")
    if not isinstance(archive, dict) or archive.get("bytes") != ARCHIVE_BYTES or archive.get("md5") != ARCHIVE_MD5:
        raise ValueError("DIODE asset receipt archive identity mismatch")
    expected_opacity = {
        "compressed_stream_only": True,
        "tar_listed": False,
        "tar_extracted": False,
        "target_array_loaded": False,
        "target_statistics_computed": False,
        "target_preview_generated": False,
    }
    if opacity != expected_opacity:
        raise ValueError("DIODE asset receipt does not prove target opacity")
    return value


def create_sentinel(
    *, selector_path: Path, asset_receipt_path: Path, provenance_path: Path, sentinel: Path, registry_root: Path
) -> dict[str, Any]:
    selector = validate_selector(selector_path)
    if selector["final_authorized"] is not True:
        raise ValueError("cannot open DIODE without one authorized survivor")
    asset = validate_asset_receipt(asset_receipt_path)
    provenance = load_json(provenance_path)
    preregistration_sha256 = str(provenance["preregistration_sha256"])
    sentinel = sentinel.resolve()
    for existing in registry_root.resolve().rglob("FRESH_FINAL_OPENED.json"):
        try:
            prior = load_json(existing)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"unreadable prior final-open sentinel blocks execution: {existing}") from None
        if _sentinel_preregistration_sha256(prior) == preregistration_sha256:
            raise RuntimeError(f"DIODE was already opened for this preregistration: {existing}")
    value = {
        "schema_version": SENTINEL_SCHEMA,
        "fresh_final_opened": True,
        "opened_utc": datetime.now(UTC).isoformat(),
        "execution_id": provenance["execution_id"],
        "git_commit": provenance["git_commit"],
        "preregistration_sha256": preregistration_sha256,
        "slurm": provenance["slurm"],
        "selector": file_identity(selector_path, schema=SELECTOR_SCHEMA),
        "survivor": selector["survivor"],
        "asset_receipt": file_identity(asset_receipt_path, schema=ASSET_SCHEMA),
        "archive": asset["archive"],
        "execution_provenance": provenance,
    }
    reject_secrets(value)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with sentinel.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return value


def assert_registry_clear(registry_root: Path, preregistration_sha256: str) -> None:
    root = registry_root.resolve()
    if not root.exists():
        return
    for existing in root.rglob("FRESH_FINAL_OPENED.json"):
        try:
            prior = load_json(existing)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"unreadable prior final-open sentinel blocks execution: {existing}") from None
        if _sentinel_preregistration_sha256(prior) == preregistration_sha256:
            raise RuntimeError(f"DIODE was already opened for this preregistration: {existing}")


def write_no_survivor(selector_path: Path, provenance_path: Path, output: Path) -> dict[str, Any]:
    selector = validate_selector(selector_path)
    if selector["final_authorized"] is not False:
        raise ValueError("no-survivor skip is valid only when final_authorized=false")
    provenance = load_json(provenance_path)
    result: dict[str, Any] = {
        "schema_version": FINAL_SCHEMA,
        "status": "no_survivor",
        "created_utc": datetime.now(UTC).isoformat(),
        "fresh_final_opened": False,
        "final_authorized": False,
        "survivor": None,
        "selector": file_identity(selector_path, schema=SELECTOR_SCHEMA),
        "scientific_gate": {"passed": False, "reason": "no_development_survivor"},
        "claim_boundary": "DIODE archive and targets were not opened.",
        "execution_provenance": provenance,
    }
    receipt = atomic_json(output, result)
    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    wandb = publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type="external-final",
        run_name=f"{execution_id}-external-final-skip-{job_id}",
        config={"execution_id": execution_id, "git_commit": provenance["git_commit"]},
        summary={"status": "no_survivor", "fresh_final_opened": False},
        artifact_name=f"phase2f-external-final-skip-{execution_id}",
        artifact_files=(receipt,),
    )
    result["wandb"] = wandb
    atomic_json(receipt, result)
    atomic_json(receipt.parent / "wandb_receipt.json", wandb)
    (receipt.parent / "SUCCESS").write_text("no_survivor\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decision = sub.add_parser("decision")
    decision.add_argument("--selector", type=Path, required=True)
    prepare = sub.add_parser("open")
    prepare.add_argument("--selector", type=Path, required=True)
    prepare.add_argument("--asset-receipt", type=Path, required=True)
    prepare.add_argument("--provenance", type=Path, required=True)
    prepare.add_argument("--sentinel", type=Path, required=True)
    prepare.add_argument("--registry-root", type=Path, required=True)
    skip = sub.add_parser("no-survivor")
    skip.add_argument("--selector", type=Path, required=True)
    skip.add_argument("--provenance", type=Path, required=True)
    skip.add_argument("--output", type=Path, required=True)
    registry = sub.add_parser("registry-clear")
    registry.add_argument("--registry-root", type=Path, required=True)
    registry.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "decision":
        selector = validate_selector(args.selector)
        print("authorized" if selector["final_authorized"] else "no_survivor")
    elif args.command == "open":
        value = create_sentinel(
            selector_path=args.selector,
            asset_receipt_path=args.asset_receipt,
            provenance_path=args.provenance,
            sentinel=args.sentinel,
            registry_root=args.registry_root,
        )
        print(json.dumps({"fresh_final_opened": True, "survivor": value["survivor"]}, sort_keys=True))
    elif args.command == "no-survivor":
        value = write_no_survivor(args.selector, args.provenance, args.output)
        print(json.dumps({"status": value["status"], "fresh_final_opened": False}, sort_keys=True))
    else:
        assert_registry_clear(args.registry_root, sha256_file(args.preregistration.resolve(strict=True)))
        print(json.dumps({"status": "clear"}, sort_keys=True))


if __name__ == "__main__":
    main()
