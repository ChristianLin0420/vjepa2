#!/usr/bin/env python3
"""Metadata-only opacity and external-seal gates for formal Phase 2g-A."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2g_data import require_canonical_record_sha256
from jepa4d.training.phase2g_runtime import hardened_wandb_settings
from slurm.phase2g_contract import (
    WANDB_SCHEMA,
    atomic_json,
    file_identity,
    load_json,
    reject_credentials,
    reject_diode_paths,
)


def publish_online_wandb(
    *,
    provenance: dict[str, Any],
    job_type: str,
    artifact_files: tuple[Path, ...],
    summary: dict[str, Any],
) -> dict[str, Any]:
    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("formal Phase 2g requires WANDB_MODE=online")
    reject_credentials(summary)
    reject_diode_paths(summary)
    import wandb

    execution_id = str(provenance["execution_id"])
    logical_label = str(provenance["job_label"])
    logical_id = str(provenance["slurm"]["job_id"])
    entity = os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358")
    project = os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel")
    group = f"phase2g-quality-{execution_id}"
    run_name = f"{execution_id}-{logical_label.lower()}-{logical_id.replace('_', '-')}"
    artifact_name = f"phase2g-{execution_id}-{logical_label.lower()}-{logical_id.replace('_', '-')}"
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        job_type=job_type,
        name=run_name,
        config={
            "execution_id": execution_id,
            "git_commit": provenance["git_commit"],
            "job_label": logical_label,
            "dependency_graph_sha256": provenance["dependency_graph_sha256"],
        },
        mode="online",
        reinit=True,
        settings=hardened_wandb_settings(wandb),
    )
    if run is None or bool(getattr(run, "offline", True)):
        raise RuntimeError("formal Phase 2g W&B run did not initialize online")
    artifact = wandb.Artifact(artifact_name, type=f"phase2g-{job_type}")
    files = []
    for path in artifact_files:
        resolved = path.resolve(strict=True)
        artifact.add_file(str(resolved), name=resolved.name)
        identity = file_identity(resolved)
        files.append({"name": resolved.name, **identity})
    for key, value in summary.items():
        run.summary[key] = value
    run.summary["status"] = "success"
    logged = run.log_artifact(artifact)
    logged.wait()
    receipt = {
        "schema_version": WANDB_SCHEMA,
        "mode": "online",
        "status": "success",
        "entity": str(run.entity),
        "project": str(run.project),
        "group": group,
        "job_type": job_type,
        "run_name": run_name,
        "run_id": str(run.id),
        "run_url": str(run.url),
        "artifact_name": artifact_name,
        "artifact_id": str(logged.id),
        "artifact_version": str(logged.version),
        "artifact_digest": str(logged.digest),
        "files": files,
    }
    run.finish(exit_code=0)
    if any(not receipt[key] for key in ("run_id", "run_url", "artifact_id", "artifact_version", "artifact_digest")):
        raise RuntimeError("formal Phase 2g W&B returned incomplete identities")
    return receipt


def _reject_external_environment() -> None:
    bad = []
    for key, value in os.environ.items():
        if "DIODE" in key.upper() or re.search(r"(?i)(?:^|[/\\])diode(?:[/\\]|$)", value):
            bad.append(key)
    if bad:
        raise ValueError(f"formal Phase 2g allocation received forbidden external-final environment: {sorted(bad)}")


def run_opacity(provenance_path: Path, output: Path) -> dict[str, Any]:
    _reject_external_environment()
    provenance = load_json(provenance_path)
    summary = {
        "schema_version": "jepa4d-phase2g-opacity-summary-v1",
        "status": "pass",
        "archive_path_received": False,
        "archive_touched": False,
        "external_final_authorized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    summary_path = atomic_json(output / "opacity_summary.json", summary)
    receipt: dict[str, Any] = {
        "schema_version": "jepa4d-phase2g-opacity-receipt-v1",
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        **{key: summary[key] for key in ("archive_path_received", "archive_touched", "external_final_authorized")},
        "execution_provenance": provenance,
    }
    receipt["wandb"] = publish_online_wandb(
        provenance=provenance,
        job_type="opacity-audit",
        artifact_files=(summary_path,),
        summary={"archive_path_received": False, "archive_touched": False, "external_final_authorized": False},
    )
    atomic_json(output / "opacity_receipt.json", receipt)
    atomic_json(output / "wandb_receipt.json", receipt["wandb"])
    (output / "SUCCESS").write_text("pass\n", encoding="utf-8")
    return receipt


def run_seal(selector_path: Path, provenance_path: Path, output: Path) -> dict[str, Any]:
    _reject_external_environment()
    selector = load_json(selector_path)
    if selector.get("status") not in {"pass", "success"}:
        raise ValueError("Phase 2g selector did not complete")
    if selector.get("selection_sha256_scope") != "complete-record-excluding-selection_sha256-and-post-upload-wandb":
        raise ValueError("Phase 2g selector has an unknown selection hash scope")
    require_canonical_record_sha256(
        selector,
        field="selection_sha256",
        excluded_fields=("selection_sha256", "wandb"),
    )
    if selector.get("external_final_authorized") is not False:
        raise ValueError("Phase 2g-A selector must always keep external final unauthorized")
    provenance = load_json(provenance_path)
    summary = {
        "schema_version": "jepa4d-phase2g-external-seal-summary-v1",
        "status": "pass",
        "selector_sha256": file_identity(selector_path)["sha256"],
        "archive_path_received": False,
        "archive_touched": False,
        "external_final_authorized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    summary_path = atomic_json(output / "seal_summary.json", summary)
    receipt: dict[str, Any] = {
        "schema_version": "jepa4d-phase2g-external-seal-receipt-v1",
        "status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "selector_sha256": summary["selector_sha256"],
        "archive_path_received": False,
        "archive_touched": False,
        "external_final_authorized": False,
        "execution_provenance": provenance,
    }
    receipt["wandb"] = publish_online_wandb(
        provenance=provenance,
        job_type="external-seal",
        artifact_files=(summary_path,),
        summary={"archive_path_received": False, "archive_touched": False, "external_final_authorized": False},
    )
    atomic_json(output / "seal_receipt.json", receipt)
    atomic_json(output / "wandb_receipt.json", receipt["wandb"])
    (output / "SUCCESS").write_text("pass\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("opacity", "seal"))
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "opacity":
        if args.selector is not None:
            raise ValueError("opacity mode does not receive a selector")
        result = run_opacity(args.provenance, args.output)
    else:
        if args.selector is None:
            raise ValueError("seal mode requires --selector")
        result = run_seal(args.selector, args.provenance, args.output)
    print(json.dumps({"status": result["status"], "mode": args.mode}, sort_keys=True))


if __name__ == "__main__":
    main()
