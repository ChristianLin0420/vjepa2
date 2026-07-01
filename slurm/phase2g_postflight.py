#!/usr/bin/env python3
"""Strict terminal audit for all 151 predecessors in formal Phase 2g-A."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jepa4d.evaluation.phase2g_data import require_canonical_record_sha256
from jepa4d.validation._content import sha256_value, write_content_addressed_json
from slurm.phase2g_contract import (
    SHA256_PATTERN,
    atomic_json,
    load_graph,
    load_json,
    reject_credentials,
    reject_diode_paths,
    scheduler_completed_many,
    sha256_file,
    validate_wandb,
)
from slurm.phase2g_stage_gate import publish_online_wandb


class WandbBackendNotReady(RuntimeError):
    """The backend has not made a just-finished immutable artifact visible yet."""


def _receipt_files(wandb_receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    files = wandb_receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("W&B receipt has no artifact file manifest")
    result: dict[str, dict[str, Any]] = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            raise ValueError("W&B artifact file identity must be an object")
        name_value = raw.get("name")
        if name_value is None and isinstance(raw.get("path"), str):
            name_value = Path(str(raw["path"])).name
        if (
            not isinstance(name_value, str)
            or not name_value
            or name_value.startswith("/")
            or ".." in Path(name_value).parts
        ):
            raise ValueError("W&B artifact file name is unsafe")
        normalized = {
            "name": name_value,
            "bytes": raw.get("bytes"),
            "sha256": raw.get("sha256"),
        }
        if (
            name_value in result
            or isinstance(normalized["bytes"], bool)
            or not isinstance(normalized["bytes"], int)
            or int(normalized["bytes"]) < 0
            or SHA256_PATTERN.fullmatch(str(normalized["sha256"])) is None
        ):
            raise ValueError("W&B artifact file identity is invalid or duplicated")
        result[name_value] = normalized
    return result


def _backend_identity_once(wandb_receipt: Mapping[str, Any], api: Any) -> dict[str, Any]:
    version = wandb_receipt.get("artifact_version")
    if not isinstance(version, str) or re.fullmatch(r"v[0-9]+", version) is None:
        raise ValueError("W&B artifact version must be immutable numeric vN")
    run_path = f"{wandb_receipt['entity']}/{wandb_receipt['project']}/{wandb_receipt['run_id']}"
    artifact_path = f"{wandb_receipt['entity']}/{wandb_receipt['project']}/{wandb_receipt['artifact_name']}:{version}"
    try:
        run = api.run(run_path)
    except Exception as error:
        raise WandbBackendNotReady("W&B run is not queryable") from error
    expected_run = {
        "entity": wandb_receipt["entity"],
        "project": wandb_receipt["project"],
        "id": wandb_receipt["run_id"],
        "name": wandb_receipt["run_name"],
        "group": wandb_receipt["group"],
        "job_type": wandb_receipt.get("job_type"),
        "url": wandb_receipt["run_url"],
    }
    observed_run = {key: getattr(run, key, None) for key in expected_run}
    if observed_run != expected_run:
        raise RuntimeError("queried W&B run identity differs from its local receipt")
    if getattr(run, "state", None) != "finished":
        raise WandbBackendNotReady("W&B run is not finished")
    try:
        artifact = api.artifact(artifact_path)
    except Exception as error:
        raise WandbBackendNotReady("W&B artifact is not queryable") from error
    if (
        getattr(artifact, "id", None) != wandb_receipt["artifact_id"]
        or getattr(artifact, "version", None) != version
        or getattr(artifact, "digest", None) != wandb_receipt["artifact_digest"]
    ):
        raise RuntimeError("queried W&B artifact identity differs from its local receipt")
    if getattr(artifact, "state", None) != "COMMITTED":
        raise WandbBackendNotReady("W&B artifact is not committed")
    expected_files = _receipt_files(wandb_receipt)
    manifest = getattr(artifact, "manifest", None)
    entries = getattr(manifest, "entries", None)
    if not isinstance(entries, Mapping) or set(map(str, entries)) != set(expected_files):
        raise RuntimeError("W&B backend manifest file set differs from its local receipt")
    for name, entry in entries.items():
        normalized_name = str(name)
        if (
            getattr(entry, "ref", None) is not None
            or getattr(entry, "size", None) != expected_files[normalized_name]["bytes"]
        ):
            raise RuntimeError(f"W&B backend manifest entry differs for {normalized_name}")
    with tempfile.TemporaryDirectory(prefix="jepa4d-phase2g-backend-") as temporary:
        fresh_root = Path(temporary).resolve(strict=True)
        try:
            downloaded = artifact.download(
                root=str(fresh_root),
                allow_missing_references=False,
                skip_cache=True,
                multipart=False,
            )
        except Exception as error:
            raise WandbBackendNotReady("W&B artifact is not downloadable") from error
        downloaded_root = Path(downloaded).resolve(strict=True)
        if downloaded_root != fresh_root or downloaded_root.is_symlink():
            raise RuntimeError("W&B artifact downloaded outside its fresh verification root")
        entries_on_disk = set(downloaded_root.rglob("*"))
        if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in entries_on_disk):
            raise RuntimeError("W&B artifact download contains a link or special file")
        files_on_disk = {
            path.relative_to(downloaded_root).as_posix(): path for path in entries_on_disk if path.is_file()
        }
        if set(files_on_disk) != set(expected_files):
            raise RuntimeError("downloaded W&B file set differs from its local receipt")
        artifact.verify(str(downloaded_root))
        verified = []
        for name in sorted(expected_files):
            path = files_on_disk[name]
            identity = {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if identity != expected_files[name]:
                raise RuntimeError(f"downloaded W&B content hash differs for {name}")
            verified.append(identity)
    if getattr(artifact, "digest", None) != wandb_receipt["artifact_digest"]:
        raise RuntimeError("W&B artifact digest changed during backend verification")
    return {
        "schema_version": "jepa4d-phase2g-wandb-backend-verification-v1",
        "status": "verified",
        "entity": wandb_receipt["entity"],
        "project": wandb_receipt["project"],
        "run_id": wandb_receipt["run_id"],
        "artifact_name": wandb_receipt["artifact_name"],
        "artifact_id": wandb_receipt["artifact_id"],
        "artifact_version": version,
        "artifact_digest": wandb_receipt["artifact_digest"],
        "files_sha256": sha256_value({"files": verified}),
        "files": verified,
    }


def verify_wandb_backend(wandb_receipt: Mapping[str, Any], *, api: Any | None = None) -> dict[str, Any]:
    """Independently download and hash every file in one immutable task artifact."""

    if api is not None:
        return _backend_identity_once(wandb_receipt, api)
    if os.environ.get("WANDB_MODE") != "online":
        raise RuntimeError("formal Phase 2g backend verification requires WANDB_MODE=online")
    import wandb

    delays = (1, 2, 4, 8, 16, 30, 30, 30, 30, 30, 30, 30)
    last_error: WandbBackendNotReady | None = None
    for attempt in range(len(delays) + 1):
        try:
            return _backend_identity_once(wandb_receipt, wandb.Api(timeout=30))
        except WandbBackendNotReady as error:
            last_error = error
            if attempt < len(delays):
                time.sleep(delays[attempt])
    raise RuntimeError("W&B backend evidence did not become visible within the bounded retry window") from last_error


def _manifest_digests(value: Any, location: str = "receipt") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            next_location = f"{location}.{key}"
            identity_location = next_location.casefold()
            if (
                any(token in identity_location for token in ("manifest", "membership"))
                and isinstance(child, str)
                and SHA256_PATTERN.fullmatch(child)
            ):
                found.append(child)
            found.extend(_manifest_digests(child, next_location))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            found.extend(_manifest_digests(child, f"{location}[{index}]"))
    return found


def validate_postflight(
    graph_path: Path,
    output_root: Path,
    *,
    backend_verifier: Any = verify_wandb_backend,
) -> dict[str, Any]:
    graph, graph_sha256 = load_graph(graph_path)
    jobs = graph["jobs"]
    predecessors = [label for label in jobs if label != "Z"]
    if len(predecessors) != 151:
        raise ValueError("formal Phase 2g postflight requires exactly 151 predecessors")
    if not scheduler_completed_many([str(jobs[label]["job_id"]) for label in predecessors]):
        raise ValueError("one or more formal Phase 2g predecessors are not scheduler COMPLETED 0:0")

    validated: dict[str, Any] = {}
    run_ids: set[str] = set()
    artifact_ids: set[str] = set()
    backend_evidence: dict[str, dict[str, Any]] = {}
    for label in predecessors:
        job = jobs[label]
        receipt_path = Path(str(job["expected_receipt"]))
        success_path = Path(str(job["expected_success"]))
        if not receipt_path.is_file() or receipt_path.is_symlink() or not success_path.is_file():
            raise ValueError(f"job {label} lacks its regular receipt/SUCCESS")
        receipt = load_json(receipt_path)
        if receipt.get("status") not in {"pass", "success"}:
            raise ValueError(f"job {label} did not complete successfully")
        validate_wandb(receipt)
        for item in receipt["wandb"]["files"]:
            if not isinstance(item, Mapping) or isinstance(item.get("path"), str):
                continue
            candidate = receipt_path.parent / str(item.get("name", ""))
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != item.get("bytes")
                or sha256_file(candidate) != item.get("sha256")
            ):
                raise ValueError(f"job {label} local W&B artifact source differs from its receipt")
        reject_credentials(receipt, label)
        reject_diode_paths(receipt, label)
        provenance = receipt.get("execution_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"job {label} lacks execution provenance")
        expected = {
            "execution_id": graph["execution_id"],
            "job_label": label,
            "git_commit": graph["git_commit"],
            "preregistration_sha256": graph["preregistration"]["sha256"],
            "preflight_sha256": graph["preflight"]["sha256"],
            "dependency_graph_sha256": graph_sha256,
            "external_final_authorized": False,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"job {label} provenance identity mismatch")
        parent_labels = [row.get("label") for row in provenance.get("parents", []) if isinstance(row, Mapping)]
        if parent_labels != job["parents"]:
            raise ValueError(f"job {label} provenance parent mapping mismatch")
        wandb = receipt["wandb"]
        run_id = str(wandb["run_id"])
        artifact_id = str(wandb["artifact_id"])
        if run_id in run_ids or artifact_id in artifact_ids:
            raise ValueError("formal Phase 2g W&B run/artifact identities must be unique per logical task")
        run_ids.add(run_id)
        artifact_ids.add(artifact_id)
        verified_backend = backend_verifier(wandb)
        if verified_backend.get("status") != "verified":
            raise ValueError(f"job {label} lacks verified W&B backend evidence")
        backend_evidence[label] = verified_backend
        validated[label] = {
            "job_id": job["job_id"],
            "status": receipt["status"],
            "receipt_sha256": __import__("hashlib").sha256(receipt_path.read_bytes()).hexdigest(),
            "wandb_run_id": run_id,
            "wandb_artifact_id": artifact_id,
            "wandb_backend_sha256": sha256_value(verified_backend),
        }

    cache = load_json(Path(str(jobs["C"]["expected_receipt"])))
    audit = load_json(Path(str(jobs["Q"]["expected_receipt"])))
    if not _manifest_digests(cache) or not _manifest_digests(audit):
        raise ValueError("cache and audit receipts must both bind the generated target-free membership manifest")
    membership = cache.get("membership")
    if not isinstance(membership, Mapping) or SHA256_PATTERN.fullmatch(str(membership.get("sha256", ""))) is None:
        raise ValueError("cache receipt lacks the target-free membership identity")
    membership_sha256 = str(membership["sha256"])
    stage_reports: dict[str, dict[str, Any]] = {}
    for label, report_name in (("C", "cache_report.json"), ("Q", "audit_report.json")):
        stage_receipt = cache if label == "C" else audit
        file_rows = stage_receipt["wandb"]["files"]
        report_rows = [
            row for row in file_rows if isinstance(row, Mapping) and Path(str(row.get("path", ""))).name == report_name
        ]
        if len(report_rows) != 1:
            raise ValueError(f"{label} W&B receipt lacks one {report_name}")
        report = load_json(Path(str(report_rows[0]["path"])))
        stage_reports[label] = report
        if report.get("membership_sha256") != membership_sha256:
            raise ValueError(f"{label} backend-bound report differs from the local membership identity")
    cache_source = cache.get("source_materialization")
    audit_source = audit.get("source_materialization")
    expected_archive_sha256 = "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c"
    if (
        not isinstance(cache_source, Mapping)
        or not isinstance(audit_source, Mapping)
        or cache_source.get("archive_sha256") != expected_archive_sha256
        or audit_source.get("archive_sha256") != expected_archive_sha256
        or cache_source.get("files_manifest_sha256") != audit_source.get("files_manifest_sha256")
        or cache_source.get("file_count") != 31_005
        or audit_source.get("file_count") != 31_005
        or cache_source.get("sha256") != audit_source.get("receipt_sha256")
        or audit_source.get("selected_source_identities_revalidated") != 12_288
    ):
        raise ValueError("cache/audit receipts do not preserve the archive-derived SUN materialization chain")
    for label in ("C", "Q"):
        report = stage_reports[label]
        if (
            report.get("source_archive_sha256") != expected_archive_sha256
            or report.get("materialization_manifest_sha256") != cache_source.get("files_manifest_sha256")
            or report.get("materialized_file_count") != 31_005
        ):
            raise ValueError(f"{label} W&B report lacks the archive-derived materialization identity")
    selector = load_json(Path(str(jobs["S"]["expected_receipt"])))
    if selector.get("selection_sha256_scope") != "complete-record-excluding-selection_sha256-and-post-upload-wandb":
        raise ValueError("Phase 2g selector has an unknown selection hash scope")
    require_canonical_record_sha256(
        selector,
        field="selection_sha256",
        excluded_fields=("selection_sha256", "wandb"),
    )
    if selector.get("external_final_authorized") is not False:
        raise ValueError("Phase 2g-A selector attempted to authorize an external final")
    for label in ("O", "G"):
        receipt = load_json(Path(str(jobs[label]["expected_receipt"])))
        if any(
            receipt.get(key) is not expected
            for key, expected in (
                ("archive_path_received", False),
                ("archive_touched", False),
                ("external_final_authorized", False),
            )
        ):
            raise ValueError(f"{label} did not preserve external-final opacity")
    resolved_output = output_root.resolve(strict=True)
    if resolved_output != Path(str(graph["output_root"])).resolve(strict=True):
        raise ValueError("postflight output root differs from the dependency graph")
    return {
        "schema_version": "jepa4d-phase2g-postflight-summary-v1",
        "status": "pass",
        "integrity_status": "pass",
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": graph["execution_id"],
        "git_commit": graph["git_commit"],
        "dependency_graph_sha256": graph_sha256,
        "expected_logical_jobs": 152,
        "validated_predecessors": len(validated),
        "unique_wandb_runs": len(run_ids),
        "unique_wandb_artifacts": len(artifact_ids),
        "formal_training_cells": len([label for label in validated if label.startswith("F-")]),
        "heldout_evaluation_cells": len([label for label in validated if label.startswith("V-")]),
        "tuning_cells": len([label for label in validated if label.startswith("H-")]),
        "membership_manifest_bound": True,
        "archive_path_received": False,
        "archive_touched": False,
        "external_final_authorized": False,
        "jobs": validated,
        "wandb_backend": backend_evidence,
        "wandb_backend_sha256": sha256_value(backend_evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = load_json(args.provenance)
    result = validate_postflight(args.graph, args.output_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary_path = atomic_json(output / "postflight_summary.json", result)
    receipt: dict[str, Any] = {
        **result,
        "schema_version": "jepa4d-phase2g-postflight-receipt-v1",
        "execution_provenance": provenance,
    }
    receipt["wandb"] = publish_online_wandb(
        provenance=provenance,
        job_type="postflight",
        artifact_files=(summary_path,),
        summary={
            "integrity_pass": True,
            "validated_predecessors": result["validated_predecessors"],
            "external_final_authorized": False,
        },
    )
    terminal_backend = verify_wandb_backend(receipt["wandb"])
    if terminal_backend.get("status") != "verified":
        raise RuntimeError("postflight terminal W&B artifact lacks backend verification")
    receipt["terminal_wandb_backend"] = terminal_backend
    receipt["terminal_wandb_backend_sha256"] = sha256_value(terminal_backend)
    atomic_json(output / "postflight_receipt.json", receipt)
    atomic_json(output / "wandb_receipt.json", receipt["wandb"])
    terminal = {
        "schema_version": "jepa4d-phase2g-terminal-v1",
        "status": "pass",
        "terminal_status": "postflight-pass",
        "execution_id": result["execution_id"],
        "git_commit": result["git_commit"],
        "dependency_graph_sha256": result["dependency_graph_sha256"],
        "postflight_receipt_sha256": sha256_file(output / "postflight_receipt.json"),
        "predecessor_wandb_backend_sha256": result["wandb_backend_sha256"],
        "terminal_wandb_backend_sha256": receipt["terminal_wandb_backend_sha256"],
        "verified_task_artifacts": 152,
        "archive_path_received": False,
        "archive_touched": False,
        "external_final_authorized": False,
    }
    terminal_address = write_content_addressed_json(terminal, output / "terminal", prefix="terminal")
    atomic_json(
        output / "terminal_receipt.json",
        {
            **terminal,
            "content_addressed_path": terminal_address.path.name,
            "content_sha256": terminal_address.sha256,
            "bytes": terminal_address.bytes,
        },
    )
    (output / "SUCCESS").write_text("pass\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "validated_predecessors": 151,
                "verified_task_artifacts": 152,
                "terminal_sha256": terminal_address.sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
