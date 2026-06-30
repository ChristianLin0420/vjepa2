#!/usr/bin/env python3
"""Strictly revalidate the complete 73-job Phase 2f DAG and every hash-bound receipt."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from jepa4d.evaluation.phase2f_metrics import publish_online_wandb, self_contained_html
from slurm.phase2f_contract import (
    WANDB_SCHEMA,
    atomic_json,
    load_graph,
    load_json,
    reject_secrets,
    scheduler_completed,
    sha256_file,
    validate_test_receipt,
    validate_wandb,
)
from slurm.phase2f_final_guard import FINAL_SCHEMA, SENTINEL_SCHEMA, validate_asset_receipt

SCHEMA = "jepa4d-phase2f-strict-postflight-v1"


Identity = tuple[Mapping[str, Any], bool]
VerifiedIdentity = tuple[str, str, int | None]


def _walk_identities(value: Any, *, wandb_snapshot: bool = False) -> list[Identity]:
    found: list[Identity] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found.append((value, wandb_snapshot))
        for key, child in value.items():
            # A receipt is uploaded before its returned W&B identity can be embedded
            # in the final local receipt.  Therefore W&B ``files`` describe the
            # immutable uploaded snapshot, not necessarily the subsequently finalized
            # local receipt.  Mark that role explicitly so a superseded snapshot can be
            # distinguished from a current local-file identity without dropping checks
            # for W&B-only reports, figures, and arrays.
            child_is_snapshot = wandb_snapshot or (value.get("schema_version") == WANDB_SCHEMA and key == "files")
            found.extend(_walk_identities(child, wandb_snapshot=child_is_snapshot))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.extend(_walk_identities(child, wandb_snapshot=wandb_snapshot))
    return found


def _verify_hash_identities(
    receipt: Mapping[str, Any], receipt_path: Path, verified: set[VerifiedIdentity] | None = None
) -> int:
    checked = 0
    seen: set[tuple[str, str]] = set()
    memo = set() if verified is None else verified
    identities = _walk_identities(receipt)
    current_paths = {Path(str(identity["path"])).resolve() for identity, is_snapshot in identities if not is_snapshot}
    for identity, is_snapshot in identities:
        path = Path(str(identity["path"]))
        key = (str(path), str(identity["sha256"]))
        if key in seen or path.resolve() == receipt_path.resolve():
            continue
        seen.add(key)
        if is_snapshot and path.resolve() in current_paths:
            # The same enclosing receipt carries the finalized current identity.  The
            # W&B entry remains useful external artifact provenance, but its pre-final
            # snapshot must not be compared with the now-finalized local receipt bytes.
            continue
        resolved = path.resolve(strict=True)
        expected_bytes = identity.get("bytes")
        memo_key: VerifiedIdentity = (
            str(resolved),
            str(identity["sha256"]),
            expected_bytes if isinstance(expected_bytes, int) else None,
        )
        if memo_key not in memo:
            if not resolved.is_file() or sha256_file(resolved) != identity["sha256"]:
                raise ValueError(f"postflight file hash mismatch: {path}")
            if "bytes" in identity and resolved.stat().st_size != identity["bytes"]:
                raise ValueError(f"postflight file size mismatch: {path}")
            memo.add(memo_key)
        checked += 1
    return checked


def validate_postflight_inputs(graph_path: Path, output_root: Path) -> dict[str, Any]:
    graph, graph_sha256 = load_graph(graph_path)
    if len(graph["jobs"]) != 73:
        raise ValueError("strict postflight expected exactly 73 jobs")
    test_path = Path(str(graph["test_receipt"]))
    test_receipt = validate_test_receipt(test_path, graph, graph_sha256)
    test_hash = sha256_file(test_path)
    validated: dict[str, Any] = {}
    verified_identities: set[VerifiedIdentity] = set()
    hashes_checked = _verify_hash_identities(test_receipt, test_path, verified_identities)
    for label, job in graph["jobs"].items():
        if label == "Z":
            continue
        receipt_path = Path(str(job["expected_receipt"]))
        if not scheduler_completed(str(job["job_id"])):
            raise ValueError(f"job {label} is not scheduler COMPLETED 0:0")
        if not receipt_path.is_file() or not Path(str(job["expected_success"])).is_file():
            raise ValueError(f"job {label} lacks expected receipt/SUCCESS")
        receipt = load_json(receipt_path)
        validate_wandb(receipt)
        if label != "T":
            provenance = receipt.get("execution_provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"job {label} lacks embedded execution_provenance")
            expected = {
                "execution_id": graph["execution_id"],
                "git_commit": graph["git_commit"],
                "preregistration_sha256": graph["preregistration"]["sha256"],
                "test_receipt_sha256": test_hash,
                "dependency_graph_sha256": graph_sha256,
                "job_label": label,
            }
            if any(provenance.get(key) != value for key, value in expected.items()):
                raise ValueError(f"job {label} execution provenance mismatch")
            parent_labels = {item.get("label") for item in provenance.get("parents", []) if isinstance(item, Mapping)}
            if parent_labels != set(job["parents"]):
                raise ValueError(f"job {label} parent provenance mismatch")
        hashes_checked += _verify_hash_identities(receipt, receipt_path, verified_identities)
        validated[label] = {
            "job_id": job["job_id"],
            "status": receipt.get("status"),
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": sha256_file(receipt_path),
            "wandb_run_id": receipt["wandb"]["run_id"],
            "wandb_artifact_id": receipt["wandb"]["artifact_id"],
        }
    if len(validated) != 72 or set(validated) != set(graph["jobs"]) - {"Z"}:
        raise ValueError("strict postflight did not validate exactly 72 completed predecessors")
    final_job = graph["jobs"]["E"]
    final_path = Path(str(final_job["expected_receipt"]))
    final = load_json(final_path)
    if final.get("schema_version") != FINAL_SCHEMA:
        raise ValueError("external final receipt schema mismatch")
    sentinel = output_root.resolve() / "final" / "FRESH_FINAL_OPENED.json"
    if final.get("status") in {"no_survivor", "skipped_no_survivor"}:
        if final.get("fresh_final_opened") is not False or sentinel.exists():
            raise ValueError("no-survivor final must leave DIODE unopened")
    else:
        if final.get("fresh_final_opened") is not True or not sentinel.is_file():
            raise ValueError("executed external final lacks the immutable open sentinel")
        opened = load_json(sentinel)
        if opened.get("schema_version") != SENTINEL_SCHEMA or opened.get("execution_id") != graph["execution_id"]:
            raise ValueError("fresh-final sentinel identity mismatch")
    asset_path = Path(str(graph["jobs"]["A"]["expected_receipt"]))
    validate_asset_receipt(asset_path)
    scientific_value = final.get("scientific_gate", False)
    scientific = (
        dict(scientific_value)
        if isinstance(scientific_value, Mapping)
        else {"passed": bool(scientific_value), "reason": final.get("outcome", final.get("status"))}
    )
    return {
        "schema_version": SCHEMA,
        "status": "pass",
        "integrity_status": "pass",
        "scientific_gate": scientific,
        "created_utc": datetime.now(UTC).isoformat(),
        "execution_id": graph["execution_id"],
        "git_commit": graph["git_commit"],
        "dependency_graph_sha256": graph_sha256,
        "expected_job_count": 73,
        "validated_predecessor_count": 72,
        "file_hash_references_checked": hashes_checked,
        "jobs": validated,
        "final_status": final.get("status"),
        "fresh_final_opened": final.get("fresh_final_opened"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_postflight_inputs(args.graph, args.output_root)
    provenance = load_json(args.provenance)
    result["execution_provenance"] = provenance
    reject_secrets(result)
    summary_path = args.output.with_name("postflight_summary.json")
    atomic_json(summary_path, result)
    figure = args.output.with_name("postflight_gate.png")
    image = Image.new("RGB", (1000, 420), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((35, 25), "Phase 2f strict postflight", fill="#17202e")
    cards = (
        ("Integrity", "PASS", "#238636"),
        (
            "Scientific gate",
            "PASS" if bool(result["scientific_gate"].get("passed", False)) else "NOT PROMOTED",
            "#238636" if bool(result["scientific_gate"].get("passed", False)) else "#cf4a4a",
        ),
        ("Predecessors", f"{result['validated_predecessor_count']}/72", "#1f6feb"),
        ("Final targets", "OPENED" if result["fresh_final_opened"] else "SEALED", "#8957e5"),
    )
    for index, (title, value, color) in enumerate(cards):
        left = 35 + (index % 2) * 480
        top = 85 + (index // 2) * 145
        draw.rounded_rectangle((left, top, left + 445, top + 110), radius=12, fill=color)
        draw.text((left + 20, top + 18), title, fill="white")
        draw.text((left + 20, top + 58), value, fill="white")
    image.save(figure)
    report = args.output.with_name("report.html")
    report.write_text(
        self_contained_html(
            "Phase 2f strict postflight",
            {
                "integrity status": result["integrity_status"],
                "scientific gate": bool(result["scientific_gate"].get("passed", False)),
                "validated predecessors": result["validated_predecessor_count"],
                "file hash references": result["file_hash_references_checked"],
                "final status": result["final_status"],
                "fresh final opened": result["fresh_final_opened"],
            },
            images=(("Integrity and scientific decision", figure),),
            claim_boundary="Integrity completion is separate from scientific promotion.",
        ),
        encoding="utf-8",
    )
    execution_id = str(provenance["execution_id"])
    job_id = str(provenance["slurm"]["job_id"])
    wandb = publish_online_wandb(
        entity=os.environ.get("JEPA4D_WANDB_ENTITY", "crlc112358"),
        project=os.environ.get("JEPA4D_WANDB_PROJECT", "jepa4d-worldmodel"),
        group=f"phase2f-{execution_id}",
        job_type="postflight",
        run_name=f"{execution_id}-postflight-{job_id}",
        config={"execution_id": execution_id, "git_commit": provenance["git_commit"]},
        summary={
            "integrity_status": "pass",
            "scientific_gate_passed": bool(result["scientific_gate"].get("passed", False)),
            "fresh_final_opened": result["fresh_final_opened"],
        },
        artifact_name=f"phase2f-postflight-{execution_id}",
        artifact_files=(summary_path, figure, report, args.graph),
    )
    result["wandb"] = wandb
    receipt = atomic_json(args.output, result)
    atomic_json(receipt.parent / "wandb_receipt.json", wandb)
    (receipt.parent / "SUCCESS").write_text("pass\n", encoding="utf-8")
    print(json.dumps({"integrity_status": "pass", "scientific_gate": result["scientific_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
