"""Recompute and enforce every authorization bound by a Phase 2b preflight."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from jepa4d.benchmarks.geometry.tum_rgbd import load_tum_indices, validate_archive
from scripts.run_phase2b_geometry_distillation import _dataset_fingerprint
from slurm.phase2b_gate import (
    asset_inventory,
    environment_fingerprint,
    repository_fingerprint,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-implementation", type=Path, required=True)
    parser.add_argument("--vggt-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolved(path: Path) -> str:
    return str(path.resolve(strict=True))


def _validate_test_receipt(path: Path, repository: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    _require(receipt.get("schema_version") == "jepa4d-phase2b-tests-v1", "unexpected test receipt schema")
    _require(receipt.get("status") == "pass", "test receipt does not pass")
    _require(bool(receipt.get("slurm_job_id")), "test receipt was not produced by a Slurm job")
    _require(receipt.get("repository") == repository, "repository differs from passing Slurm tests")
    _require(receipt.get("environment") == environment, "environment differs from passing Slurm tests")
    cuda = receipt.get("cuda_report", {})
    cuda_path = Path(str(cuda.get("path", "")))
    _require(cuda_path.is_file(), "test CUDA report is missing")
    _require(sha256(cuda_path) == cuda.get("sha256"), "test CUDA report content has changed")
    _require(cuda.get("summary", {}).get("status") == "pass", "test CUDA report does not pass")
    return receipt


def validate(args: argparse.Namespace) -> dict[str, Any]:
    _require(bool(os.getenv("SLURM_JOB_ID")), "formal Phase 2b authorization must run in Slurm")
    report = json.loads(args.preflight_report.read_text())
    _require(report.get("schema_version") == "jepa4d-phase2b-preflight-v3", "unexpected preflight schema")
    _require(report.get("status") == "pass", "preflight report does not pass")

    wandb_probe = report.get("wandb_probe", {})
    required_wandb = ("run_url", "run_id", "artifact_name", "artifact_version", "artifact_digest")
    _require(wandb_probe.get("mode") == "online", "preflight W&B probe was not online")
    _require(all(wandb_probe.get(key) for key in required_wandb), "preflight W&B artifact receipt is incomplete")
    _require(
        report.get("probe_smoke", {}).get("checkpoint_reload") == "pass",
        "preflight one-step probe checkpoint did not reload",
    )
    expected_chunks = [1, 8]
    _require(
        report.get("vjepa_smoke", {}).get("chunk_invariance", {}).get("compared_chunk_sizes") == expected_chunks,
        "preflight did not compare V-JEPA chunk sizes 1 and 8",
    )
    _require(
        report.get("vggt_smoke", {}).get("chunk_invariance", {}).get("compared_chunk_sizes") == expected_chunks,
        "preflight did not compare VGGT chunk sizes 1 and 8",
    )

    _require(torch.cuda.is_available(), "formal training allocation has no CUDA device")
    preflight_memory = int(report.get("gpu", {}).get("total_memory_bytes", 0))
    training_memory = int(torch.cuda.get_device_properties(0).total_memory)
    _require(preflight_memory > 0, "preflight did not record GPU memory")
    _require(
        training_memory >= int(0.95 * preflight_memory),
        f"training GPU memory {training_memory} is below passing preflight memory {preflight_memory}",
    )

    current_paths = {
        "dataset": _resolved(args.dataset_root),
        "archive": _resolved(args.archive),
        "manifest": _resolved(args.manifest),
        "vjepa_checkpoint": _resolved(args.vjepa_checkpoint),
        "vjepa_implementation": _resolved(args.vjepa_implementation),
        "vggt_checkpoint": _resolved(args.vggt_checkpoint),
    }
    recorded_paths = {
        "dataset": str(Path(report["dataset"]["root"]).resolve(strict=True)),
        "archive": str(Path(report["dataset"]["archive"]).resolve(strict=True)),
        "manifest": str(Path(report["dataset"]["manifest"]).resolve(strict=True)),
        "vjepa_checkpoint": str(Path(report["assets"]["vjepa_checkpoint"]["path"]).resolve(strict=True)),
        "vjepa_implementation": str(Path(report["assets"]["vjepa_implementation"]["path"]).resolve(strict=True)),
        "vggt_checkpoint": str(Path(report["assets"]["vggt_checkpoint"]["path"]).resolve(strict=True)),
    }
    _require(recorded_paths == current_paths, f"preflight paths differ from training paths: {recorded_paths}")

    repository = repository_fingerprint(args.repo_root)
    environment = environment_fingerprint()
    authorization = report.get("authorization", {})
    _require(authorization.get("repository") == repository, "repository changed after preflight")
    _require(authorization.get("environment") == environment, "Python environment changed after preflight")
    receipt = _validate_test_receipt(args.test_report, repository, environment)
    recorded_receipt = authorization.get("test_receipt", {})
    _require(_resolved(args.test_report) == recorded_receipt.get("path"), "test receipt path differs from preflight")
    _require(sha256(args.test_report) == recorded_receipt.get("sha256"), "test receipt changed after preflight")
    _require(receipt["slurm_job_id"] == recorded_receipt.get("slurm_job_id"), "test Slurm job identity changed")

    _require(report.get("assets", {}).get("hash_mode") == "full", "preflight did not use full asset hashes")
    current_assets = {
        "vjepa_checkpoint": asset_inventory(args.vjepa_checkpoint),
        "vjepa_implementation": asset_inventory(args.vjepa_implementation),
        "vggt_checkpoint": asset_inventory(args.vggt_checkpoint),
    }
    for name, current in current_assets.items():
        _require(report["assets"].get(name) == current, f"{name} content changed after preflight")

    manifest = validate_archive(args.archive, args.manifest)
    split_indices = {
        split: [int(value) for value in manifest[f"{split}_indices"]] for split in ("train", "validation", "test")
    }
    samples = {split: load_tum_indices(args.dataset_root, indices) for split, indices in split_indices.items()}
    counts = {name: len(values) for name, values in samples.items()}
    _require(counts == {"train": 64, "validation": 16, "test": 8}, f"unexpected split counts: {counts}")
    fingerprint = _dataset_fingerprint(args.dataset_root, samples, args.archive)
    _require(
        report.get("dataset", {}).get("extraction_fingerprint") == fingerprint,
        "dataset/archive content changed after preflight",
    )
    _require(report["dataset"].get("manifest_sha256") == sha256(args.manifest), "manifest content changed")

    return {
        "schema_version": "jepa4d-phase2b-authorization-v1",
        "status": "pass",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "preflight_report": str(args.preflight_report.resolve()),
        "preflight_sha256": sha256(args.preflight_report),
        "test_report": str(args.test_report.resolve()),
        "test_sha256": sha256(args.test_report),
        "test_slurm_job_id": receipt["slurm_job_id"],
        "repository_sha256": repository["sha256"],
        "environment_sha256": environment["sha256"],
        "asset_sha256": {name: value["sha256"] for name, value in current_assets.items()},
        "dataset_archive_sha256": fingerprint["archive"]["sha256"],
        "wandb_preflight_url": wandb_probe["run_url"],
        "wandb_preflight_artifact_digest": wandb_probe["artifact_digest"],
        "preflight_gpu_memory_bytes": preflight_memory,
        "training_gpu_memory_bytes": training_memory,
    }


def main() -> None:
    args = parse_args()
    result = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
