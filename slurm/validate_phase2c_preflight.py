"""Recompute and enforce every authorization bound by Phase 2c preflight."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from slurm.phase2b_gate import (  # noqa: E402
    asset_inventory,
    environment_fingerprint,
    repository_fingerprint,
    sha256,
)
from slurm.phase2c_gate import (  # noqa: E402
    atomic_json,
    bundle_identity,
    protocol_contract,
    validated_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-parent", type=Path, required=True)
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


def _validate_fusion_artifacts(fusion: dict[str, Any], expected_sample_ids: list[str] | None = None) -> None:
    for label in ("normalization", "checkpoint", "report"):
        path = Path(str(fusion.get(label, "")))
        _require(path.is_file(), f"fusion smoke {label} artifact is missing")
        _require(
            sha256(path) == fusion.get(f"{label}_sha256"),
            f"fusion smoke {label} artifact content changed",
        )

    profiles = fusion.get("end_to_end_profile_smoke")
    expected_variants = {"vjepa_final": [], "vjepa_learned_fusion": [2, 5, 8]}
    _require(
        isinstance(profiles, dict) and set(profiles) == set(expected_variants),
        "fusion smoke end-to-end profile coverage is incomplete",
    )
    profile_map = cast(dict[str, Any], profiles)
    common_sample_ids: list[str] | None = None
    for variant, capture_layers in expected_variants.items():
        profile = profile_map[variant]
        _require(isinstance(profile, dict), f"fusion smoke {variant} profile is not an object")
        _require(
            profile.get("profile") == "co-resident-batch1-encoder-normalization-fusion-probe-v1",
            f"fusion smoke {variant} profile protocol changed",
        )
        _require(
            profile.get("input_boundary") == "preloaded RGBInputBatch before device transfer and model preprocessing",
            f"fusion smoke {variant} input boundary changed",
        )
        _require(
            profile.get("capture_layers") == capture_layers,
            f"fusion smoke {variant} capture layers are invalid",
        )
        _require(
            profile.get("warmup_iterations") == 1
            and profile.get("measured_iterations_per_repetition") == 2
            and profile.get("repetitions") == 1,
            f"fusion smoke {variant} iteration contract changed",
        )
        repetitions = profile.get("repetition_ms_per_frame")
        _require(
            isinstance(repetitions, list)
            and len(repetitions) == 1
            and isinstance(repetitions[0], (int, float))
            and not isinstance(repetitions[0], bool)
            and math.isfinite(float(repetitions[0]))
            and float(repetitions[0]) > 0,
            f"fusion smoke {variant} repetition timing is invalid",
        )
        median = profile.get("median_ms_per_frame")
        _require(
            isinstance(median, (int, float))
            and not isinstance(median, bool)
            and math.isclose(float(median), float(repetitions[0]), rel_tol=0.0, abs_tol=1e-12),
            f"fusion smoke {variant} median timing is invalid",
        )
        peak_memory = profile.get("peak_end_to_end_memory_gb")
        _require(
            isinstance(peak_memory, (int, float))
            and not isinstance(peak_memory, bool)
            and math.isfinite(float(peak_memory))
            and float(peak_memory) > 0,
            f"fusion smoke {variant} peak memory is invalid",
        )
        sample_ids = profile.get("sample_ids")
        _require(
            isinstance(sample_ids, list)
            and len(sample_ids) == 8
            and all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids)
            and len(set(sample_ids)) == 8,
            f"fusion smoke {variant} sample IDs are invalid",
        )
        if common_sample_ids is None:
            common_sample_ids = sample_ids
        else:
            _require(sample_ids == common_sample_ids, "fusion smoke profile sample IDs differ")
    if expected_sample_ids is not None:
        _require(
            isinstance(expected_sample_ids, list)
            and len(expected_sample_ids) == 8
            and all(isinstance(sample_id, str) and sample_id for sample_id in expected_sample_ids)
            and len(set(expected_sample_ids)) == 8,
            "preflight V-JEPA smoke sample IDs are invalid",
        )
        _require(common_sample_ids == expected_sample_ids, "fusion profile sample IDs differ from V-JEPA smoke")


def _validate_test_receipt(
    path: Path,
    repository: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    _require(receipt.get("schema_version") == "jepa4d-phase2c-tests-v1", "unexpected test receipt schema")
    _require(receipt.get("status") == "pass", "test receipt does not pass")
    _require(bool(receipt.get("slurm_job_id")), "test receipt was not produced in Slurm")
    _require(receipt.get("protocol") == protocol_contract(), "test receipt protocol changed")
    _require(receipt.get("repository") == repository, "repository differs from passing Phase 2c tests")
    _require(receipt.get("environment") == environment, "environment differs from passing Phase 2c tests")
    cuda = receipt.get("cuda_report", {})
    cuda_path = Path(str(cuda.get("path", "")))
    _require(cuda_path.is_file(), "test CUDA report is missing")
    _require(sha256(cuda_path) == cuda.get("sha256"), "test CUDA report content changed")
    _require(cuda.get("summary", {}).get("status") == "pass", "test CUDA report does not pass")
    return receipt


def validate(args: argparse.Namespace) -> dict[str, Any]:
    _require(bool(os.getenv("SLURM_JOB_ID")), "formal Phase 2c authorization must run in Slurm")
    report = json.loads(args.preflight_report.read_text())
    _require(report.get("schema_version") == "jepa4d-phase2c-preflight-v1", "unexpected preflight schema")
    _require(report.get("status") == "pass", "preflight report does not pass")
    _require(report.get("protocol") == protocol_contract(), "preflight protocol differs from formal contract")

    wandb_probe = report.get("wandb_probe", {})
    required_wandb = ("run_url", "run_id", "artifact_name", "artifact_version", "artifact_digest")
    _require(wandb_probe.get("mode") == "online", "preflight W&B probe was not online")
    _require(all(wandb_probe.get(key) for key in required_wandb), "preflight W&B receipt is incomplete")
    fusion = report.get("fusion_smoke", {})
    _require(fusion.get("initialization") == "exact_final_layer", "fusion did not initialize as final layer")
    _require(fusion.get("checkpoint_reload") == "pass", "fusion checkpoint did not reload")
    _require(float(fusion.get("gate_gradient_norm", 0.0)) > 0, "fusion gates did not receive a gradient")
    vjepa_smoke = report.get("vjepa_smoke", {})
    _validate_fusion_artifacts(fusion, vjepa_smoke.get("sample_ids"))
    _require(
        report.get("vggt_smoke", {}).get("sample_ids") == vjepa_smoke.get("sample_ids"),
        "preflight V-JEPA and VGGT smoke sample IDs differ",
    )
    _require(
        vjepa_smoke.get("chunk_invariance", {}).get("compared_chunk_sizes") == [1, 8],
        "preflight did not compare V-JEPA chunk sizes 1 and 8",
    )
    _require(
        report.get("vggt_smoke", {}).get("chunk_invariance", {}).get("compared_chunk_sizes") == [1, 8],
        "preflight did not compare VGGT chunk sizes 1 and 8",
    )
    _require(
        len(set(vjepa_smoke.get("sequence_ids", []))) == 2,
        "preflight smoke batch did not span both training sequences",
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
        "dataset_parent": _resolved(args.dataset_parent),
        "manifest": _resolved(args.manifest),
        "vjepa_checkpoint": _resolved(args.vjepa_checkpoint),
        "vjepa_implementation": _resolved(args.vjepa_implementation),
        "vggt_checkpoint": _resolved(args.vggt_checkpoint),
    }
    recorded_paths = {
        "dataset_parent": str(Path(report["dataset"]["dataset_parent"]).resolve(strict=True)),
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
    _require(_resolved(args.test_report) == recorded_receipt.get("path"), "test receipt path differs")
    _require(sha256(args.test_report) == recorded_receipt.get("sha256"), "test receipt changed after preflight")
    _require(receipt["slurm_job_id"] == recorded_receipt.get("slurm_job_id"), "test Slurm job changed")

    _require(report.get("assets", {}).get("hash_mode") == "full", "preflight did not use full asset hashes")
    current_assets = {
        "vjepa_checkpoint": asset_inventory(args.vjepa_checkpoint),
        "vjepa_implementation": asset_inventory(args.vjepa_implementation),
        "vggt_checkpoint": asset_inventory(args.vggt_checkpoint),
    }
    for name, current in current_assets.items():
        _require(report["assets"].get(name) == current, f"{name} content changed after preflight")

    bundle = validated_bundle(args.dataset_parent, args.manifest)
    current_bundle = bundle_identity(bundle)
    recorded_bundle = {key: report["dataset"].get(key) for key in current_bundle}
    _require(recorded_bundle == current_bundle, "bundle manifest/archive/extracted content changed after preflight")

    return {
        "schema_version": "jepa4d-phase2c-authorization-v1",
        "status": "pass",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "protocol_sha256": protocol_contract()["sha256"],
        "preflight_report": str(args.preflight_report.resolve(strict=True)),
        "preflight_sha256": sha256(args.preflight_report),
        "test_report": str(args.test_report.resolve(strict=True)),
        "test_sha256": sha256(args.test_report),
        "test_slurm_job_id": receipt["slurm_job_id"],
        "repository_sha256": repository["sha256"],
        "environment_sha256": environment["sha256"],
        "asset_sha256": {name: value["sha256"] for name, value in current_assets.items()},
        "bundle_sha256": current_bundle["sha256"],
        "split_hash": bundle.split_hash,
        "wandb_preflight_url": wandb_probe["run_url"],
        "wandb_preflight_artifact_digest": wandb_probe["artifact_digest"],
        "preflight_gpu_memory_bytes": preflight_memory,
        "training_gpu_memory_bytes": training_memory,
    }


def main() -> None:
    args = parse_args()
    result = validate(args)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
