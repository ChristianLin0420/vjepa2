import json
from pathlib import Path

import pytest

from jepa4d.evaluation.phase2c_source import sha256_file, validate_phase2c_source


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _formal_source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "formal"
    root.mkdir()
    dataset_manifest = tmp_path / "dataset.yaml"
    dataset_manifest.write_text("protocol: frozen\n")
    checkpoint = tmp_path / "vjepa"
    implementation = tmp_path / "implementation"
    checkpoint.mkdir()
    implementation.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"frozen weights")
    (implementation / "model.py").write_text("# frozen implementation\n")
    split_hash = "a" * 64
    payloads = {
        "asset_manifest.json": {
            "vjepa_checkpoint": {
                "files": [
                    {
                        "path": "weights.bin",
                        "bytes": (checkpoint / "weights.bin").stat().st_size,
                        "sha256": sha256_file(checkpoint / "weights.bin"),
                    }
                ]
            },
            "vjepa_implementation": {
                "files": [
                    {
                        "path": "model.py",
                        "bytes": (implementation / "model.py").stat().st_size,
                        "sha256": sha256_file(implementation / "model.py"),
                    }
                ]
            },
        },
        "comparison.json": {
            "schema_version": "jepa4d-phase2c-cross-sequence-comparison-v1",
            "split_hash": split_hash,
            "failures": [],
        },
        "completion_gate.json": {
            "status": "success",
            "seed_failures": 0,
            "probe_checkpoints": 12,
            "result_rows": 13,
        },
        "dataset_fingerprint.json": {
            "manifest": {"sha256": sha256_file(dataset_manifest)},
        },
        "environment.json": {"git_commit": "b" * 40, "git_status": ""},
        "promotion_gate.json": {
            "schema_version": "jepa4d-phase2c-promotion-v1",
            "decision": "retain_final_layer",
            "promoted": False,
        },
        "resolved_config.json": {
            "protocol": "phase2c-cross-sequence-v1",
            "seeds": [0, 1, 2],
            "epochs": 60,
            "split_hash": split_hash,
            "split_counts": {"train": 128, "validation": 64, "test": 128},
        },
        "wandb_artifact_receipt.json": {
            "schema_version": "jepa4d-phase2c-wandb-artifact-v1",
            "status": "success",
            "mode": "online",
            "run_id": "run123",
            "run_path": "entity/project/run123",
            "run_url": "https://wandb.invalid/run123",
            "artifact_name": "formal:v0",
            "artifact_version": "v0",
            "artifact_digest": "c" * 32,
        },
    }
    for name, value in payloads.items():
        _write(root / name, value)
    artifact_manifest = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)} for name in payloads
    }
    _write(root / "artifact_manifest.json", artifact_manifest)
    return root, dataset_manifest, checkpoint, implementation


def test_validate_phase2c_source_binds_formal_output_and_assets(tmp_path: Path) -> None:
    root, manifest, checkpoint, implementation = _formal_source(tmp_path)
    result = validate_phase2c_source(
        root,
        dataset_manifest=manifest,
        vjepa_checkpoint=checkpoint,
        vjepa_implementation=implementation,
    )
    assert result["schema_version"] == "jepa4d-phase2c-source-identity-v1"
    assert result["phase2c_git_commit"] == "b" * 40
    assert result["wandb"]["artifact_digest"] == "c" * 32


def test_validate_phase2c_source_rejects_asset_tampering(tmp_path: Path) -> None:
    root, manifest, checkpoint, implementation = _formal_source(tmp_path)
    (checkpoint / "weights.bin").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="byte count changed|SHA-256 changed"):
        validate_phase2c_source(
            root,
            dataset_manifest=manifest,
            vjepa_checkpoint=checkpoint,
            vjepa_implementation=implementation,
        )
