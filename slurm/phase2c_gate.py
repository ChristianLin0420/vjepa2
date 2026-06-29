"""Shared immutable contract and identities for the Phase 2c Slurm chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jepa4d.benchmarks.geometry.tum_rgbd_bundle import TUMCrossSequenceBundle, load_cross_sequence_bundle
from slurm.phase2b_gate import canonical_sha256

SCHEMA_VERSION = "jepa4d-phase2c-contract-v1"
SPLIT_COUNTS = {"train": 128, "validation": 64, "test": 128}
SEQUENCE_SPLITS = {"train": 2, "validation": 1, "test": 2}
VARIANT_SEEDS: dict[str, list[int | None]] = {
    "vggt_teacher": [None],
    "rgb_probe": [0, 1, 2],
    "vjepa_final": [0, 1, 2],
    "vjepa_multilayer": [0, 1, 2],
    "vjepa_learned_fusion": [0, 1, 2],
}
EPOCHS = 60


def protocol_contract() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence_splits": SEQUENCE_SPLITS,
        "split_counts": SPLIT_COUNTS,
        "epochs": EPOCHS,
        "variant_seeds": VARIANT_SEEDS,
        "result_rows": 13,
        "probe_checkpoints": 12,
        "normalization_artifacts": [
            "rgb_probe-normalization.pt",
            "vjepa_final-normalization.pt",
            "vjepa_multilayer-normalization.pt",
            "vjepa_learned_fusion-normalization.pt",
        ],
        "wandb_mode": "online",
        "primary_metric": "equal-weight test-sequence macro metric_abs_rel",
    }
    return {**identity, "sha256": canonical_sha256(identity)}


def validated_bundle(dataset_parent: Path, manifest: Path) -> TUMCrossSequenceBundle:
    bundle = load_cross_sequence_bundle(dataset_parent.resolve(strict=True), manifest.resolve(strict=True))
    counts = {name: len(values) for name, values in bundle.splits.items()}
    if counts != SPLIT_COUNTS:
        raise RuntimeError(f"Phase 2c split counts differ from the formal contract: {counts}")
    sequence_counts = {
        split: sum(selection.split == split for selection in bundle.selections) for split in SEQUENCE_SPLITS
    }
    if sequence_counts != SEQUENCE_SPLITS:
        raise RuntimeError(f"Phase 2c sequence roles differ from the formal contract: {sequence_counts}")
    return bundle


def bundle_identity(bundle: TUMCrossSequenceBundle) -> dict[str, Any]:
    identity = {
        "dataset_parent": str(bundle.selections[0].root.parent.resolve(strict=True)),
        "manifest": str(bundle.manifest_path.resolve(strict=True)),
        "split_hash": bundle.split_hash,
        "fingerprint": bundle.fingerprint,
    }
    return {**identity, "sha256": canonical_sha256(identity)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
