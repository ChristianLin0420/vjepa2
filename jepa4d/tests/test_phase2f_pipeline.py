from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from jepa4d.evaluation.phase2f_metrics import (
    atomic_json,
    evaluate_depth_predictions,
    fit_variance_multiplier,
)
from scripts.aggregate_phase2f_qualification import (
    LATENCY_REPLICA_SCHEMA,
    aggregate_latency_receipts,
    aggregate_pilot_receipts,
)
from scripts.evaluate_phase2f_final import (
    FINAL_SCHEMA,
    SELECTOR_SCHEMA,
    _preprocess_sample,
    create_open_sentinel,
)
from scripts.select_phase2f_survivor import select_survivor


def _provenance(job_id: str) -> dict[str, Any]:
    return {
        "execution_id": "phase2f-test-execution",
        "git_commit": "a" * 40,
        "preregistration_sha256": "b" * 64,
        "test_receipt_sha256": "c" * 64,
        "dependency_graph_sha256": "d" * 64,
        "slurm": {"job_id": job_id, "job_name": f"test-{job_id}"},
    }


def _write(path: Path, value: dict[str, Any]) -> Path:
    return atomic_json(path, value)


def test_metrics_calibration_and_group_macro_are_finite() -> None:
    target = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    prediction = target.log()
    log_variance = torch.zeros_like(prediction)
    calibration = fit_variance_multiplier(prediction, log_variance, target)
    assert calibration["multiplier"] == pytest.approx(1e-3)
    metrics = evaluate_depth_predictions(
        prediction,
        log_variance,
        target,
        variance_multiplier=calibration["multiplier"],
        frame_ids=("a", "b"),
        group_ids=("indoor", "outdoor"),
    )
    assert metrics["group_macro"]["raw_abs_rel"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["group_macro"]["aligned_abs_rel"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["group_macro"]["absolute_log_scale_error"] == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(metrics["group_macro"]["nll"])
    assert metrics["risk_coverage"]["pixel_ause"] >= 0


def _latency_receipt(replica: int) -> dict[str, Any]:
    counts = {"M0": 86_402, "M1": 92_820, "M2": 92_916, "M3": 93_685}
    ratios = {"M0": 1.0, "M1": 1.02, "M2": 1.04, "M3": 1.08}
    return {
        "schema_version": LATENCY_REPLICA_SCHEMA,
        "status": "success",
        "replica": replica,
        "hardware": {"gpu_name": "NVIDIA A100-SXM4-80GB"},
        "config": {
            "initialization_seed": 260629,
            "warmups_per_path": 30,
            "blocks": 30,
            "iterations_per_block": 100,
            "batch_size": 1,
        },
        "arms": {
            arm: {
                "complete_head_wall_ms": [ratios[arm] * (1 + replica * 1e-3)] * 30,
                "parameter_count": counts[arm],
                "peak_allocation_bytes": 1024,
            }
            for arm in counts
        },
        "execution_provenance": _provenance(f"latency-{replica}"),
    }


def test_latency_and_pilot_qualification_use_current_job_provenance(tmp_path: Path) -> None:
    latency_paths = [_write(tmp_path / f"latency-{replica}.json", _latency_receipt(replica)) for replica in range(12)]
    latency = aggregate_latency_receipts(
        latency_paths,
        current_provenance=_provenance("latency-aggregate"),
        resamples=2_000,
    )
    assert latency["qualified_arms"] == ["M0", "M1", "M2", "M3"]
    assert latency["execution_provenance"]["slurm"]["job_id"] == "latency-aggregate"
    latency_path = _write(tmp_path / "latency-gate.json", latency)
    pilot_paths = []
    for arm in ("M0", "M1", "M2", "M3"):
        camera = {
            "raw_abs_rel": {"updated": 0.1, "stale": 0.2, "wrong": 0.3, "permuted": 0.4},
            "permutation_bijective": True,
            "permutation_change_fraction": 1.0,
            "minimum_output_delta_m": 1e-3,
        }
        pilot_paths.append(
            _write(
                tmp_path / f"pilot-{arm}.json",
                {
                    "schema_version": "jepa4d-phase2f-training-run-v1",
                    "status": "success",
                    "stage": "pilot",
                    "arm": arm,
                    "rotation": "R0",
                    "seed": 0,
                    "finite": True,
                    "exact_reload": True,
                    "maximum_forbidden_gradient_norm": 0.0,
                    "camera_controls": camera if arm in {"M2", "M3"} else {},
                    "execution_provenance": _provenance(f"pilot-{arm}"),
                },
            )
        )
    pilot = aggregate_pilot_receipts(
        latency_path,
        pilot_paths,
        current_provenance=_provenance("pilot-gate"),
    )
    assert pilot["formal_allowlist"] == ["M0", "M1", "M2", "M3"]
    assert pilot["execution_provenance"]["slurm"]["job_id"] == "pilot-gate"


def _formal_receipt(arm: str, rotation: str, seed: int, metrics: dict[str, float]) -> dict[str, Any]:
    camera = {
        "raw_abs_rel": {"updated": 0.1, "stale": 0.2, "wrong": 0.3, "permuted": 0.4},
        "permutation_bijective": True,
        "permutation_change_fraction": 1.0,
        "minimum_output_delta_m": 1e-3,
    }
    return {
        "schema_version": "jepa4d-phase2f-training-run-v1",
        "status": "success",
        "stage": "formal",
        "arm": arm,
        "rotation": rotation,
        "seed": seed,
        "finite": True,
        "exact_reload": True,
        "maximum_forbidden_gradient_norm": 0.0,
        "optimizer_steps": 1,
        "wandb": {"mode": "online", "status": "success"},
        "metrics": {"development_test": {"group_macro": metrics}},
        "camera_controls": camera if arm in {"M2", "M3"} else {},
        "checkpoint": {"path": f"/{arm}-{rotation}-{seed}.pt", "sha256": "e" * 64},
        "feature_normalization": {"path": f"/{rotation}-normalization.pt", "sha256": "f" * 64},
        "validation_variance_calibration": {"multiplier": 1.0},
        "execution_provenance": _provenance(f"formal-{arm}-{rotation}-{seed}"),
    }


def test_selector_applies_one_e_minus_twelve_tie_break_and_freezes_one_survivor(tmp_path: Path) -> None:
    arms = {
        "M0": {
            "raw_abs_rel": 0.20,
            "absolute_log_scale_error": 0.20,
            "aligned_abs_rel": 0.10,
            "nll": 0.50,
            "ause": 0.10,
        },
        "M1": {
            "raw_abs_rel": 0.15,
            "absolute_log_scale_error": 0.10,
            "aligned_abs_rel": 0.10,
            "nll": 0.40,
            "ause": 0.09,
        },
        "M2": {
            "raw_abs_rel": 0.1500000000005,
            "absolute_log_scale_error": 0.08,
            "aligned_abs_rel": 0.10,
            "nll": 0.40,
            "ause": 0.09,
        },
        "M3": {
            "raw_abs_rel": 0.16,
            "absolute_log_scale_error": 0.09,
            "aligned_abs_rel": 0.10,
            "nll": 0.40,
            "ause": 0.09,
        },
    }
    latency = {
        "schema_version": "jepa4d-phase2f-latency-qualification-v1",
        "status": "pass",
        "qualified_arms": list(arms),
        "arms": {arm: {"qualified": True, "ratio_ci95": [0.9, 1.08], "parameter_count": 90_000} for arm in arms},
        "execution_provenance": _provenance("latency-aggregate"),
    }
    pilot = {
        "schema_version": "jepa4d-phase2f-pilot-qualification-v1",
        "status": "pass",
        "formal_allowlist": list(arms),
        "execution_provenance": _provenance("pilot-gate"),
    }
    latency_path = _write(tmp_path / "latency.json", latency)
    pilot_path = _write(tmp_path / "pilot.json", pilot)
    formal_paths = [
        _write(
            tmp_path / f"formal-{arm}-{rotation}-{seed}.json",
            _formal_receipt(arm, rotation, seed, arms[arm]),
        )
        for arm in arms
        for rotation in ("R0", "R1", "R2", "R3")
        for seed in (0, 1, 2)
    ]
    result = select_survivor(
        latency_path,
        pilot_path,
        formal_paths,
        current_provenance=_provenance("selector"),
    )
    assert result["survivor"] == "M2"
    assert result["final_authorized"] is True
    assert set(result["checkpoint_set"]) == {"M0", "M2"}
    assert all(len(result["checkpoint_set"][arm]) == 12 for arm in ("M0", "M2"))
    assert result["execution_provenance"]["slurm"]["job_id"] == "selector"


def test_fresh_final_sentinel_is_atomic_and_single_use(tmp_path: Path) -> None:
    selector = _write(tmp_path / "selector.json", {"schema_version": SELECTOR_SCHEMA})
    sentinel = tmp_path / "FRESH_FINAL_OPENED.json"
    payload = create_open_sentinel(sentinel, selector_path=selector, provenance=_provenance("external-final"))
    assert payload["fresh_final_opened"] is True
    assert json.loads(sentinel.read_text())["execution_id"] == "phase2f-test-execution"
    with pytest.raises(FileExistsError):
        create_open_sentinel(sentinel, selector_path=selector, provenance=_provenance("external-final"))


def test_masked_nan_depth_is_zeroed_before_area_reduction(tmp_path: Path) -> None:
    rgb = np.zeros((768, 1024, 3), dtype=np.uint8)
    depth = np.full((768, 1024), 2.0, dtype=np.float32)
    mask = np.ones((768, 1024), dtype=bool)
    depth[100, 300] = np.nan
    mask[100, 300] = False
    rgb_path = tmp_path / "frame.png"
    depth_path = tmp_path / "frame_depth.npy"
    mask_path = tmp_path / "frame_depth_mask.npy"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    np.save(mask_path, mask)
    _, reduced, valid, _ = _preprocess_sample(
        {"sample_id": "frame", "rgb": rgb_path, "depth": depth_path, "mask": mask_path}
    )
    assert torch.isfinite(reduced[valid]).all()
    assert float(reduced[valid].mean()) == pytest.approx(2.0, abs=1e-5)


def test_external_final_no_survivor_does_not_touch_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import evaluate_phase2f_final as final_module

    selector = {
        "schema_version": SELECTOR_SCHEMA,
        "status": "success",
        "final_authorized": False,
        "survivor": None,
        "wandb": {"mode": "online", "status": "success"},
        "execution_provenance": _provenance("selector"),
    }
    selector_path = _write(tmp_path / "selector.json", selector)
    provenance_path = _write(tmp_path / "provenance.json", _provenance("external-final"))
    nonexistent_archive = tmp_path / "must-not-be-touched.tar.gz"
    output = tmp_path / "output"

    class FakeRun:
        def log(self, _value: Any) -> None:
            pass

    monkeypatch.setattr(final_module, "_initialize_run", lambda *_args, **_kwargs: FakeRun())
    monkeypatch.setattr(
        final_module,
        "_finish_wandb",
        lambda *_args, **_kwargs: {"mode": "online", "status": "success"},
    )
    monkeypatch.setenv("SLURM_JOB_ID", "external-final")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_phase2f_final.py",
            "--archive",
            str(nonexistent_archive),
            "--asset-seal",
            str(tmp_path / "absent-seal.json"),
            "--diode-meta",
            str(tmp_path / "absent-meta.json"),
            "--intrinsics",
            str(tmp_path / "absent-intrinsics.txt"),
            "--devkit-license",
            str(tmp_path / "absent-license"),
            "--selector",
            str(selector_path),
            "--sentinel",
            str(tmp_path / "FRESH_FINAL_OPENED.json"),
            "--vjepa-checkpoint",
            str(tmp_path / "absent-checkpoint"),
            "--vjepa-implementation",
            str(tmp_path / "absent-implementation"),
            "--provenance",
            str(provenance_path),
            "--output",
            str(output),
        ],
    )
    final_module.main()
    receipt = json.loads((output / "final_receipt.json").read_text())
    assert receipt["schema_version"] == FINAL_SCHEMA
    assert receipt["archive_touched"] is False
    assert not nonexistent_archive.exists()
