from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from slurm.phase2c_gate import SPLIT_COUNTS, VARIANT_SEEDS, protocol_contract
from slurm.validate_phase2c_preflight import _validate_fusion_artifacts

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip"
PARTITIONS = "#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value)
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest(output: Path) -> dict[str, dict[str, Any]]:
    excluded = {"artifact_manifest.json", "wandb_artifact_receipt.json"}
    return {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and str(path.relative_to(output)) not in excluded
    }


def _preflight_fusion_record(root: Path) -> dict[str, Any]:
    fusion: dict[str, Any] = {}
    for label in ("normalization", "checkpoint", "report"):
        path = root / f"{label}.artifact"
        _write(path, f"valid-{label}")
        fusion[label] = str(path)
        fusion[f"{label}_sha256"] = _sha256(path)

    def profile(capture_layers: list[int]) -> dict[str, Any]:
        return {
            "profile": "co-resident-batch1-encoder-normalization-fusion-probe-v1",
            "input_boundary": "preloaded RGBInputBatch before device transfer and model preprocessing",
            "capture_layers": capture_layers,
            "sample_ids": [f"training-smoke-{index}" for index in range(8)],
            "warmup_iterations": 1,
            "measured_iterations_per_repetition": 2,
            "repetitions": 1,
            "repetition_ms_per_frame": [11.0],
            "median_ms_per_frame": 11.0,
            "peak_end_to_end_memory_gb": 8.0,
        }

    fusion["end_to_end_profile_smoke"] = {
        "vjepa_final": profile([]),
        "vjepa_learned_fusion": profile([2, 5, 8]),
    }
    return fusion


def _formal_output(root: Path) -> Path:
    output = root / "formal"
    normalization_hashes = {}
    for name in protocol_contract()["normalization_artifacts"]:
        path = output / name
        _write(path, f"normalization-{name}".encode())
        normalization_hashes[name] = _sha256(path)

    common_metrics = {
        "metric_abs_rel": 0.1,
        "metric_rmse_m": 0.2,
        "metric_delta_1": 0.9,
        "aligned_abs_rel": 0.08,
        "metric_abs_log_scale_error": 0.03,
        "raw_log_depth_nll": -1.0,
        "calibrated_log_depth_nll": -1.1,
    }
    common_runtime = {
        "encoder_ms_per_frame": 10.0,
        "head_ms_per_frame": 1.0,
        "total_ms_per_frame": 11.0,
        "peak_encoder_memory_gb": 8.0,
        "peak_head_memory_gb": 1.0,
        "end_to_end_ms_per_frame": 11.0,
        "peak_end_to_end_memory_gb": 8.0,
    }
    sequence_metrics = {
        "freiburg3_long_office_household": common_metrics,
        "freiburg3_structure_texture_far": common_metrics,
    }
    test_samples_by_sequence = {
        sequence: [f"{sequence}-sample-{index:03d}" for index in range(64)] for sequence in sequence_metrics
    }
    formal_profile_sample_ids = [sample_id for values in test_samples_by_sequence.values() for sample_id in values][
        ::16
    ][:8]
    variants = []
    rows_by_key = []
    end_to_end_profiles = []
    for variant, seeds in VARIANT_SEEDS.items():
        for seed in seeds:
            checkpoint: str | None = None
            checkpoint_hash: str | None = None
            trainable = 0 if seed is None else 100
            model_metadata: dict[str, Any] = {}
            if seed is not None:
                model_metadata["checkpoint_reload"] = "strict-prediction-equality-pass"
                model_metadata["probe_initial_sha256"] = (
                    f"paired-final-fusion-seed-{seed}"
                    if variant in {"vjepa_final", "vjepa_learned_fusion"}
                    else f"{variant}-seed-{seed}"
                )
                if variant.startswith("vjepa_"):
                    profile = {
                        "profile": "co-resident-batch1-encoder-normalization-fusion-probe-v1",
                        "input_boundary": "preloaded RGBInputBatch before device transfer and model preprocessing",
                        "capture_layers": [] if variant == "vjepa_final" else [2, 5, 8],
                        "sample_ids": formal_profile_sample_ids,
                        "warmup_iterations": 30,
                        "measured_iterations_per_repetition": 30,
                        "repetitions": 3,
                        "repetition_ms_per_frame": [10.0, 11.0, 12.0],
                        "median_ms_per_frame": 11.0,
                        "peak_end_to_end_memory_gb": 8.0,
                        "variant": variant,
                        "seed": seed,
                    }
                    model_metadata["end_to_end_profile"] = profile
                    end_to_end_profiles.append(profile)
                if variant == "vjepa_learned_fusion":
                    trainable = 103
                    fusion_state = {
                        "layer_order": [2, 5, 8],
                        "final_coefficient": 1.0,
                        "coefficient_layer_2": 0.0,
                        "coefficient_layer_5": 0.0,
                        "coefficient_layer_8": 0.0,
                    }
                    model_type = "ResidualFusionGeometryProbe"
                    model_metadata.update({"additional_trainable_parameters": 3, "fusion_state": fusion_state})
                else:
                    fusion_state = None
                    model_type = "DenseGeometryProbe"
                path = output / "checkpoints" / f"{variant}-seed{seed}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "variant": variant,
                        "seed": seed,
                        "model_type": model_type,
                        "state_dict": {"weight": torch.ones(1)},
                        "fusion_state": fusion_state,
                    },
                    path,
                )
                checkpoint, checkpoint_hash = str(path), _sha256(path)
                history = []
                for epoch in range(60):
                    row = {"variant": variant, "seed": seed, "epoch": epoch, "loss": 1.0}
                    if variant == "vjepa_learned_fusion":
                        row.update({"gate_gradient_norm": 0.1, "coefficient_layer_2": 0.0})
                    history.append(json.dumps(row, sort_keys=True))
                _write(output / "histories" / f"{variant}-seed{seed}.jsonl", "\n".join(history) + "\n")
            row = {
                "variant_id": variant,
                "seed": seed,
                "metrics": common_metrics,
                "runtime": common_runtime,
                "trainable_parameters": trainable,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "model_metadata": model_metadata,
                "sequence_metrics": sequence_metrics,
            }
            variants.append(row)
            rows_by_key.append((variant, seed))

    run_url = "https://wandb.ai/test/project/runs/phase2c"
    _write(
        output / "comparison.json",
        {
            "schema_version": "jepa4d-phase2c-cross-sequence-comparison-v1",
            "split_hash": "split-hash",
            "variants": variants,
            "failures": [],
            "aggregates": {},
            "artifacts": normalization_hashes,
            "wandb_url": run_url,
        },
    )
    _write(output / "failures.json", [])
    _write(
        output / "formal_authorization.json",
        {
            "schema_version": "jepa4d-phase2c-authorization-v1",
            "status": "pass",
            "protocol_sha256": protocol_contract()["sha256"],
            "split_hash": "split-hash",
        },
    )
    _write(
        output / "resolved_config.json",
        {
            "protocol": "phase2c-cross-sequence-v1",
            "split_hash": "split-hash",
            "split_counts": SPLIT_COUNTS,
            "epochs": 60,
            "seeds": [0, 1, 2],
            "authorization": {"sha256": _sha256(output / "formal_authorization.json")},
            "wandb": {"enabled": True, "mode": "online"},
        },
    )
    _write(
        output / "dataset_fingerprint.json",
        {
            "split_counts": SPLIT_COUNTS,
            "sequences": [
                {"sequence_id": "fr1_a", "split": "train"},
                {"sequence_id": "fr1_b", "split": "train"},
                {"sequence_id": "fr2_a", "split": "validation"},
                *[
                    {
                        "sequence_id": sequence,
                        "split": "test",
                        "samples": [{"sample_id": sample_id} for sample_id in sample_ids],
                    }
                    for sequence, sample_ids in test_samples_by_sequence.items()
                ],
            ],
        },
    )
    per_sequence = [
        {"variant": variant, "seed": seed, "sequence_id": sequence, **common_metrics}
        for variant, seed in rows_by_key
        for sequence in ("freiburg3_long_office_household", "freiburg3_structure_texture_far")
    ]
    _write(output / "per_sequence_metrics.json", per_sequence)
    per_frame = [
        {
            "variant": variant,
            "seed": seed,
            "sequence_id": sequence,
            "frame_id": f"{sequence}-{index}",
            **common_metrics,
        }
        for variant, seed in rows_by_key
        for sequence in ("freiburg3_long_office_household", "freiburg3_structure_texture_far")
        for index in range(64)
    ]
    _write(output / "per_frame_metrics.json", per_frame)
    _write(
        output / "end_to_end_profiles.json",
        end_to_end_profiles,
    )
    _write(
        output / "promotion_gate.json",
        {
            "schema_version": "jepa4d-phase2c-promotion-v1",
            "decision": "retain_final_layer",
            "promoted": False,
            "conditions": {
                "primary_macro_absrel_strictly_better": False,
                "no_sequence_regression_above_5pct": True,
                "latency_at_most_1p10x_final": True,
                "peak_inference_memory_at_most_1p10x_final": True,
                "all_results_finite_valid_and_checkpointed": True,
                "zero_failures": True,
            },
            "primary": {
                "final_macro_absrel": 0.1,
                "candidate_macro_absrel": 0.1,
                "relative_change": 0.0,
            },
            "per_sequence": {
                sequence: {
                    "final_absrel": 0.1,
                    "candidate_absrel": 0.1,
                    "relative_regression": 0.0,
                    "passes_maximum_5pct_regression": True,
                }
                for sequence in ("freiburg3_long_office_household", "freiburg3_structure_texture_far")
            },
            "latency": {"final_ms_per_frame": 11.0, "candidate_ms_per_frame": 11.0, "ratio": 1.0},
            "peak_inference_memory": {"final_gib": 8.0, "candidate_gib": 8.0, "ratio": 1.0},
        },
    )
    _write(
        output / "completion_gate.json",
        {
            "status": "success",
            "result_rows": 13,
            "probe_checkpoints": 12,
            "seed_failures": 0,
            "promotion_decision": "retain_final_layer",
        },
    )
    _write(
        output / "geometry_student_report.html",
        ("<html><script>const topojsonURL='https://cdn.plot.ly/un/';Plotly.newPlot('plot', [], {});</script></html>"),
    )
    _write(output / "artifact_manifest.json", _manifest(output))
    _write(
        output / "wandb_artifact_receipt.json",
        {
            "schema_version": "jepa4d-phase2c-wandb-artifact-v1",
            "status": "success",
            "mode": "online",
            "run_id": "phase2c",
            "run_url": run_url,
            "artifact_name": "phase2c-comparison",
            "artifact_version": "v0",
            "artifact_digest": "server-digest",
            "artifact_manifest_sha256": _sha256(output / "artifact_manifest.json"),
        },
    )
    return output


def _validate(output: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "slurm" / "validate_phase2c_output.py"),
            "--output",
            str(output),
            "--report",
            str(report),
            "--require-wandb",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_phase2c_protocol_contract_is_explicit() -> None:
    contract = protocol_contract()
    assert contract["sequence_splits"] == {"train": 2, "validation": 1, "test": 2}
    assert contract["split_counts"] == SPLIT_COUNTS
    assert contract["result_rows"] == 13
    assert contract["probe_checkpoints"] == 12
    assert contract["epochs"] == 60
    assert len(contract["normalization_artifacts"]) == 4
    assert len(contract["sha256"]) == 64


def test_phase2c_sbatch_resources_and_online_lock_are_exact() -> None:
    for name in ("phase2c_tests.sbatch", "phase2c_preflight.sbatch", "phase2c_train.sbatch"):
        text = (ROOT / "slurm" / name).read_text()
        assert ACCOUNT in text
        assert PARTITIONS in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --mem=220G" in text
        assert "#SBATCH --time=" in text
        hours = int(text.split("#SBATCH --time=", 1)[1].split(":", 1)[0])
        assert hours <= 4
    train = (ROOT / "slurm" / "phase2c_train.sbatch").read_text()
    assert "#SBATCH --time=04:00:00" in train
    assert '[[ "$EPOCHS" == "60" ]]' in train
    assert "export WANDB_MODE=online" in train
    assert '--dataset-root "$DATA_PARENT"' in train
    assert '--manifest "$MANIFEST"' in train
    assert '--authorization "$JEPA4D_JOB_LOG_DIR/formal-authorization.json"' in train
    assert "--archive" not in train
    assert "validate_phase2c_preflight.py" in train
    assert "validate_phase2c_output.py" in train


def test_login_preparation_does_no_gpu_work() -> None:
    text = (ROOT / "slurm" / "prepare_phase2c_login.sh").read_text()
    assert "download_phase2c_assets.py" in text
    assert "check_cuda.py" not in text
    assert "nvidia-smi" not in text
    assert "sbatch" not in text


def test_phase2c_output_validator_accepts_complete_online_output(tmp_path: Path) -> None:
    result = _validate(_formal_output(tmp_path), tmp_path / "validation.json")
    assert result.returncode == 0, result.stdout + result.stderr


def test_phase2c_output_validator_rejects_external_plotly_script(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    _write(
        output / "geometry_student_report.html",
        (
            '<html><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
            "<script>Plotly.newPlot('plot', [], {});</script></html>"
        ),
    )
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "loads an external script" in result.stdout


def test_phase2c_output_validator_rejects_missing_sequence_evidence(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    rows = json.loads((output / "per_sequence_metrics.json").read_text())
    _write(output / "per_sequence_metrics.json", rows[:-1])
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "per-sequence" in result.stdout


def test_phase2c_output_validator_rejects_stale_artifact_manifest(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    (output / "geometry_student_report.html").write_text("mutated")
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "artifact manifest" in result.stdout


def test_phase2c_output_validator_rejects_non_macro_primary_metric(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    comparison = json.loads((output / "comparison.json").read_text())
    comparison["variants"][0]["metrics"]["metric_abs_rel"] = 0.2
    _write(output / "comparison.json", comparison)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "arithmetic mean" in result.stdout


def test_phase2c_output_validator_recomputes_promotion_decision(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    promotion = json.loads((output / "promotion_gate.json").read_text())
    promotion["decision"] = "promote_learned_fusion"
    promotion["promoted"] = True
    _write(output / "promotion_gate.json", promotion)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "promotion gate" in result.stdout


def test_phase2c_output_validator_rejects_tampered_end_to_end_profile(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    profiles = json.loads((output / "end_to_end_profiles.json").read_text())
    profiles[0]["median_ms_per_frame"] = 999.0
    profiles[0]["peak_end_to_end_memory_gb"] = 777.0
    _write(output / "end_to_end_profiles.json", profiles)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "persisted end-to-end profile differs" in result.stdout


def test_phase2c_output_validator_rejects_invalid_end_to_end_capture_protocol(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    profiles = json.loads((output / "end_to_end_profiles.json").read_text())
    target = next(profile for profile in profiles if profile["variant"] == "vjepa_final" and profile["seed"] == 0)
    target["capture_layers"] = [2, 5, 8]
    comparison = json.loads((output / "comparison.json").read_text())
    comparison_target = next(
        row for row in comparison["variants"] if row["variant_id"] == "vjepa_final" and row["seed"] == 0
    )
    comparison_target["model_metadata"]["end_to_end_profile"] = target
    _write(output / "comparison.json", comparison)
    _write(output / "end_to_end_profiles.json", profiles)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "capture layers are invalid" in result.stdout


def test_phase2c_output_validator_rejects_nonpositive_end_to_end_values(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    profiles = json.loads((output / "end_to_end_profiles.json").read_text())
    target = next(profile for profile in profiles if profile["variant"] == "vjepa_final" and profile["seed"] == 0)
    target.update(
        {
            "repetition_ms_per_frame": [-1.0, 0.0, 1.0],
            "median_ms_per_frame": 0.0,
            "peak_end_to_end_memory_gb": 0.0,
        }
    )
    comparison = json.loads((output / "comparison.json").read_text())
    comparison_target = next(
        row for row in comparison["variants"] if row["variant_id"] == "vjepa_final" and row["seed"] == 0
    )
    comparison_target["model_metadata"]["end_to_end_profile"] = target
    comparison_target["runtime"].update(
        {"total_ms_per_frame": 0.0, "end_to_end_ms_per_frame": 0.0, "peak_end_to_end_memory_gb": 0.0}
    )
    _write(output / "comparison.json", comparison)
    _write(output / "end_to_end_profiles.json", profiles)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "end-to-end profile repetitions are invalid" in result.stdout
    assert "end-to-end profile peak memory is invalid" in result.stdout


@pytest.mark.parametrize(
    "sample_ids",
    ([f"arbitrary-{index}" for index in range(8)], ["duplicate"] * 8),
)
def test_phase2c_output_validator_rejects_wrong_profile_sample_ids(tmp_path: Path, sample_ids: list[str]) -> None:
    output = _formal_output(tmp_path)
    profiles = json.loads((output / "end_to_end_profiles.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    comparison_by_key = {(row["variant_id"], row["seed"]): row for row in comparison["variants"]}
    for profile in profiles:
        profile["sample_ids"] = sample_ids
        key = (profile["variant"], profile["seed"])
        comparison_by_key[key]["model_metadata"]["end_to_end_profile"] = profile
    _write(output / "comparison.json", comparison)
    _write(output / "end_to_end_profiles.json", profiles)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "sample IDs differ from deterministic formal selection" in result.stdout


def test_phase2c_output_validator_rejects_tampered_per_sequence_nll(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    rows = json.loads((output / "per_sequence_metrics.json").read_text())
    rows[0]["raw_log_depth_nll"] = 123.0
    rows[0]["calibrated_log_depth_nll"] = 456.0
    _write(output / "per_sequence_metrics.json", rows)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "persisted per-sequence metrics differ" in result.stdout


def test_phase2c_output_validator_recomputes_result_integrity_condition(tmp_path: Path) -> None:
    output = _formal_output(tmp_path)
    comparison = json.loads((output / "comparison.json").read_text())
    target = next(row for row in comparison["variants"] if row["variant_id"] == "vjepa_final" and row["seed"] == 0)
    target["model_metadata"]["checkpoint_reload"] = "unverified"
    _write(output / "comparison.json", comparison)
    _write(output / "artifact_manifest.json", _manifest(output))
    receipt = json.loads((output / "wandb_artifact_receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = _sha256(output / "artifact_manifest.json")
    _write(output / "wandb_artifact_receipt.json", receipt)
    result = _validate(output, tmp_path / "validation.json")
    assert result.returncode != 0
    assert "all_results_finite_valid_and_checkpointed" in result.stdout


def test_phase2c_preflight_validator_rejects_tampered_normalization(tmp_path: Path) -> None:
    fusion = _preflight_fusion_record(tmp_path)
    _validate_fusion_artifacts(fusion)
    Path(fusion["normalization"]).write_text("tampered")
    with pytest.raises(RuntimeError, match="normalization artifact content changed"):
        _validate_fusion_artifacts(fusion)


def test_phase2c_preflight_validator_rejects_invalid_profile_smoke_capture(tmp_path: Path) -> None:
    fusion = _preflight_fusion_record(tmp_path)
    fusion["end_to_end_profile_smoke"]["vjepa_learned_fusion"]["capture_layers"] = []
    with pytest.raises(RuntimeError, match="learned_fusion capture layers are invalid"):
        _validate_fusion_artifacts(fusion)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("repetition_ms_per_frame", [-1.0], "repetition timing is invalid"),
        ("peak_end_to_end_memory_gb", 0.0, "peak memory is invalid"),
        ("sample_ids", ["duplicate"] * 8, "sample IDs are invalid"),
    ),
)
def test_phase2c_preflight_validator_rejects_invalid_profile_smoke_values(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    fusion = _preflight_fusion_record(tmp_path)
    fusion["end_to_end_profile_smoke"]["vjepa_final"][field] = value
    with pytest.raises(RuntimeError, match=message):
        _validate_fusion_artifacts(fusion)


def test_phase2c_preflight_validator_binds_profile_smoke_sample_ids(tmp_path: Path) -> None:
    fusion = _preflight_fusion_record(tmp_path)
    with pytest.raises(RuntimeError, match="profile sample IDs differ from V-JEPA smoke"):
        _validate_fusion_artifacts(fusion, [f"different-{index}" for index in range(8)])
