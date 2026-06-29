from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jepa4d.evaluation.fusion_attribution import (
    PHASE2C_SCHEMA,
    build_attribution_record,
    build_gate_interventions,
    evaluate_checkpoint_attribution,
    load_phase2c_artifacts,
    normalize_phase2c_feature_grids,
    sha256,
    write_full_predictions_npz,
    write_qualitative_examples_npz,
)
from jepa4d.models.geometry_student import ResidualFusionGeometryProbe
from jepa4d.visualization.fusion_attribution_report import build_fusion_attribution_report
from scripts.run_phase2d_fusion_attribution import _select_qualitative_indices


@pytest.fixture
def single_torch_thread() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _phase2c_fixture(root: Path, input_dim: int = 8) -> Path:
    root.mkdir()
    (root / "checkpoints").mkdir()
    normalization = {
        key: {"mean": torch.zeros(1, input_dim, 1, 1), "std": torch.ones(1, input_dim, 1, 1)}
        for key in ("vjepa_final", "vjepa_layer_2", "vjepa_layer_5", "vjepa_layer_8")
    }
    normalization_path = root / "vjepa_learned_fusion-normalization.pt"
    torch.save(normalization, normalization_path)
    artifact_hashes = {normalization_path.name: sha256(normalization_path)}
    rows = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = ResidualFusionGeometryProbe(input_dim, hidden_dim=8)
        with torch.no_grad():
            model.fusion.raw_gates.copy_(torch.tensor([0.5 + seed * 0.01, -0.3, 0.2]))
        checkpoint_path = root / "checkpoints" / f"vjepa_learned_fusion-seed{seed}.pt"
        torch.save(
            {
                "variant": "vjepa_learned_fusion",
                "seed": seed,
                "input_dim": input_dim,
                "model_type": "ResidualFusionGeometryProbe",
                "state_dict": model.state_dict(),
                "validation_abs_rel": 0.2 + seed * 0.01,
                "best_epoch": 3,
                "probe_initial_sha256": "synthetic",
                "fusion_state": model.fusion_state(),
            },
            checkpoint_path,
        )
        digest = sha256(checkpoint_path)
        relative = str(checkpoint_path.relative_to(root))
        artifact_hashes[relative] = digest
        rows.append(
            {
                "variant_id": "vjepa_learned_fusion",
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": digest,
                "metrics": {"metric_abs_rel": 0.4 + seed * 0.01, "variance_multiplier": 2.0},
                "model_metadata": {"fusion_state": model.fusion_state()},
            }
        )
    comparison = {
        "schema_version": PHASE2C_SCHEMA,
        "experiment_id": "synthetic-phase2c",
        "split_hash": "synthetic-split",
        "failures": [],
        "variants": rows,
        "artifacts": artifact_hashes,
        "wandb_url": "https://example.invalid/run",
    }
    (root / "comparison.json").write_text(json.dumps(comparison))
    return root


def test_gate_intervention_registry_is_complete_and_fixed_average_is_exact() -> None:
    interventions = build_gate_interventions(torch.tensor([0.3, -0.2, 0.1]))
    assert len(interventions) == 15
    assert len({value.intervention_id for value in interventions}) == 15
    assert [value.family for value in interventions].count("layer_permutation") == 5
    assert [value.family for value in interventions].count("sign_flip") == 7
    assert interventions[0].raw_gates == pytest.approx((0.3, -0.2, 0.1))
    assert next(value for value in interventions if value.intervention_id == "zero").effective_coefficients == (
        0,
        0,
        0,
    )
    fixed = next(value for value in interventions if value.intervention_id == "fixed_average")
    assert fixed.raw_gates == pytest.approx((math.atanh(0.75),) * 3)
    assert fixed.effective_coefficients == pytest.approx((0.25, 0.25, 0.25))
    assert fixed.final_coefficient == pytest.approx(0.25)


def test_phase2c_artifact_loader_checks_hashes_and_normalization(tmp_path: Path) -> None:
    root = _phase2c_fixture(tmp_path / "phase2c")
    artifacts = load_phase2c_artifacts(root)
    assert set(artifacts.checkpoints) == {0, 1, 2}
    assert artifacts.normalization["vjepa_final"]["mean"].shape == (1, 8, 1, 1)

    comparison_path = root / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["artifacts"]["checkpoints/vjepa_learned_fusion-seed1.pt"] = "0" * 64
    comparison_path.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_phase2c_artifacts(root)


def test_same_checkpoint_attribution_and_self_contained_report(tmp_path: Path, single_torch_thread: None) -> None:
    root = _phase2c_fixture(tmp_path / "phase2c")
    artifacts = load_phase2c_artifacts(root)
    generator = torch.Generator().manual_seed(11)
    validation_features = torch.randn((3, 4, 8, 4, 4), generator=generator).half()
    test_features = torch.randn((4, 4, 8, 4, 4), generator=generator).half()
    model = ResidualFusionGeometryProbe(8, hidden_dim=8)
    model.load_state_dict(artifacts.checkpoints[0]["state_dict"], strict=True)
    with torch.inference_mode():
        validation_depth = model(validation_features)[0].exp()
        test_depth = model(test_features)[0].exp()
    validation_targets = (validation_depth * 1.05).clamp(0.11, 9.0)
    test_targets_24 = (test_depth * 1.1).clamp(0.11, 9.0)
    test_targets_full = F.interpolate(test_targets_24.unsqueeze(1), size=(8, 8), mode="nearest")[:, 0]
    result = evaluate_checkpoint_attribution(
        artifacts.checkpoints[0],
        artifacts.learned_rows[0],
        validation_features,
        test_features,
        validation_targets,
        test_targets_24,
        test_targets_full,
        ["sequence_a", "sequence_a", "sequence_b", "sequence_b"],
        batch_size=2,
    )
    assert len(result["interventions"]) == 15
    original = result["interventions"][0]
    zero = next(value for value in result["interventions"] if value["intervention"]["intervention_id"] == "zero")
    assert original["macro"]["prediction_delta_relative"] == pytest.approx(0.0)
    assert zero["macro"]["prediction_delta_relative"] > 0
    assert zero["macro"]["residual_total_norm_ratio"] == pytest.approx(0.0)
    assert {value["sequence_id"] for value in original["per_sequence"]} == {"sequence_a", "sequence_b"}
    assert math.isfinite(original["macro"]["calibrated_log_depth_nll"])

    seed_results = []
    for seed in (0, 1, 2):
        copied = copy.deepcopy(result)
        copied["seed"] = seed
        seed_results.append(copied)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema_version: synthetic\n")
    record = build_attribution_record(
        artifacts=artifacts,
        seed_results=seed_results,
        dataset_manifest=manifest,
        dataset_split_hash="synthetic-split",
        sample_ids=["a0", "a1", "b0", "b1"],
        output_directory=tmp_path / "phase2d",
    )
    qualitative_path = write_qualitative_examples_npz(
        tmp_path / "phase2d" / "qualitative_examples.npz",
        predictions=[test_depth[:2], test_depth[:2] * 0.95, test_depth[:2] * 1.05],
        log_variances=[torch.zeros_like(test_depth[:2])] * 3,
        calibrated_log_depth_sigmas=[torch.ones_like(test_depth[:2]) * value for value in (0.2, 0.3, 0.4)],
        target=test_targets_24[:2],
        sample_ids=["a0", "a1"],
        sequence_ids=["sequence_a", "sequence_a"],
        variant_ids=["seed0:original", "seed0:zero", "seed0:fixed_average"],
        seeds=[0, 0, 0],
    )
    record["qualitative_handoff"] = {
        "sample_ids": ["a0", "a1"],
        "variant_ids": ["seed0:original", "seed0:zero", "seed0:fixed_average"],
    }
    report = build_fusion_attribution_report(
        record,
        tmp_path / "phase2d" / "report.html",
        qualitative_npz=qualitative_path,
    )
    content = report.read_text()
    assert "Phase 2d" in content
    assert "plotly.js" in content.lower()
    assert "<script src=" not in content.lower()
    assert "Freiburg-3 was already consumed" in content
    assert "Fixed qualitative prediction" in content
    assert "Calibrated log-depth σ" in content
    assert "data:image/png;base64," in content


def test_phase2c_normalization_stacks_final_then_intermediate_layers() -> None:
    features = {
        "vjepa_final": torch.full((2, 3, 2, 2), 1.0),
        "vjepa_layer_2": torch.full((2, 3, 2, 2), 2.0),
        "vjepa_layer_5": torch.full((2, 3, 2, 2), 3.0),
        "vjepa_layer_8": torch.full((2, 3, 2, 2), 4.0),
    }
    normalization = {key: {"mean": torch.zeros(1, 3, 1, 1), "std": torch.ones(1, 3, 1, 1)} for key in features}
    stacked = normalize_phase2c_feature_grids(features, normalization)
    assert stacked.shape == (2, 4, 3, 2, 2)
    assert stacked[:, :, 0, 0, 0].tolist() == [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]


def test_full_prediction_handoff_matches_shared_schema(tmp_path: Path) -> None:
    target = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.5, 2.5], [3.5, 4.5]],
        ]
    )
    predictions = [target * 0.9, target, target * 1.1]
    path = write_full_predictions_npz(
        tmp_path / "full_predictions.npz",
        predictions=predictions,
        target=target,
        sample_ids=["frame_0", "frame_1"],
        sequence_ids=["sequence_a", "sequence_b"],
        variant_ids=["seed0:original", "seed0:zero", "seed0:fixed_average"],
        seeds=[0, 0, 0],
    )
    with np.load(path) as payload:
        assert str(payload["schema_version"]) == "jepa4d-phase2d-depth-predictions-v1"
        assert payload["prediction_m"].shape == (3, 2, 2, 2)
        assert payload["target_m"].shape == (2, 2, 2)
        assert payload["sample_ids"].tolist() == ["frame_0", "frame_1"]
        assert payload["sequence_ids"].tolist() == ["sequence_a", "sequence_b"]
        assert payload["variant_ids"].tolist() == ["seed0:original", "seed0:zero", "seed0:fixed_average"]
        assert payload["seeds"].tolist() == [0, 0, 0]
        assert str(payload["audit_scope"]) == "full_phase2c_test"


def test_qualitative_handoff_is_bounded_and_retains_log_variance(tmp_path: Path) -> None:
    target = torch.ones((2, 3, 3))
    path = write_qualitative_examples_npz(
        tmp_path / "qualitative.npz",
        predictions=[target * 0.9, target * 1.1],
        log_variances=[torch.full_like(target, -0.5), torch.full_like(target, 0.5)],
        calibrated_log_depth_sigmas=[torch.full_like(target, 0.3), torch.full_like(target, 0.6)],
        target=target,
        sample_ids=["a", "b"],
        sequence_ids=["sequence_a", "sequence_b"],
        variant_ids=["seed0:original", "seed0:zero"],
        seeds=[0, 0],
    )
    with np.load(path, allow_pickle=False) as payload:
        assert str(payload["schema_version"]) == "jepa4d-phase2d-qualitative-v1"
        assert payload["prediction_m"].shape == (2, 2, 3, 3)
        assert payload["log_variance"].shape == (2, 2, 3, 3)
        assert payload["calibrated_log_depth_sigma"].shape == (2, 2, 3, 3)
        assert payload["sample_ids"].tolist() == ["a", "b"]

    too_many = torch.ones((9, 3, 3))
    with pytest.raises(ValueError, match="between one and eight"):
        write_qualitative_examples_npz(
            tmp_path / "too-many.npz",
            predictions=[too_many],
            log_variances=[too_many],
            calibrated_log_depth_sigmas=[too_many],
            target=too_many,
            sample_ids=[str(index) for index in range(9)],
            sequence_ids=["sequence"] * 9,
            variant_ids=["seed0:original"],
            seeds=[0],
        )


def test_qualitative_selection_is_deterministic_bounded_and_sequence_balanced() -> None:
    sequence_ids = ["b"] * 6 + ["a"] * 6
    selected = _select_qualitative_indices(sequence_ids, maximum=4)
    assert selected == _select_qualitative_indices(sequence_ids, maximum=4)
    assert len(selected) == 4
    assert {sequence_ids[index] for index in selected} == {"a", "b"}
    assert len(_select_qualitative_indices(["only"] * 10, maximum=4)) == 4
