from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from jepa4d.benchmarks.geometry.tum_rgbd_bundle import _greedy_one_to_one, rank_midpoint_indices
from jepa4d.evaluation.comparison import VariantResult
from jepa4d.models.geometry_student import DenseGeometryProbe, ResidualFusionGeometryProbe
from scripts import run_phase2b_geometry_distillation as runner


def _rows(timestamps: list[float]) -> list[tuple[float, list[str]]]:
    return [(value, [f"frame-{index}"]) for index, value in enumerate(timestamps)]


def test_global_greedy_association_is_one_to_one_and_prefers_smallest_delta() -> None:
    left = _rows([0.000, 0.010, 0.040])
    right = _rows([0.009, 0.041])
    matches = _greedy_one_to_one(left, right, 0.02)
    assert matches == {1: (0, pytest.approx(0.001)), 2: (1, pytest.approx(0.001))}
    assert len({right_index for right_index, _ in matches.values()}) == len(matches)


def test_rank_midpoint_sampling_is_deterministic_unique_and_chronological() -> None:
    valid = list(range(790))
    selected = rank_midpoint_indices(valid, 64)
    assert selected[:4] == [6, 18, 30, 43]
    assert len(selected) == len(set(selected)) == 64
    assert selected == sorted(selected)
    assert rank_midpoint_indices(valid, 64) == selected
    with pytest.raises(ValueError, match="at least"):
        rank_midpoint_indices(list(range(63)), 64)


def test_cross_sequence_manifest_rejects_missing_file(tmp_path: Path) -> None:
    from jepa4d.benchmarks.geometry.tum_rgbd_bundle import load_cross_sequence_bundle

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_cross_sequence_bundle(tmp_path, missing)


def test_phase2c_normalization_is_train_only_and_shares_exact_final_tensor() -> None:
    train = {"vjepa_final": torch.randn(5, 8, 4, 4)}
    validation = {"vjepa_final": torch.randn(3, 8, 4, 4) + 100}
    test = {"vjepa_final": torch.randn(2, 8, 4, 4) - 100}
    for layer in (2, 5, 8):
        train[f"vjepa_layer_{layer}"] = torch.randn(5, 8, 4, 4) + layer
        validation[f"vjepa_layer_{layer}"] = torch.randn(3, 8, 4, 4) + layer + 100
        test[f"vjepa_layer_{layer}"] = torch.randn(2, 8, 4, 4) + layer - 100
    variants, statistics = runner._normalize_phase2c_layers(train, validation, test)
    mutated_validation = {key: value + 10_000 for key, value in validation.items()}
    mutated_test = {key: value - 10_000 for key, value in test.items()}
    _, mutated_statistics = runner._normalize_phase2c_layers(train, mutated_validation, mutated_test)
    for key in statistics:
        assert torch.equal(statistics[key]["mean"], mutated_statistics[key]["mean"])
        assert torch.equal(statistics[key]["std"], mutated_statistics[key]["std"])
    for split_index in range(3):
        final = variants["vjepa_final"][split_index]
        learned = variants["vjepa_learned_fusion"][split_index]
        fixed = variants["vjepa_multilayer"][split_index]
        assert torch.equal(learned[:, 0], final)
        assert torch.equal(fixed, learned.float().mean(dim=1).half())


def test_final_and_learned_candidate_share_probe_initialization() -> None:
    torch.manual_seed(7)
    final = DenseGeometryProbe(8)
    torch.manual_seed(7)
    learned = ResidualFusionGeometryProbe(8)
    assert runner._state_dict_sha256(final.state_dict()) == runner._state_dict_sha256(learned.probe.state_dict())


@pytest.mark.parametrize(
    ("field", "value"),
    [("root_name", "../rgbd_dataset_freiburg1_xyz"), ("depth_scale", 500.0)],
)
def test_formal_bundle_rejects_path_aliases_and_depth_scale_changes(tmp_path: Path, field: str, value: object) -> None:
    from jepa4d.benchmarks.geometry.tum_rgbd_bundle import load_cross_sequence_bundle

    source = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "benchmarks"
        / "manifests"
        / "tum_rgbd_phase2c_cross_sequence_v1.yaml"
    )
    manifest = yaml.safe_load(source.read_text())
    manifest["sequences"][0][field] = value
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="formal sequence contract|depth scale"):
        load_cross_sequence_bundle(tmp_path, mutated)


def test_sequence_macro_weights_sequences_equally() -> None:
    target = torch.ones(4, 12, 12)
    prediction = target.clone()
    prediction[1:] *= 2
    samples = [
        SimpleNamespace(sequence_id="short"),
        SimpleNamespace(sequence_id="long"),
        SimpleNamespace(sequence_id="long"),
        SimpleNamespace(sequence_id="long"),
    ]
    macro, per_sequence = runner._evaluate_sequence_macro(prediction, target, samples)
    assert per_sequence["short"]["metric_abs_rel"] == 0.0
    assert per_sequence["long"]["metric_abs_rel"] == 1.0
    assert macro["metric_abs_rel"] == 0.5


def test_promotion_gate_enforces_primary_sequence_and_efficiency_conditions() -> None:
    results = []
    sequence_ids = ("freiburg3_long_office_household", "freiburg3_structure_texture_far")
    for variant, metric, latency, memory in (
        ("vjepa_final", 0.10, 10.0, 1.0),
        ("vjepa_learned_fusion", 0.09, 10.5, 1.05),
    ):
        for seed in (0, 1, 2):
            results.append(
                VariantResult(
                    variant_id=variant,
                    family="vjepa",
                    role="candidate" if "learned" in variant else "reference_default",
                    seed=seed,
                    metrics={"metric_abs_rel": metric},
                    runtime={
                        "total_ms_per_frame": latency,
                        "peak_encoder_memory_gb": memory,
                        "peak_head_memory_gb": 0.1,
                        "peak_end_to_end_memory_gb": memory,
                    },
                    parameters=1,
                    sequence_metrics={name: {"metric_abs_rel": metric} for name in sequence_ids},
                )
            )
    gate = runner._phase2c_promotion_gate(results, [])
    assert gate["decision"] == "promote_learned_fusion"
    assert all(gate["conditions"].values())
    gate = runner._phase2c_promotion_gate(results, [], results_integrity_valid=False)
    assert gate["decision"] == "retain_final_layer"
    assert not gate["conditions"]["all_results_finite_valid_and_checkpointed"]
    results[-1].sequence_metrics[sequence_ids[0]]["metric_abs_rel"] = 0.2
    gate = runner._phase2c_promotion_gate(results, [])
    assert gate["decision"] == "retain_final_layer"
    assert not gate["conditions"]["no_sequence_regression_above_5pct"]
