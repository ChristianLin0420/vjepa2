import pytest
import torch

from jepa4d.evaluation.comparison import ComparisonRecord, VariantResult
from jepa4d.models.geometry_student import (
    BoundedResidualLayerFusion,
    DenseGeometryProbe,
    ResidualFusionGeometryProbe,
    geometry_probe_loss,
    rgb_grid_features,
)


def test_dense_geometry_probe_and_loss_are_finite() -> None:
    features = torch.randn(2, 8, 12, 12)
    target = torch.rand(2, 12, 12) + 0.5
    valid = torch.ones_like(target, dtype=torch.bool)
    probe = DenseGeometryProbe(8, hidden_dim=16)
    log_depth, logvar = probe(features)
    loss, parts = geometry_probe_loss(log_depth, logvar, target, valid, teacher_depth=target)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_rgb_baseline_and_comparison_schema() -> None:
    features = rgb_grid_features(torch.rand(3, 3, 32, 48), size=8)
    assert features.shape == (3, 5, 8, 8)
    variant = VariantResult("rgb", "rgb", "baseline", 0, {"abs_rel": 1.0}, {"runtime_ms": 1.0}, 10)
    record = ComparisonRecord("test", "v1", "manifest.yaml", "abc", {}, [variant], [])
    assert record.to_serializable()["variants"][0]["role"] == "baseline"


def test_bounded_residual_fusion_starts_as_exact_final_and_updates_every_gate() -> None:
    final = torch.randn(2, 8, 4, 4)
    intermediates = torch.randn(2, 3, 8, 4, 4)
    fusion = BoundedResidualLayerFusion(8)
    fused = fusion(final, intermediates)
    assert torch.equal(fused, final.float())
    fused.square().mean().backward()
    assert fusion.raw_gates.grad is not None
    assert torch.isfinite(fusion.raw_gates.grad).all()
    assert (fusion.raw_gates.grad != 0).all()


def test_bounded_residual_fusion_contains_fixed_four_layer_average() -> None:
    final = torch.randn(2, 8, 4, 4)
    intermediates = torch.randn(2, 3, 8, 4, 4)
    fusion = BoundedResidualLayerFusion(8)
    with torch.no_grad():
        fusion.raw_gates.fill_(torch.atanh(torch.tensor(0.75)))
    expected = torch.cat((final.unsqueeze(1), intermediates), dim=1).float().mean(dim=1)
    assert torch.allclose(fusion(final, intermediates), expected, rtol=1e-6, atol=1e-6)
    assert bool((fusion.effective_coefficients().abs() <= 1 / 3).all())


def test_residual_fusion_probe_adds_exactly_three_parameters_and_rejects_bad_input() -> None:
    baseline = DenseGeometryProbe(8)
    candidate = ResidualFusionGeometryProbe(8)
    baseline_parameters = sum(value.numel() for value in baseline.parameters())
    candidate_parameters = sum(value.numel() for value in candidate.parameters())
    assert candidate_parameters == baseline_parameters + 3
    features = torch.randn(2, 4, 8, 4, 4)
    log_depth, logvar = candidate(features)
    assert log_depth.shape == logvar.shape == (2, 4, 4)
    with torch.no_grad():
        candidate.fusion.raw_gates.fill_(10)
    assert all(abs(float(value)) <= 1 / 3 + 1e-7 for value in candidate.fusion.effective_coefficients())
    with pytest.raises(ValueError, match="expected"):
        candidate(torch.randn(2, 3, 8, 4, 4))
    features[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        candidate(features)
