import torch

from jepa4d.evaluation.comparison import ComparisonRecord, VariantResult
from jepa4d.models.geometry_student import DenseGeometryProbe, geometry_probe_loss, rgb_grid_features


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
