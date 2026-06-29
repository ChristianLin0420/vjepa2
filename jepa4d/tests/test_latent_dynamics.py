import torch

from jepa4d.models.latent_dynamics import ActionConditionedLatentDynamics


def test_mock_dynamics_preserves_interface() -> None:
    output = ActionConditionedLatentDynamics()(torch.randn(2, 5, 8), torch.randn(2, 3))
    assert output.next_tokens.shape == (2, 5, 8)
    assert output.uncertainty.shape == (2, 5)
    assert output.value.shape == (2,)
