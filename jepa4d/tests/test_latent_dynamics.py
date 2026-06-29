import torch

from jepa4d.models.latent_dynamics import ActionConditionedLatentDynamics
from jepa4d.planning.latent_mpc import CEMConfig, CEMPlanner


def test_mock_dynamics_preserves_interface() -> None:
    output = ActionConditionedLatentDynamics()(torch.randn(2, 5, 8), torch.randn(2, 3))
    assert output.next_tokens.shape == (2, 5, 8)
    assert output.uncertainty.shape == (2, 5)
    assert output.value.shape == (2,)


def test_learned_dynamics_is_action_and_proprioception_conditioned() -> None:
    model = ActionConditionedLatentDynamics(8, 3, 2, hidden_dim=16, backend="learned")
    tokens = torch.randn(2, 5, 8)
    output = model(tokens, torch.randn(2, 3), torch.randn(2, 2))
    assert output.next_tokens.shape == tokens.shape
    assert torch.isfinite(output.next_tokens).all()
    assert torch.all((output.uncertainty >= 0) & (output.uncertainty <= 1))
    output.next_tokens.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cem_plan_is_bounded_reproducible_and_uncertainty_aware() -> None:
    dynamics = ActionConditionedLatentDynamics()
    config = CEMConfig(horizon=3, population=32, iterations=3, seed=7)
    planner = CEMPlanner(2, config)
    tokens = torch.zeros(1, 4, 8)
    first = planner.plan(tokens, dynamics)
    second = planner.plan(tokens, dynamics)
    assert torch.equal(first.actions, second.actions)
    assert first.actions.shape == (3, 2)
    assert torch.all((first.actions >= -1) & (first.actions <= 1))
    assert 0 <= first.predicted_uncertainty <= 1
