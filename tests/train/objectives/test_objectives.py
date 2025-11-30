import pytest
import torch
import torch.nn as nn
from src.train.objectives.supervised_regression import SupervisedRegressionObjective
from src.train.objectives.unsupervised import UnsupervisedObjective


class MockConfig(dict):
    def __getattr__(self, name):
        return self.get(name)


@pytest.fixture
def mock_config():
    return MockConfig(
        {"n_probes": 5, "stability_coeff": 0.1, "entropy_coeff": 0.1, "balance_coeff": 0.1, "refine_steps": 2}
    )


@pytest.fixture
def dummy_data():
    batch_size = 4
    dim = 10
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    s0 = torch.randn(batch_size, dim)
    s_final = torch.randn(batch_size, dim)
    decoded = torch.randn(batch_size, dim)

    # Mock metadata
    probs = torch.softmax(torch.randn(batch_size, 5), dim=1)
    s_history = [torch.randn(batch_size, dim) for _ in range(3)]

    metadata = {"probs": probs, "s_history": s_history}

    return x, y, s0, s_final, decoded, metadata


def test_supervised_regression_objective(mock_config, dummy_data):
    objective = SupervisedRegressionObjective(mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_data

    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0
    assert "recon_loss" in components
    assert "stability_loss" in components
    assert "entropy_loss" in components
    assert "balance_loss" in components


def test_supervised_regression_objective_no_target(mock_config, dummy_data):
    objective = SupervisedRegressionObjective(mock_config, device="cpu")
    x, _, s0, s_final, decoded, metadata = dummy_data

    with pytest.raises(ValueError):
        objective.compute_loss(
            x=x,
            y=None,
            s0=s0,
            s_final=s_final,
            decoded_s_final=decoded,
            metadata=metadata,
            num_refine_steps=mock_config.refine_steps,
        )


def test_unsupervised_objective(mock_config, dummy_data):
    objective = UnsupervisedObjective(mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_data

    # In unsupervised, y should be ignored, but we pass it to verify it doesn't crash
    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0
    # Unsupervised uses x as target for recon_loss
    expected_recon = nn.MSELoss()(decoded, x).item()
    assert abs(components["recon_loss"] - expected_recon) < 1e-6
