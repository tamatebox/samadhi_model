import pytest
import torch
from src.presets.tabular import create_mlp_samadhi


@pytest.fixture
def tabular_config():
    return {
        "dim": 32,
        "input_dim": 64,
        "output_dim": 64,  # Same as input for reconstruction
        "n_probes": 5,
        "gate_threshold": -1.0,
        "refine_steps": 2,
        "vicara_type": "standard",
    }


def test_create_mlp_samadhi(tabular_config):
    model = create_mlp_samadhi(tabular_config)
    assert model is not None

    # Check forward pass
    x = torch.randn(2, tabular_config["input_dim"])
    output, s_final, meta = model(x)

    assert output.shape == (2, tabular_config["input_dim"])
    assert s_final.shape == (2, tabular_config["dim"])
