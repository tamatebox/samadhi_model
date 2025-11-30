import pytest
import torch
from src.presets.vision import create_conv_samadhi


@pytest.fixture
def vision_config():
    return {
        "dim": 32,
        "channels": 3,
        "img_size": 32,
        "n_probes": 5,
        "gate_threshold": -1.0,
        "refine_steps": 2,
        "vicara_type": "standard",
    }


def test_create_conv_samadhi(vision_config):
    model = create_conv_samadhi(vision_config)
    assert model is not None

    # Check forward pass
    # Input: (Batch, Channels, Height, Width)
    x = torch.randn(2, vision_config["channels"], vision_config["img_size"], vision_config["img_size"])
    output, s_final, meta = model(x)

    # Output should match input shape for reconstruction
    assert output.shape == x.shape
    assert s_final.shape == (2, vision_config["dim"])
