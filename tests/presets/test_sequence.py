import pytest
import torch
from src.presets.sequence import create_lstm_samadhi, create_transformer_samadhi


@pytest.fixture
def sequence_config():
    return {
        "dim": 32,
        "input_dim": 10,
        "seq_len": 20,
        "n_probes": 5,
        "gate_threshold": -1.0,
        "refine_steps": 2,
        "vicara_type": "standard",
        "adapter_hidden_dim": 64,
        "lstm_layers": 1,
        "transformer_layers": 1,
        "transformer_heads": 2,
    }


def test_create_lstm_samadhi(sequence_config):
    model = create_lstm_samadhi(sequence_config)
    assert model is not None

    x = torch.randn(2, sequence_config["seq_len"], sequence_config["input_dim"])
    output, s_final, meta = model(x)

    assert output.shape == x.shape
    assert s_final.shape == (2, sequence_config["dim"])


def test_create_transformer_samadhi(sequence_config):
    model = create_transformer_samadhi(sequence_config)
    assert model is not None

    x = torch.randn(2, sequence_config["seq_len"], sequence_config["input_dim"])
    output, s_final, meta = model(x)

    assert output.shape == x.shape
    assert s_final.shape == (2, sequence_config["dim"])
