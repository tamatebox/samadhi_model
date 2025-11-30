from dataclasses import dataclass
from src.configs.base import BaseConfig
from src.configs.enums import AdapterType


@dataclass
class BaseAdapterConfig(BaseConfig):
    dim: int = 64
    type: AdapterType = AdapterType.MLP
    dropout: float = 0.1

    def validate(self):
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")


@dataclass
class MlpAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.MLP
    input_dim: int = 10
    adapter_hidden_dim: int = 256


@dataclass
class CnnAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.CNN
    channels: int = 3
    img_size: int = 32

    def validate(self):
        super().validate()
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")


@dataclass
class LstmAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.LSTM
    input_dim: int = 10
    seq_len: int = 50
    adapter_hidden_dim: int = 128
    lstm_layers: int = 1


@dataclass
class TransformerAdapterConfig(BaseAdapterConfig):
    type: AdapterType = AdapterType.TRANSFORMER
    input_dim: int = 10
    seq_len: int = 50
    adapter_hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
