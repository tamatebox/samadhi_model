from dataclasses import dataclass
from src.configs.base import BaseConfig
from src.configs.enums import DecoderType


@dataclass
class BaseDecoderConfig(BaseConfig):
    dim: int = 64
    type: DecoderType = DecoderType.RECONSTRUCTION


@dataclass
class ReconstructionDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.RECONSTRUCTION
    input_dim: int = 10  # Target dimension (usually same as input)
    decoder_hidden_dim: int = 64


@dataclass
class CnnDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.CNN
    channels: int = 3
    img_size: int = 32
    decoder_hidden_dim: int = 64


@dataclass
class LstmDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.LSTM
    output_dim: int = 10
    seq_len: int = 50
    decoder_hidden_dim: int = 128
    lstm_layers: int = 1


@dataclass
class SimpleSequenceDecoderConfig(BaseDecoderConfig):
    type: DecoderType = DecoderType.SIMPLE_SEQUENCE
    output_dim: int = 10
    seq_len: int = 50
    decoder_hidden_dim: int = 128
