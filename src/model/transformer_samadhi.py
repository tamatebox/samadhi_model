from typing import Dict, Any
import torch
import torch.nn as nn
import math
from src.model.samadhi import SamadhiModel


class TransformerSamadhiModel(SamadhiModel):
    """
    Transformer Samadhi Model (Sequence Samadhi with Attention).

    Uses Transformer Encoder to compress sequence into latent vector,
    and Transformer Decoder (or MLP) to reconstruct it.
    """

    def __init__(self, config: Dict[str, Any]):
        self.input_dim = config.get("input_dim")
        self.seq_len = config.get("seq_len")

        if self.input_dim is None or self.seq_len is None:
            raise ValueError("Config must contain 'input_dim' and 'seq_len'.")

        super().__init__(config)

        # Replace Adapters
        self.vitakka.adapter = self._build_transformer_adapter()
        # Decoder built by base class via _build_decoder

    def _build_transformer_adapter(self) -> nn.Module:
        return TransformerEncoderAdapter(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            hidden_dim=self.config.get("adapter_hidden_dim", 128),
            latent_dim=self.dim,
            num_layers=self.config.get("transformer_layers", 2),
            nhead=self.config.get("transformer_heads", 4),
        )

    def _build_decoder(self) -> nn.Module:
        # For simplicity, using a simple MLP decoder that maps latent -> full sequence
        # Alternatively, could use a proper autoregressive Transformer Decoder,
        # but for anomaly detection reconstruction, this often suffices and is faster.
        return SimpleSequenceDecoder(
            latent_dim=self.dim,
            seq_len=self.seq_len,
            output_dim=self.input_dim,
            hidden_dim=self.config.get("adapter_hidden_dim", 128),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderAdapter(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, latent_dim, num_layers=2, nhead=4):
        super().__init__()
        # Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len + 100)

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Flatten and map to latent
        # Aggregation strategy: Average Pooling or [CLS] token equivalent.
        # Using Average Pooling for simplicity
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: (Batch, Seq, Input)
        x = self.input_proj(x)  # (Batch, Seq, Hidden)
        x = self.pos_encoder(x)

        # Transformer Pass
        x = self.transformer_encoder(x)  # (Batch, Seq, Hidden)

        # Pooling (Average over sequence)
        x = x.mean(dim=1)  # (Batch, Hidden)

        z = self.to_latent(x)
        return self.activation(z)


class SimpleSequenceDecoder(nn.Module):
    """
    Decodes latent z back to (Batch, Seq, Input)
    """

    def __init__(self, latent_dim, seq_len, output_dim, hidden_dim):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        # Map latent to flattened sequence size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, seq_len * output_dim)
        )

    def forward(self, z):
        # z: (Batch, Latent)
        out = self.fc(z)  # (Batch, Seq * Output)
        out = out.view(-1, self.seq_len, self.output_dim)
        return out
