from typing import Dict, Any
import torch.nn as nn
from src.model.samadhi import SamadhiModel


class MlpSamadhiModel(SamadhiModel):
    """
    MLP Samadhi Model (Tabular Samadhi).

    Designed for tabular data or flat vectors.
    Compresses input data into a latent vector using an MLP Adapter,
    performs "search (Vitakka)" and "purification (Vicara)" within that latent space,
    and reconstructs the original data using an MLP Decoder.
    """

    def __init__(self, config: Dict[str, Any]):
        # Load input dimension (feature size)
        self.input_dim = config.get("input_dim")
        if self.input_dim is None:
            raise ValueError("Config must contain 'input_dim' for MlpSamadhiModel.")

        super().__init__(config)

        # Replace Vitakka's default Adapter with an MLP-based Adapter.
        self.vitakka.adapter = self._build_mlp_adapter()

    def _build_mlp_adapter(self) -> nn.Module:
        """
        MLP Encoder for Vitakka.
        Input (input_dim) -> Latent Vector (Dim)
        """
        hidden_dim = self.config.get("adapter_hidden_dim", 256)

        return nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.dim),
            nn.Tanh(),  # Normalize to [-1, 1] for latent space stability
        )

    def _build_decoder(self) -> nn.Module:
        """
        [Override] MLP Decoder for SamadhiModel.
        Latent Vector (Dim) -> Output (input_dim)
        """
        hidden_dim = self.config.get("adapter_hidden_dim", 64)

        return nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            # Revert to linear output, as input data may not be strictly in [-1, 1] after StandardScaler.
        )
