import torch
import torch.nn as nn
from typing import Dict, Any
from src.components.decoders.base import BaseDecoder


class CnnDecoder(BaseDecoder):
    """
    CNN Decoder for Vision tasks (Image Reconstruction).
    Converts Latent Vector (Batch, Dim) -> Image (Batch, C, H, W).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.channels = config.get("channels", 3)
        self.img_size = config.get("img_size", 32)

        feature_map_size = self.img_size // 16
        hidden_dim = 256 * feature_map_size * feature_map_size

        self.decoder = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.Unflatten(1, (256, feature_map_size, feature_map_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.decoder(s)
