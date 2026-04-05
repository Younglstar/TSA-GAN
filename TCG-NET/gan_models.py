import torch
import torch.nn as nn
from typing import List


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, hidden_layers: int, hidden_dim: int):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
        layers.append(nn.Linear(hidden_dim, w_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)


class Generator(nn.Module):
    def __init__(self, w_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = w_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        x = self.backbone(w)
        return self.output_layer(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        use_spectral_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.final_linear = nn.Linear(prev_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return self.final_linear(feat)
