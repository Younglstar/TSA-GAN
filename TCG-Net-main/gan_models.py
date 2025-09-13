# file: gan_models.py (最终升级版)

import torch
import torch.nn as nn
from typing import List
from gan_config import GANConfig



class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, hidden_layers: int, hidden_dim: int):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(hidden_layers - 1): layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)])
        layers.append(nn.Linear(hidden_dim, w_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z): return self.mapping(z)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = embed_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        Q = self.query(x)  # (B, F)
        K = self.key(x)    # (B, F)
        V = self.value(x)  # (B, F)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn_weights = self.softmax(attn_scores)
        out = torch.matmul(attn_weights, V)
        return out + x  # 残差连接

class Generator(nn.Module):
    def __init__(self, w_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = w_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            # 在中间层加入注意力
            if i == len(hidden_dims) // 2:
                layers.append(SelfAttention(h_dim))
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        x = self.mlp(w)
        return self.output_layer(x)



class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            linear = nn.Linear(last, h)
            if (GANConfig, "USE_SPECTRAL_NORM", True):
                linear = nn.utils.spectral_norm(linear)
            layers += [linear, nn.LeakyReLU(0.2, inplace=False)]
            last = h

        # 把“特征提取器”和“最后一层”分开，便于 feature matching
        self.feature_extractor = nn.Sequential(*layers)
        self.final_linear = nn.Linear(last, 1)  # WGAN critic 输出

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # 返回最后一层线性层之前的“判别器特征”
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        out = self.final_linear(feat)
        return out