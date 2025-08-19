# file: gan_models.py (最终升级版)

import torch
import torch.nn as nn
from typing import List


# --- 新增模块 (保持不变) ---
class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, hidden_layers: int, hidden_dim: int):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(hidden_layers - 1): layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)])
        layers.append(nn.Linear(hidden_dim, w_dim));
        self.mapping = nn.Sequential(*layers)

    def forward(self, z): return self.mapping(z)


class SelfAttention(nn.Module):
    # ... (这个类保持不变，为了简洁此处省略)
    def __init__(self, in_dim: int):
        super().__init__();
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1));
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, length = x.size();
        proj_query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, length);
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy);
        proj_value = self.value_conv(x).view(batch_size, -1, length)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1));
        out = out.view(batch_size, C, length)
        out = self.gamma * out + x;
        return out, attention


# ==============================================================================
# == 核心修改：全新设计的、基于转置卷积的生成器 ==
# ==============================================================================
# file: gan_models.py (只替换 Generator 类)

class Generator(nn.Module):
    def __init__(self, w_dim: int, output_dim: int, hidden_channels: int = 128):
        super().__init__()

        # 定义一个转置卷积块
        def block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(True)
            )

        # --- 核心修改：将网络分为注意力前和注意力后两部分 ---
        self.part1 = nn.Sequential(
            # block 1: (B, w_dim, 1) -> (B, hidden_channels * 4, 4)
            block(w_dim, hidden_channels * 4, 4, 1, 0),
            # block 2: (B, hidden_channels * 4, 4) -> (B, hidden_channels * 2, 8)
            block(hidden_channels * 4, hidden_channels * 2),
        )

        # 将自注意力层单独定义
        self.attention = SelfAttention(hidden_channels * 2)

        self.part2 = nn.Sequential(
            # block 3: (B, hidden_channels * 2, 8) -> (B, hidden_channels, 16)
            block(hidden_channels * 2, hidden_channels),
            # 最终输出层
            nn.ConvTranspose1d(hidden_channels, 1, 4, 2, 1),  # -> (B, 1, 32)
            nn.Flatten(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )
        # --- 修正结束 ---

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w 的形状是 (B, w_dim)
        x = w.unsqueeze(2)  # -> (B, w_dim, 1)

        # --- 核心修改：手动串联网络，并只取注意力模块的第一个输出 ---
        x = self.part1(x)
        x, attention_map = self.attention(x)  # 调用独立的自注意力层
        x = self.part2(x)  # 将正确的张量x传递给后续层
        # --- 修正结束 ---

        return x

# --- 判别器保持不变 ---
class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)