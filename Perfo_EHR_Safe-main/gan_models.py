# file: gan_models.py (修改后)

import torch
import torch.nn as nn
from typing import List


# ==========================================================================================
# == 新增模块：我们将 MappingNetwork 和 SelfAttention 添加到这个文件的顶部 ==
# ==========================================================================================

class MappingNetwork(nn.Module):
    """
    将噪声向量z (来自正态分布) 映射到中间潜在向量w。
    这个w向量将作为生成器的输入，它的分布比z更适合学习。
    """

    def __init__(self, z_dim: int, w_dim: int, hidden_layers: int, hidden_dim: int):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)])
        layers.append(nn.Linear(hidden_dim, w_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)


class SelfAttention(nn.Module):
    """
    自注意力层。它可以帮助模型在生成序列时，考虑到序列不同部分之间的内部关联。
    """

    def __init__(self, in_dim: int, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        # 使用1x1卷积来模拟全连接层，处理序列数据更高效
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        输入 x 的形状应为 (批量大小, 通道数, 序列长度)
        例如: (64, 128, 8)
        """
        batch_size, C, length = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, length)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, length)

        out = self.gamma * out + x  # 残差连接
        return out, attention


# ==========================================================================================
# == 改造后的 Generator，它现在使用 w 作为输入，并内嵌了 SelfAttention ==
# ==========================================================================================

class Generator(nn.Module):
    def __init__(
            self,
            w_dim: int,  # <-- 输入维度从 noise_dim 变为 w_dim
            output_dim: int,
            hidden_dims: List[int],
            # 新增参数，用于将扁平的特征向量重塑为 "伪序列"
            attention_channels: int,
            attention_seq_len: int,
            dropout_rate: float = 0.2,
            use_batch_norm: bool = True
    ):
        super().__init__()

        # --- 改造点 1：输入层 ---
        # 网络的输入现在是 w_dim
        self.initial_layer = nn.Sequential(
            nn.Linear(w_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        input_dim = hidden_dims[0]

        # --- 改造点 2：在网络中间插入自注意力 ---
        # 我们需要在某个地方将1D向量重塑为3D的 "伪序列" 以应用自注意力
        # 这里，我们将第一个隐藏层的输出作为自注意力的输入
        # 我们需要确保 hidden_dims[0] == attention_channels * attention_seq_len
        if hidden_dims[0] != attention_channels * attention_seq_len:
            raise ValueError(
                f"第一个隐藏层维度 ({hidden_dims[0]}) 必须等于 attention_channels * attention_seq_len ({attention_channels * attention_seq_len})")

        self.attention_channels = attention_channels
        self.attention_seq_len = attention_seq_len
        self.self_attention = SelfAttention(in_dim=attention_channels)

        # --- 改造点 3：自注意力之后的层 ---
        # 网络的后续部分
        layers = []
        # 注意：这里的 input_dim 仍然是 hidden_dims[0]，因为自注意力的输入输出维度相同
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout_rate))

        self.later_layers = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Tanh()  # 将输出缩放到 [-1, 1]
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # 1. 初始层处理
        x = self.initial_layer(w)

        # 2. 重塑以应用自注意力
        # (B, H1) -> (B, C, L) where H1 = C * L
        x_reshaped = x.view(x.size(0), self.attention_channels, self.attention_seq_len)

        # 3. 应用自注意力
        x_attended, _ = self.self_attention(x_reshaped)

        # 4. 展平以进行后续处理
        # (B, C, L) -> (B, C * L)
        x_flat = x_attended.view(x.size(0), -1)

        # 5. 通过网络的其余部分
        x_final = self.later_layers(x_flat)

        # 6. 输出层
        return self.output_layer(x_final)


# ==========================================================================================
# == Discriminator 保持不变 ==
# ==========================================================================================
class Discriminator(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            dropout_rate: float = 0.2,
            use_batch_norm: bool = True
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)