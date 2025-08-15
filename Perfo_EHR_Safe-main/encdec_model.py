# file: encdec_model.py (重构为CausalVAE)

import torch
import torch.nn as nn
from tcn import TemporalConvNet  # 确保 tcn.py 在您的项目中


# ==============================================================================
# == 新的CVAE编码器，包含潜在头和因果头 ==
# ==============================================================================

class _CVAE_Encoder(nn.Module):
    def __init__(self, input_dim, tcn_channels, latent_dim, num_total_features, dropout):
        super().__init__()

        # 1. 共享的TCN主干网络
        self.tcn_backbone = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )

        last_channel_dim = tcn_channels[-1]

        # 2. 并行的输出头
        # 2a. VAE的潜在分布头
        self.latent_head_mu = nn.Linear(last_channel_dim, latent_dim)
        self.latent_head_logvar = nn.Linear(last_channel_dim, latent_dim)

        # 2b. 因果邻接矩阵头
        # 注意：这里的隐藏层可以根据需要添加
        self.causal_head = nn.Sequential(
            nn.Linear(last_channel_dim, last_channel_dim // 2),
            nn.ReLU(),
            nn.Linear(last_channel_dim // 2, num_total_features * num_total_features),
            nn.Sigmoid()  # 输出概率
        )
        self.num_total_features = num_total_features

    def forward(self, x):
        # x: (N, L, C_in)
        x = x.permute(0, 2, 1)  # -> (N, C_in, L)

        # 通过TCN主干网络
        h_sequence = self.tcn_backbone(x)  # -> (N, C_out, L)

        # 全局平均池化得到摘要向量
        h_summary = torch.mean(h_sequence, dim=2)  # -> (N, C_out)

        # 通过两个并行的头
        mu = self.latent_head_mu(h_summary)
        logvar = self.latent_head_logvar(h_summary)

        A_pred_flat = self.causal_head(h_summary)
        A_pred = A_pred_flat.view(-1, self.num_total_features, self.num_total_features)

        return mu, logvar, A_pred


# ==============================================================================
# == 新的CVAE解码器，可以重构数据和掩码 ==
# ==============================================================================

class _CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, tcn_channels, output_dim, sequence_length, dropout):
        super().__init__()
        self.sequence_length = sequence_length
        tcn_input_channel = tcn_channels[0]

        # 初始全连接层，用于投影和序列展开
        self.fc_initial = nn.Linear(latent_dim, tcn_input_channel * sequence_length)

        decoder_tcn_channels = list(reversed(tcn_channels))

        # TCN网络用于精炼时间模式
        self.tcn_refiner = TemporalConvNet(
            num_inputs=tcn_input_channel,
            num_channels=decoder_tcn_channels,
            kernel_size=3,
            dropout=dropout
        )

        last_channel_dim = decoder_tcn_channels[-1]

        # 多任务输出头
        # 3a. 重构原始数据 X
        self.output_head_data = nn.Conv1d(last_channel_dim, output_dim, 1)
        # 3b. 重构掩码 M
        self.output_head_mask = nn.Sequential(
            nn.Conv1d(last_channel_dim, output_dim, 1),
            nn.Sigmoid()  # 掩码是0到1的概率
        )

    def forward(self, z):
        # z: (N, latent_dim)
        x = self.fc_initial(z)
        x = x.view(z.size(0), -1, self.sequence_length)  # -> (N, C_inter, L)

        refined_sequence = self.tcn_refiner(x)  # -> (N, C_out, L)

        # 通过两个输出头
        reconstructed_data = self.output_head_data(refined_sequence).permute(0, 2, 1)  # -> (N, L, D_out)
        reconstructed_mask = self.output_head_mask(refined_sequence).permute(0, 2, 1)  # -> (N, L, D_out)

        return reconstructed_data, reconstructed_mask


# ==============================================================================
# == 最终的CausalVAE模型，整合了编码器和解码器 ==
# ==============================================================================

class CausalVAE(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_length, tcn_channels, latent_dim, num_total_features,
                 dropout=0.2):
        super().__init__()

        self.encoder = _CVAE_Encoder(
            input_dim=input_dim,
            tcn_channels=tcn_channels,
            latent_dim=latent_dim,
            num_total_features=num_total_features,
            dropout=dropout
        )

        self.decoder = _CVAE_Decoder(
            latent_dim=latent_dim,
            tcn_channels=tcn_channels,
            output_dim=output_dim,
            sequence_length=sequence_length,
            dropout=dropout
        )

    def reparameterize(self, mu, logvar):
        """
        执行重参数化技巧 VAE的核心
        z = μ + σ * ε, 其中 ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, x):
        # 1. 编码
        mu, logvar, A_pred = self.encoder(x)

        # 2. 从学习到的分布中采样z
        z = self.reparameterize(mu, logvar)

        # 3. 解码
        reconstructed_data, reconstructed_mask = self.decoder(z)

        # 返回所有用于计算复合损失函数的部分
        return reconstructed_data, reconstructed_mask, mu, logvar, A_pred