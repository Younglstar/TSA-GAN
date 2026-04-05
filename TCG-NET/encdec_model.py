from __future__ import annotations

import torch
import torch.nn as nn
from tcn import TemporalConvNet


class _CVAE_Encoder(nn.Module):
    """
    论文对齐版：
    - 编码器输出 mu / logvar
    - 并行因果头输出 feature-level A_logits
    - A 不参与 SEM 混合，只作为辅助监督分支
    """

    def __init__(self, input_dim, tcn_channels, latent_dim, num_graph_features, dropout):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_graph_features = int(num_graph_features)

        self.tcn_backbone = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout,
        )

        last_channel_dim = int(tcn_channels[-1])
        hidden_dim = max(last_channel_dim // 2, 16)

        self.latent_head_mu = nn.Linear(last_channel_dim, self.latent_dim)
        self.latent_head_logvar = nn.Linear(last_channel_dim, self.latent_dim)

        self.causal_head = nn.Sequential(
            nn.Linear(last_channel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_graph_features * self.num_graph_features),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.latent_head_mu.weight)
        nn.init.zeros_(self.latent_head_mu.bias)

        nn.init.xavier_uniform_(self.latent_head_logvar.weight)
        nn.init.constant_(self.latent_head_logvar.bias, -1.0)

        first = self.causal_head[0]
        final = self.causal_head[2]
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)

        # 比 -4.0 更宽松，避免 A_pred 过早塌成接近全零的弱图。
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(final.bias, -2.0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h_sequence = self.tcn_backbone(x)
        h_summary = torch.mean(h_sequence, dim=2)

        mu = self.latent_head_mu(h_summary)
        logvar = self.latent_head_logvar(h_summary)

        A_logits = self.causal_head(h_summary)
        A_logits = A_logits.view(-1, self.num_graph_features, self.num_graph_features)
        return mu, logvar, A_logits


class _CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, tcn_channels, output_dim, sequence_length, dropout):
        super().__init__()
        self.sequence_length = int(sequence_length)

        decoder_input_channels = int(tcn_channels[-1])
        self.fc_initial = nn.Linear(latent_dim, decoder_input_channels * self.sequence_length)

        decoder_tcn_channels = list(reversed(tcn_channels))
        self.tcn_refiner = TemporalConvNet(
            num_inputs=decoder_input_channels,
            num_channels=decoder_tcn_channels,
            kernel_size=3,
            dropout=dropout,
        )

        last_channel_dim = int(decoder_tcn_channels[-1])
        self.output_head_data = nn.Conv1d(last_channel_dim, output_dim, kernel_size=1)
        self.output_head_mask_logits = nn.Conv1d(last_channel_dim, output_dim, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc_initial.weight)
        nn.init.zeros_(self.fc_initial.bias)
        nn.init.xavier_uniform_(self.output_head_data.weight)
        nn.init.zeros_(self.output_head_data.bias)
        nn.init.xavier_uniform_(self.output_head_mask_logits.weight)
        nn.init.zeros_(self.output_head_mask_logits.bias)

    def forward(self, z):
        x = self.fc_initial(z)
        x = x.view(z.size(0), -1, self.sequence_length)
        refined = self.tcn_refiner(x)
        reconstructed_data = self.output_head_data(refined).permute(0, 2, 1)
        mask_logits = self.output_head_mask_logits(refined).permute(0, 2, 1)
        return reconstructed_data, mask_logits


class CausalVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        sequence_length,
        tcn_channels,
        latent_dim,
        num_total_features,
        dropout=0.2,
        logvar_min=-10.0,
        logvar_max=10.0,
    ):
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.encoder = _CVAE_Encoder(
            input_dim=input_dim,
            tcn_channels=tcn_channels,
            latent_dim=latent_dim,
            num_graph_features=num_total_features,
            dropout=dropout,
        )
        self.decoder = _CVAE_Decoder(
            latent_dim=latent_dim,
            tcn_channels=tcn_channels,
            output_dim=output_dim,
            sequence_length=sequence_length,
            dropout=dropout,
        )

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar, A_logits = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_data, mask_logits = self.decoder(z)
        return reconstructed_data, mask_logits, mu, logvar, A_logits
