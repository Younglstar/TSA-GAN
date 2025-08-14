# encdec_model.py
import torch
import torch.nn as nn
from tcn import TemporalConvNet  # 导入我们新的TCN模块


# file: encdec_model.py


# 在 train_encdec.py 或 encdec_model.py 中修改/替换

# ... (其他 import) ...
from tcn import TemporalConvNet  # 导入我们新创建的TCN模块

# file: encdec_model.py (推荐的文件名)

import torch
import torch.nn as nn
from tcn import TemporalConvNet  # 确保 tcn.py 在您的项目中


# ----------------------------------------------------------------------------
# 我们将 Encoder 和 Decoder 定义为内部类，以保持代码的模块化和清晰性。
# 它们不会被直接从外部调用，而是作为 TCNAutoencoder 的一部分。
# ----------------------------------------------------------------------------

class _TCNEncoder(nn.Module):
    """
    内部编码器模块 (Internal Encoder Module).
    """

    def __init__(self, input_dim, tcn_channels, latent_dim, dropout):
        super(_TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        self.fc = nn.Linear(tcn_channels[-1], latent_dim)

    def forward(self, x):
        # 输入 x: (N, L, C_in)
        # TCN 需要 (N, C_in, L)
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)
        # 取最后一个时间步的输出并映射到潜在空间
        latent_representation = self.fc(tcn_out[:, :, -1])
        return latent_representation


class _TCNDecoder(nn.Module):
    """
    内部解码器模块 (Internal Decoder Module).
    """

    def __init__(self, latent_dim, tcn_channels, output_dim, sequence_length, dropout):
        super(_TCNDecoder, self).__init__()
        self.sequence_length = sequence_length
        # TCN输入通道应与编码器TCN的最后一个通道匹配，但为了灵活性，我们在这里独立定义
        tcn_input_channel = tcn_channels[0]

        self.fc = nn.Linear(latent_dim, tcn_input_channel * sequence_length)

        # 解码器的TCN通道可以是编码器的逆序
        decoder_tcn_channels = list(reversed(tcn_channels))

        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_channel,
            num_channels=decoder_tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        self.out_conv = nn.Conv1d(decoder_tcn_channels[-1], output_dim, 1)

    def forward(self, z):
        # z: (N, latent_dim)
        x = self.fc(z)
        # 重塑为TCN的输入格式 (N, C, L)
        x = x.view(z.size(0), -1, self.sequence_length)
        x = self.tcn(x)
        x = self.out_conv(x)
        # 转换回 (N, L, C_out)
        x = x.permute(0, 2, 1)
        return x


# ----------------------------------------------------------------------------
# 这是您将在 train_encdec.py 中直接使用的主要模型类。
# ----------------------------------------------------------------------------

class TCNAutoencoder(nn.Module):
    """
    一个集成了TCN编码器和解码器的自编码器模型。
    """

    def __init__(self, input_dim, output_dim, sequence_length, tcn_channels, latent_dim, dropout=0.2):
        """
        Args:
            input_dim (int): 输入序列中每个时间步的特征数量。
            output_dim (int): 输出序列中每个时间步的特征数量。
            sequence_length (int): 序列的长度。
            tcn_channels (list): TCN每个残差块的输出通道列表。
            latent_dim (int): 潜在空间的维度。
            dropout (float): Dropout概率。
        """
        super(TCNAutoencoder, self).__init__()

        self.encoder = _TCNEncoder(
            input_dim=input_dim,
            tcn_channels=tcn_channels,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.decoder = _TCNDecoder(
            latent_dim=latent_dim,
            tcn_channels=tcn_channels,  # 注意：这里我们将原始通道列表传入
            output_dim=output_dim,
            sequence_length=sequence_length,
            dropout=dropout
        )

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (Batch_Size, Sequence_Length, Input_Dim)。

        Returns:
            tuple: 包含重构数据和潜在表示的元组 (reconstructed_x, latent_z)。
        """
        # 编码过程
        latent_z = self.encoder(x)

        # 解码过程
        reconstructed_x = self.decoder(latent_z)

        return reconstructed_x, latent_z

