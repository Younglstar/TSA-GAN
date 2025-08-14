# file: gan_config.py (修改后)

from dataclasses import dataclass, field
from typing import List


@dataclass
class GANConfig:
    # --- 核心架构维度 ---
    # GAN的输入是上一阶段生成的 tcn_encoded_data.pkl
    # 我们需要知道这个数据的维度，这里先放一个占位符
    # 这个值应该等于 TCNAutoencoder 的 latent_dim
    INPUT_DIM: int = 64
    NOISE_DIM: int = 128
    GENERATOR_HIDDEN_DIMS: List[int] = field(
        default_factory=lambda: [256, 512, 512]  # 注意：第一个维度将用于自注意力
    )
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(
        default_factory=lambda: [512, 256, 128]
    )

    # --- 新增：映射网络 (Mapping Network) 参数 ---
    W_DIM: int = 128  # 中间潜在向量w的维度 (通常与NOISE_DIM相同)
    MAPPING_HIDDEN_LAYERS: int = 4  # 映射网络的隐藏层数量
    MAPPING_HIDDEN_DIM: int = 128  # 映射网络的隐藏层维度

    # --- 新增：自注意力 (Self-Attention) 参数 ---
    # 这两个参数用于将生成器第一个隐藏层的输出重塑为 "伪序列"
    # 必须满足: ATTENTION_CHANNELS * ATTENTION_SEQ_LEN == GENERATOR_HIDDEN_DIMS[0]
    ATTENTION_CHANNELS: int = 32  # 伪序列的通道数
    ATTENTION_SEQ_LEN: int = 8  # 伪序列的长度 (32 * 8 = 256)

    # --- 新增：多样性损失 (Diversity Loss) 权重 ---
    DIVERSITY_LAMBDA: float = 0.5  # 多样性正则化项的权重系数

    # --- 训练参数 ---
    LEARNING_RATE_G: float = 0.0002
    LEARNING_RATE_D: float = 0.0002
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 500
    N_CRITIC: int = 5  # 判别器更新次数与生成器更新次数之比
    GRAD_PENALTY_WEIGHT: float = 10.0  # WGAN-GP的梯度惩罚权重

    # --- 模型参数 ---
    DROPOUT_RATE: float = 0.2
    USE_BATCH_NORM: bool = True

    # --- 文件路径 (更新为新版本) ---
    ENCODED_DATA_PATH: str = 'tcn_encoded_data.pkl'  # GAN的输入数据
    MODEL_SAVE_PATH: str = 'gan_v2_model.pkl'
    SYNTHETIC_DATA_PATH: str = 'synthetic_v2_data.pkl'

    # --- 生成参数 ---
    NUM_SYNTHETIC_SAMPLES: int = 1000