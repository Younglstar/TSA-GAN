# gan_config.py (已修改，移除硬编码的输入维度)
from dataclasses import dataclass, field
from typing import List

@dataclass
class GANConfig:
    """
    (已重写)
    本配置文件现在只包含与 GAN 模型本身及其训练过程相关的超参数。
    GAN 的输入维度 (INPUT_DIM) 将在训练脚本中通过加载编码数据来动态确定，
    以确保与 CausalVAE 阶段的输出维度 (LATENT_DIM) 自动保持一致。
    """

    # --- 1. 模型核心架构参数 ---
    # 从中采样以生成数据的噪声向量的维度
    NOISE_DIM: int = 128
    # 判别器的隐藏层维度
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512, 512, 256])

    # --- 2. 映射网络参数 (用于 StyleGAN 风格的生成器) ---
    # StyleGAN中的中间潜在空间 W 的维度
    W_DIM: int = 128
    # 映射网络的层数
    MAPPING_HIDDEN_LAYERS: int = 4
    # 映射网络的隐藏层维度
    MAPPING_HIDDEN_DIM: int = 128

    # --- 3. 多样性损失权重 (用于 StyleGAN 风格的生成器) ---
    DIVERSITY_LAMBDA: float = 0.1

    # --- 4. 训练过程参数 ---
    LEARNING_RATE_G: float = 0.00002  # 生成器的学习率
    LEARNING_RATE_D: float = 0.00001  # 判别器的学习率
    ADAM_BETA1: float = 0.5  # Adam 优化器的 beta1 参数，0.5 有助于稳定 GAN 训练
    ADAM_BETA2: float = 0.9

    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 1000
    N_CRITIC: int = 5  # 每次更新生成器前，判别器要训练的次数 (WGAN-GP)
    GRAD_PENALTY_WEIGHT: float = 10.0  # 梯度惩罚的权重 (WGAN-GP)
    DROPOUT_RATE: float = 0.2

    # --- 5. 文件路径和样本数量 ---
    # 输入文件：来自 CausalVAE 阶段的编码后数据
    ENCODED_DATA_PATH: str = 'cvae_encoded_data.pkl'
    MODEL_SAVE_PATH: str = 'gan_final_model.pkl'
    SYNTHETIC_DATA_PATH: str = 'synthetic_final_data.pkl'
    # 最终要生成的合成样本数量
    NUM_SYNTHETIC_SAMPLES: int = 1000