# file: gan_config.py (最终优化版)

from dataclasses import dataclass, field
from typing import List


@dataclass
class GANConfig:
    INPUT_DIM: int = 64
    NOISE_DIM: int = 128
    # 生成器不再使用hidden_dims，但为保持兼容性，我们保留它
    GENERATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [])
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [512,512, 256])  # 判别器可以适当简化

    # --- 映射网络参数 ---
    W_DIM: int = 128
    MAPPING_HIDDEN_LAYERS: int = 4
    MAPPING_HIDDEN_DIM: int = 128

    # --- 多样性损失权重 (可以适当提高) ---
    DIVERSITY_LAMBDA: float = 0.1

    # --- 训练参数 (核心修改) ---
    # 恢复一个相对平衡的学习率
    LEARNING_RATE_G: float = 0.0002
    LEARNING_RATE_D: float = 0.0001
    # 调整Adam优化器的beta1参数，这在GAN训练中很常见，可以增加稳定性
    ADAM_BETA1: float = 0.5
    ADAM_BETA2: float = 0.9

    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 2000
    N_CRITIC: int = 5
    GRAD_PENALTY_WEIGHT: float = 15.0

    # ... 其他参数 ...
    DROPOUT_RATE: float = 0.2
    ENCODED_DATA_PATH: str = 'cvae_encoded_data.pkl'
    MODEL_SAVE_PATH: str = 'gan_final_model.pkl'
    SYNTHETIC_DATA_PATH: str = 'synthetic_final_data.pkl'
    NUM_SYNTHETIC_SAMPLES: int = 1000