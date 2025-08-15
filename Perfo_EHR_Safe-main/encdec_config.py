# file: encdec_config.py (修改后以适配CausalVAE)

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EncoderDecoderConfig:
    # --- TCN 核心架构参数 (保持不变) ---
    TCN_CHANNELS: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    LATENT_DIM: int = 64

    # --- 新增：因果头 (Causal Head) 参数 ---
    # 这个值代表了数据中的总特征数 (静态+时序)，用于构建 N x N 的因果矩阵。
    # 我们在这里先设置一个占位符，实际值将在运行时由DataProcessor确定。
    NUM_TOTAL_FEATURES: int = 70  # 这是一个示例值，请根据您的数据调整或在代码中动态设置

    # --- 新增：VAE 损失函数权重 ---
    # KL散度损失的权重，用于平衡重构损失和潜在空间正则化
    BETA_KL: float = 1.0
    # 因果损失的权重，用于平衡重构损失和因果矩阵学习
    GAMMA_CAUSAL: float = 1.0

    # --- 训练参数 ---
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    DROPOUT_RATE: float = 0.2

    # --- 文件路径 (更新为新版本) ---
    MODEL_SAVE_PATH: str = 'causal_vae_model.pkl'
    ENCODED_DATA_PATH: str = 'causal_vae_encoded_data.pkl'