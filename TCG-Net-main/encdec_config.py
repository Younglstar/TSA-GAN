# file: encdec_config.py (最终优化版)

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EncoderDecoderConfig:
    # --- TCN 核心架构参数 ---
    TCN_CHANNELS: List[int] = field(default_factory=lambda: [64, 128, 256])
    LATENT_DIM: int = 64
    NUM_TOTAL_FEATURES: int = 70

    # --- VAE 损失函数权重 ---
    BETA_KL_FINAL: float = 1.0
    GAMMA_CAUSAL: float = 1.0

    # --- 周期性KL退火 (Cyclical Annealing) 参数 ---
    # 我们将一个“工作-休假”周期的长度固定为50轮
    KL_ANNEALING_CYCLE_EPOCHS: int = 50

    # --- 训练参数 ---
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 300 # 保持一个较长的总轮数
    DROPOUT_RATE: float = 0.2

    # --- 学习率调度器和早停法参数 (核心修改) ---
    SCHEDULER_PATIENCE: int = 15 # 调度器的耐心可以适当增加
    SCHEDULER_FACTOR: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 50 # <-- 大幅增加早停的耐心

    # --- 文件路径 ---
    MODEL_SAVE_PATH: str = 'cvae_model.pkl' # 更新名称
    ENCODED_DATA_PATH: str = 'cvae_encoded_data.pkl'