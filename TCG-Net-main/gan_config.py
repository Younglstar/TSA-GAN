# file: gan_config.py (最终优化版)

from dataclasses import dataclass, field
from typing import List


@dataclass
class GANConfig:
    INPUT_DIM: int = 64
    NOISE_DIM: int = 128
    # 生成器不再使用hidden_dims，但为保持兼容性，我们保留它
    GENERATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128,256,512])
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [ 256,256,128])

    # --- 映射网络参数 ---
    W_DIM: int = 64
    MAPPING_HIDDEN_LAYERS: int = 2
    MAPPING_HIDDEN_DIM: int = 64

    # --- 多样性损失权重 (可以适当提高) ---
    DIVERSITY_LAMBDA: float = 0.5
    FM_LOSS_WEIGHT: float = 1            # Feature Matching loss 权重
    MOMENT_MATCHING_WEIGHT: float = 2.0     # 矩匹配（含偏度峰度）权重
    MOMENT_ORDERS: List[int] = field(default_factory=lambda: [1, 2, 3])  # 对齐到四阶

    # --- 训练参数 (核心修改) ---
    # 恢复一个相对平衡的学习率
    LEARNING_RATE_G: float = 0.00001
    LEARNING_RATE_D: float = 0.00002
    LR_DECAY_EPOCHS: int = 100
    LR_DECAY_FACTOR = 0.5  # 学习率衰减倍率
    LR_MULT = 2   # 每次重启周期扩大倍率
    LR_MIN = 1e-6  # 最低学习率
    # 调整Adam优化器的beta1参数，这在GAN训练中很常见，可以增加稳定性
    ADAM_BETA1: float = 0.0
    ADAM_BETA2: float = 0.9  # 每多少个 epoch 衰减一次


    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 12000
    N_CRITIC: int = 3
    GRAD_PENALTY_WEIGHT: float = 10.0

    # 定义要使用的增强策略，用逗号分隔。可选: 'noise', 'cutout'
    AUGMENTATION_POLICY = "noise,cutout,mixup,jitter"   # ,time_warp
    AUGMENT_NOISE_STD: float = 0.02    # 'noise' 策略的参数：噪声的标准差
    # 'cutout' 策略的参数：遮挡窗口大小占总序列长度的比例
    AUGMENT_CUTOUT_RATIO: float = 0.05
    AUGMENTATION_FACTOR: int = 8

    # ... 其他参数 ...
    DROPOUT_RATE: float = 0.2
    INSTANCE_NOISE_STD_INIT: float = 0.05   # 判别器输入“实例噪声”起始标准差
    INSTANCE_NOISE_STD_FINAL: float = 0.0   # 线性衰减到 0
    MIXUP_PROB: float = 0.1                # real/fake 小概率mixup，WGAN下的“软标签”替代
    MIXUP_ALPHA: float = 0.2                # mixup强度

    # —— 判别器稳定性 ——
    USE_SPECTRAL_NORM: bool = False         # 判别器线性层加谱归一化

    # —— (可选) DP-SGD差分隐私 ——
    DP_ENABLE: bool = True               # 先默认False，确认性能后再开
    DP_MAX_GRAD_NORM: float = 1.0
    DP_NOISE_MULTIPLIER: float = 0.8
    DP_TARGET_EPSILON: float = 10.0
    DP_TARGET_DELTA: float = 1e-4

    ENCODED_DATA_PATH: str = 'cvae_encoded_data.pkl'
    MODEL_SAVE_PATH: str = 'gan_final_model.pkl'
    SYNTHETIC_DATA_PATH: str = 'synthetic_final_data.pkl'
    NUM_SYNTHETIC_SAMPLES: int = 1000

