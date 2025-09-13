# encdec_config.py (修改后)
from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderDecoderConfig:
    """
    (已重写)
    本配置文件现在只包含与 CausalVAE 模型本身及其训练过程相关的超参数。
    输入数据的维度等信息将在运行时从数据中动态获取。
    """

    # --- 1. 模型核心架构参数 ---
    # TCN (时间卷积网络) 的通道数定义了网络的深度和宽度
    TCN_CHANNELS: List[int] = field(default_factory=lambda: [64, 128,256])
    # 潜在空间的维度，即数据被压缩到的维度大小
    LATENT_DIM: int = 64

    # --- 2. VAE 损失函数权重 ---
    # KL 散度项的最大权重，用于平衡重建损失和正则化
    BETA_KL_FINAL: float = 0.05  # <--- 修改：从一个更小、更合理的值开始
    # 因果关系损失项的权重
    GAMMA_CAUSAL: float = 2.0
    # 因果损失的目标值
    CAUSAL_TARGET_VALUE: float = 0.02  # <--- 新增：将硬编码的目标值移入配置
    # "Free Bits" 技术的阈值，为KL损失提供一个“豁免区”
    FREE_BITS_THRESHOLD: float = 0.02  # <--- 新增：引入Free Bits以防止KL完全消失

    # --- 3. 周期性KL退火 (Cyclical Annealing) 参数 ---
    # 在训练开始时，完全不使用KL损失的预热期（以 epoch 为单位）
    KL_ANNEALING_WARMUP_EPOCHS: int = 10  # <--- 新增：引入预热期，稳定初期训练
    # KL 权重在一个周期内从0增长到 BETA_KL_FINAL 所需的长度
    KL_ANNEALING_CYCLE_EPOCHS: int = 50  # <--- 修改：适当延长周期，使增长更平缓

    # --- 4. 训练过程参数 ---
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 300
    DROPOUT_RATE: float = 0.3


    # --- 5. 学习率调度器和早停法参数 ---
    # 当验证损失在 SCHEDULER_PATIENCE 个 epoch 内没有改善时，降低学习率
    SCHEDULER_PATIENCE: int = 15
    SCHEDULER_FACTOR: float = 0.1
    # 当验证损失在 EARLY_STOPPING_PATIENCE 个 epoch 内没有改善时，提前终止训练
    EARLY_STOPPING_PATIENCE: int = 40  # <--- 修改：收紧早停耐心值，避免浪费算力

    # --- 6. 文件路径 ---
    MODEL_SAVE_PATH: str = 'cvae_model.pkl'
    ENCODED_DATA_PATH: str = 'cvae_encoded_data.pkl'
    # 用于保存数据维度信息的文件，将在 processor 中创建
    FEATURE_DIMS_FILE: str = 'feature_dims.pkl'
    EMBEDDED_DATA_PATH : str = 'categorical_embeddings.pkl'