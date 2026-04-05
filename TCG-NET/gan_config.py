from dataclasses import dataclass, field
from typing import List


@dataclass
class GANConfig:
    ENCODED_DATA_PATH: str = "cvae_encoded_data.pkl"
    MODEL_SAVE_PATH: str = "gan_best_model.pt"
    LAST_MODEL_SAVE_PATH: str = "gan_last_model.pt"
    SYNTHETIC_DATA_PATH: str = "synthetic_final_data.pkl"
    TEST_DATA_PATH: str = "test_data_encoded.pkl"
    SCALING_PARAMS_PATH: str = "gan_scaling_params.pkl"
    HISTORY_SAVE_PATH: str = "gan_history.pkl"
    LOSS_CURVE_PATH: str = "gan_loss_curves.png"
    NUM_SYNTHETIC_SAMPLES: int = 200000

    NOISE_DIM: int = 64
    W_DIM: int = 32
    MAPPING_HIDDEN_LAYERS: int = 2
    MAPPING_HIDDEN_DIM: int = 64

    GENERATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [64, 64])
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128, 128])

    DROPOUT_RATE: float = 0.0
    USE_SPECTRAL_NORM: bool = True

    DIVERSITY_LAMBDA: float = 0.10
    FM_LOSS_WEIGHT: float = 0.50
    MOMENT_MATCHING_WEIGHT: float = 0.05
    MOMENT_ORDERS: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    LEARNING_RATE_G: float = 1e-4
    LEARNING_RATE_D: float = 2e-4
    ADAM_BETA1: float = 0.0
    ADAM_BETA2: float = 0.9

    BATCH_SIZE: int = 512
    NUM_EPOCHS: int = 300
    N_CRITIC: int = 3
    GRAD_PENALTY_WEIGHT: float = 5.0

    USE_SCHEDULER: bool = True
    LR_MIN: float = 1e-6

    INSTANCE_NOISE_STD_INIT: float = 0.01
    INSTANCE_NOISE_STD_FINAL: float = 0.0
    MIXUP_PROB: float = 0.0
    MIXUP_ALPHA: float = 0.2

    VALID_SPLIT: float = 0.2
    SEED: int = 3
    SAVE_EVERY_EPOCHS: int = 20

    EARLY_STOPPING_ENABLE: bool = False
    EARLY_STOPPING_PATIENCE: int = 40
    EARLY_STOPPING_MIN_DELTA: float = 1e-4
    EARLY_STOPPING_MONITOR: str = "wasserstein_dist"
    EARLY_STOPPING_MODE: str = "min"

    DP_ENABLE: bool = False
    DP_MAX_GRAD_NORM: float = 1.0
    DP_NOISE_MULTIPLIER: float = 0.8
    DP_TARGET_EPSILON: float = 10.0
    DP_TARGET_DELTA: float = 1e-4

    # ===== checkpoint / resume =====
    CHECKPOINT_DIR: str = "./cache_gan/checkpoints"
    SAVE_LAST_EVERY_EPOCH: bool = True
    SAVE_EVERY_N_EPOCHS: int = 5
    SAVE_BEST: bool = True
    BEST_METRIC: str = "g_loss"   # 可选: g_loss / d_loss / wasserstein_dist
    BEST_MODE: str = "min"        # g_loss/d_loss 用 min, wasserstein_dist 常用 max
    RESUME_TRAINING: bool = False
    RESUME_PATH: str = ""
    RESET_OPTIMIZER_ON_RESUME: bool = False
