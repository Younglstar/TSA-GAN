from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderDecoderConfig:
    # --- 1. 模型核心架构参数 ---
    TCN_CHANNELS: List[int] = field(default_factory=lambda: [32, 64, 128])
    LATENT_DIM: int = 12

    # --- 2. 损失函数权重 ---
    BETA_KL_FINAL: float = 0.003
    GAMMA_CAUSAL: float = 5.0
    DAG_LOSS_WEIGHT: float = 0.0
    CAUSAL_TARGET_VALUE: float = 0.0
    FREE_BITS_THRESHOLD: float = 0.05
    RECON_MASK_WEIGHT: float = 1.0

    # --- 2.1 latent 几何正则（新增） ---
    LAMBDA_LATENT_CORR: float = 1e-2
    LAMBDA_LATENT_VAR: float = 5e-3
    LATENT_VAR_TARGET: float = 0.8
    LATENT_REG_START_EPOCH: int = 5

    # --- 3. KL 退火参数 ---
    USE_CYCLICAL_ANNEALING: bool = False
    KL_ANNEALING_WARMUP_EPOCHS: int = 10
    KL_ANNEALING_RAMP_EPOCHS: int = 100
    KL_ANNEALING_CYCLE_EPOCHS: int = 50

    # --- 4. 训练过程参数 ---
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 300
    DROPOUT_RATE: float = 0.10
    INPUT_NOISE_STD: float = 0.05
    GRAD_CLIP_NORM: float = 1.0
    HUGE_LOSS_THRESHOLD: float = 100.0
    MAX_SKIPPED_BATCHES_PER_EPOCH: int = 1000000

    # --- 5. 调度器 / 早停 / 选模 ---
    SCHEDULER_PATIENCE: int = 12
    SCHEDULER_FACTOR: float = 0.1
    SCHEDULER_METRIC: str = "recon_loss"
    EARLY_STOPPING_PATIENCE: int = 40
    BEST_SCORE_START_EPOCH: int = 15
    RESET_BEST_ON_RESUME: bool = True

    # --- 6. 因果目标图相关 ---
    CAUSAL_TARGET_PATH: str = "./cache_encdec/A_target.npy"
    CAUSAL_TARGET_KEY: str = "A_target"
    CAUSAL_TARGET_CLAMP_01: bool = True
    CAUSAL_SUP_LOSS: str = "mse"
    CAUSAL_IGNORE_DIAGONAL: bool = True

    # --- 6.1 因果图构造参数（与 causal_target_builder.py 配套） ---
    CAUSAL_USE_ONLY_TEMPORAL_FEATURES: bool = True
    CAUSAL_STANDARDIZE_PER_SUBJECT: bool = False
    CAUSAL_MAX_LAG: int = 1
    CAUSAL_MIN_VALID_POINTS: int = 10
    CAUSAL_RIDGE_ALPHA: float = 1e-4
    CAUSAL_THRESHOLD: float = 0.01
    CAUSAL_KEEP_TOPK_PER_TARGET: int = 5
    CAUSAL_SYMMETRIZE: bool = False

    # --- 7. 编码输出相关 ---
    ENCODE_SEQUENCE: bool = False
    LAMBDA_MU_SMOOTH: float = 0.0
    LATENT_LOG_INTERVAL: int = 20

    # --- 8. 文件路径 ---
    MODEL_SAVE_PATH: str = "cvae_model.pkl"
    ENCODED_DATA_PATH: str = "cvae_encoded_data.pkl"
    FEATURE_DIMS_FILE: str = "feature_dims.pkl"
    EMBEDDED_DATA_PATH: str = "categorical_embeddings.pkl"

    # --- 9. 额外开关 ---
    MASK_LOSS_ONLY_ON_OBSERVED: bool = False
    DAG_WARMUP_EPOCHS: int = 20
    DAG_LOSS_EVERY_STEPS: int = 50
    VAL_EVERY_EPOCHS: int = 2
    DAG_USE_BATCH_MEAN: bool = True
    DAG_DISABLE_IN_VAL: bool = True

    # --- 10. processor / cache ---
    CACHE_DIR: str = "./cache_encdec"
    MAX_SEQ_LEN: int = 60
    TRUNCATE_MODE: str = "tail"
    VAL_RATIO: float = 0.2
    SEED: int = 42
    NUM_WORKERS: int = 0
    PREFETCH_FACTOR: int = 2
    PIN_MEMORY: bool = True
    PROCESSOR_CHUNK_SIZE: int = 1_000_000
    RESUME_TRAINING: bool = False
    RESUME_PATH: str = "cvae_model.pkl"
    RESET_OPTIMIZER_ON_RESUME: bool = False
    SAVE_LAST_EVERY_EPOCH: bool = True
    SAVE_EVERY_EPOCH: bool = False
    SAVE_EVERY_N_EPOCHS: int = 5
    CHECKPOINT_DIR: str = "./cache_encdec/checkpoints"

    # --- 11. AMP / 性能 / 调试 ---
    USE_AMP: bool = True
    AMP_DTYPE: str = "float16"  # "float16" or "bfloat16"
    USE_FUSED_ADAM: bool = True
    TQDM_POSTFIX_EVERY: int = 100
    TF32: bool = True

    DEBUG_CAUSAL_PRINT: bool = False
    DEBUG_CAUSAL_PRINT_EVERY: int = 20
    DEBUG_CAUSAL_PRINT_ON_VAL_ONLY: bool = True
    DEBUG_CAUSAL_PRINT_STATS: bool = False

    # --- 12. 稳定性保护 ---
    USE_SAFE_DAG_LOSS: bool = True
    A_PROB_CLAMP_MIN: float = 0.0
    A_PROB_CLAMP_MAX: float = 0.30
    LOGVAR_MIN: float = -10.0
    LOGVAR_MAX: float = 10.0
    INVALID_LOSS_PENALTY: float = 1e6
    MAX_DAG_VALUE: float = 1e4

    # --- 13. TRACE 定位 ---
    TRACE_ENABLE: bool = False
    TRACE_STALL_EPOCH: int = 21
    TRACE_FIRST_BATCH_ONLY: bool = True
    TRACE_FORCE_CUDA_SYNC: bool = False