# normalizer_config.py (已修改，移除了冗余的特征定义)
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class NormalizerConfig:
    """
    (已重写)
    本配置文件现在只包含与归一化 *过程* 相关的参数。
    需要被归一化的具体特征列表将从主配置 (config.py) 中动态获取，
    以遵循单一数据源原则，避免配置冗余。
    """

    # 1. 归一化算法的参数
    RANDOM_SEED: int = 42
    EPSILON: float = 1e-10  # 用于防止除以零的极小值

    # 2. 文件路径
    # preprocess 输出
    STATIC_INPUT_PATH: str = "static_data.parquet"
    TEMPORAL_INPUT_PATH: str = "temporal_data.parquet"
    STATIC_MASK_INPUT_PATH: str = "static_mask.parquet"
    TEMPORAL_MASK_INPUT_PATH: str = "temporal_mask.parquet"

    # normalize 输出
    NORM_STATIC_OUTPUT_PATH: str = "norm_static.parquet"
    NORM_TEMPORAL_OUTPUT_PATH: str = "norm_temporal.parquet"
    NORM_STATIC_MASK_OUTPUT_PATH: str = "norm_static_mask.parquet"
    NORM_TEMPORAL_MASK_OUTPUT_PATH: str = "norm_temporal_mask.parquet"

    # 归一化参数文件
    NORMALIZATION_PARAMS_FILE: str = "normalization_params.pkl"
    # 3. 用于存储归一化参数的字典 (例如均值和标准差)
    # 这个字段由 DataNormalizationProcessor 内部使用和填充。
    normalization_params: Dict = field(default_factory=dict, repr=False)