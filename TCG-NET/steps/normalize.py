from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from config import DataConfig
from normalizer_config import NormalizerConfig
from normalizer_processor import DataNormalizationProcessor
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def normalize(
    data_config: DataConfig | None = None,
    norm_config: NormalizerConfig | None = None,
) -> Dict[str, Any]:
    """
    Normalization step.

    从 preprocess 阶段生成的 parquet 文件中读取：
      - static_data / temporal_data
      - static_mask / temporal_mask

    对数值特征进行归一化，同时保持 mask 语义：
      - 缺失值位置保持为 0
      - 对应 mask 仍为 0

    输出：
      - 归一化后的 static/temporal parquet
      - 对应 mask parquet
      - 更新后的 normalization params（写入 NORMALIZATION_PARAMS_FILE）
    """

    cfg = data_config or DataConfig()
    ncfg = norm_config or NormalizerConfig()

    logger.info("Step: normalize | loading parquet inputs...")

    # ======================================================
    # 1. 从 preprocess 输出加载数据（使用配置中的路径）
    # ======================================================
    static_data = pd.read_parquet(ncfg.STATIC_INPUT_PATH)
    temporal_data = pd.read_parquet(ncfg.TEMPORAL_INPUT_PATH)
    static_mask = pd.read_parquet(ncfg.STATIC_MASK_INPUT_PATH)
    temporal_mask = pd.read_parquet(ncfg.TEMPORAL_MASK_INPUT_PATH)

    logger.info(
        "Loaded preprocess outputs | static=%s temporal=%s",
        static_data.shape,
        temporal_data.shape,
    )

    processor = DataNormalizationProcessor(ncfg, cfg)

    # ======================================================
    # 2. 归一化，同时使用/更新 mask
    #    - 缺失值位置填 0，但用 mask 记录真实缺失
    # ======================================================
    norm_static, static_mask = processor.normalize_static_features(
        static_data,
        static_mask,
    )

    norm_temporal, temporal_mask = processor.normalize_temporal_features(
        temporal_data,
        temporal_mask,
    )

    # 归一化参数落盘（每个特征一个 param_key）
    processor.save_normalization_params()
    logger.info("Saved normalization params -> %s", ncfg.NORMALIZATION_PARAMS_FILE)

    # ======================================================
    # 3. 分文件保存归一化后的数据和 mask（适配大规模数据）
    # ======================================================
    norm_static.to_parquet(ncfg.NORM_STATIC_OUTPUT_PATH)
    norm_temporal.to_parquet(ncfg.NORM_TEMPORAL_OUTPUT_PATH)
    static_mask.to_parquet(ncfg.NORM_STATIC_MASK_OUTPUT_PATH)
    temporal_mask.to_parquet(ncfg.NORM_TEMPORAL_MASK_OUTPUT_PATH)

    logger.info("Saved normalized parquet files:")
    logger.info("  static  -> %s", ncfg.NORM_STATIC_OUTPUT_PATH)
    logger.info("  temporal-> %s", ncfg.NORM_TEMPORAL_OUTPUT_PATH)
    logger.info("  static_mask   -> %s", ncfg.NORM_STATIC_MASK_OUTPUT_PATH)
    logger.info("  temporal_mask -> %s", ncfg.NORM_TEMPORAL_MASK_OUTPUT_PATH)

    # 返回一个轻量级 meta，后续 embed/encdec 可以直接用
    return {
        "static_path": ncfg.NORM_STATIC_OUTPUT_PATH,
        "temporal_path": ncfg.NORM_TEMPORAL_OUTPUT_PATH,
        "mask_static_path": ncfg.NORM_STATIC_MASK_OUTPUT_PATH,
        "mask_temporal_path": ncfg.NORM_TEMPORAL_MASK_OUTPUT_PATH,
    }
