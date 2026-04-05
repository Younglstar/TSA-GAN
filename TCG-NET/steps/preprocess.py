from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from config import DataConfig
from data_loader import DataLoader
from missing_pattern_analyzer import MissingPatternAnalyzer
from utils.io import save_pickle
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def preprocess(config: DataConfig | None = None) -> Dict[str, Any]:

    cfg = config or DataConfig()

    logger.info("Step: preprocess | input=%s", cfg.INPUT_FILE)

    loader = DataLoader(cfg)
    static_data, temporal_data, static_mask, temporal_mask = loader.split_static_temporal()

    analyzer = MissingPatternAnalyzer(static_data, temporal_data, cfg)
    missing_rates_static, missing_rates_temporal = analyzer.calculate_missing_rates()

    # ★ missing_patterns 很大 -> 仅按需启用 或 采样
    try:
        missing_patterns = analyzer.generate_missing_patterns()
    except MemoryError:
        logger.warning("missing_patterns too large -> skipped")
        missing_patterns = None

    subject_id_col = cfg.SUBJECT_ID_COL
    logger.info("Unique subjects (%s): %d", subject_id_col, len(static_data))

    # ======================================================
    # ★★★ 核心变化：大对象改为 parquet 分文件保存 ★★★
    # ======================================================

    static_data.to_parquet("static_data.parquet")
    temporal_data.to_parquet("temporal_data.parquet")
    static_mask.to_parquet("static_mask.parquet")
    temporal_mask.to_parquet("temporal_mask.parquet")

    logger.info("Saved:")
    logger.info("  static_data.parquet")
    logger.info("  temporal_data.parquet")
    logger.info("  static_mask.parquet")
    logger.info("  temporal_mask.parquet")

    # ======================================================
    # ★★★ 仅把小对象写入 metadata.pkl ★★★
    # ======================================================

    metadata = {
        "missing_rates": {
            "static": missing_rates_static,
            "temporal": missing_rates_temporal,
        },
        "missing_patterns": missing_patterns,  # 可能为 None
        "subject_count": len(static_data),
    }

    save_pickle(metadata, "preprocess_metadata.pkl")
    logger.info("Saved metadata -> preprocess_metadata.pkl")

    return metadata
