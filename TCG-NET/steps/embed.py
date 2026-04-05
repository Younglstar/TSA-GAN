# file: steps/embed.py
from __future__ import annotations

# [MOD] 1) 必须在 import pandas/pyarrow 之前设置线程环境变量
# 原因：pandas/pyarrow/numpy/BLAS 默认可能开很多线程，容易造成隐性阻塞/调度卡住
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from typing import Any, Dict
import gc
import time
import pandas as pd

from config import DataConfig
from normalizer_config import NormalizerConfig
from embedding_config import EmbeddingConfig
from embedding_processor import CategoricalEmbeddingProcessor
from utils.io import save_pickle
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _flush_logger(logger_obj) -> None:
    """
    [MOD] 2) 显式 flush logger handler
    原因：某些情况下日志缓冲/IO 调度会影响线程推进；显式 flush 能增强稳定性
    """
    try:
        for h in getattr(logger_obj, "handlers", []):
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass


def _safe_to_parquet(df: pd.DataFrame, path: str, desc: str) -> None:
    """
    [MOD] 3) 封装安全 parquet 写入：
    - 使用 pyarrow 引擎（稳定且快）
    - 使用 snappy 压缩（写入/读取更稳定、IO更少）
    - 显式 index=False（避免 index 写入导致额外开销/不一致）
    - 写入前后做日志 + flush
    - 写完后 gc.collect() 释放内存，降低后续写入卡住风险
    """
    logger.info("Start writing %s -> %s | shape=%s", desc, path, df.shape)
    _flush_logger(logger)

    t0 = time.time()
    # engine="pyarrow" 在大表写入更稳；snappy 能显著减少磁盘吞吐压力
    df.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    # [MOD] 写完强制回收，减少“下一步写入卡住”
    gc.collect()

    logger.info("Done writing %s | %.2fs", desc, time.time() - t0)
    _flush_logger(logger)


def embed(
    data_config: DataConfig | None = None,
    embed_config: EmbeddingConfig | None = None,
) -> Dict[str, Any]:
    """
    Categorical embedding step.

    输入（来自 normalize 阶段 parquet）：
      - static / temporal
      - static_mask / temporal_mask

    输出（parquet）：
      - embed_static.parquet / embed_temporal.parquet
      - embed_static_mask.parquet / embed_temporal_mask.parquet
      - embeddings_meta.pkl（仅包含路径字典）
    """

    cfg = data_config or DataConfig()
    ecfg = embed_config or EmbeddingConfig()
    ncfg = NormalizerConfig()

    logger.info("Step: embed | loading normalized parquet...")
    _flush_logger(logger)

    # ======================================================
    # 1. 从 normalize 输出读取（避免载入大 pickle）
    # ======================================================
    # [MOD] 4) 读取后立刻做一次轻量 gc，避免后面嵌入时内存高水位导致卡住
    norm_static = pd.read_parquet(ncfg.NORM_STATIC_OUTPUT_PATH, engine="pyarrow")
    norm_temporal = pd.read_parquet(ncfg.NORM_TEMPORAL_OUTPUT_PATH, engine="pyarrow")
    static_mask = pd.read_parquet(ncfg.NORM_STATIC_MASK_OUTPUT_PATH, engine="pyarrow")
    temporal_mask = pd.read_parquet(ncfg.NORM_TEMPORAL_MASK_OUTPUT_PATH, engine="pyarrow")
    gc.collect()

    logger.info("Loaded normalized data | static=%s temporal=%s", norm_static.shape, norm_temporal.shape)
    _flush_logger(logger)

    processor = CategoricalEmbeddingProcessor(ecfg, cfg)

    # ======================================================
    # 2. 静态特征嵌入 + mask 同步扩展
    # ======================================================
    logger.info("Start embedding STATIC features...")
    _flush_logger(logger)

    static_emb, static_mask = processor.fit_transform_static_features(norm_static, static_mask)

    # [MOD] 5) 释放原始大对象引用，减少峰值内存
    del norm_static
    gc.collect()

    logger.info("Done embedding STATIC | static_emb=%s static_mask=%s", static_emb.shape, static_mask.shape)
    _flush_logger(logger)

    # ======================================================
    # 3. 时序特征嵌入 + mask 同步扩展
    # ======================================================
    logger.info("Start embedding TEMPORAL features...")
    _flush_logger(logger)

    temporal_emb, temporal_mask = processor.fit_transform_temporal_features(norm_temporal, temporal_mask)

    # [MOD] 6) 同样释放原始时序数据
    del norm_temporal
    gc.collect()

    logger.info("Done embedding TEMPORAL | temporal_emb=%s temporal_mask=%s", temporal_emb.shape, temporal_mask.shape)
    _flush_logger(logger)

    # ======================================================
    # 4. 保存嵌入模型（label_encoders + state_dict）
    # ======================================================
    processor.save_models()
    logger.info("Saved embedding models -> %s", ecfg.EMBEDDING_MODELS_FILE)
    _flush_logger(logger)

    # ======================================================
    # 5. 写 parquet（最容易“卡住”的阶段，重点增强稳定性）
    # ======================================================
    _safe_to_parquet(static_emb, ecfg.EMBED_STATIC_OUTPUT_PATH, desc="static_emb")
    _safe_to_parquet(temporal_emb, ecfg.EMBED_TEMPORAL_OUTPUT_PATH, desc="temporal_emb")
    _safe_to_parquet(static_mask, ecfg.EMBED_STATIC_MASK_OUTPUT_PATH, desc="static_mask")
    _safe_to_parquet(temporal_mask, ecfg.EMBED_TEMPORAL_MASK_OUTPUT_PATH, desc="temporal_mask")

    # [MOD] 7) 写完后释放大对象，避免后续 encdec 阶段启动时内存处于高水位
    del static_emb, temporal_emb, static_mask, temporal_mask
    gc.collect()

    logger.info("Saved embedded parquet files OK.")
    logger.info("  static_emb   -> %s", ecfg.EMBED_STATIC_OUTPUT_PATH)
    logger.info("  temporal_emb -> %s", ecfg.EMBED_TEMPORAL_OUTPUT_PATH)
    logger.info("  static_mask  -> %s", ecfg.EMBED_STATIC_MASK_OUTPUT_PATH)
    logger.info("  temporal_mask-> %s", ecfg.EMBED_TEMPORAL_MASK_OUTPUT_PATH)
    _flush_logger(logger)

    # ======================================================
    # 6. 保存轻量 meta（路径字典），供 encdec_processor 使用
    # ======================================================
    meta: Dict[str, Any] = {
        "static_path": ecfg.EMBED_STATIC_OUTPUT_PATH,
        "temporal_path": ecfg.EMBED_TEMPORAL_OUTPUT_PATH,
        "mask_static_path": ecfg.EMBED_STATIC_MASK_OUTPUT_PATH,
        "mask_temporal_path": ecfg.EMBED_TEMPORAL_MASK_OUTPUT_PATH,
    }

    save_pickle(meta, ecfg.EMBEDDINGS_FILE)
    logger.info("Saved embeddings meta -> %s", ecfg.EMBEDDINGS_FILE)
    _flush_logger(logger)

    return meta
