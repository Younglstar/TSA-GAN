from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from config import DataConfig
from embedding_config import EmbeddingConfig
from embedding_processor import CategoricalEmbeddingProcessor
from normalizer_config import NormalizerConfig
from normalizer_processor import DataNormalizationProcessor
from stochastic_normalizer import StochasticNormalizer
from utils.io import ensure_exists
from utils.logging_utils import get_logger


logger = get_logger(__name__)


# --------------------------------------------------------
# 1. 工具函数：按列名自动找某个特征的 embedding 列
# --------------------------------------------------------
def _find_embedding_cols(df: pd.DataFrame, feature: str) -> List[str]:
    prefix = f"{feature}_emb_"
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    # 保证顺序：_emb_0, _emb_1, ...
    emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_emb_")[-1]))
    return emb_cols


# --------------------------------------------------------
# 2. 反 embedding：把 *_emb_* 维度还原成类别值
#    - 支持大数据：按 batch 分块
#    - 缺失位置 (mask=0) 还原为 NaN
# --------------------------------------------------------
def inverse_embeddings_df(
    df: pd.DataFrame,
    mask_df: Optional[pd.DataFrame],
    embed_proc: CategoricalEmbeddingProcessor,
    feature_list: List[str],
    batch_size: int = 8192,
) -> pd.DataFrame:
    if df.empty:
        return df

    restored = df.copy()

    for feature in feature_list:
        emb_cols = _find_embedding_cols(restored, feature)
        if not emb_cols:
            logger.info("inverse_embeddings_df: feature '%s' has no embedding cols, skip.", feature)
            continue

        if feature not in embed_proc.label_encoders:
            logger.warning("inverse_embeddings_df: no label_encoder for feature '%s', skip.", feature)
            continue

        if feature not in embed_proc.embed_config.category_mappings:
            logger.warning("inverse_embeddings_df: no category mapping for feature '%s', skip.", feature)
            continue

        logger.info("Inverse embedding for feature '%s' with %d dims", feature, len(emb_cols))

        le = embed_proc.label_encoders[feature]
        num_cats = embed_proc.embed_config.category_mappings[feature]["num_categories"]

        model = embed_proc.embedding_models.get(feature)
        if model is None:
            logger.warning("inverse_embeddings_df: no embedding model instance for '%s', skip.", feature)
            continue

        model.eval()
        device = next(model.parameters()).device

        # 预先算好每个类别的“原型 embedding 向量”，以便做最近邻
        with torch.no_grad():
            cat_indices = torch.arange(num_cats, dtype=torch.long, device=device)
            cat_emb = model.get_embeddings(cat_indices).cpu().numpy().astype("float32")  # (num_cats, emb_dim)

        n = len(restored)
        restored_vals = np.empty(n, dtype=object)

        # 缺失 mask：如果给了 mask_df，就用；否则用 emb 全 0 作为缺失
        if mask_df is not None and set(emb_cols).issubset(mask_df.columns):
            missing_mask_all = (mask_df[emb_cols].sum(axis=1) == 0).to_numpy()
        else:
            missing_mask_all = (restored[emb_cols].abs().sum(axis=1) == 0).to_numpy()

        # 按 batch 处理，节省内存
        num_batches = math.ceil(n / batch_size)
        logger.info("  feature '%s': n=%d, batch_size=%d, num_batches=%d",
                    feature, n, batch_size, num_batches)

        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, n)
            if s >= e:
                break

            batch_emb = restored.loc[restored.index[s:e], emb_cols].to_numpy(dtype="float32")  # (B, emb_dim)

            # 对缺失位置，直接标记 NaN 不做最近邻
            batch_missing = missing_mask_all[s:e]
            if ( ~batch_missing ).sum() == 0:
                restored_vals[s:e][batch_missing] = np.nan
                continue

            # 只对非缺失行做最近邻
            batch_valid_emb = batch_emb[~batch_missing]  # (B_valid, emb_dim)

            # 计算到每个类别 embedding 的欧式距离： (B_valid, num_cats)
            # d^2 = ||x||^2 + ||c||^2 - 2 x·c
            x2 = (batch_valid_emb ** 2).sum(axis=1, keepdims=True)
            c2 = (cat_emb ** 2).sum(axis=1, keepdims=True).T  # (1, num_cats)
            xc = batch_valid_emb @ cat_emb.T
            dists = x2 + c2 - 2 * xc

            best_idx = dists.argmin(axis=1)  # 每个样本最近的类别 index

            # map 回原始类别字符串
            cats = le.inverse_transform(best_idx)

            # 写回
            restored_vals[s:e][batch_missing] = np.nan
            restored_vals[s:e][~batch_missing] = cats

        # 新增原始特征列
        restored[feature] = restored_vals

        # 删除 embedding 列
        restored = restored.drop(columns=emb_cols)

    return restored


# --------------------------------------------------------
# 3. 反归一化：用保存好的 params 构造 normalizer，再 inverse_transform
#    - prefix: "static_" / "temporal_"
#    - features: 数值特征列名列表
#    - mask_df: 0=缺失 → inverse 后再恢复为 NaN
#    - 支持按 chunk 反归一化，节省内存
# --------------------------------------------------------
def inverse_normalize_df(
    df: pd.DataFrame,
    mask_df: Optional[pd.DataFrame],
    norm_proc: DataNormalizationProcessor,
    features: List[str],
    prefix: str,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    if df.empty or not features:
        return df

    restored = df.copy()

    for feat in features:
        if feat not in restored.columns:
            logger.info("inverse_normalize_df: feature '%s' not found, skip.", feat)
            continue

        param_key = f"{prefix}{feat}"
        if param_key not in norm_proc.norm_config.normalization_params:
            logger.warning("inverse_normalize_df: no params for key '%s', skip.", param_key)
            continue

        logger.info("Inverse normalize feature '%s' (key=%s)", feat, param_key)

        normalizer = StochasticNormalizer(norm_proc.norm_config)
        # 手动注入训练时保存的参数
        normalizer.params = norm_proc.norm_config.normalization_params[param_key]

        col = restored[feat]
        n = len(col)

        # 按 chunk 反归一化，避免一次性在 3000 万行上开大数组
        inv_vals = np.empty(n, dtype=object)
        num_batches = math.ceil(n / chunk_size)

        logger.info("  feature '%s': n=%d, chunk_size=%d, num_batches=%d",
                    feat, n, chunk_size, num_batches)

        for b in range(num_batches):
            s = b * chunk_size
            e = min((b + 1) * chunk_size, n)
            if s >= e:
                break

            chunk = col.iloc[s:e]
            inv_chunk = normalizer.inverse_transform(chunk)
            inv_vals[s:e] = inv_chunk.to_numpy()

        restored[feat] = inv_vals

        # 恢复缺失位置：mask==0 → NaN
        if mask_df is not None and feat in mask_df.columns:
            restored.loc[mask_df[feat] == 0, feat] = np.nan

    return restored


# --------------------------------------------------------
# 4. 总入口：读取“合成的、仍在模型空间中的”数据 → 反 embedding → 反归一化 → 输出 CSV
# --------------------------------------------------------
def postprocess_synthetic(
    data_config: DataConfig | None = None,
    norm_config: NormalizerConfig | None = None,
    embed_config: EmbeddingConfig | None = None,
    # 这些路径按你 GAN/decoding 输出习惯来，如果叫别的名字可以改一下
    static_path: str = "synthetic_static.parquet",
    temporal_path: str = "synthetic_temporal.parquet",
    static_mask_path: str = "synthetic_static_mask.parquet",
    temporal_mask_path: str = "synthetic_temporal_mask.parquet",
    # 输出 CSV
    static_csv_out: str = "synthetic_static.csv",
    temporal_csv_out: str = "synthetic_temporal.csv",
    # 性能相关
    emb_batch_size: int = 8192,
    norm_chunk_size: int = 200_000,
) -> None:
    """
    最终后处理步骤：
      1) 读取合成的、已解码的 static / temporal（仍在 embedding + 归一化空间）
      2) 反 embedding → 恢复类别特征
      3) 反归一化 → 恢复数值特征到原始尺度（带随机性）
      4) mask==0 的位置恢复为 NaN
      5) 输出两个可读 CSV：静态 / 时序
    """

    cfg = data_config or DataConfig()
    ncfg = norm_config or NormalizerConfig()
    ecfg = embed_config or EmbeddingConfig()

    # ---------- 读入合成数据 ----------
    for p in [static_path, temporal_path, static_mask_path, temporal_mask_path]:
        ensure_exists(p, hint="Synthetic data or mask parquet not found")

    logger.info("Postprocess: loading synthetic parquet...")
    synth_static = pd.read_parquet(static_path)
    synth_temporal = pd.read_parquet(temporal_path)
    static_mask = pd.read_parquet(static_mask_path)
    temporal_mask = pd.read_parquet(temporal_mask_path)

    logger.info("  synthetic_static shape=%s", synth_static.shape)
    logger.info("  synthetic_temporal shape=%s", synth_temporal.shape)

    # ---------- 准备 embedding 模型 ----------
    embed_proc = CategoricalEmbeddingProcessor(ecfg, cfg)
    embed_proc.load_models()  # 恢复 label_encoders + embedding_models

    # ---------- 准备 normalizer 参数 ----------
    norm_proc = DataNormalizationProcessor(ncfg, cfg)
    norm_proc.load_normalization_params()

    # ---------- 1) 先反 embedding ----------
    logger.info("Step 1/3: inverse embeddings (static)...")
    synth_static = inverse_embeddings_df(
        synth_static,
        static_mask,
        embed_proc,
        cfg.STATIC_CATEGORICAL_FEATURES,
        batch_size=emb_batch_size,
    )

    logger.info("Step 1/3: inverse embeddings (temporal)...")
    synth_temporal = inverse_embeddings_df(
        synth_temporal,
        temporal_mask,
        embed_proc,
        cfg.TEMPORAL_CATEGORICAL_FEATURES,
        batch_size=emb_batch_size,
    )

    # ---------- 2) 再反归一化 ----------
    logger.info("Step 2/3: inverse normalization (static)...")
    synth_static = inverse_normalize_df(
        synth_static,
        static_mask,
        norm_proc,
        cfg.STATIC_NUMERICAL_FEATURES,
        prefix="static_",
        chunk_size=norm_chunk_size,
    )

    logger.info("Step 2/3: inverse normalization (temporal)...")
    synth_temporal = inverse_normalize_df(
        synth_temporal,
        temporal_mask,
        norm_proc,
        cfg.TEMPORAL_NUMERICAL_FEATURES,
        prefix="temporal_",
        chunk_size=norm_chunk_size,
    )

    # ---------- 3) 输出 CSV ----------
    logger.info("Step 3/3: save final CSVs...")
    synth_static.to_csv(static_csv_out, index=False)
    synth_temporal.to_csv(temporal_csv_out, index=False)

    logger.info("Saved synthetic static CSV -> %s", static_csv_out)
    logger.info("Saved synthetic temporal CSV -> %s", temporal_csv_out)
