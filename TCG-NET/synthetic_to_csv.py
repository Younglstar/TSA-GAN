# file: synthetic_to_csv_argparse.py
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import DataConfig
from embedding_config import EmbeddingConfig
from embedding_processor import CategoricalEmbeddingProcessor
from encdec_config import EncoderDecoderConfig
from gan_config import GANConfig
from normalizer_config import NormalizerConfig
from normalizer_processor import DataNormalizationProcessor


def _read_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {what}: {path}")


def _read_parquet_columns(path: str) -> List[str]:
    _ensure_exists(path, "parquet 文件")
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        return schema.names
    except Exception:
        df0 = pd.read_parquet(path).head(0)
        return list(df0.columns)


def _find_embedding_cols(df: pd.DataFrame, feature: str) -> List[str]:
    prefix = f"{feature}_emb_"
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_emb_")[-1]))
    return emb_cols


def _manual_inverse_transform_from_params(values: pd.Series, params: dict, feat: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=object)

    valid_mask = x.notna()
    if valid_mask.sum() == 0:
        return out

    x_valid = x.loc[valid_mask].to_numpy(dtype=np.float64)

    if ("bounds" in params) and ("unique_values" in params):
        bounds = np.asarray(params["bounds"], dtype=np.float64)
        uniq = np.asarray(params["unique_values"], dtype=object)
        if bounds.ndim != 1 or uniq.ndim != 1:
            raise RuntimeError(f"[{feat}] params 结构非法：bounds/unique_values 不是一维。")
        if len(bounds) != len(uniq) + 1:
            raise RuntimeError(
                f"[{feat}] params 结构非法：len(bounds)={len(bounds)} 应等于 len(unique_values)+1={len(uniq)+1}"
            )

        lower = float(bounds[0])
        upper = np.nextafter(float(bounds[-1]), -np.inf)
        x_valid = np.clip(x_valid, lower, upper)

        idx = np.searchsorted(bounds, x_valid, side="right") - 1
        idx = np.clip(idx, 0, len(uniq) - 1)

        restored = uniq[idx]
        restored = pd.to_numeric(pd.Series(restored), errors="coerce").to_numpy()
        out.loc[valid_mask] = restored
        return out

    if ("mean" in params) and ("std" in params):
        restored = x_valid * float(params["std"]) + float(params["mean"])
        out.loc[valid_mask] = restored
        return out

    if ("min" in params) and ("max" in params):
        xmin = float(params["min"])
        xmax = float(params["max"])
        vmin = np.nanmin(x_valid)
        vmax = np.nanmax(x_valid)

        if vmin >= -0.1 and vmax <= 1.1:
            restored = x_valid * (xmax - xmin) + xmin
        elif vmin >= -1.1 and vmax <= 1.1:
            restored = (x_valid + 1.0) / 2.0 * (xmax - xmin) + xmin
        else:
            raise RuntimeError(f"[{feat}] params 含 min/max，但当前值域无法判断，vmin={vmin}, vmax={vmax}")

        out.loc[valid_mask] = restored
        return out

    raise RuntimeError(f"[{feat}] 暂不支持的 normalization params 结构，keys={list(params.keys())}")


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
            continue

        if feature not in embed_proc.label_encoders:
            print(f"[WARN] inverse_embeddings_df: no label_encoder for '{feature}', skip.")
            continue

        if feature not in embed_proc.embed_config.category_mappings:
            print(f"[WARN] inverse_embeddings_df: no category mapping for '{feature}', skip.")
            continue

        print(f"[INFO] inverse embedding for feature '{feature}', emb_dim={len(emb_cols)}")

        le = embed_proc.label_encoders[feature]
        num_cats = embed_proc.embed_config.category_mappings[feature]["num_categories"]
        model = embed_proc.embedding_models.get(feature)
        if model is None:
            print(f"[WARN] inverse_embeddings_df: no embedding model for '{feature}', skip.")
            continue

        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            cat_indices = torch.arange(num_cats, dtype=torch.long, device=device)
            cat_emb = model.get_embeddings(cat_indices).cpu().numpy().astype("float32")

        n = len(restored)
        restored_vals = np.empty(n, dtype=object)

        if mask_df is not None and set(emb_cols).issubset(mask_df.columns):
            missing_mask_all = (mask_df[emb_cols].sum(axis=1) == 0).to_numpy()
        else:
            missing_mask_all = (restored[emb_cols].abs().sum(axis=1) == 0).to_numpy()

        num_batches = math.ceil(n / batch_size)
        for b in range(num_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, n)
            batch_emb = restored.iloc[s:e][emb_cols].to_numpy(dtype="float32")
            batch_missing = missing_mask_all[s:e]

            if (~batch_missing).sum() == 0:
                restored_vals[s:e][batch_missing] = np.nan
                continue

            batch_valid_emb = batch_emb[~batch_missing]
            x2 = (batch_valid_emb ** 2).sum(axis=1, keepdims=True)
            c2 = (cat_emb ** 2).sum(axis=1, keepdims=True).T
            xc = batch_valid_emb @ cat_emb.T
            dists = x2 + c2 - 2 * xc

            best_idx = dists.argmin(axis=1)
            cats = le.inverse_transform(best_idx)

            restored_vals[s:e][batch_missing] = np.nan
            restored_vals[s:e][~batch_missing] = cats

        restored[feature] = restored_vals
        restored = restored.drop(columns=emb_cols)

    return restored


def inverse_normalize_df(
    df: pd.DataFrame,
    mask_df: Optional[pd.DataFrame],
    norm_proc,
    features: List[str],
    prefix: str,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    if df.empty or not features:
        return df

    restored = df.copy()

    for feat in features:
        if feat not in restored.columns:
            continue

        param_key = f"{prefix}{feat}"
        if param_key not in norm_proc.norm_config.normalization_params:
            print(f"[WARN] inverse_normalize_df: no params for '{param_key}', skip.")
            continue

        params = norm_proc.norm_config.normalization_params[param_key]
        print(f"[INFO] inverse normalize feature '{feat}' with key='{param_key}'")

        col = restored[feat]
        n = len(col)
        inv_vals = np.empty(n, dtype=object)
        num_batches = math.ceil(n / chunk_size)

        for b in range(num_batches):
            s = b * chunk_size
            e = min((b + 1) * chunk_size, n)
            chunk = col.iloc[s:e].copy()
            inv_chunk = _manual_inverse_transform_from_params(values=chunk, params=params, feat=feat)
            inv_vals[s:e] = inv_chunk.to_numpy() if hasattr(inv_chunk, "to_numpy") else np.asarray(inv_chunk)

        restored[feat] = pd.to_numeric(pd.Series(inv_vals), errors="coerce").to_numpy()

        if mask_df is not None and feat in mask_df.columns:
            restored.loc[mask_df[feat] == 0, feat] = np.nan

    return restored


class SyntheticDataConverter:
    def __init__(
        self,
        decode_dir: Optional[str] = None,
        out_dir: Optional[str] = None,
        decoded_data_pkl: Optional[str] = None,
        mask_binary_pkl: Optional[str] = None,
        mask_logits_pkl: Optional[str] = None,
        mask_probs_pkl: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encdec_conf = EncoderDecoderConfig()
        self.gan_conf = GANConfig()
        self.embed_conf = EmbeddingConfig()
        self.data_conf = DataConfig()
        self.norm_conf = NormalizerConfig()

        self.cache_dir = getattr(self.encdec_conf, "CACHE_DIR", ".")
        self.decode_dir = decode_dir or os.path.join(self.cache_dir, "decode_outputs")
        self.output_dir = out_dir or os.path.join(self.cache_dir, "final_synthetic_output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.decoded_data_pkl = decoded_data_pkl or os.path.join(self.decode_dir, "decoded_data.pkl")
        self.mask_binary_pkl = mask_binary_pkl or os.path.join(self.decode_dir, "mask_binary.pkl")
        self.mask_logits_pkl = mask_logits_pkl or os.path.join(self.decode_dir, "mask_logits.pkl")
        self.mask_probs_pkl = mask_probs_pkl or os.path.join(self.decode_dir, "mask_probs.pkl")

        print("[INFO] loading metadata / decoded outputs...")

        self.feature_dims = self._load_feature_dims()
        self.decoded_data = self._load_decoded_data()
        self.mask_binary = self._load_mask_binary()

        self.meta_embed = self._load_embedding_meta()
        self.static_parquet_path, self.temporal_parquet_path = self._resolve_embedded_parquet_paths()

        self.subject_id_col = self.data_conf.SUBJECT_ID_COL
        self.static_feature_cols, self.temporal_feature_cols = self._infer_embedded_feature_columns()
        self.static_dim = len(self.static_feature_cols)
        self.temporal_dim = len(self.temporal_feature_cols)

        model_dim = self.decoded_data.shape[2]
        if self.static_dim + self.temporal_dim != model_dim:
            raise ValueError(
                f"静态/时序列数与 decoded tensor 维度不一致: static_dim={self.static_dim}, "
                f"temporal_dim={self.temporal_dim}, sum={self.static_dim + self.temporal_dim}, model_dim={model_dim}"
            )

        self.embed_proc = CategoricalEmbeddingProcessor(self.embed_conf, self.data_conf)
        self.embed_proc.load_models()

        self.norm_proc = DataNormalizationProcessor(self.norm_conf, self.data_conf)
        self.norm_proc.load_normalization_params()

        self.static_parquet_out = os.path.join(self.output_dir, "synthetic_static.parquet")
        self.temporal_parquet_out = os.path.join(self.output_dir, "synthetic_temporal.parquet")
        self.static_mask_parquet_out = os.path.join(self.output_dir, "synthetic_static_mask.parquet")
        self.temporal_mask_parquet_out = os.path.join(self.output_dir, "synthetic_temporal_mask.parquet")
        self.static_csv_out = os.path.join(self.output_dir, "synthetic_static.csv")
        self.temporal_csv_out = os.path.join(self.output_dir, "synthetic_temporal.csv")
        self.merged_long_csv_out = os.path.join(self.output_dir, "synthetic_merged_long.csv")
        self.meta_json_out = os.path.join(self.output_dir, "synthetic_to_csv_meta.json")

    def _load_feature_dims(self):
        candidates = [
            os.path.join(self.cache_dir, getattr(self.encdec_conf, "FEATURE_DIMS_FILE", "feature_dims.pkl")),
            getattr(self.encdec_conf, "FEATURE_DIMS_FILE", "feature_dims.pkl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                print(f"[INFO] feature_dims -> {p}")
                return _read_pickle(p)
        raise FileNotFoundError(f"未找到 feature_dims.pkl, tried: {candidates}")

    def _load_decoded_data(self) -> np.ndarray:
        _ensure_exists(self.decoded_data_pkl, "decoded_data.pkl")
        x = np.asarray(_read_pickle(self.decoded_data_pkl), dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"decoded_data 必须是 3D 张量，当前 shape={x.shape}")
        return x

    def _load_mask_binary(self) -> np.ndarray:
        _ensure_exists(self.mask_binary_pkl, "mask_binary.pkl")
        x = np.asarray(_read_pickle(self.mask_binary_pkl), dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"mask_binary 必须是 3D 张量，当前 shape={x.shape}")
        return x

    def _load_embedding_meta(self):
        meta_path = self.embed_conf.EMBEDDINGS_FILE
        _ensure_exists(meta_path, "embedding meta")
        print(f"[INFO] embedding meta -> {meta_path}")
        return _read_pickle(meta_path)

    def _resolve_embedded_parquet_paths(self) -> Tuple[str, str]:
        static_path = self.meta_embed.get("static_path", "embed_static.parquet")
        temporal_path = self.meta_embed.get("temporal_path", "embed_temporal.parquet")
        _ensure_exists(temporal_path, "embed_temporal.parquet")
        if static_path and not os.path.exists(static_path):
            print(f"[WARN] static parquet not found: {static_path}, treat as empty static.")
            static_path = ""
        return static_path, temporal_path

    def _infer_embedded_feature_columns(self) -> Tuple[List[str], List[str]]:
        static_feature_cols: List[str] = []
        if self.static_parquet_path:
            static_cols = _read_parquet_columns(self.static_parquet_path)
            static_feature_cols = [c for c in static_cols if c != self.subject_id_col]

        temporal_cols = _read_parquet_columns(self.temporal_parquet_path)
        meta_cols = set(getattr(self.data_conf, "IDDATA_COLS", []) + getattr(self.data_conf, "TIMEDATA_COLS", []))
        meta_cols.add(self.subject_id_col)
        temporal_feature_cols = [c for c in temporal_cols if c not in meta_cols]
        return static_feature_cols, temporal_feature_cols

    def _tensor_to_modelspace_tables(self):
        N, T, _ = self.decoded_data.shape

        if self.static_dim > 0:
            static_arr = self.decoded_data[:, 0, :self.static_dim]
            static_mask_arr = self.mask_binary[:, 0, :self.static_dim]
            static_df = pd.DataFrame(static_arr, columns=self.static_feature_cols)
            static_mask_df = pd.DataFrame(static_mask_arr, columns=self.static_feature_cols)
        else:
            static_df = pd.DataFrame(index=np.arange(N))
            static_mask_df = pd.DataFrame(index=np.arange(N))

        static_df[self.subject_id_col] = [f"SYNTH_{i:06d}" for i in range(N)]
        static_mask_df[self.subject_id_col] = static_df[self.subject_id_col].values

        temporal_arr = self.decoded_data[:, :, self.static_dim:]
        temporal_mask_arr = self.mask_binary[:, :, self.static_dim:]
        temporal_flat = temporal_arr.reshape(N * T, self.temporal_dim)
        temporal_mask_flat = temporal_mask_arr.reshape(N * T, self.temporal_dim)

        temporal_df = pd.DataFrame(temporal_flat, columns=self.temporal_feature_cols)
        temporal_mask_df = pd.DataFrame(temporal_mask_flat, columns=self.temporal_feature_cols)

        synth_ids = np.repeat(static_df[self.subject_id_col].values, T)
        seq_pos = np.tile(np.arange(T, dtype=np.int64), N)

        temporal_df[self.subject_id_col] = synth_ids
        temporal_df["SEQ_POS"] = seq_pos
        temporal_mask_df[self.subject_id_col] = synth_ids
        temporal_mask_df["SEQ_POS"] = seq_pos

        static_cols_order = [self.subject_id_col] + [c for c in static_df.columns if c != self.subject_id_col]
        temporal_cols_order = [self.subject_id_col, "SEQ_POS"] + [c for c in temporal_df.columns if c not in {self.subject_id_col, "SEQ_POS"}]
        return static_df[static_cols_order], temporal_df[temporal_cols_order], static_mask_df[static_cols_order], temporal_mask_df[temporal_cols_order]

    def _save_modelspace_tables(self, static_df, temporal_df, static_mask_df, temporal_mask_df):
        static_df.to_parquet(self.static_parquet_out, index=False)
        temporal_df.to_parquet(self.temporal_parquet_out, index=False)
        static_mask_df.to_parquet(self.static_mask_parquet_out, index=False)
        temporal_mask_df.to_parquet(self.temporal_mask_parquet_out, index=False)

    def _postprocess_to_final_csv(self, static_df, temporal_df, static_mask_df, temporal_mask_df):
        static_restored = inverse_embeddings_df(
            static_df, static_mask_df, self.embed_proc,
            getattr(self.data_conf, "STATIC_CATEGORICAL_FEATURES", []), batch_size=8192
        )
        temporal_restored = inverse_embeddings_df(
            temporal_df, temporal_mask_df, self.embed_proc,
            getattr(self.data_conf, "TEMPORAL_CATEGORICAL_FEATURES", []), batch_size=8192
        )

        static_restored = inverse_normalize_df(
            static_restored, static_mask_df, self.norm_proc,
            getattr(self.data_conf, "STATIC_NUMERICAL_FEATURES", []),
            prefix="static_", chunk_size=200_000
        )
        temporal_restored = inverse_normalize_df(
            temporal_restored, temporal_mask_df, self.norm_proc,
            getattr(self.data_conf, "TEMPORAL_NUMERICAL_FEATURES", []),
            prefix="temporal_", chunk_size=200_000
        )

        static_restored.to_csv(self.static_csv_out, index=False)
        temporal_restored.to_csv(self.temporal_csv_out, index=False)
        return static_restored, temporal_restored

    def _merge_long(self, static_final: pd.DataFrame, temporal_final: pd.DataFrame):
        merged = temporal_final.merge(static_final, on=self.subject_id_col, how="left", suffixes=("", "_static"))
        front_cols = [self.subject_id_col] + (["SEQ_POS"] if "SEQ_POS" in merged.columns else [])
        remain_cols = [c for c in merged.columns if c not in front_cols]
        merged = merged[front_cols + remain_cols]
        merged.to_csv(self.merged_long_csv_out, index=False)
        return merged

    def _save_meta(self):
        meta = {
            "decoded_data_pkl": self.decoded_data_pkl,
            "mask_binary_pkl": self.mask_binary_pkl,
            "mask_logits_pkl": self.mask_logits_pkl,
            "mask_probs_pkl": self.mask_probs_pkl,
            "decoded_data_shape": list(self.decoded_data.shape),
            "mask_binary_shape": list(self.mask_binary.shape),
            "outputs": {
                "static_parquet": self.static_parquet_out,
                "temporal_parquet": self.temporal_parquet_out,
                "static_mask_parquet": self.static_mask_parquet_out,
                "temporal_mask_parquet": self.temporal_mask_parquet_out,
                "static_csv": self.static_csv_out,
                "temporal_csv": self.temporal_csv_out,
                "merged_long_csv": self.merged_long_csv_out,
            },
        }
        with open(self.meta_json_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def generate_and_save_csv(self):
        print("[INFO] step1: tensor -> model-space tables")
        static_df, temporal_df, static_mask_df, temporal_mask_df = self._tensor_to_modelspace_tables()

        print("[INFO] step2: save model-space parquet")
        self._save_modelspace_tables(static_df, temporal_df, static_mask_df, temporal_mask_df)

        print("[INFO] step3: inverse embeddings + inverse normalization -> final csv")
        static_final, temporal_final = self._postprocess_to_final_csv(static_df, temporal_df, static_mask_df, temporal_mask_df)

        print("[INFO] step4: merge long")
        merged = self._merge_long(static_final, temporal_final)

        self._save_meta()

        print("\n✅ 全部完成")
        print(f"  static csv   shape = {static_final.shape}")
        print(f"  temporal csv shape = {temporal_final.shape}")
        print(f"  merged  csv shape  = {merged.shape}")


def build_argparser():
    parser = argparse.ArgumentParser(description="把 decode outputs 逆处理成最终 CSV")
    parser.add_argument("--decode_dir", type=str, default=None, help="decode 输出目录，内含 decoded_data.pkl / mask_binary.pkl")
    parser.add_argument("--decoded_data_pkl", type=str, default=None, help="decoded_data.pkl 路径")
    parser.add_argument("--mask_binary_pkl", type=str, default=None, help="mask_binary.pkl 路径")
    parser.add_argument("--mask_logits_pkl", type=str, default=None, help="mask_logits.pkl 路径（当前仅记录到 meta）")
    parser.add_argument("--mask_probs_pkl", type=str, default=None, help="mask_probs.pkl 路径（当前仅记录到 meta）")
    parser.add_argument("--out_dir", type=str, default=None, help="最终 CSV 输出目录")
    return parser


def main():
    args = build_argparser().parse_args()
    converter = SyntheticDataConverter(
        decode_dir=args.decode_dir,
        out_dir=args.out_dir,
        decoded_data_pkl=args.decoded_data_pkl,
        mask_binary_pkl=args.mask_binary_pkl,
        mask_logits_pkl=args.mask_logits_pkl,
        mask_probs_pkl=args.mask_probs_pkl,
    )
    converter.generate_and_save_csv()


if __name__ == "__main__":
    main()
