from __future__ import annotations

import gc
import os
import pickle
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import DataConfig
from encdec_config import EncoderDecoderConfig
from embedding_config import EmbeddingConfig


class DataProcessor:
    def __init__(
        self,
        encdec_config: EncoderDecoderConfig,
        data_config: DataConfig,
        embed_config: EmbeddingConfig,
    ):
        self.encdec_config = encdec_config
        self.data_config = data_config
        self.embed_config = embed_config
        self.feature_dims: Dict[str, Any] | None = None

    def _feature_dims_path(self) -> str:
        cache_dir = getattr(self.encdec_config, "CACHE_DIR", ".")
        os.makedirs(cache_dir, exist_ok=True)
        filename = getattr(self.encdec_config, "FEATURE_DIMS_FILE", "feature_dims.pkl")
        return os.path.join(cache_dir, filename)

    def _memmap_paths(self) -> Tuple[str, str]:
        cache_dir = getattr(self.encdec_config, "CACHE_DIR", ".")
        os.makedirs(cache_dir, exist_ok=True)
        return (
            os.path.join(cache_dir, "encdec_X.f32"),
            os.path.join(cache_dir, "encdec_M.u8"),
        )

    def _load_embedded_data(self) -> Dict[str, Any]:
        meta_path = self.embed_config.EMBEDDINGS_FILE
        print(f"  -> 正在从 '{meta_path}' 加载嵌入元信息...")

        meta: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        else:
            print("  [警告] 未找到嵌入元信息文件，将尝试默认 parquet 路径。")

        static_path = meta.get("static_path", "embed_static.parquet")
        temporal_path = meta.get("temporal_path", "embed_temporal.parquet")
        mask_static_path = meta.get("mask_static_path", "static_mask.parquet")
        mask_temporal_path = meta.get("mask_temporal_path", "temporal_mask.parquet")

        static_data = pd.read_parquet(static_path)
        temporal_data = pd.read_parquet(temporal_path)
        static_mask = pd.read_parquet(mask_static_path) if os.path.exists(mask_static_path) else pd.DataFrame()
        temporal_mask = pd.read_parquet(mask_temporal_path) if os.path.exists(mask_temporal_path) else pd.DataFrame()

        print("  -> 嵌入 parquet 加载完成：")
        print(f"     static_data    shape={static_data.shape}")
        print(f"     temporal_data  shape={temporal_data.shape}")
        print(f"     static_mask    shape={static_mask.shape}")
        print(f"     temporal_mask  shape={temporal_mask.shape}")

        return {
            "meta": meta,
            "static_data": static_data,
            "temporal_data": temporal_data,
            "static_mask": static_mask,
            "temporal_mask": temporal_mask,
        }

    @staticmethod
    def _default_feature_slices(columns: list[str], offset: int = 0) -> OrderedDict[str, tuple[int, int]]:
        out: OrderedDict[str, tuple[int, int]] = OrderedDict()
        for i, col in enumerate(columns):
            out[str(col)] = (offset + i, offset + i + 1)
        return out

    @staticmethod
    def _normalize_group_spec(group_spec: Any, local_columns: list[str], offset: int, prefix: str = "") -> OrderedDict[str, tuple[int, int]]:
        result: OrderedDict[str, tuple[int, int]] = OrderedDict()
        if not isinstance(group_spec, dict):
            return result

        col_to_idx = {str(c): i for i, c in enumerate(local_columns)}
        for raw_name, spec in group_spec.items():
            name = f"{prefix}{raw_name}" if prefix else str(raw_name)
            idxs: list[int] = []
            if isinstance(spec, dict):
                if "columns" in spec:
                    idxs = [col_to_idx[str(c)] for c in spec["columns"] if str(c) in col_to_idx]
                elif "indices" in spec:
                    idxs = [int(i) for i in spec["indices"] if 0 <= int(i) < len(local_columns)]
                elif "slice" in spec:
                    s, e = spec["slice"]
                    idxs = list(range(int(s), int(e)))
                elif "start" in spec and "end" in spec:
                    idxs = list(range(int(spec["start"]), int(spec["end"])))
            elif isinstance(spec, (list, tuple)):
                if len(spec) == 2 and all(isinstance(v, (int, np.integer)) for v in spec):
                    s, e = int(spec[0]), int(spec[1])
                    idxs = list(range(s, e))
                else:
                    idxs = [col_to_idx[str(c)] for c in spec if str(c) in col_to_idx]

            idxs = sorted(set(i for i in idxs if 0 <= i < len(local_columns)))
            if not idxs:
                continue
            contiguous = idxs == list(range(idxs[0], idxs[-1] + 1))
            if contiguous:
                result[name] = (offset + idxs[0], offset + idxs[-1] + 1)
            else:
                for k, i in enumerate(idxs):
                    result[f"{name}__{k}"] = (offset + i, offset + i + 1)
        return result

    def _build_graph_feature_slices(self, meta: Dict[str, Any], static_feature_cols: list[str], temporal_feature_cols: list[str]):
        static_dim = len(static_feature_cols)
        static_groups = meta.get("static_feature_groups") or meta.get("static_feature_slices")
        temporal_groups = meta.get("temporal_feature_groups") or meta.get("temporal_feature_slices")
        graph_groups = meta.get("graph_feature_slices") or meta.get("feature_slices") or meta.get("raw_feature_slices")

        if isinstance(graph_groups, dict):
            all_cols = static_feature_cols + temporal_feature_cols
            graph_feature_slices = self._normalize_group_spec(graph_groups, all_cols, offset=0)
        else:
            graph_feature_slices = OrderedDict()

        if not graph_feature_slices:
            static_feature_slices = self._normalize_group_spec(static_groups, static_feature_cols, offset=0)
            temporal_feature_slices = self._normalize_group_spec(temporal_groups, temporal_feature_cols, offset=static_dim)
            if not static_feature_slices:
                static_feature_slices = self._default_feature_slices(static_feature_cols, offset=0)
            if not temporal_feature_slices:
                temporal_feature_slices = self._default_feature_slices(temporal_feature_cols, offset=static_dim)
            graph_feature_slices = OrderedDict()
            graph_feature_slices.update(static_feature_slices)
            graph_feature_slices.update(temporal_feature_slices)
            return graph_feature_slices, static_feature_slices, temporal_feature_slices

        static_feature_slices = OrderedDict((name, spec) for name, spec in graph_feature_slices.items() if spec[1] <= static_dim)
        temporal_feature_slices = OrderedDict((name, spec) for name, spec in graph_feature_slices.items() if spec[0] >= static_dim)
        return graph_feature_slices, static_feature_slices, temporal_feature_slices

    def process_data(self) -> Tuple[np.memmap, np.memmap]:
        print("1. 加载嵌入后的数据...")
        data = self._load_embedded_data()

        meta: Dict[str, Any] = data["meta"]
        static_data: pd.DataFrame = data["static_data"]
        temporal_data: pd.DataFrame = data["temporal_data"]
        static_mask: Optional[pd.DataFrame] = data.get("static_mask")
        temporal_mask: Optional[pd.DataFrame] = data.get("temporal_mask")

        subject_id_col = self.data_config.SUBJECT_ID_COL
        time_cols = [c for c in self.data_config.TIMEDATA_COLS if c in temporal_data.columns]

        if subject_id_col in static_data.columns:
            static_data = static_data.set_index(subject_id_col)
        if subject_id_col in getattr(static_mask, "columns", []):
            static_mask = static_mask.set_index(subject_id_col)

        subjects_from_temporal = pd.Index(temporal_data[subject_id_col].unique(), name=subject_id_col)
        subject_index = static_data.index.union(subjects_from_temporal)
        static_data = static_data.reindex(subject_index)
        static_feature_cols = list(static_data.columns)

        if static_mask is None or static_mask.empty:
            static_mask_np = (~static_data.isna()).to_numpy(dtype=np.float32)
        else:
            static_mask = static_mask.reindex(index=subject_index, columns=static_feature_cols, fill_value=1.0)
            static_mask_np = static_mask.to_numpy(dtype=np.float32)

        static_features_np = static_data.fillna(0.0).to_numpy(dtype=np.float32)
        num_subjects, static_dim = static_features_np.shape
        subject_order = subject_index.to_numpy()
        sid_to_row = pd.Series(np.arange(num_subjects, dtype=np.int64), index=subject_index)

        del static_data, static_mask
        gc.collect()

        meta_cols = set(self.data_config.IDDATA_COLS + self.data_config.TIMEDATA_COLS)
        meta_cols.add(subject_id_col)
        temporal_feature_cols = [c for c in temporal_data.columns if c not in meta_cols]
        temporal_dim = len(temporal_feature_cols)

        sort_cols = [subject_id_col] + time_cols
        temporal_data = temporal_data.sort_values(sort_cols, kind="mergesort").copy()
        temporal_data["_orig_index"] = temporal_data.index.to_numpy()
        temporal_data["_pos"] = temporal_data.groupby(subject_id_col, sort=False).cumcount()

        if temporal_mask is not None and not temporal_mask.empty:
            temporal_mask = temporal_mask.reindex(index=temporal_data["_orig_index"].to_numpy(), columns=temporal_feature_cols, fill_value=1.0)
            temporal_mask.index = temporal_data.index

        real_max_seq_len = int(temporal_data["_pos"].max() + 1) if len(temporal_data) else 0
        t_cap = int(getattr(self.encdec_config, "MAX_SEQ_LEN", 60))
        t_final = min(real_max_seq_len, t_cap)
        truncate_mode = str(getattr(self.encdec_config, "TRUNCATE_MODE", "tail")).lower()

        if real_max_seq_len > t_final:
            if truncate_mode == "tail":
                gsize = temporal_data.groupby(subject_id_col, sort=False)["_pos"].transform("max") + 1
                keep_mask = temporal_data["_pos"] >= (gsize - t_final)
                temporal_data = temporal_data.loc[keep_mask].copy()
                if temporal_mask is not None and not temporal_mask.empty:
                    temporal_mask = temporal_mask.loc[keep_mask].copy()
                temporal_data["_pos"] = temporal_data.groupby(subject_id_col, sort=False).cumcount()
            else:
                keep_mask = temporal_data["_pos"] < t_final
                temporal_data = temporal_data.loc[keep_mask].copy()
                if temporal_mask is not None and not temporal_mask.empty:
                    temporal_mask = temporal_mask.loc[keep_mask].copy()

        if t_final <= 0:
            t_final = 1

        graph_feature_slices, static_feature_slices, temporal_feature_slices = self._build_graph_feature_slices(
            meta=meta,
            static_feature_cols=static_feature_cols,
            temporal_feature_cols=temporal_feature_cols,
        )

        all_feature_cols = static_feature_cols + temporal_feature_cols
        total_dim = static_dim + temporal_dim
        num_all_graph_features = len(graph_feature_slices)
        num_temporal_graph_features = len(temporal_feature_slices)
        use_only_temporal = bool(getattr(self.encdec_config, "CAUSAL_USE_ONLY_TEMPORAL_FEATURES", False))
        active_graph_slices = temporal_feature_slices if use_only_temporal else graph_feature_slices
        num_total_features = len(active_graph_slices)

        x_path, m_path = self._memmap_paths()
        x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(num_subjects, t_final, total_dim))
        m_mm = np.memmap(m_path, dtype="uint8", mode="w+", shape=(num_subjects, t_final, total_dim))
        x_mm[:] = 0.0
        m_mm[:] = 0

        if static_dim > 0:
            x_mm[:, :, :static_dim] = static_features_np[:, None, :]
            m_mm[:, :, :static_dim] = (static_mask_np[:, None, :] > 0).astype(np.uint8)

        if temporal_dim > 0 and len(temporal_data) > 0:
            rows_all = sid_to_row.reindex(temporal_data[subject_id_col].to_numpy()).to_numpy()
            pos_all = temporal_data["_pos"].to_numpy(dtype=np.int64)
            chunk_size = int(getattr(self.encdec_config, "PROCESSOR_CHUNK_SIZE", 1_000_000))
            total_rows = len(temporal_data)
            print(f"   -> 开始分块写入 memmap，总行数: {total_rows}, 分块大小: {chunk_size}")
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                valid = ~np.isnan(rows_all[start_idx:end_idx])
                if not np.any(valid):
                    continue
                curr_rows = rows_all[start_idx:end_idx][valid].astype(np.int64)
                curr_pos = pos_all[start_idx:end_idx][valid].astype(np.int64)
                feat_chunk = temporal_data[temporal_feature_cols].iloc[start_idx:end_idx].to_numpy(copy=True)
                feat_chunk = feat_chunk[valid].astype(np.float32)
                if temporal_mask is not None and not temporal_mask.empty:
                    mask_chunk = temporal_mask[temporal_feature_cols].iloc[start_idx:end_idx].to_numpy(copy=True)
                    mask_chunk = (mask_chunk[valid].astype(np.float32) > 0).astype(np.uint8)
                    feat_chunk = np.where(mask_chunk > 0, np.nan_to_num(feat_chunk, nan=0.0), 0.0)
                else:
                    mask_chunk = (~np.isnan(feat_chunk)).astype(np.uint8)
                    np.nan_to_num(feat_chunk, copy=False, nan=0.0)
                x_mm[curr_rows, curr_pos, static_dim:] = feat_chunk
                m_mm[curr_rows, curr_pos, static_dim:] = mask_chunk
                if start_idx % (chunk_size * 5) == 0:
                    print(f"      已处理: {end_idx} / {total_rows}")

        x_mm.flush()
        m_mm.flush()

        self.feature_dims = {
            "input_dim": total_dim,
            "sequence_length": int(t_final),
            "num_subjects": int(num_subjects),
            "num_total_features": int(num_total_features),
            "num_all_graph_features": int(num_all_graph_features),
            "num_temporal_graph_features": int(num_temporal_graph_features),
            "graph_scope": "temporal_only" if use_only_temporal else "all",
            "static_dim": int(static_dim),
            "temporal_dim": int(temporal_dim),
            "subject_id_col": subject_id_col,
            "time_cols": time_cols,
            "static_feature_cols": static_feature_cols,
            "temporal_feature_cols": temporal_feature_cols,
            "all_feature_cols": all_feature_cols,
            "subject_order_preview": subject_order[: min(20, len(subject_order))].tolist(),
            "graph_feature_slices": dict(graph_feature_slices),
            "feature_slices": dict(graph_feature_slices),
            "active_graph_feature_slices": dict(active_graph_slices),
            "static_feature_slices": dict(static_feature_slices),
            "temporal_feature_slices": dict(temporal_feature_slices),
            "memmap_x_path": x_path,
            "memmap_m_path": m_path,
        }
        self.save_feature_dims()

        del temporal_data, temporal_mask, static_features_np, static_mask_np
        if "feat_chunk" in locals():
            del feat_chunk
        if "mask_chunk" in locals():
            del mask_chunk
        gc.collect()

        print(
            f"✅ 完成：X={tuple(x_mm.shape)}, M={tuple(m_mm.shape)}, "
            f"graph_nodes={num_total_features}, feature_dims={self._feature_dims_path()}"
        )
        return x_mm, m_mm

    def save_feature_dims(self) -> None:
        path = self._feature_dims_path()
        print(f"正在将 feature_dims 保存到 '{path}'...")
        with open(path, "wb") as f:
            pickle.dump(self.feature_dims, f)

    def load_feature_dims(self) -> Dict[str, Any]:
        path = self._feature_dims_path()
        with open(path, "rb") as f:
            self.feature_dims = pickle.load(f)
        return self.feature_dims
