import pandas as pd
from typing import Tuple, Dict
from config import DataConfig


class MissingPatternAnalyzer:
    def __init__(self, static_data: pd.DataFrame, temporal_data: pd.DataFrame, config: DataConfig):
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.config = config

        self.static_missing_rates = None
        self.temporal_missing_rates = None
        self.missing_patterns = None

    # ------------------------------------------------
    # 1) 缺失率（按列统计）
    # ------------------------------------------------
    def calculate_missing_rates(self) -> Tuple[pd.Series, pd.Series]:

        # ---- 静态 ----
        if not self.static_data.empty:
            self.static_missing_rates = self.static_data.isna().sum() / len(self.static_data)
        else:
            self.static_missing_rates = pd.Series(dtype=float)

        # ---- 时序 ----
        if not self.temporal_data.empty:

            meta_cols = self.config.IDDATA_COLS + self.config.TIMEDATA_COLS
            feature_cols = [c for c in self.temporal_data.columns if c not in meta_cols]

            if feature_cols:
                temporal_features_df = self.temporal_data[feature_cols]
                self.temporal_missing_rates = temporal_features_df.isna().sum() / len(temporal_features_df)
            else:
                self.temporal_missing_rates = pd.Series(dtype=float)

        else:
            self.temporal_missing_rates = pd.Series(dtype=float)

        return self.static_missing_rates, self.temporal_missing_rates

    # ------------------------------------------------
    # 2) 缺失模式 —— 设计为 **每个 subject × 每个特征的缺失率**
    # ------------------------------------------------
    def generate_missing_patterns(self) -> Dict[str, pd.DataFrame]:

        # ---- 静态（0/1 是否缺失）----
        static_patterns = (
            self.static_data.isna()
            .astype("int8")
            if not self.static_data.empty
            else pd.DataFrame()
        )

        # ---- 时序（避免生成巨大矩阵）----
        if not self.temporal_data.empty:

            meta_cols = self.config.IDDATA_COLS + self.config.TIMEDATA_COLS
            feature_cols = [c for c in self.temporal_data.columns if c not in meta_cols]

            if feature_cols:

                sid = self.config.SUBJECT_ID_COL

                # grouping by subject
                grouped = self.temporal_data.groupby(sid)[feature_cols]

                # 缺失率 = 在时间维度上 isna().mean()
                temporal_patterns = (
                    grouped.apply(lambda df: df.isna().mean())
                    .astype("float32")
                )

                temporal_patterns.index.name = sid

            else:
                temporal_patterns = pd.DataFrame()

        else:
            temporal_patterns = pd.DataFrame()

        self.missing_patterns = {
            "static": static_patterns,
            "temporal": temporal_patterns
        }

        return self.missing_patterns
