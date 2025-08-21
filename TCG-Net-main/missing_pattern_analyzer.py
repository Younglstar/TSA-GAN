# missing_pattern_analyzer.py (已修改，以处理通用的 DataFrame 结构)
import pandas as pd
from typing import Tuple, Dict
from config import DataConfig  # 引入 DataConfig 以获取元数据信息


class MissingPatternAnalyzer:
    def __init__(self, static_data: pd.DataFrame, temporal_data: pd.DataFrame, config: DataConfig):
        """
        (已重写) 构造函数现在接收 config 和 DataFrame 类型的 temporal_data。
        """
        self.static_data = static_data
        self.temporal_data = temporal_data
        self.config = config
        self.static_missing_rates = None
        self.temporal_missing_rates = None
        self.missing_patterns = None

    def calculate_missing_rates(self) -> Tuple[pd.Series, pd.Series]:
        """
        (已重写) 为静态和时序特征计算缺失率。
        返回的第二个值现在是 pd.Series 而不是字典。
        """
        # 1. 静态特征缺失率 (逻辑不变，但增加了空值检查)
        if not self.static_data.empty and not self.static_data.columns.empty:
            self.static_missing_rates = self.static_data.isna().sum() / len(self.static_data)
        else:
            self.static_missing_rates = pd.Series(dtype=float)

        # 2. 时序特征缺失率 (通用化)
        if not self.temporal_data.empty:
            # 确定哪些是元数据列，以便在计算中排除它们
            meta_cols = self.config.ID_COLUMNS + self.config.DATE_COLUMNS
            # 只选择真正的特征列进行计算
            feature_cols = [col for col in self.temporal_data.columns if col not in meta_cols]

            if feature_cols:
                temporal_features_df = self.temporal_data[feature_cols]
                self.temporal_missing_rates = temporal_features_df.isna().sum() / len(temporal_features_df)
            else:
                self.temporal_missing_rates = pd.Series(dtype=float)
        else:
            self.temporal_missing_rates = pd.Series(dtype=float)

        return self.static_missing_rates, self.temporal_missing_rates

    def generate_missing_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        (已重写) 为所有特征生成二进制缺失模式。
        'temporal' 键的值现在是单个 DataFrame。
        """
        # 1. 静态模式 (逻辑不变)
        static_patterns = self.static_data.isna().astype(int)

        # 2. 时序模式 (通用化)
        if not self.temporal_data.empty:
            # 我们希望保留ID和日期作为索引，只对特征列生成模式
            meta_cols = self.config.ID_COLUMNS + self.config.DATE_COLUMNS
            # 检查元数据列是否存在于DataFrame中
            existing_meta_cols = [col for col in meta_cols if col in self.temporal_data.columns]

            if existing_meta_cols:
                # 将元数据列设为索引
                temp_df_with_index = self.temporal_data.set_index(existing_meta_cols)
                # 对剩余的特征列计算缺失模式
                temporal_patterns = temp_df_with_index.isna().astype(int)
            else:
                # 如果没有元数据列，直接对整个DataFrame计算
                temporal_patterns = self.temporal_data.isna().astype(int)
        else:
            temporal_patterns = pd.DataFrame()

        self.missing_patterns = {
            'static': static_patterns,
            'temporal': temporal_patterns
        }

        return self.missing_patterns