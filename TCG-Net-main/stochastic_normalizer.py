# file: stochastic_normalizer.py (最终修正版 - 解决类型冲突)

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
from normalizer_base import BaseNormalizer
from normalizer_config import NormalizerConfig


class StochasticNormalizer(BaseNormalizer):
    def __init__(self, config: NormalizerConfig):
        super().__init__(config)

    def _calculate_distribution_params(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        """Calculate distribution parameters for stochastic normalization"""
        values = data.values if isinstance(data, pd.Series) else data

        # --- 核心修正：在处理前，将所有元素统一转换为字符串类型 ---
        # 1. 先将NaN替换为'MISSING'
        # 2. 然后将所有元素（包括数字）都转换为字符串
        values_str = pd.Series(values).fillna('MISSING').astype(str).values
        # --- 修正结束 ---

        # 现在可以安全地对一个纯字符串数组进行操作了
        unique_vals, counts = np.unique(values_str, return_counts=True)
        frequencies = counts / len(values_str)
        cum_probs = np.cumsum(frequencies)

        return {
            'unique_values': unique_vals,
            'frequencies': frequencies,
            'cum_probs': cum_probs,
            'bounds': np.concatenate(([0], cum_probs))
        }

    def fit(self, data: Union[pd.DataFrame, pd.Series]) -> None:
        """Fit the stochastic normalizer to the data"""
        self.params = self._calculate_distribution_params(data)

    def transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Transform data using stochastic normalization"""
        values = data.values if isinstance(data, pd.Series) else data
        transformed = np.zeros_like(values, dtype=float)

        for i, val in enumerate(values):
            # 同样地，将要查找的值也转换为字符串格式
            val_to_find = 'MISSING' if pd.isna(val) else str(val)

            try:
                idx = np.where(self.params['unique_values'] == val_to_find)[0][0]
                lower_bound = self.params['bounds'][idx]
                upper_bound = self.params['bounds'][idx + 1]
                transformed[i] = np.random.uniform(lower_bound, upper_bound)
            except IndexError:
                # 安全保障
                transformed[i] = np.nan

        if isinstance(data, pd.Series):
            return pd.Series(transformed, index=data.index)
        return transformed

    def inverse_transform(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Inverse transform normalized data back to original scale"""
        values = data.values if isinstance(data, pd.Series) else data
        inverse_transformed = np.empty_like(values, dtype=object)

        for i, val in enumerate(values):
            bin_idx = np.digitize(val, self.params['bounds']) - 1
            bin_idx = np.clip(bin_idx, 0, len(self.params['unique_values']) - 1)

            original_val_str = self.params['unique_values'][bin_idx]

            # 还原时，将'MISSING'转换回np.nan
            inverse_transformed[i] = np.nan if original_val_str == 'MISSING' else original_val_str

        # 尝试将结果转换为数值类型，如果失败则保持为对象类型
        result = pd.to_numeric(inverse_transformed, errors='ignore')

        if isinstance(data, pd.Series):
            return pd.Series(result, index=data.index)
        return result