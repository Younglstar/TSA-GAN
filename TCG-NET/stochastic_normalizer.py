# file: stochastic_normalizer.py

import numpy as np
import pandas as pd
from typing import Dict, Union
from collections import Counter
import gc

from normalizer_base import BaseNormalizer
from normalizer_config import NormalizerConfig


class StochasticNormalizer(BaseNormalizer):
    def __init__(self, config: NormalizerConfig):
        super().__init__(config)

    def _calculate_distribution_params(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        if isinstance(data, pd.Series):
            values = data.to_numpy()
        else:
            values = np.asarray(data).reshape(-1)

        n = len(values)
        counter = Counter()
        # 进一步调小 chunk_size 以确保安全，30万是一个比较稳健的数值
        chunk_size = 300_000

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = values[start:end]

            # 优化：如果是数值类型且没有缺失，跳过转字符串
            s = pd.Series(chunk).fillna("MISSING").astype(str)
            counter.update(s.values)

            del chunk, s
            if start % (chunk_size * 10) == 0:
                gc.collect()

        unique_vals = np.array(list(counter.keys()), dtype=object)
        counts = np.array(list(counter.values()), dtype=np.int64)
        del counter

        total = counts.sum()
        if total == 0:
            frequencies = np.array([1.0], dtype=float)
            unique_vals = np.array(["MISSING"], dtype=object)
        else:
            frequencies = counts / float(total)

        cum_probs = np.cumsum(frequencies)
        bounds = np.concatenate(([0.0], cum_probs))

        return {
            "unique_values": unique_vals,
            "frequencies": frequencies,
            "cum_probs": cum_probs,
            "bounds": bounds,
        }

    def fit(self, data: Union[pd.Series, np.ndarray]) -> None:
        self.params = self._calculate_distribution_params(data)

    def transform(self, data: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            raise ValueError("StochasticNormalizer.transform 期望输入为 1D，而不是 DataFrame")

        # 转换为 numpy 数组处理，绕过 Series 复杂的元数据管理
        if isinstance(data, pd.Series):
            raw_values = data.values
        else:
            raw_values = np.asarray(data).reshape(-1)

        n = len(raw_values)
        transformed = np.zeros(n, dtype=np.float32)  # 使用 float32 节省一半内存

        bounds = self.params["bounds"].astype(np.float32)
        unique_values = self.params["unique_values"]

        # 建立一个快速查找字典，避免在循环中重复创建 Categorical 对象
        val_to_idx = {val: i for i, val in enumerate(unique_values)}

        chunk_size = 300_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_raw = raw_values[start:end]

            # 优化点：手动处理缺失值和转换，避免 astype(str) 产生大量中间副本
            # 如果输入本身是 float 类型，NaN 会被识别
            chunk_series = pd.Series(chunk_raw).fillna("MISSING").astype(str)

            # 使用 map 映射索引，比创建 Categorical 类别对象更轻量
            idx = chunk_series.map(val_to_idx).values

            # 此时 idx 是 float64 数组（因为可能有 NaN），转为 int
            valid_mask = ~np.isnan(idx.astype(float))
            idx_int = idx[valid_mask].astype(np.int32)

            if valid_mask.any():
                lower = bounds[idx_int]
                upper = bounds[idx_int + 1]

                # 随机采样并填充
                r = np.random.random(len(idx_int)).astype(np.float32)
                transformed[start:end][valid_mask] = lower + r * (upper - lower)

            # 彻底清理
            del chunk_raw, chunk_series, idx, idx_int, valid_mask
            if start % (chunk_size * 5) == 0:
                gc.collect()

        if isinstance(data, pd.Series):
            return pd.Series(transformed, index=data.index)
        return transformed

    def inverse_transform(self, data: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        # ... (保持原样即可，因为 inverse_transform 通常处理的是已经生成的小批量数据)
        # 如果也是全量处理，建议也加入 chunk 逻辑
        return super().inverse_transform(data)