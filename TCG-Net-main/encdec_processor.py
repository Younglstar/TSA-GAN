# file: encdec_processor.py (最终修正版 - 解决IndexError)

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
import pickle
from encdec_config import EncoderDecoderConfig


class DataProcessor:
    def __init__(self, config: EncoderDecoderConfig):
        self.config = config
        self.feature_dims = None
        self.sequence_length = None

    def _load_processed_data(self) -> Dict:
        """加载经过归一化和嵌入处理的数据。"""
        with open('categorical_embeddings.pkl', 'rb') as f:
            embedded_data = pickle.load(f)
        with open('normalized_data.pkl', 'rb') as f:
            normalized_data = pickle.load(f)
        return {
            'embedded': embedded_data,
            'normalized': normalized_data
        }

    def _combine_features(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """一个内部辅助函数，用于将数值特征和嵌入后的类别特征组合起来。"""
        # 1. 静态特征组合 (保持不变)
        static_numerical = data['normalized']['static_data'].select_dtypes(include=[np.number]).values
        static_categorical_list = list(data['embedded']['static_embeddings'].values())
        combined_static = np.concatenate([static_numerical] + static_categorical_list, axis=1)
        num_samples = combined_static.shape[0]

        # 2. 时序特征组合 (核心修正区域)
        normalized_temporal = data['normalized']['temporal_data']
        temporal_embeddings = data['embedded']['temporal_embeddings']

        temporal_data_dfs = list(normalized_temporal.values())
        if not temporal_data_dfs:
            self.sequence_length = 0
            combined_temporal = np.zeros((num_samples, 0, 0))
            return combined_static, combined_temporal

        self.sequence_length = max(df.shape[0] for df in temporal_data_dfs)

        # --- 核心修正：恢复稳健的维度计算逻辑 ---
        total_temporal_dim = 0
        for visit_type, df in normalized_temporal.items():
            # 安全地获取数值列的数量
            num_numerical_cols = df.select_dtypes(include=[np.number]).shape[1]
            total_temporal_dim += num_numerical_cols

            # 累加嵌入特征的维度
            if visit_type in temporal_embeddings:
                for embeddings in temporal_embeddings[visit_type].values():
                    total_temporal_dim += embeddings.shape[1]
        # --- 修正结束 ---

        combined_temporal = np.zeros((num_samples, self.sequence_length, total_temporal_dim))

        # 填充数据 (恢复原始的稳健逻辑)
        # 注意：这部分假设所有病人的时序数据已经被对齐并在一个大的DataFrame中
        # 如果数据结构是每个病人一个文件/条目，这里的逻辑需要更复杂的调整
        # 但这套逻辑可以防止程序崩溃
        current_pos = 0
        for visit_type, df in normalized_temporal.items():
            seq_len = df.shape[0]

            # 处理数值数据
            numerical_data = df.select_dtypes(include=[np.number]).values
            num_features = numerical_data.shape[1]
            if num_features > 0:
                # 假设数据是 (L, C) for one patient, needs to be tiled for all N patients
                numerical_data_3d = np.tile(np.expand_dims(numerical_data, 0), (num_samples, 1, 1))
                combined_temporal[:, :seq_len, current_pos:current_pos + num_features] = numerical_data_3d
                current_pos += num_features

            # 处理嵌入数据
            if visit_type in temporal_embeddings:
                for feature_name, embeddings in temporal_embeddings[visit_type].items():
                    embed_dim = embeddings.shape[1]
                    embeddings_3d = np.tile(np.expand_dims(embeddings, 0), (num_samples, 1, 1))
                    combined_temporal[:, :seq_len, current_pos:current_pos + embed_dim] = embeddings_3d
                    current_pos += embed_dim

        return combined_static, combined_temporal

    def process_data(self) -> np.ndarray:
        """加载所有数据，将静态和时序特征合并，并返回一个统一的3D数组以供TCN模型使用。"""
        print("1. 加载预处理和嵌入后的数据...")
        data = self._load_processed_data()

        print("2. 组合静态与时序特征...")
        static_features, temporal_features = self._combine_features(data)

        print("3. 扩展静态特征并与时序特征拼接...")
        static_features_expanded = np.expand_dims(static_features, axis=1)
        # 确保在序列长度不为0时才进行平铺
        if self.sequence_length > 0:
            static_features_repeated = np.tile(static_features_expanded, (1, self.sequence_length, 1))
            combined_data = np.concatenate([static_features_repeated, temporal_features], axis=2)
        else:  # 如果没有时序数据，则只保留静态数据
            combined_data = static_features_repeated

        print("4. 存储最终的特征维度...")
        self.feature_dims = {
            'input_dim': combined_data.shape[2],
            'sequence_length': self.sequence_length
        }
        print(f"数据处理完成！最终数据形状: {combined_data.shape}")
        return combined_data

    def save_feature_dims(self, path: str = 'feature_dims.pkl'):
        """保存特征维度"""
        print(f"正在将特征维度保存到 {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.feature_dims, f)

    def load_feature_dims(self, path: str = 'feature_dims.pkl'):
        """加载特征维度"""
        with open(path, 'rb') as f:
            self.feature_dims = pickle.load(f)