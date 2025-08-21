# file: encdec_processor.py (最终修正版 - 解决内存错误)

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
        return {'embedded': embedded_data, 'normalized': normalized_data}

    def process_data(self) -> np.ndarray:
        """
        (已重写) 加载所有数据，将静态和时序特征合并，并返回一个统一的3D数组。
        """
        print("1. 加载预处理和嵌入后的数据...")
        data = self._load_processed_data()

        # --- 2. 组合静态特征 (逻辑保持不变) ---
        static_numerical = data['normalized']['static_data'].select_dtypes(include=[np.number]).values
        static_categorical_list = list(data['embedded']['static_embeddings'].values())
        combined_static = np.concatenate([static_numerical] + static_categorical_list, axis=1)
        num_samples, static_dim = combined_static.shape

        # --- 3. 组合时序特征 (核心修正) ---
        normalized_temporal = data['normalized']['temporal_data']
        temporal_embeddings = data['embedded']['temporal_embeddings']

        # a. 获取所有就诊类型 (例如 'initial', 'followup')
        visit_types = sorted(normalized_temporal.keys())
        self.sequence_length = len(visit_types)  # 序列长度现在是就诊类型的数量

        # b. 逐个就诊类型地组合特征
        temporal_features_per_visit = []
        for visit_type in visit_types:
            df_norm = normalized_temporal[visit_type]
            df_embed = temporal_embeddings[visit_type]

            # ... inside the for loop ...
            numerical_part = df_norm.select_dtypes(include=[np.number]).values

            # --- 核心修正 ---
            # 1. 先获取嵌入值的列表
            categorical_part_list = list(df_embed.values())

            # 2. 判断列表是否为空
            if categorical_part_list:
                # 如果不为空，正常进行拼接
                categorical_part = np.concatenate(categorical_part_list, axis=1)
            else:
                # 如果为空，说明这个就诊类型没有分类特征。
                # 我们创建一个“空”的数组，它有正确的行数，但有0列。
                num_samples_for_visit = numerical_part.shape[0]
                categorical_part = np.empty((num_samples_for_visit, 0))

            # 3. 最终拼接（这行代码现在是安全的）
            combined_visit_features = np.concatenate([numerical_part, categorical_part], axis=1)
            temporal_features_per_visit.append(combined_visit_features)

        # c. 将不同就诊类型的数据堆叠成一个3D张量
        # 形状: (样本数, 序列长度, 时序特征维度)
        combined_temporal = np.stack(temporal_features_per_visit, axis=1)
        temporal_dim = combined_temporal.shape[2]

        # --- 4. 扩展静态特征并与时序特征拼接 ---
        # a. 扩展静态特征 -> (样本数, 1, 静态特征维度)
        static_features_expanded = np.expand_dims(combined_static, axis=1)
        # b. 沿序列维度重复 -> (样本数, 序列长度, 静态特征维度)
        static_features_repeated = np.tile(static_features_expanded, (1, self.sequence_length, 1))

        # c. 最终拼接 -> (样本数, 序列长度, 静态维度 + 时序维度)
        combined_data = np.concatenate([static_features_repeated, combined_temporal], axis=2)

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
        with open(path, 'wb') as f: pickle.dump(self.feature_dims, f)

    def load_feature_dims(self, path: str = 'feature_dims.pkl'):
        """加载特征维度"""
        with open(path, 'rb') as f: self.feature_dims = pickle.load(f)