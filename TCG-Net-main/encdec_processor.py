# encdec_processor.py (修改后的版本)

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple

# 导入所有需要的配置文件
from config import DataConfig
from encdec_config import EncoderDecoderConfig
from embedding_config import EmbeddingConfig  # <-- 1. 导入 EmbeddingConfig


class DataProcessor:
    def __init__(self, encdec_config: EncoderDecoderConfig, data_config: DataConfig,
                 embed_config: EmbeddingConfig):  # <-- 2. 增加 embed_config 参数
        """
        (已重写) 构造函数现在接收所有相关的配置文件。
        """
        self.encdec_config = encdec_config
        self.data_config = data_config
        self.embed_config = embed_config  # <-- 3. 保存 embed_config
        self.feature_dims = None

    def _load_embedded_data(self) -> Dict:
        """从嵌入阶段加载处理好的数据。"""
        # 现在的 self.embed_config 已经存在，可以安全访问
        path = self.embed_config.EMBEDDINGS_FILE
        print(f"  -> 正在从 '{path}' 加载嵌入数据...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    # ... 类的其余部分 (process_data, save_feature_dims 等) 保持不变 ...
    def process_data(self) -> np.ndarray:
        """
        (核心重写逻辑)
        将经过归一化和嵌入的数据（静态和时序DataFrame）转换为一个
        适合输入到序列模型的、统一的3D NumPy 张量。
        """
        print("1. 加载嵌入后的数据...")
        data = self._load_embedded_data()
        static_data = data['static_data']
        temporal_data = data['temporal_data']

        subject_id_col = self.data_config.SUBJECT_ID_COL

        # --- 2. 准备静态数据 ---
        # 确保静态数据的索引是我们期望的主体ID
        if static_data.index.name != subject_id_col:
            # 如果数据中已有ID列，则设为索引；否则假定索引就是ID
            if subject_id_col in static_data.columns:
                static_data = static_data.set_index(subject_id_col)
            else:
                static_data.index.name = subject_id_col

        # 获取所有主体的有序列表
        subject_order = static_data.index.tolist()
        static_features_np = static_data.values
        num_subjects, static_dim = static_features_np.shape
        print(f"静态数据准备完成。发现 {num_subjects} 个独立主体，每个主体有 {static_dim} 个静态特征。")

        # --- 3. 准备时序数据 (分组、填充、堆叠) ---
        print("2. 正在将时序DataFrame转换为3D张量...")
        # 按主体ID分组
        print("Columns in temporal_data:", temporal_data.columns)

        grouped = temporal_data.groupby(subject_id_col)

        # 计算最大序列长度
        max_seq_len = grouped.size().max()
        # 从列名中排除元数据列，计算纯特征数量
        temporal_feature_cols = [col for col in temporal_data.columns if col not in self.data_config.TIMEDATA_COLS+self.data_config.IDDATA_COLS]
        temporal_dim = len(temporal_feature_cols)

        print(f"  - 数据中的最大序列长度为: {max_seq_len}")
        print(f"  - 每个时间步有 {temporal_dim} 个时序特征。")

        # 创建一个空的3D数组用于存放结果
        temporal_features_np = np.zeros((num_subjects, max_seq_len, temporal_dim))
        # 遍历每个主体，填充3D数组
        for i, subject_id in enumerate(subject_order):
            if subject_id in grouped.groups:
                subject_data = grouped.get_group(subject_id)
                # 丢弃元数据列，只保留特征
                subject_features = subject_data[temporal_feature_cols].values
                seq_len = len(subject_features)
                # 将该主体的数据放入3D数组，短于max_len的序列会自动被0填充
                temporal_features_np[i, :seq_len, :] = subject_features

        print("时序数据已成功转换为填充后的3D张量。")

        # --- 4. 扩展静态特征并与时序特征拼接 ---
        print("3. 正在合并静态和时序张量...")
        # a. 扩展静态特征 -> (主体数, 1, 静态特征维度)
        static_expanded = np.expand_dims(static_features_np, axis=1)
        # b. 沿序列维度重复 -> (主体数, 最大序列长度, 静态特征维度)
        static_repeated = np.tile(static_expanded, (1, max_seq_len, 1))

        # c. 最终拼接 -> (主体数, 最大序列长度, 静态维度 + 时序维度)
        combined_data = np.concatenate([static_repeated, temporal_features_np], axis=2)

        # --- 5. 存储最终的特征维度 ---
        print("4. 正在保存最终的数据维度信息...")
        self.feature_dims = {
            'input_dim': combined_data.shape[2],
            'sequence_length': max_seq_len,
            'num_total_features': static_dim + temporal_dim  # 用于因果头
        }
        self.save_feature_dims()

        print("\n--- 数据处理完成！---")
        print(f"最终输入模型的3D数据形状为: {combined_data.shape}")
        return combined_data

    def save_feature_dims(self):
        """保存特征维度信息到文件。"""
        path = self.encdec_config.FEATURE_DIMS_FILE
        print(f"正在将维度信息 {self.feature_dims} 保存到 {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.feature_dims, f)

    def load_feature_dims(self) -> Dict:
        """从文件加载特征维度信息。"""
        path = self.encdec_config.FEATURE_DIMS_FILE
        with open(path, 'rb') as f:
            self.feature_dims = pickle.load(f)
        return self.feature_dims