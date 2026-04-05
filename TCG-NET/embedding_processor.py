# embedding_processor.py (已重写，实现动态、通用的嵌入逻辑)
import pandas as pd
import numpy as np
import torch
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm  # 引入tqdm来显示训练进度条

from config import DataConfig
from embedding_config import EmbeddingConfig
from embedding_model import CategoricalEmbedding, EmbeddingTrainer


class CategoricalEmbeddingProcessor:
    def __init__(self, embed_config: EmbeddingConfig, data_config: DataConfig):
        """
        (已重写) 构造函数现在接收两个配置文件。
        """
        self.embed_config = embed_config
        self.data_config = data_config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.embedding_models: Dict[str, CategoricalEmbedding] = {}
        self.embedding_trainers: Dict[str, EmbeddingTrainer] = {}

    def _get_embedding_dim(self, num_categories: int) -> int:
        """根据类别数量动态确定嵌入维度。"""
        # 使用"四次方根法则"作为启发式规则
        dim = int(np.ceil(num_categories ** 0.25))
        return min(self.embed_config.MAX_EMBEDDING_DIM, max(self.embed_config.MIN_EMBEDDING_DIM, dim))

    def _create_and_train_model(self, feature_name: str, data_series: pd.Series):
        """为单个特征创建LabelEncoder，创建并训练嵌入模型。"""
        # 1. 创建 LabelEncoder
        # 总是包含'MISSING'以处理未见过的类别或NaN
        unique_values = data_series.fillna('MISSING').astype(str).unique().tolist()
        if 'MISSING' not in unique_values:
            unique_values.append('MISSING')

        le = LabelEncoder().fit(unique_values)
        self.label_encoders[feature_name] = le
        self.embed_config.category_mappings[feature_name] = {
            'categories': le.classes_.tolist(),
            'num_categories': len(le.classes_)
        }
        print(f"  - 为特征 '{feature_name}' 创建了 LabelEncoder，包含 {len(le.classes_)} 个唯一类别。")

        # 2. 创建模型和训练器
        num_categories = len(le.classes_)
        embedding_dim = self._get_embedding_dim(num_categories)

        model = CategoricalEmbedding(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            hidden_dims=self.embed_config.HIDDEN_LAYERS.copy(),  # 传入副本避免reverse影响原配置
            dropout_rate=self.embed_config.DROPOUT_RATE
        )
        trainer = EmbeddingTrainer(model=model, learning_rate=self.embed_config.LEARNING_RATE)
        self.embedding_models[feature_name] = model
        self.embedding_trainers[feature_name] = trainer

        # 3. 准备数据并训练
        encoded_data = le.transform(data_series.fillna('MISSING').astype(str))
        tensor_data = torch.LongTensor(encoded_data)

        print(f"  - 正在为 '{feature_name}' 训练嵌入模型...")
        for epoch in tqdm(range(self.embed_config.NUM_EPOCHS), desc=f"  Training {feature_name}", leave=False):
            trainer.train_step(tensor_data)

        # 4. 获取嵌入向量
        with torch.no_grad():
            embeddings = model.get_embeddings(tensor_data).cpu().numpy()

        return embeddings

    def _fit_transform(self, data: pd.DataFrame, features: List[str], mask: pd.DataFrame = None):
        if data is None or data.empty or not features:
            if mask is None:
                mask = pd.DataFrame(1.0, index=data.index, columns=data.columns)
            return data, mask

        output_df = data.copy()

        # 对齐/初始化 mask
        if mask is None:
            output_mask = pd.DataFrame(1.0, index=output_df.index, columns=output_df.columns).astype("float32")
            for f in features:
                if f in output_df.columns:
                    output_mask[f] = (~output_df[f].isna()).astype("float32")
        else:
            output_mask = mask.reindex(index=output_df.index, columns=output_df.columns, fill_value=1.0).astype(
                "float32")

        for feature in features:
            if feature not in output_df.columns:
                print(f"  - 警告: 特征 '{feature}' 在数据中未找到，已跳过。")
                continue

            print(f"\n正在处理特征: '{feature}'...")

            feat_mask = output_mask[feature].astype("float32")  # 1/0

            # 训练并获取 embedding（训练仍用 fillna('MISSING') 没问题）
            embeddings = self._create_and_train_model(feature, output_df[feature])

            emb_cols = [f"{feature}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=emb_cols, index=output_df.index)

            # ✅ 缺失位置 embedding 置 0（关键）
            embedding_df = embedding_df.mul(feat_mask.values.reshape(-1, 1))

            # ✅ embedding mask 扩维：每个维度继承原列 mask
            embedding_mask_df = pd.DataFrame(
                np.repeat(feat_mask.values.reshape(-1, 1), embeddings.shape[1], axis=1),
                columns=emb_cols,
                index=output_df.index,
            ).astype("float32")

            # 同步更新 data 和 mask：drop 原列 + concat 新列
            output_df = pd.concat([output_df.drop(columns=[feature]), embedding_df], axis=1)
            output_mask = pd.concat([output_mask.drop(columns=[feature]), embedding_mask_df], axis=1)

            print(f"  - 特征 '{feature}' 已被替换为 {embeddings.shape[1]} 维的嵌入向量。")

        return output_df, output_mask

    def fit_transform_static_features(self, static_data: pd.DataFrame, static_mask: pd.DataFrame = None):
        categorical_features = self.data_config.STATIC_CATEGORICAL_FEATURES
        return self._fit_transform(static_data, categorical_features, static_mask)

    def fit_transform_temporal_features(self, temporal_data: pd.DataFrame, temporal_mask: pd.DataFrame = None):
        categorical_features = self.data_config.TEMPORAL_CATEGORICAL_FEATURES
        return self._fit_transform(temporal_data, categorical_features, temporal_mask)

    def save_models(self) -> None:
        """保存所有训练好的模型和编码器。"""
        save_dict = {
            'label_encoders': self.label_encoders,
            'embedding_models': {name: model.state_dict() for name, model in self.embedding_models.items()},
            'category_mappings': self.embed_config.category_mappings
        }
        with open(self.embed_config.EMBEDDING_MODELS_FILE, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_models(self) -> None:
        """加载模型和编码器。"""
        with open(self.embed_config.EMBEDDING_MODELS_FILE, 'rb') as f:
            save_dict = pickle.load(f)

        self.label_encoders = save_dict['label_encoders']
        self.embed_config.category_mappings = save_dict['category_mappings']

        # 重新创建模型实例并加载状态
        for feature, state_dict in save_dict['embedding_models'].items():
            num_cats = self.embed_config.category_mappings[feature]['num_categories']
            emb_dim = self._get_embedding_dim(num_cats)
            model = CategoricalEmbedding(
                num_categories=num_cats,
                embedding_dim=emb_dim,
                hidden_dims=self.embed_config.HIDDEN_LAYERS.copy(),
                dropout_rate=self.embed_config.DROPOUT_RATE
            )
            model.load_state_dict(state_dict)
            self.embedding_models[feature] = model