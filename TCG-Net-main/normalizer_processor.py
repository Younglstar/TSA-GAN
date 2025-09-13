# normalizer_processor.py (已修改，实现动态、通用的归一化逻辑)
import pandas as pd
import pickle
from typing import Dict
from config import DataConfig
from normalizer_config import NormalizerConfig
from stochastic_normalizer import StochasticNormalizer


class DataNormalizationProcessor:
    def __init__(self, norm_config: NormalizerConfig, data_config: DataConfig):
        """
        (已重写) 构造函数现在接收两个配置文件。
        - norm_config: 包含归一化算法本身的参数。
        - data_config: 提供需要被归一化的特征列名 (单一事实来源)。
        """
        self.norm_config = norm_config
        self.data_config = data_config
        self.normalizers: Dict[str, StochasticNormalizer] = {}

    def normalize_static_features(self, static_data: pd.DataFrame) -> pd.DataFrame:
        """
        (已重写) 动态地归一化在 DataConfig 中定义的静态数值特征。
        """
        # 从 data_config 获取需要处理的列
        numerical_features = self.data_config.STATIC_NUMERICAL_FEATURES

        if not numerical_features or static_data.empty:
            print("没有要归一化的静态数值特征或静态数据为空，直接返回。")
            return static_data

        normalized_static = static_data.copy()
        print(f"将要归一化的静态数值特征: {numerical_features}")

        for feature in numerical_features:
            if feature in normalized_static.columns:
                print(f"  - 正在处理静态特征: '{feature}'")
                normalizer = StochasticNormalizer(self.norm_config)
                # fit_transform 会处理缺失值 (NaN)
                normalized_static[feature] = normalizer.fit_transform(normalized_static[feature])

                # 存储 normalizer 实例及其参数
                param_key = f'static_{feature}'
                self.normalizers[param_key] = normalizer
                normalizer.save_params(param_key)
            else:
                print(f"  - 警告: 特征 '{feature}' 在静态数据中未找到，已跳过。")

        return normalized_static
    def normalize_temporal_features(self, temporal_data: pd.DataFrame) -> pd.DataFrame:
        """
        (优化版) 批量归一化 DataConfig 中定义的时序数值特征。
        - 数值特征: 使用 StochasticNormalizer 进行归一化
        - 类别特征 & 元数据: 保留
        """

        numerical_features = self.data_config.TEMPORAL_NUMERICAL_FEATURES

        if not numerical_features or temporal_data.empty:
            print("没有要归一化的时序数值特征或时序数据为空，直接返回。")
            return temporal_data

        # 过滤出实际存在的数值特征，避免重复判断
        available_features = [f for f in numerical_features if f in temporal_data.columns]

        if not available_features:
            print("未找到任何可归一化的时序数值特征，直接返回。")
            return temporal_data

        print(f"将要归一化的时序数值特征: {available_features}")

        # 避免整体 copy，只对需要归一化的列单独 copy
        normalized_temporal = temporal_data.copy()

        # 循环处理，但减少对象创建
        for feature in available_features:
            print(f"  - 正在处理时序特征: '{feature}'")
            normalizer = StochasticNormalizer(self.norm_config)
            normalized_values = normalizer.fit_transform(temporal_data[feature].to_numpy())

            # 直接替换列，避免逐元素赋值
            normalized_temporal[feature] = normalized_values

            # 保存 normalizer 参数
            param_key = f'temporal_{feature}'
            self.normalizers[param_key] = normalizer
            normalizer.save_params(param_key)

        # 打印缺失特征一次性提示
        missing_features = set(numerical_features) - set(available_features)
        if missing_features:
            print(f"警告: 下列特征在时序数据中未找到，已跳过: {list(missing_features)}")

        return normalized_temporal
    def save_normalization_params(self) -> None:
        """将所有特征的归一化参数保存到文件。"""
        with open(self.norm_config.NORMALIZATION_PARAMS_FILE, 'wb') as f:
            # 现在保存的是 norm_config 内部的字典
            pickle.dump(self.norm_config.normalization_params, f)

    def load_normalization_params(self) -> None:
        """从文件加载归一化参数。"""
        with open(self.norm_config.NORMALIZATION_PARAMS_FILE, 'rb') as f:
            self.norm_config.normalization_params = pickle.load(f)
'''    def normalize_temporal_features(self, temporal_data: pd.DataFrame) -> pd.DataFrame:
        """
        (已重写) 动态地归一化在 DataConfig 中定义的时序数值特征。
        类别特征和元数据列将被保留，不做改动。
        """
        # 从 data_config 获取需要处理的列
        numerical_features = self.data_config.TEMPORAL_NUMERICAL_FEATURES

        if not numerical_features or temporal_data.empty:
            print("没有要归一化的时序数值特征或时序数据为空，直接返回。")
            return temporal_data

        normalized_temporal = temporal_data.copy()
        print(f"将要归一化的时序数值特征: {numerical_features}")

        for feature in numerical_features:
            if feature in normalized_temporal.columns:
                print(f"  - 正在处理时序特征: '{feature}'")
                normalizer = StochasticNormalizer(self.norm_config)
                normalized_temporal[feature] = normalizer.fit_transform(normalized_temporal[feature])

                param_key = f'temporal_{feature}'
                self.normalizers[param_key] = normalizer
                normalizer.save_params(param_key)
            else:
                print(f"  - 警告: 特征 '{feature}' 在时序数据中未找到，已跳过。")

        return normalized_temporal'''
