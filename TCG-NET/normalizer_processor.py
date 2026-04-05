# normalizer_processor.py (已修改，实现动态、通用的归一化逻辑)
import pandas as pd
import pickle
from config import DataConfig
from normalizer_config import NormalizerConfig
from stochastic_normalizer import StochasticNormalizer
from typing import Dict, Tuple, Optional
from utils.logging_utils import get_logger
logger = get_logger(__name__)
from utils.logging_utils import setup_logging
setup_logging()




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

    def normalize_static_features(
            self,
            static_data: pd.DataFrame,
            static_mask: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        归一化静态数值特征，并保留 mask：
        - mask: 1=观测，0=缺失
        - 缺失位置归一化后强制置 0
        """
        numerical_features = self.data_config.STATIC_NUMERICAL_FEATURES

        if static_data is None or static_data.empty:
            logger.info("静态数据为空，直接返回。")
            if static_mask is None:
                static_mask = pd.DataFrame(index=static_data.index, columns=static_data.columns).fillna(1.0)
            return static_data, static_mask

        # 对齐/构造 mask
        if static_mask is None:
            # 兜底：用 NaN 推断缺失
            static_mask = (~static_data.isna()).astype("float32")
        else:
            # 确保列对齐
            static_mask = static_mask.reindex(index=static_data.index, columns=static_data.columns,
                                              fill_value=1.0).astype("float32")

        if not numerical_features:
            logger.info("没有要归一化的静态数值特征，直接返回。")
            return static_data, static_mask

        normalized_static = static_data.copy()
        logger.info("将要归一化的静态数值特征: %s", numerical_features)

        for feature in numerical_features:
            if feature not in normalized_static.columns:
                logger.warning("特征 %s 在静态数据中未找到，已跳过", feature)
                continue

            logger.info("正在处理静态特征: %s", feature)
            normalizer = StochasticNormalizer(self.norm_config)

            # 归一化（可能会处理 NaN）
            norm_col = normalizer.fit_transform(normalized_static[feature])

            # 缺失位置归零（关键）
            feat_mask = static_mask[feature].astype("float32")
            norm_col = pd.Series(norm_col, index=normalized_static.index) * feat_mask

            normalized_static[feature] = norm_col

            # 保存参数
            param_key = f"static_{feature}"
            self.normalizers[param_key] = normalizer
            self.norm_config.normalization_params[param_key] = normalizer.params

        # 建议统一保存一次（避免循环内写文件）
        self.save_normalization_params()
        return normalized_static, static_mask

    def normalize_temporal_features(
            self,
            temporal_data: pd.DataFrame,
            temporal_mask: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        归一化时序数值特征，并保留 mask：
        - mask: 1=观测，0=缺失
        - 缺失位置归一化后强制置 0
        """
        numerical_features = self.data_config.TEMPORAL_NUMERICAL_FEATURES

        if temporal_data is None or temporal_data.empty:
            logger.info("时序数据为空，直接返回。")
            if temporal_mask is None:
                temporal_mask = pd.DataFrame(index=temporal_data.index, columns=temporal_data.columns).fillna(1.0)
            return temporal_data, temporal_mask

        # 对齐/构造 mask
        if temporal_mask is None:
            temporal_mask = (~temporal_data.isna()).astype("float32")
        else:
            temporal_mask = temporal_mask.reindex(index=temporal_data.index, columns=temporal_data.columns,
                                                  fill_value=1.0).astype("float32")

        if not numerical_features:
            logger.info("没有要归一化的时序数值特征，直接返回。")
            return temporal_data, temporal_mask

        available_features = [f for f in numerical_features if f in temporal_data.columns]
        if not available_features:
            logger.info("未找到任何可归一化的时序数值特征，直接返回。")
            return temporal_data, temporal_mask

        logger.info("开始归一化时序特征，共 %d 个特征", len(available_features))
        normalized_temporal = temporal_data.copy()

        for i, feature in enumerate(available_features, 1):
            logger.debug("处理特征 %d/%d: %s", i, len(available_features), feature)

            normalizer = StochasticNormalizer(self.norm_config)
            norm_col = normalizer.fit_transform(temporal_data[feature])

            # 缺失位置归零（关键）
            feat_mask = temporal_mask[feature].astype("float32")
            norm_col = pd.Series(norm_col, index=temporal_data.index) * feat_mask

            normalized_temporal[feature] = norm_col

            param_key = f"temporal_{feature}"
            self.normalizers[param_key] = normalizer
            self.norm_config.normalization_params[param_key] = normalizer.params

        missing_features = set(numerical_features) - set(available_features)
        if missing_features:
            logger.warning("下列特征在时序数据中未找到，已跳过: %s", sorted(missing_features))

        self.save_normalization_params()
        return normalized_temporal, temporal_mask

    def save_normalization_params(self) -> None:
        """将所有特征的归一化参数保存到文件。"""
        with open(self.norm_config.NORMALIZATION_PARAMS_FILE, 'wb') as f:
            # 现在保存的是 norm_config 内部的字典
            pickle.dump(self.norm_config.normalization_params, f)

    def load_normalization_params(self) -> None:
        """从文件加载归一化参数。"""
        with open(self.norm_config.NORMALIZATION_PARAMS_FILE, 'rb') as f:
            self.norm_config.normalization_params = pickle.load(f)

