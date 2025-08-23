# data_loader.py (已修改，增强了通用性和灵活性)
import pandas as pd
from typing import Tuple
from config import DataConfig


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config
        self.data: pd.DataFrame = None
        self.static_data: pd.DataFrame = None
        # 注意：temporal_data 的类型从 Dict 变为 DataFrame
        self.temporal_data: pd.DataFrame = None

    def load_data(self) -> pd.DataFrame:
        """加载数据，验证列的存在，并转换日期列。"""
        print(f"1. 正在从 {self.config.INPUT_FILE} 加载数据...")
        try:
            self.data = pd.read_csv(self.config.INPUT_FILE)
        except FileNotFoundError:
            print(f"错误: 输入文件未找到 {self.config.INPUT_FILE}")
            raise

        # 验证所有在 config 中定义的列都存在于文件中
        required_cols = list(set(
            self.config.IDDATA_COLS +
            self.config.TIMEDATA_COLS +
            self.config.STATIC_FEATURES +
            self.config.TEMPORAL_FEATURES
        ))
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"输入文件中缺少以下必要的列: {missing_cols}")

        # 转换日期列为 datetime 对象
        for date_col in self.config.TIMEDATA_COLS:
            self.data[date_col] = pd.to_datetime(self.data[date_col])

        print("数据加载和日期转换成功。")
        return self.data

    def _get_subject_id_col(self) -> str:
        """获取并验证主体ID列。"""
        if not self.config.IDDATA_COLS or len(self.config.IDDATA_COLS) == 0:
            raise ValueError("配置错误: `ID_COLUMNS` 必须在 config.py 中定义且不能为空。")
        # 假设列表中的第一个ID列是主要的主体标识符
        return self.config.IDDATA_COLS[0]

    def split_static_temporal(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        (已重写以实现通用化)
        根据配置将数据拆分为静态和时序 DataFrame。

        返回:
            - static_data: 一个以主体ID为索引的 DataFrame，每行对应一个唯一的主体。
                           如果未定义静态特征，则该DataFrame除了索引外是空的。
            - temporal_data: 一个包含所有主体所有时序记录的 DataFrame。
        """
        if self.data is None:
            self.load_data()

        subject_id = self._get_subject_id_col()
        print(f"2. 使用主体ID '{subject_id}' 拆分数据...")

        # --- 静态数据提取 (通用化) ---
        if self.config.STATIC_FEATURES:
            print(f"提取静态特征: {self.config.STATIC_FEATURES}")
            # 选取ID和静态特征列，并根据ID去重，确保每个主体只有一行
            self.static_data = self.data[[subject_id] + self.config.STATIC_FEATURES].drop_duplicates(
                subset=[subject_id]).set_index(subject_id)
        else:
            print("未定义静态特征，将创建一个带索引的空DataFrame。")
            # 如果没有静态特征，创建一个以所有唯一主体ID为索引的空DataFrame
            unique_subjects = self.data[subject_id].unique()
            self.static_data = pd.DataFrame(index=unique_subjects)

        self.static_data.index.name = subject_id  # 确保索引有名称

        # --- 时序数据提取 (通用化) ---
        # 移除了'initial'/'followup'的特定逻辑

        # 从时序特征列表中排除掉元数据列（ID和Date），避免重复
        temporal_feature_cols = [
            feat for feat in self.config.TEMPORAL_FEATURES
            if feat not in self.config.IDDATA_COLS and feat not in self.config.IDDATA_COLS
        ]

        # 时序数据应包含元数据列和特征列
        temporal_cols = self.config.IDDATA_COLS + self.config.TIMEDATA_COLS + temporal_feature_cols
        print(f"提取时序特征: {temporal_feature_cols}")

        self.temporal_data = self.data[temporal_cols].copy()

        # 按主体和时间排序，确保时序的正确性
        if self.config.TIMEDATA_COLS:
            sort_by_cols = [subject_id, self.config.TIMEDATA_COLS[0]]
            print(f"正在按 {sort_by_cols} 对时序数据进行排序...")
            self.temporal_data = self.temporal_data.sort_values(by=sort_by_cols).reset_index(drop=True)

        print("数据拆分完成。")
        print(f"  - 静态数据形状: {self.static_data.shape}")
        print(f"  - 时序数据形状: {self.temporal_data.shape}")

        return self.static_data, self.temporal_data