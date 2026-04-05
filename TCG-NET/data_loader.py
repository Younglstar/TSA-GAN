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
        self.static_mask: pd.DataFrame = None
        self.temporal_mask: pd.DataFrame = None

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

    def split_static_temporal(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        根据配置将数据拆分为静态和时序 DataFrame，并生成缺失掩码。
        掩码约定：1=未缺失（观测到），0=缺失

        返回:
            - static_data: index=subject_id，列=STATIC_FEATURES
            - temporal_data: 包含 ID/TIME 元数据列 + 时序特征列
            - static_mask: 与 static_data 对齐（同 index & columns）
            - temporal_mask: 与 temporal_data 对齐（同 columns），其中元数据列恒为1
        """
        if self.data is None:
            self.load_data()

        subject_id = self._get_subject_id_col()
        print(f"2. 使用主体ID '{subject_id}' 拆分数据...")

        # -----------------------
        # 1) 静态数据 + mask
        # -----------------------
        if self.config.STATIC_FEATURES:
            print(f"提取静态特征: {self.config.STATIC_FEATURES}")

            static_raw = (
                self.data[[subject_id] + self.config.STATIC_FEATURES]
                .drop_duplicates(subset=[subject_id])
                .set_index(subject_id)
            )

            self.static_data = static_raw.copy()
            # mask: 1=notna, 0=na
            self.static_mask = (~static_raw.isna()).astype("float32")

        else:
            print("未定义静态特征，将创建一个带索引的空DataFrame。")
            unique_subjects = self.data[subject_id].unique()
            self.static_data = pd.DataFrame(index=unique_subjects)
            self.static_mask = pd.DataFrame(index=unique_subjects)

        self.static_data.index.name = subject_id
        self.static_mask.index.name = subject_id

        # -----------------------
        # 2) 时序数据 + mask
        # -----------------------
        meta_cols = list(dict.fromkeys(self.config.IDDATA_COLS + self.config.TIMEDATA_COLS))  # 去重但保序
        temporal_feature_cols = [feat for feat in self.config.TEMPORAL_FEATURES if feat not in set(meta_cols)]

        temporal_cols = meta_cols + temporal_feature_cols
        print(f"提取时序特征: {temporal_feature_cols}")

        temporal_raw = self.data[temporal_cols].copy()

        # 按主体和时间排序，确保时序的正确性
        if self.config.TIMEDATA_COLS:
            sort_by_cols = [subject_id, self.config.TIMEDATA_COLS[0]]
            print(f"正在按 {sort_by_cols} 对时序数据进行排序...")
            temporal_raw = temporal_raw.sort_values(by=sort_by_cols).reset_index(drop=True)

        self.temporal_data = temporal_raw

        # mask：元数据列恒为 1；特征列按 notna 得到 1/0
        temporal_mask = pd.DataFrame(1.0, index=temporal_raw.index, columns=temporal_raw.columns).astype("float32")
        if temporal_feature_cols:
            temporal_mask[temporal_feature_cols] = (~temporal_raw[temporal_feature_cols].isna()).astype("float32")

        self.temporal_mask = temporal_mask

        print("数据拆分完成。")
        print(f"  - 静态数据形状: {self.static_data.shape}, 静态mask形状: {self.static_mask.shape}")
        print(f"  - 时序数据形状: {self.temporal_data.shape}, 时序mask形状: {self.temporal_mask.shape}")

        return self.static_data, self.temporal_data, self.static_mask, self.temporal_mask
