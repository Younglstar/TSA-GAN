# config.py (已修改，增强了通用性和灵活性)

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """
    数据处理的统一配置类。

    该设计旨在通过清晰定义列的角色和类型，灵活地处理各种数据结构。
    你可以通过修改这里的列表来适配你的数据集，例如，如果你的数据没有静态特征，
    只需将 STATIC_NUMERICAL_FEATURES 和 STATIC_CATEGORICAL_FEATURES 保持为空列表即可。
    """

    # --- 1. 核心元数据列定义 ---
    # 这些列定义了数据集的基本结构，不作为模型的特征输入。

    # 主体ID列：用于唯一标识每个独立个体（例如：患者ID，股票代码，用户ID）。
    # 这是关联同一个体的所有记录的关键。
    SUBJECT_ID_COL: str ='Country' #'Name'

    # 时间戳列：表示时间序列数据中的时间点或序列顺序。
    TIMESTAMP_COL: str = 'Date'

    # --- 2. 特征列定义 ---
    # 在这里定义哪些列是特征，以及它们的具体类型。
    # 如果某种类型的特征在你的数据集中不存在，保留为空列表 `[]` 即可。

    # 静态特征 (Static Features): 对于同一个主体，这些特征值是固定不变的。
    STATIC_NUMERICAL_FEATURES: List[str] = field(default_factory=list)
    STATIC_CATEGORICAL_FEATURES: List[str] = field(default_factory=lambda:['Country'])

    # 时序特征 (Temporal Features): 这些特征值会随时间戳变化。
    TEMPORAL_NUMERICAL_FEATURES: List[str] = field(
        default_factory=lambda:['Confirmed','Recovered','Deaths'] #['Open', 'High', 'Low', 'Close', 'Volume']
    )
    TEMPORAL_CATEGORICAL_FEATURES: List[str] = field(default_factory=list)
    # 注意：在您的例子中 'Name' 是 ID，而不是时序特征，所以我已将其移至 SUBJECT_ID_COL。

    # --- 3. 文件路径定义 ---
    INPUT_FILE: str = 'countries-aggregated.csv'
    PROCESSED_FILE: str = 'processed_data.pkl'

    # --- 4. 辅助属性 (通常无需修改) ---
    # 这些属性会根据上面的定义自动生成组合列表，方便代码调用。

    @property
    def IDDATA_COLS(self) -> List[str]:
        """返回所有用于识别和排序的元数据列的列表。"""
        return [self.SUBJECT_ID_COL]

    @property
    def TIMEDATA_COLS(self) -> List[str]:
        """返回所有用于识别和排序的元数据列的列表。"""
        return [self.TIMESTAMP_COL]

    @property
    def STATIC_FEATURES(self) -> List[str]:
        """返回所有静态特征列的合并列表。"""
        return self.STATIC_NUMERICAL_FEATURES + self.STATIC_CATEGORICAL_FEATURES

    @property
    def TEMPORAL_FEATURES(self) -> List[str]:
        """返回所有时序特征列的合并列表。"""
        return self.TEMPORAL_NUMERICAL_FEATURES + self.TEMPORAL_CATEGORICAL_FEATURES

    @property
    def ALL_FEATURES(self) -> List[str]:
        """返回数据集中所有特征列的完整列表。"""
        return self.STATIC_FEATURES + self.TEMPORAL_FEATURES