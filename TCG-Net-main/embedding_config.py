# embedding_config.py (已修改，移除了冗余的特征定义)
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EmbeddingConfig:
    """
    (已重写)
    本配置文件现在只包含与类别嵌入 *过程* 和 *模型* 相关的参数。
    需要被嵌入的具体类别特征列表将从主配置 (config.py) 中动态获取，
    以遵循单一数据源原则，避免配置冗余。
    """

    # 1. 嵌入维度参数
    # 这些参数可以根据特征的基数（唯一值的数量）进行调整
    DEFAULT_EMBEDDING_DIM: int = 8
    MIN_EMBEDDING_DIM: int = 4
    MAX_EMBEDDING_DIM: int = 32

    # 2. 训练超参数
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 100

    # 3. 嵌入模型结构参数 (例如，用于训练嵌入的预测网络)
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [64, 32])
    DROPOUT_RATE: float = 0.2

    # 4. 文件路径
    EMBEDDING_MODELS_FILE: str = 'embedding_models.pkl'
    EMBEDDINGS_FILE: str = 'categorical_embeddings.pkl'

    # 5. 用于存储类别到整数的映射
    # 这个字段由 CategoricalEmbeddingProcessor 内部使用和填充
    category_mappings: Dict = field(default_factory=dict, repr=False)