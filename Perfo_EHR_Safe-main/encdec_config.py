# encdec_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
# file: encdec_config.py (修改)

from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderDecoderConfig:
    # 移除了旧的MLP隐藏层维度 (ENCODER_HIDDEN_DIMS, DECODER_HIDDEN_DIMS)

    # 新增TCN每层的输出通道数配置
    TCN_CHANNELS: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    LATENT_DIM: int = 64
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    DROPOUT_RATE: float = 0.2

    # 更新模型和输出文件的名称，以避免与旧版本混淆
    MODEL_SAVE_PATH: str = 'tcn_encoder_decoder_model.pkl'
    ENCODED_DATA_PATH: str = 'tcn_encoded_data.pkl'
'''
def get_default_encoder_dims() -> List[int]:
    return [512, 256, 128]

def get_default_decoder_dims() -> List[int]:
    return [128, 256, 512]

@dataclass
class EncoderDecoderConfig:
    # Architecture dimensions
    ENCODER_HIDDEN_DIMS: List[int] = field(default_factory=get_default_encoder_dims)
    DECODER_HIDDEN_DIMS: List[int] = field(default_factory=get_default_decoder_dims)
    LATENT_DIM: int = 64
    
    # Training parameters
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    DROPOUT_RATE: float = 0.2
    
    # Model parameters
    USE_BATCH_NORM: bool = True
    ACTIVATION: str = 'relu'
    
    # File paths
    MODEL_SAVE_PATH: str = 'encoder_decoder_model.pkl'
    ENCODED_DATA_PATH: str = 'encoded_data.pkl'
    
    # Weights for different components in loss function
    RECONSTRUCTION_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            'static': 1.0,
            'temporal': 1.0,
            'mask': 1.0,
            'time': 0.1
        }
    )'''